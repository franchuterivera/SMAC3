import logging
import typing
from collections import Counter
from enum import Enum
import json

import numpy as np

from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import (
    InstSeedBudgetKey,
    RunInfo,
    RunHistory,
    RunValue,
    RunKey,
    StatusType
)
from smac.utils.io.traj_logging import TrajLogger
from smac.intensification.abstract_racer import (
    AbstractRacer,
    RunInfoIntent,
    _config_to_run_type,
)
from smac.optimizer.epm_configuration_chooser import EPMChooser


class EnsembleIntensifierStage(Enum):
    """Class to define different stages of intensifier"""
    RUN_NEW_CHALLENGER = 0
    # We found a very promising NEW configuration on RUN_NEW_CHALLENGER, but it is on a
    # very low repetition. We will prioritize giving it resources to move it to a high budget
    RUN_OLD_CHALLENGER_ON_HIGHER_REPEAT = 1
    # We toggle between RUN_NEW_CHALLENGER and investing resources in more repetitions
    INTENSIFY_MEMBERS_REPETITIONS = 2
    # When transitioning from level N-1 to N, CV repetitions are initially 0,
    # so we cannot trust really on this run. But if this configuration has better
    # performance than the old level N-1 we quickly make it reach the highest repetition
    # so ensemble selection can use it
    INTENSIFY_LEVEL_TO_HIGHEST_REPEAT = 3


class RobustEnsembleMembersIntensification(AbstractRacer):
    """Races challengers against a group of incumbents


    Parameters
    ----------
    stats: Stats
        stats object
    traj_logger: TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids. In this case repetitions to have a better estimate of the
        configuration
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : int
        runtime cutoff of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    use_ta_time_bound: bool,
        if true, trust time reported by the target algorithms instead of
        measuring the wallclock time for limiting the time of intensification
    run_limit : int
        Maximum number of target algorithm runs per call to intensify.
    min_chall: int
        How many configurations have to be available to start intensifying
    maxE : int
        Maximum number of incumbents to track. Can be mapped to the members of an ensemble
    """

    def __init__(
        self,
        stats: Stats,
        traj_logger: TrajLogger,
        rng: np.random.RandomState,
        instances: typing.List[str],
        instance_specifics: typing.Mapping[str, np.ndarray] = None,
        cutoff: int = None,
        deterministic: bool = False,
        run_obj_time: bool = True,
        run_limit: int = MAXINT,
        use_ta_time_bound: bool = False,
        maxE: int = 50,
        min_chall: int = 1,
        adaptive_capping_slackfactor: float = 1.2,
        performance_threshold_to_intensify_new_config: float = 1.00,
    ):
        # make sure instnaces are numeric
        self.instance2id = {element: i for i, element in enumerate(instances)}
        self.id2instance = {i: element for i, element in enumerate(instances)}
        self.lowest_level = min([json.loads(instance)['level'] for instance in instances])
        self.highest_level = max([json.loads(instance)['level'] for instance in instances])
        self.lowest_repeat = min([json.loads(instance)['repeats'] for instance in instances])
        self.highest_repeat = max([json.loads(instance)['repeats'] for instance in instances])

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         # Minimum number of repetitions is 1
                         minR=1,
                         # Highest repetition is the max run per config
                         maxR=max(list(self.id2instance.keys())),
                         min_chall=min_chall,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         )

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # Maximum number of ensemble members
        self.maxE = maxE

        # general attributes
        self.run_limit = run_limit

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")

        self.use_ta_time_bound = use_ta_time_bound
        self.elapsed_time = 0.

        self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
        self.challenger = None
        self.num_chall_run = 0
        self.chall_gen = iter([])

        # In case a new configuration looks promising
        # up to WORST_MEMBER*self.performance_threshold_to_intensify_new_config
        # we quickly intensify this config to a higher repetition
        # so that it can be used in the ensemble selection
        self.performance_threshold_to_intensify_new_config = performance_threshold_to_intensify_new_config

        if self.run_obj_time:
            raise NotImplementedError()
        self.logger.info(f"Intensifier instances={self.id2instance} for {self.maxE} incumbents")

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True,
                     num_workers: int = 1,
                     ) -> typing.Tuple[RunInfoIntent, RunInfo]:
        """
        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        intent: RunInfoIntent
            What should the smbo object do with the runinfo.
        run_info: RunInfo
            An object that encapsulates necessary information for a config run
        """
        if num_workers > 1:
            raise ValueError("Intensifier does not support more than 1 worker, yet "
                             "the argument num_workers to get_next_run is {}".format(
                                 num_workers
                             ))

        # If this function is called, it means the iteration is
        # not complete
        self.iteration_done = False

        # Run the given config recommended by BO model
        if self.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER:
            self.challenger = self._next_challenger(challengers=challengers,
                                                    chooser=chooser,
                                                    run_history=run_history,
                                                    repeat_configs=False)
            # Run the config in the new budget
            instance_id = 0
        elif self.stage == EnsembleIntensifierStage.RUN_OLD_CHALLENGER_ON_HIGHER_REPEAT:
            # Self.challenger stays the same. There is promise of a good result
            # with this configuration

            # Run the config in the new repetition
            instance_id = self.get_config_highest_instance(run_history=run_history,
                                                           challenger=self.challenger) + 1
        elif self.stage == EnsembleIntensifierStage.INTENSIFY_LEVEL_TO_HIGHEST_REPEAT:
            # Self.challenger stays the same. There is promise of a good result
            # with this configuration -- it just transitioned to a new level but repetitions of cv
            # are not high enough for it to be ensembled

            # Run the config in the new repetition
            instance_id = self.get_config_highest_instance(run_history=run_history,
                                                           challenger=self.challenger) + 1
        elif self.stage == EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS:
            ensemble_members = self.get_ensemble_members(run_history=run_history)
            repetitions = [r for l, r, c in ensemble_members]
            max_repetition = max(repetitions)
            if all([r == max_repetition for r in repetitions]) and max_repetition < max(list(self.id2instance.keys())):
                self.challenger = ensemble_members[0][2]
                instance_id = ensemble_members[0][1] + 1
            elif any([r < max_repetition for r in repetitions]):
                not_in_higest_repeat = [member for member in ensemble_members
                                        if member[1] < max_repetition]

                # List is sorted so prioritize good configs
                self.challenger = not_in_higest_repeat[0][2]
                instance_id = not_in_higest_repeat[0][1] + 1
            else:
                # All configs in maximum repetition, here search like crazy for a better
                # configuration
                self.challenger = None
                self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
        else:
            raise ValueError('No valid stage found!')

        # No new challengers are available for this iteration,
        # Move to the next iteration. This can happen
        # when all configurations for this iteration are exhausted
        # and have been run in all proposed instance/pairs.
        # Also, when we want to just keep running new configs until a more promising
        # configuration appear. This happens when all ensemble members spots
        # are taken and only a really new good config will make it's way in
        if self.challenger is None:
            return RunInfoIntent.SKIP, RunInfo(
                config=None,
                instance=None,
                instance_specific="0",
                seed=0,
                cutoff=self.cutoff,
                capped=False,
                budget=0.0,
            )

        if instance_id not in self.id2instance:
            for run_key, run_value in run_history.data.items():
                self.logger.error(f"{run_key}->{run_value}")
            raise ValueError(f"While at stage {self.stage} proposed to run {instance_id}/{self.id2instance}")

        if self.deterministic:
            seed = 0
        else:
            seed = self.rs.randint(low=0, high=MAXINT, size=1)[0]

        return RunInfoIntent.RUN, RunInfo(
            config=self.challenger,
            instance=self.id2instance[instance_id],
            instance_specific=self.instance_specifics.get(self.id2instance[instance_id], "0"),
            seed=seed,
            cutoff=self.cutoff,
            capped=False,
            budget=0.0,
        )

    def get_config_highest_instance(
        self,
        challenger: Configuration,
        run_history: RunHistory,
    ) -> int:
        runs_for_config = run_history.get_runs_for_config(config=challenger,
                                                          only_max_observed_budget=False)
        return max([self.instance2id[k.instance] for k in runs_for_config])

    def is_highest_instance_for_config(self, run_history, run_key) -> bool:
        """
        Returns true if the provided run_key corresponds to the
        highest instance available for a given configuration
        """
        max_instance = max([self.instance2id[key.instance_id] for key in run_history.data.keys()
                            # This is done for warm starting runhistory
                            if key.instance_id in self.instance2id and key.config_id == run_key.config_id])
        return max_instance == self.instance2id[run_key.instance_id]

    def get_ensemble_members(
        self,
        run_history: RunHistory,
    ) -> typing.List[typing.Tuple[float, int, Configuration]]:

        ensemble_members = []
        for run_key, run_value in run_history.data.items():
            # This means we have read a past run history
            if run_key.instance_id not in self.instance2id: continue
            if self.is_highest_instance_for_config(run_history, run_key):
                ensemble_members.append(
                    (run_value.cost, self.instance2id[run_key.instance_id], run_history.ids_config[run_key.config_id])
                )
        if len(ensemble_members) == 0:
            # No configs yet!
            return ensemble_members
        # returns a sorted list by loss
        # This is an ascending list, so lower loss is first with lowest repeat in case of tie
        ensemble_members = sorted(ensemble_members, key=lambda x: x[1])  # repetitions
        ensemble_members = sorted(ensemble_members, key=lambda x: x[0])  # loss
        return ensemble_members[:self.maxE]

    def process_results(self,
                        run_info: RunInfo,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float,
                        result: RunValue,
                        log_traj: bool = True,
                        ) -> \
            typing.Tuple[Configuration, float]:
        """

        During intensification, the following can happen:
        -> Challenger raced against incumbent
        -> Also, during a challenger run, a capped exception
           can be triggered, where no racer post processing is needed
        -> A run on the incumbent for more confidence needs to
           be processed, IntensifierStage.PROCESS_INCUMBENT_RUN
        -> The first run results need to be processed
           (PROCESS_FIRST_CONFIG_RUN)

        At the end of any run, checks are done to move to a new iteration.

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        incumbent : typing.Optional[Configuration]
            best configuration so far, None in 1st run
        run_history : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
             Contain the result (status and other methadata) of exercising
             a challenger/incumbent.
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """

        # Register the stage for debug printing
        old_stage = self.stage

        # State transition logic
        ensemble_members = self.get_ensemble_members(run_history=run_history)
        if self.stage in [EnsembleIntensifierStage.RUN_NEW_CHALLENGER,
                          EnsembleIntensifierStage.RUN_OLD_CHALLENGER_ON_HIGHER_REPEAT]:

            # Keep track of how many new challengers for the stats
            if self.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER:
                self.num_chall_run += 1

            repetitions = [r for l, r, c in ensemble_members]
            repetitions_on_max = [r for r in repetitions if r == max(list(self.id2instance))]

            if len([run_value for run_value in run_history.data.values() if run_value.status == StatusType.SUCCESS]) < self.min_chall:
                # Not enough challengers to start intensification of repetitions
                self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
            else:
                # Now we have enough challengers.
                # The question to answer is, do we keep repeating this challenger
                # or do we move to intensify new configuration

                # Get the highest loss to see if we have a promising configuration
                lower_bound_performance = max([l for l, r, c in ensemble_members]
                                              ) * self.performance_threshold_to_intensify_new_config
                # If this is a promising configuration, better than what we have as candidates.
                # But to actually use it in ensemble, we have to increase the repetitions
                losses = [l for l, r, c in ensemble_members]
                configs = [c for l, r, c in ensemble_members]
                try:
                    index_new_config = configs.index(run_info.config)
                except ValueError:
                    index_new_config = -1
                index_of_similar_loss = 0
                if len(configs) > 1:
                    # There is at least something to compare!
                    if index_new_config >= 0:
                        # This new config is so good that is on the ensemble members
                        # Even tho it is brand new
                        index_of_similar_loss = index_new_config - 1 if index_new_config > 0 else index_new_config + 1
                    else:
                        index_of_similar_loss = min(range(len(losses)),
                                                    key=lambda i: abs(losses[i] - result.cost))
                #self.logger.critical(f"c = {run_info.config.config_id} index_new_config={index_new_config} index_of_similar_loss={index_of_similar_loss} {result.cost} < {lower_bound_performance}")

                if result.cost < lower_bound_performance and self.instance2id[run_info.instance] < repetitions[index_of_similar_loss]:
                    # This means that the new config found in EnsembleIntensifierStage.RUN_NEW_CHALLENGER
                    # has better performance than the worst ensemble candidate, but sady
                    # is not going to be used by the ensemble selection as it is
                    # because we need it on a higher instance. So run this in a higher
                    # instance. What instance/repetition. Well it make sense to match it to
                    # the instance that has the most similar loss
                    self.stage = EnsembleIntensifierStage.RUN_OLD_CHALLENGER_ON_HIGHER_REPEAT
                elif len(ensemble_members) < self.maxE or len(repetitions_on_max) < self.maxE:
                    # We do not have all members yet or not all are on highest budget
                    # so we toggle between looking for new configs and repetition intensification
                    self.stage = EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS
                else:
                    # If reached this point:
                    # + all ensemble candidates have been found.
                    #   Also, they are all in the highest repetition.
                    # + Not enought number of configs to intensify repeats according to self.min_chall
                    # we want to keep finding new configurations
                    self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
        elif self.stage == EnsembleIntensifierStage.INTENSIFY_LEVEL_TO_HIGHEST_REPEAT:
            if not self.is_performance_better_than_lower_level(run_history, run_info, result) or self.is_instance_on_max_cv_repetition(run_info):
                # This configuration had a level transition, for example from level 1 to level 2
                # We transition a configuration sorted by loss. So if the loss is better than before
                # by all means we want to intensify it further. ONLY WHEN IT IS NOT GOOD ENOUGH ANYMORE
                # we continue with the default flow, in this case getting a new challenger
                # Ensemble builder will still use the old level=1 runs instead of level=2 as level=1 was better
                #
                # Also, just get it to the max repetition
                self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
        elif self.stage == EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS:
            if self.is_level_transition(run_info) and self.is_performance_better_than_lower_level(run_history, run_info, result):
                self.stage = EnsembleIntensifierStage.INTENSIFY_LEVEL_TO_HIGHEST_REPEAT
            else:
                # we already spend budget on repetition intensification,
                # toggle to find a better configuration
                self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER

        if incumbent is None:
            self.logger.info(
                "First run, no incumbent provided;"
                " challenger is assumed to be the incumbent"
            )
            incumbent = run_info.config

        self._ta_time += result.time
        self.num_run += 1

        incumbent = self._compare_configs(
            incumbent=incumbent, challenger=run_info.config,
            run_history=run_history,
            log_traj=log_traj)

        self.elapsed_time += (result.endtime - result.starttime)
        inc_perf = run_history.get_cost(incumbent)

        representation = "\n".join([str((l, i, self.id2instance[i], c.config_id)) for l, i, c in ensemble_members])
        self.logger.info(f"Ensemble Intensification \n{old_stage}->{self.stage}\n(loss, instance_id, instance, config_id): \n{representation}")

        return incumbent, inc_perf

    def is_level_transition(
        self,
        run_info: RunInfo,
    ) -> bool:
        """
        From a config we can get the instance, and from the instance we can
        see if we have a level transition. Level transition only counts if
        we moved to a new level (we are not the lowest level) and cv repeats
        are 0

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated

        Returns
        -------
        bool
            If this instance means that we have a level transition
        """
        instance_dict = json.loads(run_info.instance)
        level = instance_dict['level']
        repeat = instance_dict['repeats']
        return repeat == self.lowest_repeat and level > self.lowest_level

    def is_performance_better_than_lower_level(
        self,
        run_history: RunHistory,
        run_info: RunInfo,
        result: RunValue,
    ) -> bool:
        """
        What challenger to run next!

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        run_history : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        result: RunValue
             Contain the result (status and other methadata) of exercising
             a challenger/incumbent.

        Returns
        -------
        bool
            If the performance on this level is better that on the past level
        """
        desired_level = json.loads(run_info.instance)['level'] - 1
        for instance in self.instances:
            instance_dict = json.loads(instance)
            if int(instance_dict['level']) != int(desired_level):
                continue
            if int(instance_dict['repeats']) != int(self.highest_repeat):
                continue
            k = RunKey(run_history.config_ids[run_info.config], instance, run_info.seed, run_info.budget)
            if k not in run_history.data:
                # Exit the for loop to trigger the failure
                break
            # lower is better!!!
            return run_history.data[k].cost > result.cost
        raise ValueError(f"For RH={run_history.data.items()} and run_info={run_info} could not find lower score")

    def is_instance_on_max_cv_repetition(
        self,
        run_info: RunInfo,
    ) -> bool:
        """
        Is this run_info pointing to a config with max CV repetitions

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated

        Returns
        -------
        bool
            If this config is on the max repetition available
        """
        return json.loads(run_info.instance)['repeats'] == self.highest_repeat
