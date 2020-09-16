import unittest
from unittest import mock

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.successive_halving import SuccessiveHalving, _SuccessiveHalving
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.utils.io.traj_logging import TrajLogger

from .test_eval_utils import eval_challenger


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


def target_from_run_info(RunInfo):
    value_from_config = sum([a for a in RunInfo.config.get_dictionary().values()])
    return RunValue(
        cost=value_from_config,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=time.time(),
        endtime=time.time() + 1,
        additional_info={}
    )


class TestSuccessiveHalving(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 7, 'b': 11})
        self.config2 = Configuration(self.cs,
                                     values={'a': 13, 'b': 17})
        self.config3 = Configuration(self.cs,
                                     values={'a': 0, 'b': 7})
        self.config4 = Configuration(self.cs,
                                     values={'a': 29, 'b': 31})

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        # Create the base object
        self.SH = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            deterministic=False,
            run_obj_time=False,
            instances=[1, 2, 3, 4, 5],
            n_seeds=2,
            initial_budget=2,
            max_budget=5,
            eta=2,
        )

    def test_initialization(self):
        """Makes sure that a proper _SH is created"""

        # We initialize the SH with zero intensifier_instances
        self.assertEqual(len(self.SH.intensifier_instances), 0)

        # Add an instance to check the _SH initialization
        self.assertTrue(self.SH._add_new_instance(num_workers=1))

        # Parameters properly passed to _SH
        self.assertEqual(len(self.SH.intensifier_instances[0].inst_seed_pairs), 10)
        self.assertEqual(self.SH.intensifier_instances[0].initial_budget, 2)
        self.assertEqual(self.SH.intensifier_instances[0].max_budget, 5)

    def test_process_results_via_sourceid(self):
        """Makes sure source id is honored when deciding
        which _SH will consume the result/run_info"""
        # Mock the _SH so we can make sure the correct item is passed
        for i in range(10):
            self.SH.intensifier_instances[i] = mock.Mock()

        # randomly create run_infos and push into SH. Then we will make
        # sure they got properly allocated
        for i in np.random.choice(list(range(10)), 30):
            run_info = RunInfo(
                config=self.config1,
                instance=0,
                instance_specific="0",
                cutoff=None,
                seed=0,
                capped=False,
                budget=0.0,
                source_id=i,
            )

            # make sure results aren't messed up via magic variable
            # That is we check only the proper _SH has this
            magic = time.time()

            result = RunValue(
                cost=1,
                time=0.5,
                status=StatusType.SUCCESS,
                starttime=1,
                endtime=2,
                additional_info=magic
            )
            self.SH.process_results(
                run_info=run_info,
                incumbent=None,
                run_history=self.rh,
                time_bound=None,
                result=result,
                log_traj=False
            )

            # Check the call arguments of each sh instance and make sure
            # it is the correct one

            # First the expected one
            self.assertEqual(
                self.SH.intensifier_instances[i].process_results.call_args[1]['run_info'], run_info)
            self.assertEqual(
                self.SH.intensifier_instances[i].process_results.call_args[1]['result'], result)
            all_other_run_infos, all_other_results = [], []
            for j in range(len(self.SH.intensifier_instances)):
                # Skip the expected _SH
                if i == j:
                    continue
                if self.SH.intensifier_instances[j].process_results.call_args is None:
                    all_other_run_infos.append(None)
                else:
                    all_other_run_infos.append(
                        self.SH.intensifier_instances[j].process_results.call_args[1]['run_info'])
                    all_other_results.append(
                        self.SH.intensifier_instances[j].process_results.call_args[1]['result'])
            self.assertNotIn(run_info, all_other_run_infos)
            self.assertNotIn(result, all_other_results)

    def test_get_next_run_single_SH(self):
        """Makes sure that a single _SH returns a valid config"""

        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(30):
            intent, run_info = self.SH.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh,
                num_workers=1,
            )

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            if intent == RunInfoIntent.WAIT:
                break

            # Add the config to self.rh in order to make SH aware that this
            # config/instance was launched
            self.rh.add(
                config=run_info.config,
                cost=10,
                time=0.0,
                status=StatusType.RUNNING,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
            )

        # We should not create more _SH intensifier_instances
        self.assertEqual(len(self.SH.intensifier_instances), 1)

        # We are running with:
        # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
        # 'n_configs_in_stage': [2.0, 1.0],
        # This means we run int(2.5) + 2.0 = 4 runs before waiting
        self.assertEqual(i, 4)

    def test_get_next_run_dual_SH(self):
        """Makes sure that two  _SH can properly coexist and tag
        run_info properly"""

        # Everything here will be tested with a single _SH
        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(30):
            intent, run_info = self.SH.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh,
                num_workers=2,
            )

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            # Add the config to self.rh in order to make SH aware that this
            # config/instance was launched
            if intent == RunInfoIntent.WAIT:
                break
            self.rh.add(
                config=run_info.config,
                cost=10,
                time=0.0,
                status=StatusType.RUNNING,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
            )

        # We create a second sh intensifier_instances as after 4 runs, the _SH
        # number zero needs to wait
        self.assertEqual(len(self.SH.intensifier_instances), 2)

        # We are running with:
        # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
        # 'n_configs_in_stage': [2.0, 1.0],
        # This means we run int(2.5) + 2.0 = 4 runs before waiting
        # But we have 2 successive halvers now!
        self.assertEqual(i, 8)

    def test_add_new_instance(self):
        """Test whether we can add a _SH and when we should not"""

        # By default we do not create a _SH
        # test adding the first instance!
        self.assertEqual(len(self.SH.intensifier_instances), 0)
        self.assertTrue(self.SH._add_new_instance(num_workers=1))
        self.assertEqual(len(self.SH.intensifier_instances), 1)
        self.assertIsInstance(self.SH.intensifier_instances[0], _SuccessiveHalving)
        # A second call should not add a new _SH
        self.assertFalse(self.SH._add_new_instance(num_workers=1))

        # We try with 2 _SH active

        # We effectively return true because we added a new _SH
        self.assertTrue(self.SH._add_new_instance(num_workers=2))

        self.assertEqual(len(self.SH.intensifier_instances), 2)
        self.assertIsInstance(self.SH.intensifier_instances[1], _SuccessiveHalving)

        # Trying to add a third one should return false
        self.assertFalse(self.SH._add_new_instance(num_workers=2))
        self.assertEqual(len(self.SH.intensifier_instances), 2)

    def _exhaust_run_and_get_incumbent(self, sh, rh, num_workers=2):
        """
        Runs all provided configs on all intensifier_instances and return the incumbent
        as a nice side effect runhistory/stats are properly filled
        """
        challengers = [self.config1, self.config2, self.config3, self.config4]
        incumbent = None
        for i in range(100):
            try:
                intent, run_info = sh.get_next_run(
                    challengers=challengers,
                    incumbent=None, chooser=None, run_history=rh,
                    num_workers=num_workers,
                )
            except ValueError as e:
                # Get configurations until you run out of them
                print(e)
                break

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            if intent == RunInfoIntent.WAIT:
                break

            result = target_from_run_info(run_info)
            rh.add(
                config=run_info.config,
                cost=result.cost,
                time=result.time,
                status=result.status,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
            )
            incumbent, inc_perf = sh.process_results(
                run_info=run_info,
                incumbent=incumbent,
                run_history=rh,
                time_bound=100.0,
                result=result,
                log_traj=False,
            )
        return incumbent, inc_perf

    def test_parallel_same_as_serial_SH(self):
        """Makes sure we behave the same as a serial run at the end"""

        # Get the run_history for a _SH run:
        rh = RunHistory()
        stats = Stats(scenario=self.scen)
        stats.start_timing()
        _SH = _SuccessiveHalving(
            stats=stats,
            traj_logger=TrajLogger(output_dir=None, stats=stats),
            rng=np.random.RandomState(12345),
            deterministic=False,
            run_obj_time=False,
            instances=[1, 2, 3, 4, 5],
            n_seeds=2,
            initial_budget=2,
            max_budget=5,
            eta=2,
        )
        incumbent, inc_perf = self._exhaust_run_and_get_incumbent(_SH, rh)

        # Just to make sure nothing has changed from the _SH side to make
        # this check invalid:
        # We add config values, so config 3 with 0 and 7 should be the lesser cost
        self.assertEqual(incumbent, self.config3)
        self.assertEqual(inc_perf, 7.0)

        # Do the same for SH, but have multiple _SH in there
        # _add_new_instance returns true if it was able to add a new _SH
        # We call this method twice because we want 2 workers
        self.assertTrue(self.SH._add_new_instance(num_workers=2))
        self.assertTrue(self.SH._add_new_instance(num_workers=2))
        incumbent_psh, inc_perf_psh = self._exhaust_run_and_get_incumbent(self.SH, self.rh)
        self.assertEqual(incumbent, incumbent_psh)

        # This makes sure there is a single incumbent in SH
        self.assertEqual(inc_perf, inc_perf_psh)

        # We don't want to loose any configuration, and particularly
        # we want to make sure the values of _SH to SH match
        self.assertEqual(len(self.rh.data), len(rh.data))

        # We are comparing exhausted single vs parallel successive
        # halving runs. The number and type of configs should be the same
        # and is enforced as a dictionary key argument check. The number
        # of runs will be different ParallelSuccesiveHalving has 2 _SH intensifier_instances
        # yet we make sure that after exhaustion, the budgets a config was run
        # should match
        configs_sh_rh = {}
        for k, v in rh.data.items():
            config_sh = rh.ids_config[k.config_id]
            if config_sh not in configs_sh_rh:
                configs_sh_rh[config_sh] = []
            if v.cost not in configs_sh_rh[config_sh]:
                configs_sh_rh[config_sh].append(v.cost)
        configs_psh_rh = {}
        for k, v in self.rh.data.items():
            config_psh = self.rh.ids_config[k.config_id]
            if config_psh not in configs_psh_rh:
                configs_psh_rh[config_psh] = []
            if v.cost not in configs_psh_rh[config_psh]:
                configs_psh_rh[config_psh].append(v.cost)

        # If this dictionaries are equal it means we have all configs
        # and the values track the numbers and actual cost!
        self.assertDictEqual(configs_sh_rh, configs_psh_rh)


class Test_SuccessiveHalving(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 0, 'b': 100})
        self.config2 = Configuration(self.cs,
                                     values={'a': 100, 'b': 0})
        self.config3 = Configuration(self.cs,
                                     values={'a': 100, 'b': 100})
        self.config4 = Configuration(self.cs,
                                     values={'a': 0, 'b': 0})

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def test_init_1(self):
        """
            test parameter initializations for successive halving - instance as budget
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=False, run_obj_time=False,
            instances=[1, 2, 3], n_seeds=2, initial_budget=None, max_budget=None, eta=2)

        self.assertEqual(len(intensifier.inst_seed_pairs), 6)  # since instance-seed pairs
        self.assertEqual(len(intensifier.instances), 3)
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 6)
        self.assertListEqual(intensifier.n_configs_in_stage, [4.0, 2.0, 1.0])
        self.assertTrue(intensifier.instance_as_budget)
        self.assertTrue(intensifier.repeat_configs)

    def test_init_2(self):
        """
            test parameter initialiations for successive halving - real-valued budget
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=10, eta=2)

        self.assertEqual(len(intensifier.inst_seed_pairs), 1)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 10)
        self.assertListEqual(intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])
        self.assertListEqual(list(intensifier.all_budgets), [1.25, 2.5, 5., 10.])
        self.assertFalse(intensifier.instance_as_budget)
        self.assertFalse(intensifier.repeat_configs)

    def test_init_3(self):
        """
            test parameter initialiations for successive halving - real-valued budget, high initial budget
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=9, max_budget=10, eta=2)

        self.assertEqual(len(intensifier.inst_seed_pairs), 1)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 9)
        self.assertEqual(intensifier.max_budget, 10)
        self.assertListEqual(intensifier.n_configs_in_stage, [1.0])
        self.assertListEqual(list(intensifier.all_budgets), [10.])
        self.assertFalse(intensifier.instance_as_budget)
        self.assertFalse(intensifier.repeat_configs)

    def test_init_4(self):
        """
            test wrong parameter initializations for successive halving
        """

        # runtime as budget (no param provided)
        with self.assertRaisesRegex(ValueError,
                                    "requires parameters initial_budget and max_budget for intensification!"):
            _SuccessiveHalving(
                stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1])

        # eta < 1
        with self.assertRaisesRegex(ValueError, "eta must be greater than 1"):
            _SuccessiveHalving(
                stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1], eta=0)

        # max budget > instance-seed pairs
        with self.assertRaisesRegex(ValueError,
                                    "Max budget cannot be greater than the number of instance-seed pairs"):
            _SuccessiveHalving(
                stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1, 2, 3], initial_budget=1, max_budget=5, n_seeds=1)

    def test_top_k_1(self):
        """
            test _top_k() for configs with same instance-seed-budget keys
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1, 2], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config3, cost=3, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config3, cost=3, time=3,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config4, cost=0.5, time=0.5,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config4, cost=0.5, time=0.5,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        conf = intensifier._top_k(configs=[self.config1, self.config2, self.config3, self.config4],
                                  k=2, run_history=self.rh)

        # Check that config4 is also before config1 (as it has the lower cost)
        self.assertEqual(conf, [self.config4, self.config1])

    def test_top_k_2(self):
        """
            test _top_k() for configs with different instance-seed-budget keys
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1, 2], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=10, time=10,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)

        with self.assertRaisesRegex(ValueError, 'Cannot compare configs'):
            intensifier._top_k(configs=[self.config2, self.config1, self.config3],
                               k=1, run_history=self.rh)

    def test_top_k_3(self):
        """
            test _top_k() for not enough configs to generate for the next budget
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.CRASHED, instance_id=1, seed=None,
                    additional_info=None)
        configs = intensifier._top_k(configs=[self.config1], k=2, run_history=self.rh)

        # top_k should return whatever configuration is possible
        self.assertEqual(configs, [self.config1])

    def test_top_k_4(self):
        """
            test _top_k() for not enough configs to generate for the next budget
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None, run_obj_time=False,
            rng=np.random.RandomState(12345), eta=2, num_initial_challengers=4,
            instances=[1], initial_budget=1, max_budget=10)
        intensifier._update_stage(self.rh)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        self.rh.add(config=self.config3, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        self.rh.add(config=self.config4, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        intensifier.success_challengers.add(self.config1)
        intensifier.fail_challengers.add(self.config2)
        intensifier.fail_challengers.add(self.config3)
        intensifier.fail_challengers.add(self.config4)
        intensifier._update_stage(self.rh)
        self.assertEqual(intensifier.fail_chal_offset, 1)  # we miss one challenger for this round
        configs = intensifier._top_k(configs=[self.config1], k=2, run_history=self.rh)
        self.assertEqual(configs, [self.config1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None,
                    budget=intensifier.all_budgets[1], additional_info=None)
        intensifier.fail_challengers.add(self.config2)
        intensifier._update_stage(self.rh)
        self.assertEqual(intensifier.stage, 0)  # going back, since there are not enough to advance

    def test_get_next_run_1(self):
        """
            test get_next_run for a presently running configuration
        """
        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')

        taf.runhistory = self.rh

        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1, 2], initial_budget=1, max_budget=2, eta=2)

        # next challenger from a list
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertTrue(intensifier.new_challenger)

        # until evaluated, does not pick new challenger
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(intensifier.running_challenger, run_info.config)
        self.assertFalse(intensifier.new_challenger)

        # evaluating configuration
        self.assertIsNotNone(run_info.config)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
            log_traj=False,
        )
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        self.assertEqual(run_info.config, self.config2)
        self.assertEqual(len(intensifier.success_challengers), 1)
        self.assertTrue(intensifier.new_challenger)

    def test_get_next_run_2(self):
        """
            test get_next_run for higher stages of SH iteration
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1], initial_budget=1, max_budget=2, eta=2)

        intensifier._update_stage(run_history=None)
        intensifier.stage += 1
        intensifier.configs_to_run = [self.config1]

        # next challenger should come from configs to run
        intent, run_info = intensifier.get_next_run(
            challengers=None,
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(len(intensifier.configs_to_run), 0)
        self.assertFalse(intensifier.new_challenger)

    def test_update_stage(self):
        """
            test update_stage - initializations for all tracking variables
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1], initial_budget=1, max_budget=2, eta=2)

        # first stage update
        intensifier._update_stage(run_history=None)

        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.sh_iters, 0)
        self.assertEqual(intensifier.running_challenger, None)
        self.assertEqual(intensifier.success_challengers, set())

        # higher stages
        self.rh.add(self.config1, 1, 1, StatusType.SUCCESS)
        self.rh.add(self.config2, 2, 2, StatusType.SUCCESS)
        intensifier.success_challengers = {self.config1, self.config2}
        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.sh_iters, 0)
        self.assertEqual(intensifier.configs_to_run, [self.config1])

        # next iteration
        intensifier.success_challengers = {self.config1}
        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertIsInstance(intensifier.configs_to_run, list)
        self.assertEqual(len(intensifier.configs_to_run), 0)

    @unittest.mock.patch.object(_SuccessiveHalving, '_top_k')
    def test_update_stage_2(self, top_k_mock):
        """
            test update_stage - everything good is in state do not advance
        """

        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, initial_budget=1, max_budget=4, eta=2, instances=None)

        # update variables
        intensifier._update_stage(run_history=None)

        intensifier.success_challengers.add(self.config1)
        intensifier.success_challengers.add(self.config2)
        intensifier.do_not_advance_challengers.add(self.config3)
        intensifier.do_not_advance_challengers.add(self.config4)

        top_k_mock.return_value = [self.config1, self.config3]

        # Test that we update the stage as there is one configuration advanced to the next budget
        self.assertEqual(intensifier.stage, 0)
        intensifier._update_stage(run_history=None)
        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.configs_to_run, [self.config1])
        self.assertEqual(intensifier.fail_chal_offset, 1)
        self.assertEqual(len(intensifier.success_challengers), 0)
        self.assertEqual(len(intensifier.do_not_advance_challengers), 0)

        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, initial_budget=1, max_budget=4, eta=2, instances=None)

        # update variables
        intensifier._update_stage(run_history=None)

        intensifier.success_challengers.add(self.config1)
        intensifier.success_challengers.add(self.config2)
        intensifier.do_not_advance_challengers.add(self.config3)
        intensifier.do_not_advance_challengers.add(self.config4)

        top_k_mock.return_value = [self.config3, self.config4]

        # Test that we update the stage as there is no configuration advanced to the next budget
        self.assertEqual(intensifier.stage, 0)
        intensifier._update_stage(run_history=None)
        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.configs_to_run, [])
        self.assertEqual(intensifier.fail_chal_offset, 0)
        self.assertEqual(len(intensifier.success_challengers), 0)
        self.assertEqual(len(intensifier.do_not_advance_challengers), 0)

        top_k_mock.return_value = []

    def test_eval_challenger_1(self):
        """
           test eval_challenger with quality objective & real-valued budget
        """

        def target(x: Configuration, instance: str, seed: int, budget: float):
            return 0.1 * budget

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')
        taf.runhistory = self.rh

        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[None], initial_budget=0.25, max_budget=0.5, eta=2)
        intensifier._update_stage(run_history=None)

        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        self.rh.add(config=self.config2, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.25)
        self.rh.add(config=self.config3, cost=2, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.25)

        intensifier.success_challengers = {self.config2, self.config3}
        intensifier._update_stage(run_history=self.rh)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            incumbent=self.config1,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config2)
        self.assertEqual(inc_value, 0.05)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(list(self.rh.data.keys())[-1][0], self.rh.config_ids[self.config2])
        self.assertEqual(self.stats.inc_changed, 1)

    def test_eval_challenger_2(self):
        """
           test eval_challenger with runtime objective and adaptive capping
        """

        def target(x: Configuration, instance: str):
            if x['a'] == 100 or instance == 2:
                time.sleep(1.5)
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, cutoff=1,
            instances=[1, 2], initial_budget=1, max_budget=2, eta=2, instance_order=None)

        # config1 should be executed successfully and selected as incumbent
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            incumbent=None,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(self.stats.inc_changed, 1)

        # config2 should be capped and config1 should still be the incumbent

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(list(self.rh.data.values())[1][2], StatusType.CAPPED)

        # config1 is selected for the next stage and allowed to timeout since this is the 1st run for this instance
        intent, run_info = intensifier.get_next_run(
            challengers=[],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 3)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.TIMEOUT)

    @mock.patch.object(_SuccessiveHalving, '_top_k')
    def test_eval_challenger_capping(self, patch):
        """
            test eval_challenger with adaptive capping and all configurations capped/crashed
        """

        def target(x):
            if x['b'] == 100:
                time.sleep(1.5)
            if x['a'] == 100:
                raise ValueError('You shall not pass')
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime",
                                abort_on_first_run_crash=False)
        taf.runhistory = self.rh

        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, cutoff=1,
            instances=[1, 2], initial_budget=1, max_budget=2, eta=2, instance_order=None)

        for i in range(2):
            self.rh.add(config=self.config1, cost=.001, time=0.001,
                        status=StatusType.SUCCESS, instance_id=i + 1, seed=0,
                        additional_info=None)

        # provide configurations
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=self.config1,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.CRASHED)
        self.assertEqual(len(intensifier.success_challengers), 0)

        # provide configurations
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],
            chooser=None,
            incumbent=self.config1,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(self.stats.inc_changed, 0)
        self.assertEqual(list(self.rh.data.values())[3][2], StatusType.CAPPED)
        self.assertEqual(len(intensifier.success_challengers), 0)
        # top_k() should not be called since all configs were capped
        self.assertFalse(patch.called)

        # now the SH iteration should begin a new iteration since all configs were capped!
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(intensifier.stage, 0)

        # should raise an error as this is a new iteration but no configs were provided
        with self.assertRaisesRegex(ValueError, 'No configurations/chooser provided.'):
            intent, run_info = intensifier.get_next_run(
                challengers=None,
                chooser=None,
                incumbent=self.config1,
                run_history=self.rh
            )

    def test_eval_challenger_capping_2(self):
        """
            test eval_challenger for adaptive capping with all but one configuration capped
        """
        def target(x):
            if x['a'] + x['b'] > 0:
                time.sleep(1.5)
            return x['a']

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=False, cutoff=1,
            instances=[1, 2], n_seeds=2, initial_budget=1, max_budget=4, eta=2, instance_order=None)

        # first configuration run
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config4],
            chooser=None,
            incumbent=None,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config4)

        # remaining 3 runs should be capped
        for i in [self.config1, self.config2, self.config3]:
            intent, run_info = intensifier.get_next_run(
                challengers=[i],
                chooser=None,
                incumbent=inc,
                run_history=self.rh
            )
            result = eval_challenger(run_info, taf, self.stats, self.rh)
            inc, inc_value = intensifier.process_results(
                run_info=run_info,
                incumbent=inc,
                run_history=self.rh,
                time_bound=np.inf,
                result=result,
            )

        self.assertEqual(inc, self.config4)
        self.assertEqual(list(self.rh.data.values())[1][2], StatusType.CAPPED)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.CAPPED)
        self.assertEqual(list(self.rh.data.values())[3][2], StatusType.CAPPED)
        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.fail_chal_offset, 1)  # 2 configs expected, but 1 failure

        # run next stage - should run only 1 configuration since other 3 were capped
        # 1 runs for config1
        intent, run_info = intensifier.get_next_run(
            challengers=[],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        self.assertEqual(run_info.config, self.config4)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(intensifier.stage, 2)

        # run next stage with only config1
        # should go to next iteration since no more configurations left
        for _ in range(2):
            intent, run_info = intensifier.get_next_run(
                challengers=[],
                chooser=None,
                incumbent=inc,
                run_history=self.rh
            )
            self.assertEqual(run_info.config, self.config4)
            result = eval_challenger(run_info, taf, self.stats, self.rh)
            inc, inc_value = intensifier.process_results(
                run_info=run_info,
                incumbent=inc,
                run_history=self.rh,
                time_bound=np.inf,
                result=result,
            )

        self.assertEqual(inc, self.config4)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config4, only_max_observed_budget=True)), 4)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(intensifier.stage, 0)

        with self.assertRaisesRegex(ValueError, 'No configurations/chooser provided.'):
            intensifier.get_next_run(
                challengers=[],
                chooser=None,
                incumbent=inc,
                run_history=self.rh
            )

    def test_eval_challenger_3(self):
        """
            test eval_challenger for updating to next stage and shuffling instance order every run
        """

        def target(x: Configuration, instance: str):
            return (x['a'] + int(instance)) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[0, 1], instance_order='shuffle', eta=2,
            deterministic=True, cutoff=1)

        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.inst_seed_pairs, [(0, 0), (1, 0)])

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            incumbent=None,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.configs_to_run, [])
        self.assertEqual(intensifier.stage, 0)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.configs_to_run, [self.config1])  # Incumbent is promoted to the next stage
        self.assertEqual(intensifier.stage, 1)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)

        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 2)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(self.stats.inc_changed, 1)

        # For the 2nd SH iteration, we should still be able to run the old configurations again
        # since instance order is "shuffle"

        self.assertEqual(intensifier.inst_seed_pairs, [(1, 0), (0, 0)])

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(run_info.config, self.config2)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)), 2)

    def test_incumbent_selection_default(self):
        """
            test _compare_config for default incumbent selection design (highest budget so far)
        """
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=2, eta=2)
        intensifier.stage = 0

        # SH considers challenger as incumbent in first run in eval_challenger
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config1, incumbent=self.config1,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config1, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)

        # Adding a worse configuration
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config2, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config2, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)

        # Adding a better configuration, but the incumbent will only be changed on budget=2
        self.rh.add(config=self.config3, cost=0.5, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config3, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)
        self.rh.add(config=self.config3, cost=0.5, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config3, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config3)

        # Test that the state is only based on the runhistory
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1], initial_budget=1)
        intensifier.stage = 0

        # Adding a better configuration, but the incumbent will only be changed on budget=2
        self.rh.add(config=self.config4, cost=0.1, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config4, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config3)
        self.rh.add(config=self.config4, cost=0.1, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config4, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config4)

    def test_incumbent_selection_designs(self):
        """
            test _compare_config with different incumbent selection designs
        """

        # select best on any budget
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=2, eta=2, incumbent_selection='any_budget')
        intensifier.stage = 0

        self.rh.add(config=self.config1, instance_id=1, seed=None, budget=1,
                    cost=0.5, time=1, status=StatusType.SUCCESS, additional_info=None)
        self.rh.add(config=self.config1, instance_id=1, seed=None, budget=2,
                    cost=10, time=1, status=StatusType.SUCCESS, additional_info=None)
        self.rh.add(config=self.config2, instance_id=1, seed=None, budget=2,
                    cost=5, time=1, status=StatusType.SUCCESS, additional_info=None)

        # incumbent should be config1, since it has the best performance in one of the budgets
        inc = intensifier._compare_configs(incumbent=self.config2, challenger=self.config1,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config1, inc)
        # if config1 is incumbent already, it shouldn't change
        inc = intensifier._compare_configs(incumbent=self.config1, challenger=self.config2,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config1, inc)

        # select best on highest budget only
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=4, eta=2, incumbent_selection='highest_budget')
        intensifier.stage = 0

        # incumbent should not change, since there is no run on the highest budget,
        # though config3 is run on a higher budget
        self.rh.add(config=self.config3, instance_id=1, seed=None, budget=2,
                    cost=0.5, time=1, status=StatusType.SUCCESS, additional_info=None)
        self.rh.add(config=self.config4, instance_id=1, seed=None, budget=1,
                    cost=5, time=1, status=StatusType.SUCCESS, additional_info=None)
        inc = intensifier._compare_configs(incumbent=self.config4, challenger=self.config3,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config4, inc)
        self.assertEqual(self.stats.inc_changed, 0)

        # incumbent changes to config3 since that is run on the highest budget
        self.rh.add(config=self.config3, instance_id=1, seed=None, budget=4,
                    cost=10, time=1, status=StatusType.SUCCESS, additional_info=None)
        inc = intensifier._compare_configs(incumbent=self.config4, challenger=self.config3,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config3, inc)

    def test_count_running_instances_for_challenger(self):
        """
        Makes sure we return the proper number of running configs
        """
        def target(x):
            return 1
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')

        taf.runhistory = self.rh
        # select best on any budget
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=list(range(20)), initial_budget=3, max_budget=10, eta=2)

        # Get and process the first instance pair of config 1
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1, self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )

        # Add a check for empty run history
        self.assertEqual(len(self.rh.data), 0)
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 0)

        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
            log_traj=False,
        )
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 0)

        # This will get us the second instance of config 1
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )

        # Mark this run_info object as running and make sure that the intensifier can
        # access it
        self.rh.add(config=run_info.config,
                    instance_id=run_info.instance,
                    seed=run_info.seed,
                    budget=run_info.budget,
                    cost=10,
                    time=1,
                    status=StatusType.RUNNING,
                    additional_info=None)
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 1)

        # Now, when the intensifier is requested, to provide a new run,
        # it should provide a new instance, not the same one
        new_intent, new_run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(new_run_info.config, run_info.config)
        self.assertNotEqual(new_run_info.instance, run_info.instance)

        # Add another running instance
        self.rh.add(config=new_run_info.config,
                    instance_id=new_run_info.instance,
                    seed=new_run_info.seed,
                    budget=new_run_info.budget,
                    cost=10,
                    time=1,
                    status=StatusType.RUNNING,
                    additional_info=None)
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 2)

        # Get a new config to gain more confidence
        newer_intent, newer_run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertNotIn(newer_run_info, [new_run_info, run_info])

    def test_launched_all_configs_for_current_stage(self):
        """
        This check makes sure we can identify when all the current runs
        (config/instance/seed) pairs for a given stage have been launched
        """
        def target(x):
            return 1
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')

        taf.runhistory = self.rh
        # select best on any budget
        intensifier = _SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=list(range(10)), initial_budget=2, max_budget=10, eta=2)

        # So there are 2 instances per config.
        # self.stage=0
        # self.n_configs_in_stage=[4.0, 2.0, 1.0]
        # self.all_budgets=[ 2.5  5.  10. ]
        total_configs_in_stage = 4
        instances_per_stage = 2

        # get all configs and add them to list
        run_tracker = []
        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(total_configs_in_stage * instances_per_stage):
            intent, run_info = intensifier.get_next_run(
                challengers=challengers,
                chooser=None,
                run_history=self.rh,
                incumbent=None,
            )

            # All this runs are valid for this stage
            self.assertEqual(intent, RunInfoIntent.RUN)

            # Remove from the challengers, the launched configs
            challengers = [c for c in challengers if c != run_info.config]
            run_tracker.append((run_info.config, run_info.instance, run_info.seed))
            self.rh.add(config=run_info.config,
                        instance_id=run_info.instance,
                        seed=run_info.seed,
                        budget=run_info.budget,
                        cost=10,
                        time=1,
                        status=StatusType.RUNNING,
                        additional_info=None)

        # This will get us the second instance of config 1
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )

        # We have launched all runs, that are expected for this stage
        # not registered any, so for sure we have to wait
        # For all runs to be completed before moving to the next stage
        self.assertEqual(intent, RunInfoIntent.WAIT)


if __name__ == "__main__":
    unittest.main()
