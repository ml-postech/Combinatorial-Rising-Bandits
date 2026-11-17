"""Exercise released dispatch with recording stubs and bounded real Toy runs."""
import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'src/main.py'
NAMES = ['CRUCB', 'RUCB', 'SW_CUCB', 'SW_CTS', 'SW_UCB', 'SW_TS']
EPSILONS = ['0.0001', '0.01', '0.05', '0.125', '0.25', '0.4', '0.5']
TREE = ast.parse(SCRIPT.read_text())
BRANCH = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
              and any(isinstance(c, ast.Name) and c.id == 'algo' for c in ast.walk(n.test)))


def dispatch(name, task):
    calls, saved = [], {}
    env = object()

    class Runner:
        def run(self):
            return np.ones(1001), np.zeros((5, 1001))

    def constructor(algorithm):
        def record(*args, **kwargs):
            calls.append((algorithm, args, kwargs))
            return Runner()
        return record

    ns = {
        'algo': name, 'args': SimpleNamespace(T=1001), 'env': env, 'task': task,
        'np': SimpleNamespace(savetxt=lambda path, data: saved.update({path: data.copy()})),
        'seed': 3, 'i': 2, 'easy': True,
        'optimal_cumulative_rewards': np.full(1001, 3.0),
        'regret': {task: {name: np.zeros((3, 1001))}},
    }
    ns.update({name: constructor(name) for name in NAMES})
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=[BRANCH], type_ignores=[]), str(SCRIPT), 'exec'), ns)
    return calls, saved, env


class DispatchTests(unittest.TestCase):
    def test_constructor_and_outputs(self):
        for task in ['Toy', 'Path', 'Matching', 'Spanning', 'Max']:
            for name in NAMES + ['CRUCB_' + e for e in EPSILONS]:
                with self.subTest(task=task, algorithm=name):
                    calls, saved, env = dispatch(name, task)
                    parameterized = name.startswith('CRUCB_')
                    algorithm = 'CRUCB' if parameterized else name
                    kwargs = {'epsilon': float(name.split('_')[1])} if parameterized else {}
                    self.assertEqual(calls, [(algorithm, (1001, env, task), kwargs)])
                    if parameterized:
                        epsilon = kwargs['epsilon']
                        stem = '{}_CRUCB_{}_203_True_b'.format(task, epsilon)
                        regret_name = './results/regrets_{}_CRUCB_203_0.01_{}_True_b.txt'.format(task, epsilon)
                    else:
                        stem = '{}_{}_203_True'.format(task, name)
                        regret_name = './results/regrets_{}_{}_203_0.01_True.txt'.format(task, name)
                    arms = './results/selected_arms_' + stem + '.txt'
                    rewards = './results/cumulative_rewards_' + stem + '.txt'
                    self.assertEqual(set(saved), {arms, rewards, regret_name})
                    self.assertTrue(np.issubdtype(saved[arms].dtype, np.integer))
                    np.testing.assert_array_equal(saved[rewards], np.ones(1001))
                    expected = np.zeros((3, 1001))
                    expected[2] = 2.0
                    np.testing.assert_array_equal(saved[regret_name], expected)

    def test_invalid_identifiers(self):
        for name in ['unknown', 'SW_bad', 'CRUCB_bad', 'CRUCB_', 'CRUCB_0.1_extra',
                     'RUCB_0.2', 'SW_CUCB_0.2', 'main']:
            with self.subTest(algorithm=name):
                with self.assertRaises(ValueError):
                    dispatch(name, 'Toy')

    def test_bounded_toy_runs(self):
        # Same task, easy mode and named algorithms as scripts/Toy.sh; bounded horizon.
        for names in [['CRUCB', 'SW_CUCB', 'RUCB'], ['SW_CTS', 'SW_UCB', 'SW_TS', 'CRUCB_0.125']]:
            with self.subTest(algorithms=names), tempfile.TemporaryDirectory() as tmp:
                result_dir = Path(tmp) / 'results'
                result_dir.mkdir()
                command = [sys.executable, '-B', str(SCRIPT), '--easy', 'True', '--tasks', 'Toy',
                           '--algos'] + names + ['--n_random', '1', '--T', '1001']
                result = subprocess.run(command, cwd=tmp, capture_output=True, text=True,
                                        env=dict(os.environ, MPLBACKEND='Agg'), timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                optimal = np.loadtxt(result_dir / 'optimal_cumulative_rewards_Toy_optimal_True_b.txt')
                for name in names:
                    if name.startswith('CRUCB_'):
                        stem = 'Toy_CRUCB_0.125_0_True_b'
                        regret_file = 'regrets_Toy_CRUCB_0_0.01_0.125_True_b.txt'
                    else:
                        stem = 'Toy_{}_0_True'.format(name)
                        regret_file = 'regrets_Toy_{}_0_0.01_True.txt'.format(name)
                    rewards = np.loadtxt(result_dir / ('cumulative_rewards_' + stem + '.txt'))
                    arms = np.loadtxt(result_dir / ('selected_arms_' + stem + '.txt'))
                    regrets = np.loadtxt(result_dir / regret_file)
                    self.assertEqual(rewards.shape, (1001,))
                    self.assertEqual(arms.shape[-1], 1001)
                    self.assertTrue(np.isfinite(rewards).all())
                    np.testing.assert_allclose(regrets, optimal - rewards)


if __name__ == '__main__':
    unittest.main()
