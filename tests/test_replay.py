"""Scientific invariants: held-out data, controlled interventions, and exact valuation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

from skill_forgetting.experiments.common import DOMAINS, SUBSETS, digest, read_rows, write_json, write_rows
from skill_forgetting.experiments.prepare import prepare, problem_id
from skill_forgetting.experiments.report import shapley, analyze, collect
from skill_forgetting.experiments.scoring import judge, score
from skill_forgetting.experiments.slurm import scripts, submit
from skill_forgetting.math.plain import sft_example
from skill_forgetting.math.evaluate import build_prompt
from skill_forgetting.training.single_sft import SFTSpec


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = json.loads((Path(__file__).parents[1] / 'configs/replay_shapley_pilot.json').read_text())
        self.config.update(validation_per_domain=5, stage_samples=12, branch_samples=12,
                           replay_per_domain=2, stage_steps=3, branch_steps=3,
                           artifact_root=str(self.root / 'artifacts'), run_root=str(self.root / 'run'))
        self.raw = self.root / 'raw'
        for domain in DOMAINS:
            train = [{'problem': f'{domain} problem {i}', 'solution': r'\boxed{1}',
                      'type': domain, 'level': f'Level {i%5+1}'} for i in range(30)]
            test = [{'problem': f'{domain} test {i}', 'solution': r'\boxed{2}',
                     'type': domain, 'level': 'Level 1'} for i in range(3)]
            # A native-test duplicate must be excluded from training, without altering test.
            train.append(test[0])
            source = self.raw / 'per_type_data_with_raw' / domain
            write_rows(source / 'train_raw.jsonl', train)
            write_rows(source / 'test_raw.jsonl', test)
        self.run = self.root / 'run'
        self.manifest = prepare(self.config, self.raw, self.run)

    def test_validation_and_native_test_are_excluded_from_every_training_branch(self):
        held_out = set()
        for domain in DOMAINS:
            split = self.run / 'splits' / domain
            held_out.update(map(problem_id, read_rows(split / 'validation_raw.jsonl')))
            held_out.update(map(problem_id, read_rows(split / 'test_raw.jsonl')))
            self.assertEqual(read_rows(split / 'test_raw.jsonl'), read_rows(
                self.raw / 'per_type_data_with_raw' / domain / 'test_raw.jsonl'))
            self.assertEqual(self.manifest['counts'][domain]['excluded_train_duplicates'], 1)
            self.assertFalse(held_out & set(map(problem_id, read_rows(self.run / 'plain_sft' / domain / 'rows.jsonl'))))
        for subset in SUBSETS:
            rows = read_rows(self.run / 'branches' / subset / 'data' / 'rows.jsonl')
            self.assertEqual(len(rows), 12)
            self.assertFalse(held_out & set(map(problem_id, rows)))

    def test_only_the_added_players_fixed_slots_change(self):
        slots = self.manifest['slots']
        branches = {s: read_rows(self.run / 'branches' / s / 'data' / 'rows.jsonl') for s in SUBSETS}
        self.assertFalse(set(slots['A']) & set(slots['B']))
        for smaller, larger, added in [('empty','A','A'), ('A','AB','B'), ('AB','ABC','C'), ('B','BC','C')]:
            changed = {i for i, (left, right) in enumerate(zip(branches[smaller], branches[larger]))
                       if left['example_id'] != right['example_id']}
            self.assertEqual(changed, set(slots[added]))
        for position in slots['A']:
            self.assertEqual(branches['A'][position], branches['ABC'][position])
        seen = set(map(problem_id, read_rows(self.run / 'plain_sft' / DOMAINS[0] / 'rows.jsonl')))
        self.assertTrue(set(map(problem_id, read_rows(self.run / 'replay' / 'A.jsonl'))) <= seen)

    def test_seed_reproduces_split_data_and_positions(self):
        other = self.root / 'again'
        manifest = prepare(self.config, self.raw, other)
        self.assertEqual(manifest, self.manifest)
        with self.assertRaises(FileExistsError):
            prepare(self.config, self.raw, self.run)

    def test_training_and_inference_share_user_content(self):
        row = {'problem': '  Find x.  ', 'solution': r'\boxed{2}', 'type': ' Algebra ', 'level': ' Level 1 '}
        sft = sft_example(row)
        self.assertEqual(build_prompt(row), sft['instruction'] + '\n' + sft['input'])

    def test_branch_command_has_fixed_steps_seed_and_no_merge(self):
        spec = SFTSpec(model='checkpoint-3', data_dir=self.run / 'branches/A/data',
                       output_dir=self.root / 'adapter', max_steps=17, seed=73)
        command = spec.train_command()
        for flag, value in [('--max_steps','17'), ('--seed','73'), ('--data_seed','73'), ('--packing','false'),
                            ('--train_on_prompt','false')]:
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIsNone(spec.merge_dir)
        with self.assertRaises(ValueError):
            spec.merge_command()

    def test_exact_shapley_shares_pair_synergy_and_third_order_effect(self):
        def value(subset):
            active = set() if subset == 'empty' else set(subset)
            return .1 + sum({'A':.03,'B':.02,'C':-.01}[p] for p in active) + \
                (.06 if {'A','B'} <= active else 0) + (.03 if len(active)==3 else 0)
        values = {s:value(s) for s in SUBSETS}
        phi = shapley(values)
        for player, expected in {'A':.07,'B':.06,'C':0.0}.items():
            self.assertAlmostEqual(phi[player], expected)
        summary = analyze(values, 64)
        self.assertAlmostEqual(summary['pair_interactions']['AB'], .06)
        self.assertAlmostEqual(summary['third_order_interaction'], .03)
        self.assertEqual(summary['best_observed_at_fixed_replay_budget']['128']['subsets'], ['AB'])
        with self.assertRaises(ValueError):
            shapley({'empty':0})

    def test_equivalent_math_and_empty_answer(self):
        verdict = judge(r'\boxed{0.5}', r'\boxed{\frac{1}{2}}')
        self.assertTrue(verdict['correct'])
        self.assertFalse(verdict['legacy_correct'])
        self.assertTrue(verdict['math_verify_correct'])
        self.assertFalse(judge('',r'\boxed{1}')['correct'])
        self.assertFalse(judge(r'\boxed{2}',r'\boxed{1}')['correct'])

    def test_scoring_rejects_missing_predictions_and_test_inputs(self):
        validation = self.run / 'validation_raw.jsonl'
        expected = read_rows(validation)
        rows = [{'example_id':r['example_id'],'domain':r['domain'],'gold_solution':r['solution'],
                 'prediction':r'\boxed{1}','finish_reason':'stop'} for r in expected]
        prediction = self.root / 'predictions.jsonl'
        write_rows(prediction, rows)
        result = score(prediction, validation, self.root / 'score')
        self.assertEqual(result['v'],1)
        write_rows(prediction, rows[:-1])
        with self.assertRaises(ValueError):
            score(prediction, validation, self.root / 'bad')
        with self.assertRaises(ValueError):
            score(prediction, self.run / 'splits/algebra/test_raw.jsonl', self.root / 'bad')

    def test_truncated_reasoning_does_not_credit_incidental_numbers(self):
        unfinished = r'The counts must be 2 and 3. The arrangement is $G\_\_\_'
        verdict = judge(unfinished, r'\boxed{3}', finish_reason='length')
        self.assertFalse(verdict['correct'])
        self.assertTrue(verdict['incomplete_terminal_answer'])
        # An explicit completed answer can survive later generation truncation.
        self.assertTrue(judge(r'\boxed{1/2}', r'\boxed{0.5}', finish_reason='length')['correct'])
        complete = ('Since there are 17 teams, there must be 16 games played in order to determine a winner. '
                    'Each game eliminates one team, and we need to eliminate 16 teams in order to determine a winner.')
        self.assertTrue(judge(complete, r'\boxed{16}', finish_reason='stop')['correct'])
        validation = self.run / 'validation_raw.jsonl'
        expected = read_rows(validation)
        rows = [{'example_id': r['example_id'], 'domain': r['domain'], 'gold_solution': r['solution'],
                 'prediction': r'We have counts of 2 and 1. The arrangement is $G\_\_',
                 'finish_reason': 'length'} for r in expected]
        predictions = self.root / 'unfinished.jsonl'
        write_rows(predictions, rows)
        result = score(predictions, validation, self.root / 'unfinished_score')
        self.assertEqual(result['v'], 0)

    def test_report_refuses_missing_real_scores(self):
        with self.assertRaises(FileNotFoundError):
            collect(self.run)

    def test_slurm_respects_shared_partition_and_one_gpu_per_branch(self):
        rendered = scripts(self.config, self.root/'config.json', self.root, self.run)
        self.assertIn('--partition=acd_u', rendered['branches'])
        self.assertIn('--gres=gpu:1', rendered['branches'])
        self.assertIn('--array=0-7%8', rendered['branches'])
        self.assertIn('--gres=gpu:1', rendered['report'])
        self.assertIn('HF_HUB_OFFLINE=1', rendered['baseline'])

    def test_smoke_success_gates_the_entire_job_chain(self):
        config = copy.deepcopy(self.config)
        config.update(train_python=sys.executable, eval_python=sys.executable,
                      model=str(self.root/'model'))
        write_json(self.root/'model/config.json', {})
        write_json(self.root/'config.json', config)
        manifest = copy.deepcopy(self.manifest)
        manifest['config'] = config
        write_json(self.run/'manifest.json', manifest)
        commands = []
        def launch(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(stdout=str(100+len(commands))+'\n')
        with patch('skill_forgetting.experiments.slurm.subprocess.run', side_effect=launch):
            ids = submit(self.root/'config.json', self.root, self.run, dependency_job='99')
        self.assertIn('--dependency=afterok:99', commands[0])
        self.assertIn('--dependency=afterok:101', commands[1])
        self.assertIn('--dependency=afterok:102', commands[2])
        self.assertEqual(ids, {'baseline':'101', 'branches':'102', 'report':'103'})


if __name__ == '__main__':
    unittest.main()
