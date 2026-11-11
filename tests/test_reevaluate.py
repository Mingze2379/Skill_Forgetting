"""Protect trained artifacts and verify inference-only sensitivity comparisons."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skill_forgetting.experiments.common import file_digest, write_json
from skill_forgetting.experiments.reevaluate import prepare, run_one
from skill_forgetting.experiments import scoring


class ReevaluationTests(unittest.TestCase):
    def test_rejects_outputs_inside_original_result_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            write_json(config, {'run_root': str(root / 'source'), 'max_new_tokens': 1024})
            with self.assertRaisesRegex(ValueError, 'separate'):
                prepare(config, root / 'source/results/A/overwrite', 2048)

    def test_saved_model_goes_directly_to_evaluation_with_separate_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'length2048'
            write_json(source / 'results/A/score.json', {'metadata': {
                'subset': 'A', 'budget': {'optimizer_steps': 64},
                'cache_key': 'old', 'cache_inputs': {'generation': {'max_new_tokens': 1024}}}})
            (source / 'validation_raw.jsonl').write_text('{}\n')
            score_path = source / 'results/A/score.json'
            plan = {'source_root': str(source), 'output_root': str(output),
                    'validation_sha256': file_digest(source / 'validation_raw.jsonl'),
                    'scorer_sha256': file_digest(Path(scoring.__file__)),
                    'checkpoint': {'model': 'saved-checkpoint', 'sha256': 'checkpoint'},
                    'config': {'max_new_tokens': 2048}, 'sources': {'A': {
                        'source_score_sha256': file_digest(score_path), 'adapter': None,
                        'adapter_files': {}}}}
            original = score_path.read_bytes()
            with patch('skill_forgetting.experiments.reevaluate.evaluate') as evaluation, \
                    patch('skill_forgetting.experiments.run.run_training',
                          side_effect=AssertionError('Re-evaluation must never train')):
                run_one(plan, 'A')
            args, kwargs = evaluation.call_args
            self.assertEqual(args[0]['max_new_tokens'], 2048)
            self.assertEqual(args[3], output / 'results/A')
            self.assertEqual(kwargs['metadata']['budget'], {'optimizer_steps': 64})
            self.assertNotIn('cache_key', kwargs['metadata'])
            self.assertEqual(score_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
