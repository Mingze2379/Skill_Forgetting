"""CPU regression tests; no external models or datasets are loaded."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from skill_forgetting.math import augment, build_grpo_data, prepare
from skill_forgetting.math.constants import EVAL_TYPES, TRAIN_TYPES
from skill_forgetting.memory.pipeline import DynamicSkillMemoryPipeline
from skill_forgetting.memory.retriever import HybridSkillRetriever
from skill_forgetting.memory.schema import CanonicalSkill, SkillFamily, UpdateProposal
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.updater import UpdateManager
from skill_forgetting.memory.utils import read_jsonl, write_jsonl
from skill_forgetting.training import grpo, sft


def cli(module, *args):
    with patch("sys.argv", [module.__name__, *map(str, args)]), redirect_stdout(io.StringIO()):
        module.main()


def embeddings(texts):
    return np.array([[1.0, 0.0] if "algebra" in text.lower() else [0.0, 1.0] for text in texts])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = SkillMemoryStore(self.root / "memory")
        self.store.initialize(
            families=[SkillFamily("math", "Math", "Mathematics")],
            seed_skills=[
                CanonicalSkill("algebra", "math", "Algebra substitution", "algebra",
                               canonical_steps=["Substitute the known quantity."]),
                CanonicalSkill("geometry", "math", "Geometry angles", "geometry",
                               canonical_steps=["Use the angle sum."]),
            ],
        )

    def test_retrieval_and_memory_feedback_round_trip(self):
        with patch("skill_forgetting.memory.retriever.encode_texts", side_effect=embeddings):
            found = HybridSkillRetriever(self.store).retrieve({"problem": "algebra equation"}, top_k=1)
        self.assertEqual(found[0].skill_id, "algebra")
        UpdateManager(self.store).record_retrieval_feedback(found, correct=True)
        reloaded = SkillMemoryStore(self.root / "memory")
        self.assertEqual(reloaded.get_skill("algebra").stats.times_helped, 1)
        self.assertEqual(reloaded.get_skill("geometry").stats.times_retrieved, 0)

    def test_gold_bootstrap_preserves_support_without_inference_feedback(self):
        class Extractor:
            def propose(self, *args, **kwargs):
                return UpdateProposal("reuse", "algebra", None, "Reusable substitution", 1.0)

        source = self.root / "train.jsonl"
        write_jsonl(source, [{"problem": "algebra x + 1 = 3", "solution": r"\boxed{2}"}])
        pipeline = DynamicSkillMemoryPipeline(self.store, extractor=Extractor())
        with patch("skill_forgetting.memory.retriever.encode_texts", side_effect=embeddings):
            rows = pipeline.bootstrap_from_gold(source, self.root / "bootstrap.jsonl")
        self.assertTrue(rows[0]["correct"])
        self.assertTrue(rows[0]["update_applied"])
        reloaded = SkillMemoryStore(self.root / "memory")
        self.assertEqual(reloaded.get_skill("algebra").stats.support_count, 1)
        self.assertEqual(reloaded.get_skill("algebra").stats.times_retrieved, 0)

    def test_native_splits_augmentation_and_math_grpo_conversion(self):
        data = self.root / "source"
        for task in EVAL_TYPES:
            for split in ("train", "test"):
                write_jsonl(data / split / "by_type" / f"{task}.jsonl", [{
                    "problem": f"algebra {split} question",
                    "solution": r"\boxed{2}", "type": task, "level": "1",
                }])
        cli(prepare, "--by_type_dir", data, "--run_root", self.root)
        with patch("skill_forgetting.memory.retriever.encode_texts", side_effect=embeddings):
            cli(augment, "--run_root", self.root, "--top_k_skills", "1")
        cli(build_grpo_data, "--run_root", self.root)
        for task in TRAIN_TYPES:
            record = read_jsonl(self.root / "math_verl_skill" / task / "train.jsonl")[0]
            self.assertIn("Substitute the known quantity", record["prompt"][0]["content"])
            self.assertIn("train question", record["prompt"][0]["content"])
            self.assertNotIn("test question", record["prompt"][0]["content"])
            self.assertEqual(record["reward_model"]["ground_truth"], "2")
        for task in EVAL_TYPES:
            record = read_jsonl(self.root / "math_verl_skill" / task / "eval.jsonl")[0]
            self.assertIn("test question", record["prompt"][0]["content"])
        self.assertFalse((self.root / "per_type_data_skill_augmented" / "precalculus" / "competition_math_train.jsonl").exists())

    def test_missing_augmented_data_does_not_fall_back_to_raw(self):
        write_jsonl(self.root / "per_type_data_with_raw" / "algebra" / "train_raw.jsonl",
                    [{"problem": "algebra", "solution": r"\boxed{2}"}])
        with self.assertRaises(FileNotFoundError):
            cli(build_grpo_data, "--run_root", self.root, "--types", "algebra")

    def test_agent_augmentation_and_rewards(self):
        from skill_forgetting.agents import augment as agent_augment
        from skill_forgetting.agents import build_grpo_data as agent_grpo
        from skill_forgetting.agents.constants import ENVIRONMENT_ORDER
        from skill_forgetting.agents.dataset_builder import normalize_skillrl_row
        from skill_forgetting.agents.reward import compute_score
        for environment in ENVIRONMENT_ORDER:
            row = normalize_skillrl_row({
                "instruction": "algebra task\n\n## Retrieved Relevant Experience\nOLD SKILL\n\nNow it's your turn",
                "output": "<think>Plan</think><action>look</action>",
            }, environment)
            self.assertNotIn("OLD SKILL", row["instruction"])
            for split in ("train", "eval"):
                write_jsonl(self.root / "agent_sft_data" / environment / f"{split}_raw.jsonl", [row])
        with patch("skill_forgetting.memory.retriever.encode_texts", side_effect=embeddings):
            cli(agent_augment, "--run_root", self.root)
        cli(agent_grpo, "--run_root", self.root)
        record = read_jsonl(self.root / "agent_verl_skill" / "alfworld" / "train.jsonl")[0]
        self.assertIn("Substitute the known quantity", record["prompt"][0]["content"])
        self.assertEqual(compute_score("skillrl_alfworld", "<action>look</action>", "<action>look</action>")["score"], 1)
        self.assertEqual(compute_score("skillrl_search", "<answer>Paris</answer>", "<answer>Paris</answer>")["score"], 1)

    def test_sft_dry_run_chains_merged_lora(self):
        output = io.StringIO()
        with patch("sys.argv", ["sft", "--model", "test/model", "--run_root", str(self.root),
                               "--finetuning_type", "lora", "--dry_run"]), redirect_stdout(output):
            sft.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 8)
        self.assertIn("--model_name_or_path " + str(self.root / "checkpoints/step_01_algebra"), lines[2])
        self.assertNotIn("_adapter --template", lines[2])
        self.assertFalse((self.root / "checkpoints").exists())

    def test_grpo_chains_recorded_checkpoint_and_keeps_skill_inputs(self):
        commands = []

        def fake_run(command, *, dry_run, env=None):
            commands.append(command)
            if "verl.trainer.main_ppo" in command:
                setting = next(v for v in command if v.startswith("trainer.default_local_dir="))
                stage = Path(json.loads(setting.split("=", 1)[1]))
                actor = stage / "global_step_7" / "actor"
                actor.mkdir(parents=True)
                (stage / "latest_checkpointed_iteration.txt").write_text("7")
            else:
                target = Path(command[command.index("--target_dir") + 1])
                target.mkdir(parents=True)
                (target / "model.safetensors").write_text("synthetic test marker")

        for task in TRAIN_TYPES:
            for split in ("train", "eval"):
                write_jsonl(self.root / "math_verl_skill" / task / f"{split}.jsonl", [{"prompt": "synthetic"}])
        with patch.object(grpo, "run", side_effect=fake_run):
            cli(grpo, "--domain", "math", "--model", "test/model", "--run_root", self.root)
        self.assertEqual(len(commands), 8)
        second_stage = commands[2]
        model = next(v for v in second_stage if v.startswith("actor_rollout_ref.model.path="))
        self.assertEqual(json.loads(model.split("=", 1)[1]), str(self.root / "checkpoints/step_01_algebra/merged"))
        self.assertTrue(any("math_verl_skill" in v for v in second_stage))
        self.assertIn(str(self.root / "checkpoints/step_01_algebra/global_step_7/actor"), commands[1])

    def test_missing_checkpoint_tracker_fails(self):
        with self.assertRaises(FileNotFoundError):
            grpo.merged_checkpoint(self.root / "missing_stage", dry_run=False)


if __name__ == "__main__":
    unittest.main()
