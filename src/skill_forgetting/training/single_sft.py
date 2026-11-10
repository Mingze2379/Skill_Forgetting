"""One reproducible LlamaFactory SFT call, optionally followed by a LoRA merge."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from skill_forgetting.experiments.common import digest, file_digest, write_json


@dataclass
class SFTSpec:
    model: str
    data_dir: Path
    output_dir: Path
    max_steps: int | None = None
    seed: int = 42
    dataset: str = 'competition_math_train'
    template: str = 'qwen'
    finetuning_type: str = 'lora'
    learning_rate: float = 1e-4
    epochs: float = 5.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    cutoff_len: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    merge_dir: Path | None = None
    python: str = sys.executable

    def options(self) -> dict:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError('max_steps must be positive')
        if min(self.batch_size, self.gradient_accumulation_steps, self.cutoff_len) <= 0:
            raise ValueError('Training dimensions must be positive')
        options = {
            'stage': 'sft', 'do_train': True, 'model_name_or_path': self.model,
            'template': self.template, 'finetuning_type': self.finetuning_type,
            'dataset_dir': str(self.data_dir), 'dataset': self.dataset,
            'output_dir': str(self.output_dir), 'cutoff_len': self.cutoff_len,
            'learning_rate': self.learning_rate, 'num_train_epochs': self.epochs,
            'per_device_train_batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'seed': self.seed, 'data_seed': self.seed,
            'preprocessing_num_workers': 1, 'dataloader_num_workers': 0,
            'logging_steps': 10, 'save_strategy': 'no',
            'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.03,
            'bf16': True, 'report_to': 'none', 'packing': False,
            'train_on_prompt': False, 'mask_history': False,
            'gradient_checkpointing': True, 'flash_attn': 'sdpa',
            'overwrite_cache': True, 'save_only_model': True,
        }
        if self.max_steps is not None:
            options['max_steps'] = self.max_steps
        if self.finetuning_type == 'lora':
            options.update(lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                           lora_dropout=self.lora_dropout, lora_target='all')
        return options

    def train_command(self) -> list[str]:
        args = [self.python, '-m', 'skill_forgetting.training.llama_entry']
        for key, value in self.options().items():
            args += ['--' + key, str(value).lower() if isinstance(value, bool) else str(value)]
        return args

    def merge_command(self) -> list[str]:
        if self.merge_dir is None or self.finetuning_type != 'lora':
            raise ValueError('Merge requires a LoRA adapter and a merge destination')
        return [self.python, '-m', 'skill_forgetting.training.llama_entry', '--cl_export',
                '--model_name_or_path', self.model, '--adapter_name_or_path', str(self.output_dir),
                '--template', self.template, '--finetuning_type', 'lora',
                '--export_dir', str(self.merge_dir), '--export_size', '2',
                '--export_legacy_format', 'false']


def weights(directory: Path) -> list[Path]:
    return sorted(directory.glob('*.safetensors')) + sorted(directory.glob('pytorch_model*.bin'))


def train_once(spec: SFTSpec, *, dry_run: bool = False) -> Path:
    command = spec.train_command()
    print(shlex.join(command), flush=True)
    if not dry_run:
        for filename in ('dataset_info.json', spec.dataset + '.jsonl'):
            if not (spec.data_dir / filename).is_file():
                raise FileNotFoundError(spec.data_dir / filename)
        for directory in (spec.output_dir, spec.merge_dir):
            if directory is not None and directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f'Use a fresh output directory: {directory}')
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        record = json.loads(json.dumps(asdict(spec), default=str))
        record['dataset_sha256'] = file_digest(spec.data_dir / (spec.dataset + '.jsonl'))
        write_json(spec.output_dir / 'training_spec.json', record)
        env = {**os.environ, 'CL_LOSS_TOKEN_OUTPUT': str(spec.output_dir / 'loss_tokens.json')}
        subprocess.run(command, check=True, env=env)
        if spec.finetuning_type == 'lora':
            if not (spec.output_dir / 'adapter_config.json').is_file() or not weights(spec.output_dir):
                raise RuntimeError('SFT did not save a complete adapter')
        elif not weights(spec.output_dir):
            raise RuntimeError('SFT did not save model weights')
        budget = json.loads((spec.output_dir / 'loss_tokens.json').read_text())
        if spec.max_steps is not None and budget['optimizer_steps'] != spec.max_steps:
            raise RuntimeError('Training completed with an unexpected optimizer step budget')
        if budget['loss_tokens'] <= 0:
            raise RuntimeError('No supervised tokens were observed during training')
    if spec.merge_dir is not None:
        merge = spec.merge_command()
        print(shlex.join(merge), flush=True)
        if not dry_run:
            subprocess.run(merge, check=True)
            if not weights(spec.merge_dir):
                raise RuntimeError('Export did not produce merged model weights')
    destination = spec.merge_dir or spec.output_dir
    if not dry_run:
        write_json(spec.output_dir / 'training_complete.json', {
            'spec_sha256': digest(record), 'destination': str(destination), 'budget': budget,
            'weights_sha256': {p.name: file_digest(p) for p in weights(destination)},
        })
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--max_steps', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--merge_dir', type=Path)
    p.add_argument('--template', default='qwen')
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    train_once(SFTSpec(**{k: v for k, v in vars(args).items() if k != 'dry_run'}), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
