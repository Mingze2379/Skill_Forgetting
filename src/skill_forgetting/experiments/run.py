"""Train the common checkpoint or one adapter branch; cache generation and validation scores."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

from skill_forgetting.training.single_sft import SFTSpec, train_once, weights
from .common import DOMAINS, SUBSETS, digest, file_digest, load_config, read_rows, write_json
from .scoring import score


def spec_for(config: dict, model: str, data: Path, output: Path, steps: int, merge=None) -> SFTSpec:
    return SFTSpec(model=model, data_dir=data.resolve(), output_dir=output.resolve(),
                   max_steps=steps, seed=config['seed'], template=config['template'],
                   learning_rate=config['learning_rate'], batch_size=config['batch_size'],
                   gradient_accumulation_steps=config['gradient_accumulation_steps'],
                   cutoff_len=config['cutoff_len'], lora_rank=config['lora_rank'],
                   lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                   merge_dir=merge.resolve() if merge else None, python=config['train_python'])


def run_training(spec: SFTSpec, dry_run=False) -> dict | None:
    complete = spec.output_dir / 'training_complete.json'
    if not dry_run and complete.is_file():
        record = json.loads(json.dumps(asdict(spec), default=str))
        record['dataset_sha256'] = file_digest(spec.data_dir / (spec.dataset + '.jsonl'))
        cached = json.loads(complete.read_text())
        if cached['spec_sha256'] != digest(record):
            raise ValueError('Existing training output belongs to a different training specification')
        directory = Path(cached['destination'])
        if any(not (directory / name).is_file() or file_digest(directory / name) != sha
               for name, sha in cached['weights_sha256'].items()):
            raise ValueError('Cached training checkpoint weights changed or disappeared')
        print('Using completed training:', complete, flush=True)
        return cached
    train_once(spec, dry_run=dry_run)
    return None if dry_run else json.loads(complete.read_text())


def evaluate(config: dict, run_root: Path, checkpoint: dict, output: Path,
             adapter: Path | None = None, metadata: dict | None = None, dry_run=False):
    import skill_forgetting.experiments.scoring as scoring_module
    validation = run_root / 'validation_raw.jsonl'
    metadata = metadata or {}
    settings = {key: config[key] for key in ('max_new_tokens', 'max_model_len', 'eval_max_num_seqs')}
    inputs = {'checkpoint': checkpoint['sha256'], 'validation': file_digest(validation),
              'adapter': {p.name: file_digest(p) for p in weights(adapter)} if adapter else None,
              'generation': settings, 'scorer_code': file_digest(Path(scoring_module.__file__)),
              'scorer_version': version('math-verify'), 'metadata': metadata}
    key = digest(inputs)
    complete = output / 'score.json'
    if not dry_run and complete.is_file():
        cached = json.loads(complete.read_text())
        if cached['metadata'].get('cache_key') != key:
            raise ValueError('Cached score uses different checkpoint, data, generation settings, or scorer')
        if file_digest(output / 'predictions.jsonl') != cached['prediction_sha256']:
            raise ValueError('Cached predictions were changed')
        print('Using cached score:', complete, flush=True)
        return cached
    predictions = output / 'predictions.jsonl'
    command = [config['eval_python'], '-m', 'skill_forgetting.math.evaluate',
               '--model', checkpoint['model'], '--input_jsonl', str(validation.resolve()),
               '--output_jsonl', str(predictions.resolve()), '--max_new_tokens', str(config['max_new_tokens']),
               '--max_model_len', str(config['max_model_len']), '--max_num_seqs', str(config['eval_max_num_seqs']),
               '--temperature', '0', '--tensor_parallel_size', '1', '--gpu_memory_utilization', '0.85']
    if adapter is not None:
        command += ['--lora_path', str(adapter.resolve()), '--max_lora_rank', str(config['lora_rank'])]
    print('Evaluation:', command, flush=True)
    if dry_run:
        return None
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return score(predictions, validation, output, {**metadata, 'cache_key': key, 'cache_inputs': inputs})


def baseline(config: dict, run_root: Path, dry_run=False):
    artifact = Path(config['artifact_root'])
    model = config['model']
    last = None
    for index, domain in enumerate(DOMAINS[:3], 1):
        merge = artifact / 'checkpoints' / f'step_{index:02d}_{domain}'
        spec = spec_for(config, model, run_root / 'plain_sft' / domain,
                        artifact / 'adapters' / f'step_{index:02d}_{domain}', config['stage_steps'], merge)
        last = run_training(spec, dry_run)
        model = str(merge.resolve())
    if dry_run:
        return
    checkpoint = {'model': model, 'weights_sha256': last['weights_sha256'],
                  'sha256': digest(last['weights_sha256']), 'seen_domains': list(DOMAINS[:3]),
                  'seed': config['seed'], 'manifest_sha256': file_digest(run_root / 'manifest.json')}
    write_json(run_root / 'checkpoint.json', checkpoint)
    evaluate(config, run_root, checkpoint, run_root / 'baseline_validation',
             metadata={'stage': 'before_intermediate_algebra', 'checkpoint_sha256': checkpoint['sha256']})


def branch(config: dict, run_root: Path, subset: str, dry_run=False):
    if subset not in SUBSETS:
        raise ValueError('Unknown replay subset')
    checkpoint_path = run_root / 'checkpoint.json'
    if dry_run and not checkpoint_path.exists():
        model = str(Path(config['artifact_root']) / 'checkpoints' / 'step_03_geometry')
        checkpoint = {'model': model, 'sha256': 'DRY_RUN_ONLY'}
    else:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint['manifest_sha256'] != file_digest(run_root / 'manifest.json'):
            raise ValueError('Checkpoint belongs to a different prepared experiment')
        if any(file_digest(Path(checkpoint['model']) / name) != sha
               for name, sha in checkpoint['weights_sha256'].items()):
            raise ValueError('Shared starting checkpoint changed')
    output = Path(config['artifact_root']) / 'branches' / subset / 'adapter'
    data = run_root / 'branches' / subset / 'data'
    spec = spec_for(config, checkpoint['model'], data, output, config['branch_steps'])
    trained = run_training(spec, dry_run)
    if dry_run:
        print(f'{subset}: evaluate base={checkpoint["model"]}, adapter={output}, split=validation', flush=True)
        return
    if trained['budget']['examples_seen'] != config['branch_steps'] * config['batch_size'] * config['gradient_accumulation_steps']:
        raise ValueError('Actual sample budget differs from the configured one-GPU fixed-step budget')
    if trained['budget']['prepared_dataset_rows'] != config['branch_samples']:
        raise ValueError('LlamaFactory dropped training examples; fixed replay positions are no longer comparable')
    metadata = {'subset': subset, 'checkpoint_sha256': checkpoint['sha256'],
                'manifest_sha256': file_digest(run_root / 'manifest.json'),
                'dataset_sha256': file_digest(data / 'competition_math_train.jsonl'),
                'seed': config['seed'], 'sample_count': len(read_rows(data / 'rows.jsonl')),
                'budget': trained['budget'], 'training_spec_sha256': trained['spec_sha256']}
    result = evaluate(config, run_root, checkpoint, run_root / 'results' / subset,
                      adapter=output, metadata=metadata)
    print(f'{subset}: v={result["v"]:.4f}, accuracy={result["accuracy"]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run_root', type=Path)
    p.add_argument('--dry_run', action='store_true')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('baseline')
    b = sub.add_parser('branch')
    choice = b.add_mutually_exclusive_group(required=True)
    choice.add_argument('--subset', choices=SUBSETS)
    choice.add_argument('--subset_index', type=int, choices=range(8))
    args = p.parse_args()
    config = load_config(args.config)
    run_root = args.run_root or Path(config['run_root'])
    # A configuration change after preparation must be an explicit new experiment.
    manifest = json.loads((run_root / 'manifest.json').read_text())
    if manifest['config'] != config:
        raise ValueError('Runtime configuration differs from the prepared experiment')
    if args.action == 'baseline':
        baseline(config, run_root, args.dry_run)
    else:
        branch(config, run_root, args.subset or SUBSETS[args.subset_index], args.dry_run)


if __name__ == '__main__':
    main()
