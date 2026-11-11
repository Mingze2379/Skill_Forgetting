"""Re-evaluate saved checkpoints without training or changing the frozen experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from .common import DOMAINS, SUBSETS, file_digest, read_rows, write_json, write_rows
from .report import analyze
from .run import evaluate
from . import scoring

TARGETS = ('baseline_validation', *SUBSETS)


def source_output(source: Path, target: str) -> Path:
    return source / target if target == 'baseline_validation' else source / 'results' / target


def prepare(config_path: Path, output: Path, max_new_tokens: int) -> dict:
    config = json.loads(config_path.read_text())
    source = Path(config['run_root']).resolve()
    output = output.resolve()
    protected = (source / 'results', source / 'baseline_validation', source / 'report')
    if output == source or any(output == p or p in output.parents for p in protected):
        raise ValueError('Use a separate evaluation directory')
    if config['max_new_tokens'] != 1024 or max_new_tokens != 2048:
        raise ValueError('This check compares the frozen 1024-token evaluation with 2048 tokens')
    if (output / 'plan.json').exists():
        raise FileExistsError('Evaluation already prepared; inspect its plan and jobs before continuing')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['config'] != config:
        raise ValueError('Original frozen configuration changed')
    checkpoint = json.loads((source / 'checkpoint.json').read_text())
    if checkpoint['manifest_sha256'] != file_digest(source / 'manifest.json'):
        raise ValueError('Checkpoint and source manifest differ')
    for name, sha in checkpoint['weights_sha256'].items():
        if file_digest(Path(checkpoint['model']) / name) != sha:
            raise ValueError('Saved checkpoint weights changed')
    # Count the exact rendered prompts, so the larger output budget is not
    # silently reduced by the unchanged context limit.
    from transformers import AutoTokenizer
    from skill_forgetting.math.evaluate import build_prompt
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['model'], local_files_only=True)
    lengths = [len(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': build_prompt(row)}], tokenize=True,
        add_generation_prompt=True)) for row in read_rows(source / 'validation_raw.jsonl')]
    if max(lengths) + max_new_tokens > config['max_model_len']:
        raise ValueError('The context limit cannot accommodate every prompt plus the requested answer budget')
    records = {}
    for target in TARGETS:
        origin = source_output(source, target)
        result = json.loads((origin / 'score.json').read_text())
        if result['scorer']['code_sha256'] != file_digest(Path(scoring.__file__)):
            raise ValueError('Use the same scoring code as the original evaluation')
        if result['metadata']['cache_inputs']['generation'] != {
                key: config[key] for key in ('max_new_tokens', 'max_model_len', 'eval_max_num_seqs')}:
            raise ValueError('Original generation settings differ')
        if file_digest(origin / 'predictions.jsonl') != result['prediction_sha256']:
            raise ValueError('Original predictions changed')
        adapter = None if target == 'baseline_validation' else (
            Path(config['artifact_root']) / 'branches' / target / 'adapter')
        adapter_files = {}
        if adapter:
            for name, sha in result['metadata']['cache_inputs']['adapter'].items():
                if file_digest(adapter / name) != sha:
                    raise ValueError('Saved adapter weights changed')
                adapter_files[name] = sha
            adapter_files['adapter_config.json'] = file_digest(adapter / 'adapter_config.json')
        records[target] = {'source_score_sha256': file_digest(origin / 'score.json'),
                           'prediction_sha256': result['prediction_sha256'],
                           'adapter': str(adapter) if adapter else None,
                           'adapter_files': adapter_files}
    settings = copy.deepcopy(config)
    settings['max_new_tokens'] = max_new_tokens
    plan = {'source_root': str(source), 'output_root': str(output), 'config': settings,
            'checkpoint': checkpoint, 'targets': list(TARGETS), 'sources': records,
            'validation_sha256': file_digest(source / 'validation_raw.jsonl'),
            'max_prompt_tokens': max(lengths), 'training': 'none; reuse saved models',
            'scorer_sha256': file_digest(Path(scoring.__file__))}
    write_json(output / 'plan.json', plan)
    return plan


def run_one(plan: dict, target: str, dry_run=False):
    if target not in TARGETS:
        raise ValueError('Unknown evaluation target')
    source = Path(plan['source_root'])
    origin = source_output(source, target)
    record = plan['sources'][target]
    if file_digest(origin / 'score.json') != record['source_score_sha256']:
        raise ValueError('Original scores changed after preparing re-evaluation')
    if file_digest(source / 'validation_raw.jsonl') != plan['validation_sha256']:
        raise ValueError('Validation data changed')
    if file_digest(Path(scoring.__file__)) != plan['scorer_sha256']:
        raise ValueError('Scoring code changed')
    adapter = Path(record['adapter']) if record['adapter'] else None
    if adapter:
        for name, sha in record['adapter_files'].items():
            if file_digest(adapter / name) != sha:
                raise ValueError('Adapter changed')
    metadata = json.loads((origin / 'score.json').read_text())['metadata']
    metadata = {k: v for k, v in metadata.items() if k not in ('cache_key', 'cache_inputs')}
    metadata['reevaluation_source_score_sha256'] = record['source_score_sha256']
    return evaluate(plan['config'], source, plan['checkpoint'],
                    Path(plan['output_root']) / 'results' / target,
                    adapter=adapter, metadata=metadata, dry_run=dry_run)


def compare(plan: dict) -> dict:
    source, output = Path(plan['source_root']), Path(plan['output_root'])
    summaries, changes, values = [], [], {'1024': {}, '2048': {}}
    for target in TARGETS:
        origin = source_output(source, target)
        newdir = output / 'results' / target
        before = json.loads((origin / 'score.json').read_text())
        after = json.loads((newdir / 'score.json').read_text())
        settings = {key: plan['config'][key] for key in
                    ('max_new_tokens', 'max_model_len', 'eval_max_num_seqs')}
        if after['metadata']['cache_inputs']['generation'] != settings:
            raise ValueError('The new evaluation uses a different generation budget')
        if (file_digest(origin / 'predictions.jsonl') != before['prediction_sha256'] or
                file_digest(newdir / 'predictions.jsonl') != after['prediction_sha256']):
            raise ValueError('Predictions changed after scoring')
        if (before['validation_sha256'] != after['validation_sha256'] or
                before['scorer'] != after['scorer'] or before['n'] != after['n'] or
                before['metadata']['checkpoint_sha256'] != after['metadata']['checkpoint_sha256']):
            raise ValueError('Evaluation comparison mixes data, models or scoring')
        if target != 'baseline_validation' and before['metadata']['budget'] != after['metadata']['budget']:
            raise ValueError('Training budget changed')
        previous = {r['example_id']: r for r in read_rows(origin / 'judgments.jsonl')}
        original_predictions = {r['example_id']: r for r in read_rows(origin / 'predictions.jsonl')}
        predictions = {r['example_id']: r for r in read_rows(newdir / 'predictions.jsonl')}
        if set(predictions) != set(original_predictions):
            raise ValueError('Prediction coverage changed')
        improved = regressed = 0
        for row in read_rows(newdir / 'judgments.jsonl'):
            idx = row['example_id']
            if predictions[idx]['prompt'] != original_predictions[idx]['prompt']:
                raise ValueError('Rendered prompts changed')
            old = previous[idx]
            improved += row['correct'] and not old['correct']
            regressed += old['correct'] and not row['correct']
            if row['correct'] != old['correct']:
                changes.append({'target': target, 'example_id': idx, 'domain': row['domain'],
                                'before_correct': old['correct'], 'after_correct': row['correct'],
                                'before_finish_reason': original_predictions[idx]['finish_reason'],
                                'after_finish_reason': predictions[idx]['finish_reason'],
                                'problem': row['problem'], 'gold_answer': row['gold_answer'],
                                'before_answer': old['predicted_answer'], 'after_answer': row['predicted_answer']})
        summary = {'target': target, 'v_1024': before['v'], 'v_2048': after['v'],
                   'delta_pp': (after['v']-before['v'])*100,
                   'truncated_1024': sum(before['generation_truncated'].values()),
                   'truncated_2048': sum(after['generation_truncated'].values()),
                   'improved': improved, 'regressed': regressed}
        for domain in DOMAINS:
            summary.update({f'{domain}_1024': before['accuracy'][domain],
                            f'{domain}_2048': after['accuracy'][domain]})
        summaries.append(summary)
        if target != 'baseline_validation':
            values['1024'][target] = before['v']; values['2048'][target] = after['v']
    reports = {length: analyze(v, plan['config']['replay_per_domain']) for length, v in values.items()}
    directory = output / 'report'; directory.mkdir(exist_ok=True)
    with (directory / 'length_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    write_rows(directory / 'changed_judgments.jsonl', changes)
    write_json(directory / 'shapley_comparison.json', reports)
    table = ['| target | v 1024 | v 2048 | change pp | truncated 1024 | truncated 2048 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        table.append(f"| {row['target']} | {row['v_1024']*100:.2f}% | {row['v_2048']*100:.2f}% | "
                     f"{row['delta_pp']:+.2f} | {row['truncated_1024']} | {row['truncated_2048']} |")
    (directory / 'results.md').write_text('\n'.join(table)+'\n')
    result = {'training': 'none', 'max_new_tokens': plan['config']['max_new_tokens'],
              'summaries': summaries, 'shapley': reports, 'changed_answers': len(changes)}
    write_json(directory / 'comparison_complete.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_root', type=Path, required=True)
    sub = p.add_subparsers(dest='action', required=True)
    prepare_args = sub.add_parser('prepare')
    prepare_args.add_argument('--config', type=Path, required=True)
    prepare_args.add_argument('--max_new_tokens', type=int, default=2048)
    one = sub.add_parser('evaluate')
    one.add_argument('--target_index', type=int, choices=range(len(TARGETS)), required=True)
    one.add_argument('--dry_run', action='store_true')
    sub.add_parser('report')
    args = p.parse_args()
    if args.action == 'prepare':
        result = prepare(args.config, args.output_root, args.max_new_tokens)
    else:
        plan = json.loads((args.output_root / 'plan.json').read_text())
        result = (run_one(plan, TARGETS[args.target_index], args.dry_run)
                  if args.action == 'evaluate' else compare(plan))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
