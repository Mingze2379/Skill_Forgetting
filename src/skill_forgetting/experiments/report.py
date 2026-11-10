"""Compute exact three-player Shapley and budget-constrained rankings from eight real scores."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from .common import DOMAINS, PLAYERS, SUBSETS, file_digest, write_json


def subset_name(players) -> str:
    return ''.join(sorted(players)) or 'empty'


def shapley(values: dict[str, float]) -> dict[str, float]:
    if set(values) != set(SUBSETS):
        raise ValueError('Exact Shapley needs all eight distinct subsets')
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite subset utility')
    result = {}
    for player in PLAYERS:
        others = [p for p in PLAYERS if p != player]
        value = 0.0
        for size in range(3):
            weight = math.factorial(size) * math.factorial(2-size) / math.factorial(3)
            for subset in itertools.combinations(others, size):
                value += weight * (values[subset_name((*subset, player))] - values[subset_name(subset)])
        result[player] = value
    if not math.isclose(sum(result.values()), values['ABC']-values['empty'], abs_tol=1e-10):
        raise AssertionError('Shapley efficiency invariant failed')
    return result


def analyze(values: dict[str, float], replay_per_domain: int) -> dict:
    phi = shapley(values)
    single = {p: values[p]-values['empty'] for p in PLAYERS}
    pair_interactions = {''.join(pair): values[''.join(pair)] - values[pair[0]] - values[pair[1]]
                         + values['empty'] for pair in itertools.combinations(PLAYERS, 2)}
    best = {}
    for size in range(4):
        candidates = [s for s in SUBSETS if (0 if s == 'empty' else len(s)) == size]
        optimum = max(values[s] for s in candidates)
        best[str(size * replay_per_domain)] = {
            'replay_samples': size * replay_per_domain,
            'subsets': [s for s in candidates if math.isclose(values[s], optimum, abs_tol=1e-12)],
            'v': optimum}
    rank = lambda scores: sorted(scores, key=lambda p: (-scores[p], p))
    return {'utility': 'four-domain macro validation accuracy',
            'singleton_gain': single, 'singleton_rank': rank(single),
            'shapley': phi, 'shapley_rank': rank(phi),
            'pair_interactions': pair_interactions,
            'third_order_interaction': values['ABC']-values['AB']-values['AC']-values['BC']
                                       +values['A']+values['B']+values['C']-values['empty'],
            'best_observed_at_fixed_replay_budget': best,
            'notes': 'One checkpoint and seed; differences are descriptive, not a significance claim.'}


def collect(run_root: Path) -> dict:
    manifest = json.loads((run_root / 'manifest.json').read_text())
    expected_validation = manifest['validation_sha256']
    rows, values, checkpoints = [], {}, set()
    seeds, steps, samples, scorer_ids = set(), set(), set(), set()
    for subset in SUBSETS:
        path = run_root / 'results' / subset / 'score.json'
        if not path.is_file():
            raise FileNotFoundError(f'Cannot report eight-row results until {subset} has a real score: {path}')
        result = json.loads(path.read_text())
        metadata = result['metadata']
        if result['validation_sha256'] != expected_validation or metadata['subset'] != subset:
            raise ValueError('Score uses a different validation set or subset')
        if metadata['dataset_sha256'] != manifest['branch_sha256'][subset]:
            raise ValueError('Branch score was trained on a different dataset')
        if metadata['manifest_sha256'] != file_digest(run_root / 'manifest.json'):
            raise ValueError('Cached score belongs to another experiment manifest')
        for domain in DOMAINS:
            if result['n'][domain] != manifest['counts'][domain]['validation']:
                raise ValueError('Validation sample budget differs across branches')
            if not 0 <= result['accuracy'][domain] <= 1:
                raise ValueError('Invalid domain accuracy')
        if not math.isclose(result['v'], sum(result['accuracy'][d] for d in DOMAINS)/4, abs_tol=1e-12):
            raise ValueError('Cached utility does not match four-domain macro accuracy')
        budget = metadata['budget']
        checkpoints.add(metadata['checkpoint_sha256'])
        seeds.add(metadata['seed']); steps.add(budget['optimizer_steps']); samples.add(metadata['sample_count'])
        scorer_ids.add(json.dumps(result['scorer'], sort_keys=True))
        values[subset] = result['v']
        rows.append({'subset': subset, 'replay_samples': (0 if subset == 'empty' else len(subset))
                     * manifest['config']['replay_per_domain'],
                     **{f'acc_{d}': result['accuracy'][d] for d in DOMAINS},
                     'v': result['v'], 'v_minus_empty': 0,
                     'optimizer_steps': budget['optimizer_steps'], 'loss_tokens': budget['loss_tokens'],
                     'examples_seen': budget['examples_seen'], 'sample_count': metadata['sample_count'],
                     'legacy_v': sum(result['legacy_accuracy'][d] for d in DOMAINS)/4,
                     'scorer_disagreements': result['disagreement_count']})
    if any(len(group) != 1 for group in (checkpoints, seeds, steps, samples, scorer_ids)):
        raise ValueError('Branches do not share checkpoint, seed, step budget, sample count, and scorer')
    if next(iter(steps)) != manifest['config']['branch_steps'] or next(iter(samples)) != manifest['config']['branch_samples']:
        raise ValueError('Training budget differs from configured experiment')
    for row in rows:
        row['v_minus_empty'] = row['v']-values['empty']
    output = run_root / 'report'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'eight_subsets.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = analyze(values, manifest['config']['replay_per_domain'])
    summary['per_domain_shapley'] = {domain: shapley({r['subset']: r[f'acc_{domain}'] for r in rows})
                                      for domain in DOMAINS}
    write_json(output / 'shapley.json', summary)
    table = ['| subset | algebra | counting/probability | geometry | intermediate algebra | v | loss tokens |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append('| ' + row['subset'] + ' | ' + ' | '.join(
            f"{row[f'acc_{d}']:.3f}" for d in DOMAINS) + f" | {row['v']:.4f} | {row['loss_tokens']} |")
    (output / 'results.md').write_text('\n'.join(table) + '\n\n' +
        'Singleton order: ' + ' > '.join(summary['singleton_rank']) + '\n\n' +
        'Shapley order: ' + ' > '.join(summary['shapley_rank']) + '\n\n' + summary['notes'] + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(collect(args.run_root), indent=2))


if __name__ == '__main__':
    main()
