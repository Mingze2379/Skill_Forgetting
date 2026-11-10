"""Hold out validation from native training data; create ordinary SFT and replay slots."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from skill_forgetting.math.plain import sft_example
from .common import DOMAINS, PLAYERS, SUBSETS, digest, file_digest, load_config, read_rows, write_json, write_rows


def problem_id(row: dict) -> str:
    # Ignore type/level so duplicates across domains cannot leak into evaluation.
    return digest(' '.join(row['problem'].split()))


def stratified_validation(rows: list[dict], count: int, seed: int):
    if count >= len(rows):
        raise ValueError('Validation must leave at least one training problem')
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row.get('level', 'Unknown'))].append(row)
    rng = random.Random(seed)
    sizes = {level: len(items) * count / len(rows) for level, items in buckets.items()}
    quotas = {level: int(size) for level, size in sizes.items()}
    for level in sorted(buckets, key=lambda x: (-(sizes[x] - quotas[x]), x))[:count-sum(quotas.values())]:
        quotas[level] += 1
    train, val = [], []
    for level in sorted(buckets):
        group = sorted(buckets[level], key=problem_id)
        rng.shuffle(group)
        val.extend(group[:quotas[level]])
        train.extend(group[quotas[level]:])
    return sorted(train, key=problem_id), sorted(val, key=problem_id)


def export_sft(directory: Path, rows: list[dict]) -> None:
    write_rows(directory / 'competition_math_train.jsonl', map(sft_example, rows))
    write_json(directory / 'dataset_info.json', {'competition_math_train': {
        'file_name': 'competition_math_train.jsonl',
        'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'},
    }})


def prepare(config: dict, raw_root: Path, run_root: Path) -> dict:
    if (run_root / 'manifest.json').exists():
        raise FileExistsError('Prepared experiment already exists; use a fresh run_root')
    run_root.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test, source_hashes = {}, {}, {}
    all_test_ids = set()
    for domain in DOMAINS:
        source = raw_root / 'per_type_data_with_raw' / domain
        raw_train[domain] = read_rows(source / 'train_raw.jsonl')
        raw_test[domain] = read_rows(source / 'test_raw.jsonl')
        all_test_ids.update(map(problem_id, raw_test[domain]))
        for split in ('train', 'test'):
            source_hashes[f'{domain}/{split}'] = file_digest(source / f'{split}_raw.jsonl')

    seen, train, validation, counts = set(), {}, {}, {}
    combined_validation = []
    for index, domain in enumerate(DOMAINS):
        clean = []
        for row in sorted(raw_train[domain], key=problem_id):
            key = problem_id(row)
            if key in all_test_ids or key in seen:
                continue
            seen.add(key)
            clean.append({**row, 'example_id': key, 'domain': domain})
        train[domain], validation[domain] = stratified_validation(
            clean, config['validation_per_domain'], config['seed'] + index)
        for row in validation[domain]:
            row['split'] = 'validation'
        combined_validation.extend(validation[domain])
        directory = run_root / 'splits' / domain
        write_rows(directory / 'train_raw.jsonl', train[domain])
        write_rows(directory / 'validation_raw.jsonl', validation[domain])
        # Native test is copied unchanged and never used by branch training/valuation.
        write_rows(directory / 'test_raw.jsonl', raw_test[domain])
        counts[domain] = {'train': len(train[domain]), 'validation': len(validation[domain]),
                          'native_test': len(raw_test[domain]),
                          'excluded_train_duplicates': len(raw_train[domain]) - len(clean)}
        pool = list(train[domain])
        random.Random(config['seed'] + 100 + index).shuffle(pool)
        if config['stage_samples'] > len(pool):
            raise ValueError(f'{domain}: stage_samples exceeds unique training pool')
        stage = pool[:config['stage_samples']]
        export_sft(run_root / 'plain_sft' / domain, stage)
        write_rows(run_root / 'plain_sft' / domain / 'rows.jsonl', stage)

    write_rows(run_root / 'validation_raw.jsonl', combined_validation)
    n, k = config['branch_samples'], config['replay_per_domain']
    new_pool = list(train[DOMAINS[3]])
    random.Random(config['seed'] + 200).shuffle(new_pool)
    if len(new_pool) < n:
        raise ValueError('branch_samples exceeds the unique intermediate-algebra training pool')
    base = new_pool[:n]
    # Each old cluster is sampled from examples actually seen in the first three stages.
    replay = {}
    for index, player in enumerate(PLAYERS):
        pool = read_rows(run_root / 'plain_sft' / DOMAINS[index] / 'rows.jsonl')
        random.Random(config['seed'] + 300 + index).shuffle(pool)
        if len(pool) < k:
            raise ValueError('Replay budget exceeds the seen-domain training examples')
        replay[player] = pool[:k]
        write_rows(run_root / 'replay' / f'{player}.jsonl', replay[player])
    positions = list(range(n))
    random.Random(config['seed'] + 400).shuffle(positions)
    slots = {player: positions[i*k:(i+1)*k] for i, player in enumerate(PLAYERS)}
    write_json(run_root / 'replay' / 'slots.json', slots)

    branch_hashes = {}
    for subset in SUBSETS:
        rows = [dict(row) for row in base]
        active = () if subset == 'empty' else subset
        for player in active:
            for position, row in zip(slots[player], replay[player]):
                rows[position] = dict(row)
        audit = [{**row, 'position': i, 'slot': next(
            (p for p in PLAYERS if i in slots[p]), None),
            'replay': row['domain'] != DOMAINS[3]} for i, row in enumerate(rows)]
        directory = run_root / 'branches' / subset / 'data'
        export_sft(directory, rows)
        write_rows(directory / 'rows.jsonl', audit)
        branch_hashes[subset] = file_digest(directory / 'competition_math_train.jsonl')
    manifest = {'config': config, 'source_sha256': source_hashes, 'counts': counts,
                'validation_sha256': file_digest(run_root / 'validation_raw.jsonl'),
                'branch_sha256': branch_hashes, 'slots': slots,
                'player_domains': dict(zip(PLAYERS, DOMAINS[:3])),
                'prompt': 'skill_forgetting.math.plain.solver_prompt',
                'native_test_used_for_valuation': False}
    write_json(run_root / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--raw_root', type=Path, required=True)
    p.add_argument('--run_root', type=Path)
    args = p.parse_args()
    config = load_config(args.config)
    manifest = prepare(config, args.raw_root, args.run_root or Path(config['run_root']))
    print(manifest['counts'])


if __name__ == '__main__':
    main()
