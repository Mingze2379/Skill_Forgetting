from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DOMAINS = ('algebra', 'counting___probability', 'geometry', 'intermediate_algebra')
PLAYERS = ('A', 'B', 'C')
SUBSETS = ('empty', 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC')


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp.replace(path)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config['branch_samples'] < 3 * config['replay_per_domain']:
        raise ValueError('The three replay slots do not fit in the fixed branch sample count')
    for key in ('branch_samples', 'replay_per_domain', 'validation_per_domain',
                'stage_samples', 'stage_steps', 'branch_steps', 'batch_size',
                'gradient_accumulation_steps', 'cutoff_len'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('utility', 'macro_four_domains') != 'macro_four_domains':
        raise ValueError('This experiment uses four-domain macro validation accuracy')
    return config
