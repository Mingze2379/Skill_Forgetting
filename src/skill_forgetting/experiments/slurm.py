"""Submit one sequential baseline, eight dependent GPU branches, then a CPU report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from .common import SUBSETS, load_config, write_json


def scripts(config: dict, config_path: Path, repo: Path, run_root: Path) -> dict[str, str]:
    quote = shlex.quote
    logs = (run_root / 'slurm').resolve()
    env = ('set -euo pipefail\n'
           'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1\n'
           'export TOKENIZERS_PARALLELISM=false\n'
           f'export OMP_NUM_THREADS={config["cpus"]}\n'
           f'export PYTHONPATH={quote(str(repo / "src"))}\n'
           f'cd {quote(str(repo))}\n')
    common = ['#!/bin/bash', f'#SBATCH --partition={config["partition"]}',
              f'#SBATCH --account={config["account"]}', '#SBATCH --nodes=1', '#SBATCH --ntasks=1']
    gpu = [f'#SBATCH --cpus-per-task={config["cpus"]}', '#SBATCH --gres=gpu:1',
           f'#SBATCH --mem={config["memory"]}', f'#SBATCH --time={config["time_limit"]}']
    command = [config['train_python'], '-m', 'skill_forgetting.experiments.run',
               '--config', str(config_path.resolve()), '--run_root', str(run_root.resolve())]
    baseline = '\n'.join(common + gpu + ['#SBATCH --job-name=cl-baseline',
                 f'#SBATCH --output={logs}/baseline-%j.log']) + '\n' + env + shlex.join(command + ['baseline']) + '\n'
    branch = '\n'.join(common + gpu + ['#SBATCH --job-name=cl-replay',
             f'#SBATCH --array=0-7%{config["branch_concurrency"]}',
             f'#SBATCH --output={logs}/branch-%A_%a.log']) + '\n' + env + \
             shlex.join(command + ['branch', '--subset_index']) + ' "$SLURM_ARRAY_TASK_ID"\n'
    report = '\n'.join(common + ['#SBATCH --cpus-per-task=1', '#SBATCH --mem=2G',
             '#SBATCH --time=00:10:00', '#SBATCH --job-name=cl-report',
             f'#SBATCH --output={logs}/report-%j.log']) + '\n' + env + shlex.join(
             [config['train_python'], '-m', 'skill_forgetting.experiments.report',
              '--run_root', str(run_root.resolve())]) + '\n'
    return {'baseline': baseline, 'branches': branch, 'report': report}


def submit(config_path: Path, repo: Path, run_root: Path | None = None, dry_run=False,
           dependency_job: str | None = None) -> dict:
    if dependency_job is not None and not dependency_job.isdigit():
        raise ValueError('The prerequisite SLURM job ID must be numeric')
    config = load_config(config_path)
    run_root = (run_root or Path(config['run_root'])).resolve()
    manifest = json.loads((run_root / 'manifest.json').read_text())
    if manifest['config'] != config:
        raise ValueError('Submission configuration differs from the prepared experiment')
    directory = run_root / 'slurm'
    if (directory / 'jobs.json').exists():
        raise FileExistsError('Jobs already submitted; inspect jobs.json before submitting duplicates')
    directory.mkdir(parents=True, exist_ok=True)
    rendered = scripts(config, config_path, repo.resolve(), run_root)
    for name, script in rendered.items():
        target = directory / f'{name}.sbatch'
        target.write_text(script); target.chmod(0o700)
    plan = {'subsets': list(SUBSETS), 'config': config,
            'dependency': 'baseline afterok -> branches array afterok -> report',
            'prerequisite_job': dependency_job,
            'gpu_per_job': 1, 'maximum_concurrent_gpus': config['branch_concurrency']}
    write_json(directory / 'plan.json', plan)
    if dry_run:
        print('Rendered SLURM scripts:', directory)
        return plan
    for path in (Path(config['train_python']), Path(config['eval_python']), Path(config['model']) / 'config.json'):
        if not path.exists():
            raise FileNotFoundError(path)
    ids = {}
    for name, parent in (('baseline', None), ('branches', 'baseline'), ('report', 'branches')):
        command = ['sbatch', '--parsable']
        if parent:
            command += ['--dependency=afterok:' + ids[parent]]
        elif dependency_job:
            command += ['--dependency=afterok:' + dependency_job]
        command.append(str(directory / f'{name}.sbatch'))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        ids[name] = result.stdout.strip().split(';')[0]
        if not ids[name].isdigit():
            raise RuntimeError('Unexpected sbatch response')
        # Persist each submitted ID, even if a later submission fails.
        write_json(directory / 'jobs.json', {**plan, 'job_ids': ids})
    print(json.dumps(ids, indent=2), flush=True)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--run_root', type=Path)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--dependency_job', help='Start the baseline only after this smoke-test job succeeds')
    args = p.parse_args()
    submit(args.config, args.repo, args.run_root, args.dry_run, args.dependency_job)


if __name__ == '__main__':
    main()
