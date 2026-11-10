"""A real one-step LoRA, prompt-template, token-accounting, and adapter-inference check."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from .common import DOMAINS, load_config, read_rows, write_json, write_rows
from .prepare import export_sft
from .run import spec_for, run_training
from skill_forgetting.math.plain import solver_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    config = load_config(args.config)
    run_root = Path(config['run_root'])
    artifact = Path(config['artifact_root']) / 'smoke'
    rows = read_rows(run_root / 'branches/empty/data/rows.jsonl')[:8]
    export_sft(artifact / 'data', rows)

    from transformers import AutoTokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=config['template']))
    messages = [{'role':'user','content':solver_prompt(rows[0])},
                {'role':'assistant','content':rows[0]['solution']}]
    input_ids, _ = template.encode_oneturn(tokenizer, messages)
    inference_ids = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
    if input_ids != inference_ids:
        raise RuntimeError('SFT and generation chat templates produce different prompt token IDs')
    print('PASS: training and inference prompt token IDs are identical', flush=True)

    tiny = {**config, 'batch_size':1, 'gradient_accumulation_steps':1, 'cutoff_len':512}
    spec = spec_for(tiny, config['model'], artifact/'data', artifact/'adapter', 1)
    complete = run_training(spec)
    print('PASS: real one-step SFT and loss accounting:', complete['budget'], flush=True)

    validation = read_rows(run_root/'validation_raw.jsonl')
    selected = [r for domain in DOMAINS for r in [row for row in validation if row['domain']==domain][:2]]
    write_rows(artifact/'validation_raw.jsonl', selected)
    command = [config['eval_python'], '-m', 'skill_forgetting.math.evaluate',
               '--model', config['model'], '--lora_path', str(artifact/'adapter'),
               '--input_jsonl', str(artifact/'validation_raw.jsonl'),
               '--output_jsonl', str(artifact/'predictions.jsonl'), '--max_new_tokens','128',
               '--max_model_len','4096', '--max_num_seqs','4', '--gpu_memory_utilization','0.7']
    subprocess.run(command, check=True)
    if len(read_rows(artifact/'predictions.jsonl')) != 8:
        raise RuntimeError('Adapter generation did not cover all smoke inputs')
    write_json(artifact/'smoke_passed.json', {'training':complete, 'adapter_inference_examples':8,
               'used_for_valuation':False})
    print('SMOKE_TEST_PASSED (no smoke scores enter the valuation table)', flush=True)


if __name__ == '__main__':
    main()
