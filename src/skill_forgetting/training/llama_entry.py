"""Invoke LlamaFactory with a counter for the labels actually used by causal loss."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def main():
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp, export_model
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from skill_forgetting.experiments.common import write_json

    if '--cl_export' in sys.argv:
        sys.argv.remove('--cl_export')
        export_model()
        return

    counts = {'loss_tokens': 0, 'input_tokens': 0, 'examples_seen': 0, 'micro_batches': 0}
    prepared = {}
    original = CustomSeq2SeqTrainer.compute_loss

    def counted_loss(self, model, inputs, *args, **kwargs):
        labels = inputs.get('labels')
        mask = inputs.get('attention_mask')
        training = model.training
        result = original(self, model, inputs, *args, **kwargs)
        if training:
            if labels is None or labels.ndim != 2:
                raise RuntimeError('Token accounting requires unpacked 2D causal SFT labels')
            counts['loss_tokens'] += int((labels[:, 1:] != -100).sum().item())
            counts['input_tokens'] += int(mask.sum().item()) if mask is not None else labels.numel()
            counts['examples_seen'] += labels.shape[0]
            counts['micro_batches'] += 1
            prepared['prepared_dataset_rows'] = len(self.train_dataset)
        return result

    class BudgetCallback(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            values = torch.tensor(list(counts.values()), dtype=torch.int64,
                                  device=kwargs['model'].device if dist.is_initialized() else 'cpu')
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if dist.is_initialized():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            if state.is_world_process_zero:
                budget = dict(zip(counts, values.cpu().tolist()))
                budget.update(prepared)
                budget.update(optimizer_steps=state.global_step, world_size=world_size,
                              definition='Sum of non-ignored labels[:, 1:] in successful training compute_loss calls',
                              packing=False, prompt_tokens_in_loss=False)
                write_json(Path(os.environ['CL_LOSS_TOKEN_OUTPUT']), budget)

    CustomSeq2SeqTrainer.compute_loss = counted_loss
    try:
        run_exp(callbacks=[BudgetCallback()])
    finally:
        CustomSeq2SeqTrainer.compute_loss = original


if __name__ == '__main__':
    main()
