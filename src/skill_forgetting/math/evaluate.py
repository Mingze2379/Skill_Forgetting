#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


from skill_forgetting.memory.utils import build_example_id, compare_answers
from skill_forgetting.math.plain import solver_prompt


def build_prompt(row: dict[str, Any]) -> str:
    return solver_prompt(row)


def apply_chat_template(prompts: list[str], model_name: str) -> list[str]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return formatted


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def prepare_skill_augmented_prompts(
    rows: list[dict[str, Any]],
    *,
    skill_memory_dir: Path,
    top_k_skills: int | None,
    skill_prompt_style: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    from skill_forgetting.memory.prompts import build_solver_prompt
    from skill_forgetting.memory.retriever import HybridSkillRetriever
    from skill_forgetting.memory.store import SkillMemoryStore
    from skill_forgetting.memory.utils import build_example_id

    store = SkillMemoryStore(skill_memory_dir)
    retriever = HybridSkillRetriever(store)
    prompts: list[str] = []
    retrieval_rows: list[dict[str, Any]] = []

    for row in rows:
        retrieved_skills = retriever.retrieve(row, top_k=top_k_skills)
        prompt = build_solver_prompt(row, store, retrieved_skills, prompt_style=skill_prompt_style)
        example_id = build_example_id(
            str(row.get("problem", "")),
            str(row.get("type", "")),
            str(row.get("level", "")),
        )
        top_skill = retrieved_skills[0] if retrieved_skills else None
        retrieval_rows.append(
            {
                "example_id": example_id,
                "problem": row.get("problem", ""),
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "retrieved_skill": top_skill.skill_id if top_skill else None,
                "retrieved_skill_family": top_skill.skill_family if top_skill else None,
                "retrieval_score": top_skill.total_score if top_skill else None,
                "retrieved_skills": [item.to_dict() for item in retrieved_skills],
                "skill_prompt_style": skill_prompt_style,
                "prompt": prompt,
            }
        )
        prompts.append(prompt)

    return prompts, retrieval_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM inference on competition math JSONL data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skill_memory_dir", type=Path, default=None)
    parser.add_argument("--top_k_skills", type=int, default=None)
    parser.add_argument("--retrieval_log_jsonl", type=Path, default=None)
    parser.add_argument(
        "--skill_prompt_style",
        type=str,
        default="full",
        choices=["full", "short"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    rows = load_rows(args.input_jsonl, args.limit)
    if not rows:
        raise ValueError("Evaluation input is empty.")
    if args.retrieval_log_jsonl is not None and args.skill_memory_dir is None:
        raise ValueError("--retrieval_log_jsonl requires --skill_memory_dir.")

    retrieval_rows: list[dict[str, Any]] = []
    if args.skill_memory_dir is not None:
        prompts, retrieval_rows = prepare_skill_augmented_prompts(
            rows,
            skill_memory_dir=args.skill_memory_dir,
            top_k_skills=args.top_k_skills,
            skill_prompt_style=args.skill_prompt_style,
        )
    else:
        prompts = [build_prompt(row) for row in rows]

    prompts = apply_chat_template(prompts, args.model)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.retrieval_log_jsonl is not None:
        args.retrieval_log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    if args.lora_path:
        from vllm.lora.request import LoRARequest

        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest("competition_math_lora", 1, args.lora_path),
        )
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    if len(outputs) != len(rows):
        raise RuntimeError("Inference returned fewer predictions than evaluation examples.")

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, (row, prompt, out) in enumerate(zip(rows, prompts, outputs)):
            text = out.outputs[0].text if out.outputs else ""
            record = {
                "example_id": row.get("example_id") or build_example_id(
                    str(row.get("problem", "")), str(row.get("type", "")), str(row.get("level", ""))),
                "domain": row.get("domain"),
                "split": row.get("split"),
                "problem": row.get("problem", ""),
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "gold_solution": row.get("solution", ""),
                "prompt": prompt,
                "prediction": text,
                "generated_tokens": len(out.outputs[0].token_ids) if out.outputs else 0,
                "finish_reason": out.outputs[0].finish_reason if out.outputs else None,
            }
            if retrieval_rows:
                retrieval = retrieval_rows[idx]
                record.update(
                    {
                        "example_id": retrieval.get("example_id"),
                        "retrieved_skill": retrieval.get("retrieved_skill"),
                        "retrieved_skill_family": retrieval.get("retrieved_skill_family"),
                        "retrieval_score": retrieval.get("retrieval_score"),
                        "retrieved_skills": retrieval.get("retrieved_skills", []),
                    }
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.retrieval_log_jsonl is not None:
        with args.retrieval_log_jsonl.open("w", encoding="utf-8") as handle:
            for retrieval in retrieval_rows:
                handle.write(json.dumps(retrieval, ensure_ascii=False) + "\n")

    print(f"Wrote {len(outputs)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
