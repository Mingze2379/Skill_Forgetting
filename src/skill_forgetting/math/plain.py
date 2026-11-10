"""Plain Alpaca SFT examples with exactly the same user prompt as evaluation."""
from __future__ import annotations

INSTRUCTION = "Solve the following competition math problem. Show concise reasoning and final answer."


def sft_example(row: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": (
            f"Type: {str(row.get('type', 'Unknown')).strip()}\n"
            f"Level: {str(row.get('level', 'Unknown')).strip()}\n\n"
            f"Problem:\n{str(row['problem']).strip()}"
        ),
        "output": row['solution'],
    }


def solver_prompt(row: dict) -> str:
    example = sft_example(row)
    return example['instruction'] + '\n' + example['input']
