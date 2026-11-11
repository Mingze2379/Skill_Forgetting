"""Keep legacy judgments and add mathematical equivalence; export disagreement audits."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import random

from skill_forgetting.memory.utils import compare_answers
from .common import DOMAINS, file_digest, read_rows, write_json, write_rows


def judge(prediction: str, gold: str, *, finish_reason: str | None = None) -> dict:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    legacy = compare_answers(prediction, gold)
    # A truncated reasoning line has no completed final answer. Parsing its prose
    # can otherwise credit an incidental number from an unfinished derivation.
    incomplete_terminal = (finish_reason == 'length' and
                           legacy.prediction.source == 'terminal_line' and
                           not legacy.is_correct)
    config = [LatexExtractionConfig(), ExprExtractionConfig()]
    # Compare extracted final answers, rather than incidental expressions in the reasoning.
    def parsed(answer):
        return parse(r'\boxed{' + answer + '}', extraction_config=config,
                     fallback_mode='no_fallback', parsing_timeout=3)
    parsed_gold, parsed_prediction = [], []
    error = None
    equivalent = False
    try:
        parsed_gold = parsed(legacy.gold.extracted_answer)
        if not incomplete_terminal:
            parsed_prediction = parsed(legacy.prediction.extracted_answer)
        if parsed_gold and parsed_prediction:
            equivalent = bool(verify(parsed_gold, parsed_prediction, timeout_seconds=3))
    except Exception as exc:
        error = type(exc).__name__
    return {'correct': legacy.is_correct or equivalent,
            'legacy_correct': legacy.is_correct, 'math_verify_correct': equivalent,
            'gold_answer': legacy.gold.extracted_answer,
            'predicted_answer': legacy.prediction.extracted_answer,
            'prediction_answer_source': legacy.prediction.source,
            'incomplete_terminal_answer': incomplete_terminal,
            'gold_parsed': bool(parsed_gold), 'prediction_parsed': bool(parsed_prediction),
            'math_verify_error': error}


def score(predictions: Path, validation: Path, output: Path, metadata: dict | None = None) -> dict:
    expected = read_rows(validation)
    if not expected or any(row.get('split') != 'validation' for row in expected):
        raise ValueError('Valuation scoring accepts validation only; final test has a separate entry point')
    expected_by_id = {row['example_id']: row for row in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError('Duplicate validation example IDs')
    rows = read_rows(predictions)
    actual_ids = [row.get('example_id') for row in rows]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_by_id):
        raise ValueError('Predictions must cover exactly the validation manifest, once per example')
    judged, disagreements = [], []
    correct, legacy_correct, totals, clipped = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        gold = expected_by_id[row['example_id']]
        if row.get('domain') != gold['domain'] or row.get('gold_solution') != gold['solution']:
            raise ValueError('Prediction domain or gold solution differs from validation manifest')
        result = judge(row['prediction'], gold['solution'], finish_reason=row.get('finish_reason'))
        result.update(example_id=row['example_id'], domain=gold['domain'],
                      problem=gold['problem'], gold_solution=gold['solution'], prediction=row['prediction'])
        judged.append(result)
        domain = gold['domain']
        totals[domain] += 1
        correct[domain] += result['correct']
        legacy_correct[domain] += result['legacy_correct']
        clipped[domain] += row.get('finish_reason') == 'length'
        if result['legacy_correct'] != result['math_verify_correct']:
            disagreements.append(result)
    if set(totals) != set(DOMAINS):
        raise ValueError('All four domains must be represented in validation')
    accuracy = {d: correct[d] / totals[d] for d in DOMAINS}
    summary = {'accuracy': accuracy, 'v': sum(accuracy.values()) / 4,
               'legacy_accuracy': {d: legacy_correct[d] / totals[d] for d in DOMAINS},
               'correct': dict(correct), 'n': dict(totals), 'generation_truncated': dict(clipped),
               'disagreement_count': len(disagreements),
               'validation_sha256': file_digest(validation), 'prediction_sha256': file_digest(predictions),
               'scorer': {'math_verify_version': version('math-verify'),
                          'primary': 'legacy exact match OR Math-Verify final-answer equivalence',
                          'truncation_policy': 'No equivalence credit for unfinished terminal reasoning; explicit or exact final answers remain eligible',
                          'code_sha256': file_digest(Path(__file__))},
               'metadata': metadata or {}}
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'judgments.jsonl', judged)
    write_rows(output / 'disagreements.jsonl', disagreements)
    rng = random.Random(42)
    audit = list(disagreements)
    rng.shuffle(audit)
    write_rows(output / 'audit_sample.jsonl', audit[:20])
    # Score is written last: readers never treat partial inference or scoring as complete.
    write_json(output / 'score.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(score(args.predictions, args.validation, args.output)['accuracy'])


if __name__ == '__main__':
    main()
