#!/usr/bin/env python3
"""Compare two MCQ prediction files and summarize per-sample PREM deltas."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OPTION_RE = re.compile(r"^\s*[([]?([A-Z])[)\].:]\s+(.+?)\s*$")
ANSWER_RE = re.compile(r"(?<![A-Z])([A-E])(?![A-Z])")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
    elif text.startswith("{"):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError:
            try:
                rows = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON/JSONL objects in {path}")
    return rows


def answer_index(value: Any) -> int | None:
    match = ANSWER_RE.search(str(value).upper())
    return None if match is None else ord(match.group(1)) - ord("A")


def options_from_question(question: Any) -> list[str]:
    options = []
    for line in str(question or "").splitlines():
        match = OPTION_RE.match(line)
        if match:
            options.append(match.group(2))
    return options


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def option_jaccard(options: list[str]) -> float | None:
    if len(options) < 2:
        return None
    scores = []
    for left in range(len(options)):
        for right in range(left + 1, len(options)):
            a, b = token_set(options[left]), token_set(options[right])
            union = a | b
            scores.append(len(a & b) / len(union) if union else 1.0)
    return sum(scores) / len(scores)


def jaccard_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "<0.25"
    if value < 0.40:
        return "0.25-0.40"
    if value < 0.55:
        return "0.40-0.55"
    return ">=0.55"


def duration_bin(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 60:
        return "<60s"
    if seconds < 180:
        return "60-180s"
    if seconds < 600:
        return "180-600s"
    return ">=600s"


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    result = {}
    for name, group in sorted(grouped.items()):
        result[name] = {
            "count": len(group),
            "baseline_accuracy": sum(row["baseline_correct"] for row in group) / len(group),
            "memory_accuracy": sum(row["memory_correct"] for row in group) / len(group),
            "rescued": sum(row["delta"] == "rescued" for row in group),
            "harmed": sum(row["delta"] == "harmed" for row in group),
            "mean_option_jaccard": (
                sum(row["option_jaccard"] for row in group if row["option_jaccard"] is not None)
                / max(1, sum(row["option_jaccard"] is not None for row in group))
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = {str(row["id"]): row for row in load_rows(args.baseline)}
    memory_rows = {str(row["id"]): row for row in load_rows(args.memory)}
    common_ids = sorted(set(baseline_rows) & set(memory_rows))
    if not common_ids:
        raise ValueError("The two prediction files have no common sample ids")

    samples = []
    for sample_id in common_ids:
        base, mem = baseline_rows[sample_id], memory_rows[sample_id]
        answer = mem.get("answer", base.get("answer"))
        base_pred = answer_index(base.get("pred", base.get("prediction")))
        mem_pred = answer_index(mem.get("pred", mem.get("prediction")))
        base_correct = base_pred is not None and base_pred == answer
        mem_correct = mem_pred is not None and mem_pred == answer
        if base_correct and mem_correct:
            delta = "both_correct"
        elif not base_correct and not mem_correct:
            delta = "both_wrong"
        elif mem_correct:
            delta = "rescued"
        else:
            delta = "harmed"
        question = mem.get("question", base.get("question", ""))
        options = options_from_question(question)
        similarity = option_jaccard(options)
        row = {
            "id": sample_id,
            "answer": answer,
            "baseline_pred": base_pred,
            "memory_pred": mem_pred,
            "baseline_correct": bool(base_correct),
            "memory_correct": bool(mem_correct),
            "delta": delta,
            "option_count": len(options),
            "option_jaccard": similarity,
            "option_jaccard_bin": jaccard_bin(similarity),
            "duration_bin": duration_bin(mem.get("duration", base.get("duration"))),
        }
        for key in ("question_category", "question_type", "question_subtype", "level", "topic_category"):
            if key in mem or key in base:
                row[key] = mem.get(key, base.get(key))
        samples.append(row)

    summary = Counter(row["delta"] for row in samples)
    report = {
        "baseline": str(args.baseline),
        "memory": str(args.memory),
        "common_samples": len(samples),
        "summary": {
            "baseline_accuracy": sum(row["baseline_correct"] for row in samples) / len(samples),
            "memory_accuracy": sum(row["memory_correct"] for row in samples) / len(samples),
            "rescued": summary["rescued"],
            "harmed": summary["harmed"],
            "both_correct": summary["both_correct"],
            "both_wrong": summary["both_wrong"],
            "baseline_invalid": sum(row["baseline_pred"] is None for row in samples),
            "memory_invalid": sum(row["memory_pred"] is None for row in samples),
        },
        "groups": {
            "option_jaccard": aggregate(samples, "option_jaccard_bin"),
            "duration": aggregate(samples, "duration_bin"),
        },
        "samples": samples,
    }
    for key in ("question_category", "question_type", "question_subtype", "level", "topic_category"):
        if any(key in row for row in samples):
            report["groups"][key] = aggregate(samples, key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
