#!/usr/bin/env python3
"""Run advanced retrieval evaluation for candidate rerank/late-interaction work.

This runner is intentionally behavior-neutral: it does not enable a new model
or change serving config. It evaluates the currently running retrieval stack on
hard cases and can optionally compare against a previous advanced-eval summary.
Use it before canarying stronger local rerankers, late-interaction retrieval, or
larger candidate budgets.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "tests" / "retrieval_eval_suites" / "sprint7_advanced_retrieval_hard.jsonl"

if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_retrieval_eval_gates import (  # noqa: E402
    _as_float,
    _percentile,
    _rate,
    build_summary,
    create_session,
    evaluate_case,
    load_cases,
    post_query,
    write_outputs,
)


def _metric(row: Dict[str, Any], key: str) -> Optional[float]:
    return _as_float((row.get("metrics") or {}).get(key))


def _all_expected_hits(rows: Sequence[Dict[str, Any]], expected_key: str, hit_key: str) -> Optional[float]:
    expected = [row for row in rows if (row.get("metrics") or {}).get(expected_key)]
    return _rate(sum(1 for row in expected if (row.get("metrics") or {}).get(hit_key)), len(expected))


def _load_baseline(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if path.is_dir():
        path = path / "advanced_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None:
        return None
    return round(float(current) - float(baseline), 4)


def _comparison(current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not baseline:
        return {"enabled": False}
    current_metrics = current.get("advanced_metrics") or {}
    baseline_metrics = baseline.get("advanced_metrics") or {}
    return {
        "enabled": True,
        "baseline_label": baseline.get("label"),
        "pass_rate_delta": _delta(current_metrics.get("pass_rate"), baseline_metrics.get("pass_rate")),
        "hard_source_accuracy_delta": _delta(
            current_metrics.get("hard_source_accuracy"),
            baseline_metrics.get("hard_source_accuracy"),
        ),
        "hard_entity_hit_rate_delta": _delta(
            current_metrics.get("hard_entity_hit_rate"),
            baseline_metrics.get("hard_entity_hit_rate"),
        ),
        "hard_code_hit_rate_delta": _delta(
            current_metrics.get("hard_code_hit_rate"),
            baseline_metrics.get("hard_code_hit_rate"),
        ),
        "latency_p95_delta": _delta(current_metrics.get("latency_p95"), baseline_metrics.get("latency_p95")),
        "rerank_ms_p95_delta": _delta(
            current_metrics.get("rerank_ms_p95"),
            baseline_metrics.get("rerank_ms_p95"),
        ),
    }


def _canary_decision(
    *,
    gate_summary: Dict[str, Any],
    advanced_metrics: Dict[str, Any],
    comparison: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    blockers: List[str] = []
    if not gate_summary.get("passed"):
        blockers.append("retrieval_gate_failed")
    if float(advanced_metrics.get("pass_rate") or 0.0) < args.min_hard_pass_rate:
        blockers.append("hard_pass_rate_below_threshold")
    if (
        advanced_metrics.get("hard_source_accuracy") is not None
        and float(advanced_metrics["hard_source_accuracy"]) < args.min_hard_source_accuracy
    ):
        blockers.append("hard_source_accuracy_below_threshold")
    if (
        advanced_metrics.get("latency_p95") is not None
        and float(advanced_metrics["latency_p95"]) > args.max_hard_p95_seconds
    ):
        blockers.append("hard_latency_p95_above_threshold")
    if comparison.get("enabled"):
        if comparison.get("hard_source_accuracy_delta") is not None and comparison["hard_source_accuracy_delta"] < -args.max_accuracy_regression:
            blockers.append("source_accuracy_regressed_vs_baseline")
        if comparison.get("pass_rate_delta") is not None and comparison["pass_rate_delta"] < -args.max_accuracy_regression:
            blockers.append("pass_rate_regressed_vs_baseline")
        if comparison.get("latency_p95_delta") is not None and comparison["latency_p95_delta"] > args.max_p95_latency_regression:
            blockers.append("latency_regressed_vs_baseline")

    return {
        "eligible": not blockers,
        "blockers": blockers,
        "recommendation": (
            "eligible_for_small_canary"
            if not blockers
            else "keep_shadow_or_reject_candidate"
        ),
    }


def _augment_summary(
    *,
    rows: Sequence[Dict[str, Any]],
    gate_summary: Dict[str, Any],
    label: str,
    baseline: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    latencies = [_metric(row, "latency_seconds") for row in rows]
    latencies = [value for value in latencies if value is not None]
    rerank_ms = []
    dense_ms = []
    sparse_ms = []
    for row in rows:
        preview = row.get("response_preview") or {}
        # The lightweight gate row only stores sparse/dense in metrics today.
        metrics = row.get("metrics") or {}
        if metrics.get("sparse_total_ms") is not None:
            sparse_ms.append(float(metrics["sparse_total_ms"]))
        if metrics.get("dense_total_ms") is not None:
            dense_ms.append(float(metrics["dense_total_ms"]))
        debug = preview.get("retrieval_debug") or {}
        if debug:
            value = ((debug.get("rerank") or {}).get("total_ms"))
            if value is not None:
                rerank_ms.append(float(value))

    advanced_metrics = {
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("passed")),
        "failed": sum(1 for row in rows if not row.get("passed")),
        "pass_rate": _rate(sum(1 for row in rows if row.get("passed")), len(rows)),
        "latency_p50": _percentile(latencies, 0.50),
        "latency_p95": _percentile(latencies, 0.95),
        "dense_ms_p95": _percentile(dense_ms, 0.95),
        "sparse_ms_p95": _percentile(sparse_ms, 0.95),
        "rerank_ms_p95": _percentile(rerank_ms, 0.95),
        "hard_source_accuracy": _all_expected_hits(rows, "source_expected", "source_hit"),
        "hard_entity_hit_rate": _all_expected_hits(rows, "entity_expected", "entity_hit"),
        "hard_code_hit_rate": _all_expected_hits(rows, "code_expected", "code_hit"),
        "no_evidence_rate": _rate(sum(1 for row in rows if (row.get("metrics") or {}).get("no_evidence")), len(rows)),
        "source_limited_rate": _rate(sum(1 for row in rows if (row.get("metrics") or {}).get("source_limited")), len(rows)),
    }
    comparison = _comparison({"advanced_metrics": advanced_metrics, "label": label}, baseline)
    canary = _canary_decision(
        gate_summary=gate_summary,
        advanced_metrics=advanced_metrics,
        comparison=comparison,
        args=args,
    )
    return {
        "label": label,
        "suite": str(args.input_jsonl),
        "endpoint": args.endpoint,
        "passed": bool(gate_summary.get("passed")) and bool(canary["eligible"]),
        "advanced_metrics": advanced_metrics,
        "gate_summary": gate_summary,
        "comparison": comparison,
        "canary_decision": canary,
        "thresholds": {
            "min_hard_pass_rate": args.min_hard_pass_rate,
            "min_hard_source_accuracy": args.min_hard_source_accuracy,
            "max_hard_p95_seconds": args.max_hard_p95_seconds,
            "max_accuracy_regression": args.max_accuracy_regression,
            "max_p95_latency_regression": args.max_p95_latency_regression,
        },
    }


def _write_advanced_outputs(output_dir: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    write_outputs(output_dir, rows, summary["gate_summary"])
    (output_dir / "advanced_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "advanced_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Advanced Retrieval Evaluation\n\n")
        handle.write(f"- Label: `{summary['label']}`\n")
        handle.write(f"- Passed: {summary['passed']}\n")
        handle.write(f"- Canary recommendation: `{summary['canary_decision']['recommendation']}`\n")
        if summary["canary_decision"]["blockers"]:
            handle.write(f"- Blockers: {', '.join(summary['canary_decision']['blockers'])}\n")
        handle.write("\n## Advanced Metrics\n\n```json\n")
        handle.write(json.dumps(summary["advanced_metrics"], indent=2, sort_keys=True))
        handle.write("\n```\n\n## Comparison\n\n```json\n")
        handle.write(json.dumps(summary["comparison"], indent=2, sort_keys=True))
        handle.write("\n```\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=Path("regression_artifacts/advanced_retrieval_eval"))
    parser.add_argument("--label", default="current")
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--max-p95-seconds", type=float, default=45.0)
    parser.add_argument("--min-source-accuracy", type=float, default=0.85)
    parser.add_argument("--min-file-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-code-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-entity-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-hard-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-hard-source-accuracy", type=float, default=0.85)
    parser.add_argument("--max-hard-p95-seconds", type=float, default=45.0)
    parser.add_argument("--max-accuracy-regression", type=float, default=0.02)
    parser.add_argument("--max-p95-latency-regression", type=float, default=3.0)
    args = parser.parse_args()

    baseline = _load_baseline(args.baseline_summary)
    cases = load_cases(args.input_jsonl)
    sessions: Dict[tuple[str, str], str] = {}
    rows: List[Dict[str, Any]] = []
    for case in cases:
        group = str(case.get("session_group") or case.get("id") or case["query"])
        session_key = (case["kb_id"], group)
        if session_key not in sessions:
            sessions[session_key] = create_session(args.endpoint, case["kb_id"], min(args.timeout, 60.0))
        response = post_query(args.endpoint, case, sessions[session_key], args.timeout)
        row = evaluate_case(case, response, require_metadata=args.require_metadata)
        rows.append(row)
        print(
            f"{'PASS' if row['passed'] else 'FAIL'} {row['id']} "
            f"class={row['query_class']} latency={row['metrics']['latency_seconds']}s"
        )

    gate_args = argparse.Namespace(
        max_p95_seconds=args.max_p95_seconds,
        min_source_accuracy=args.min_source_accuracy,
        min_file_hit_rate=args.min_file_hit_rate,
        min_code_hit_rate=args.min_code_hit_rate,
        min_entity_hit_rate=args.min_entity_hit_rate,
    )
    gate_summary = build_summary(rows, gate_args)
    summary = _augment_summary(
        rows=rows,
        gate_summary=gate_summary,
        label=args.label,
        baseline=baseline,
        args=args,
    )
    _write_advanced_outputs(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
