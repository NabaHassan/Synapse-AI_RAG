#!/usr/bin/env python3
"""Run retrieval-focused evaluation gates against the multi-KB HTTP API.

This runner complements the answer regression suite. It checks whether the
retrieval path still surfaces expected sources, entities, statutes/codes, file
names, citations, and latency characteristics before recall-sensitive search
changes are enabled broadly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "tests" / "retrieval_eval_suites" / "sprint4_retrieval_gates.jsonl"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _nested_get(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                case = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not case.get("kb_id") or not case.get("query"):
                raise ValueError(f"Case at {path}:{line_no} must include kb_id and query")
            cases.append(case)
    return cases


def create_session(endpoint: str, kb_id: str, timeout: float) -> str:
    request = urllib.request.Request(
        endpoint.rstrip("/") + f"/kb/{kb_id}/session/new",
        data=json.dumps({}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read().decode("utf-8"))["session_id"])


def post_query(endpoint: str, case: Dict[str, Any], session_id: str, timeout: float) -> Dict[str, Any]:
    payload = {"query": case["query"], "session_id": session_id}
    request = urllib.request.Request(
        endpoint.rstrip("/") + f"/kb/{case['kb_id']}/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = {"status": "http_error", "status_code": exc.code, "body": body}
    except Exception as exc:  # pragma: no cover - deployment/runtime guard
        parsed = {"status": "error", "error": str(exc)}
    parsed["_runner_latency_seconds"] = round(time.time() - started, 3)
    parsed["_runner_session_id"] = session_id
    return parsed


def _citation_texts(response: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for citation in response.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        texts.extend(
            str(citation.get(key) or "")
            for key in ["source", "source_file", "text", "chunk_id", "document_id"]
        )
    return texts


def _evidence_texts(response: Dict[str, Any]) -> List[str]:
    metadata = response.get("metadata") or {}
    state = metadata.get("evidence_conversation_state") or {}
    texts: List[str] = []
    for key in ["source_families", "document_ids", "section_ids", "entities"]:
        texts.extend(str(item) for item in _as_list(state.get(key)))
    for packet in _as_list(state.get("evidence_packets")):
        if isinstance(packet, dict):
            texts.extend(
                str(packet.get(key) or "")
                for key in ["source_file", "chunk_id", "document_id", "support_type"]
            )
            packet_meta = packet.get("metadata") or {}
            if isinstance(packet_meta, dict):
                texts.extend(str(value) for value in packet_meta.values())
    for citation in _as_list(state.get("citations")):
        if isinstance(citation, dict):
            texts.extend(str(value) for value in citation.values())
    return texts


def combined_search_text(response: Dict[str, Any]) -> str:
    pieces = [
        str(response.get("answer") or ""),
        *_citation_texts(response),
        *_evidence_texts(response),
    ]
    return _norm(" ".join(pieces))


def _contains_all(haystack: str, needles: Sequence[Any]) -> bool:
    return all(_norm(needle) in haystack for needle in needles)


def _contains_any(haystack: str, needles: Sequence[Any]) -> bool:
    return any(_norm(needle) in haystack for needle in needles)


def _rank_for_expected_source(response: Dict[str, Any], expected_sources: Sequence[Any]) -> Optional[int]:
    normalized_sources = [_norm(item) for item in expected_sources]
    if not normalized_sources:
        return None
    for idx, citation in enumerate(response.get("citations") or [], start=1):
        if not isinstance(citation, dict):
            continue
        citation_text = _norm(" ".join(str(citation.get(key) or "") for key in citation.keys()))
        if any(source and source in citation_text for source in normalized_sources):
            return idx
    return None


def _first_sparse_debug(response: Dict[str, Any]) -> Dict[str, Any]:
    sparse = ((response.get("metadata") or {}).get("retrieval_debug") or {}).get("sparse")
    if isinstance(sparse, list) and sparse:
        return sparse[0] if isinstance(sparse[0], dict) else {}
    return {}


def _first_dense_debug(response: Dict[str, Any]) -> Dict[str, Any]:
    dense = ((response.get("metadata") or {}).get("retrieval_debug") or {}).get("dense")
    if isinstance(dense, list) and dense:
        return dense[0] if isinstance(dense[0], dict) else {}
    return {}


def evaluate_case(case: Dict[str, Any], response: Dict[str, Any], *, require_metadata: bool) -> Dict[str, Any]:
    expect = case.get("expect") or {}
    metadata = response.get("metadata") or {}
    search_text = combined_search_text(response)
    citations = response.get("citations") or []
    sparse_debug = _first_sparse_debug(response)
    dense_debug = _first_dense_debug(response)
    failures: List[str] = []
    warnings: List[str] = []

    if response.get("status") != expect.get("status", "success"):
        failures.append(f"status expected {expect.get('status', 'success')!r}, got {response.get('status')!r}")
    if require_metadata and not metadata:
        failures.append("metadata missing; run with internal debug response enabled for retrieval gates")

    answer_state = metadata.get("answer_state")
    if expect.get("answer_state_in") and answer_state not in set(expect["answer_state_in"]):
        failures.append(f"answer_state expected one of {expect['answer_state_in']!r}, got {answer_state!r}")
    if int(expect.get("min_citations", 0)) and len(citations) < int(expect["min_citations"]):
        failures.append(f"citations expected >= {expect['min_citations']}, got {len(citations)}")

    if expect.get("sources_any") and not _contains_any(search_text, expect["sources_any"]):
        failures.append(f"missing expected source/file any of {expect['sources_any']!r}")
    if expect.get("sources_all") and not _contains_all(search_text, expect["sources_all"]):
        failures.append(f"missing expected source/file all of {expect['sources_all']!r}")
    if expect.get("entities_all") and not _contains_all(search_text, expect["entities_all"]):
        failures.append(f"missing expected entities all of {expect['entities_all']!r}")
    if expect.get("entities_any") and not _contains_any(search_text, expect["entities_any"]):
        failures.append(f"missing expected entities any of {expect['entities_any']!r}")
    if expect.get("codes_any") and not _contains_any(search_text, expect["codes_any"]):
        failures.append(f"missing expected statute/code any of {expect['codes_any']!r}")
    if expect.get("answer_contains_any") and not _contains_any(_norm(response.get("answer")), expect["answer_contains_any"]):
        failures.append(f"answer missing any of {expect['answer_contains_any']!r}")
    if expect.get("not_contains"):
        for phrase in expect["not_contains"]:
            if _norm(phrase) in search_text:
                failures.append(f"unexpected phrase {phrase!r}")

    latency = float(response.get("_runner_latency_seconds") or 0.0)
    if expect.get("max_latency_seconds") is not None and latency > float(expect["max_latency_seconds"]):
        failures.append(f"latency {latency:.3f}s > {float(expect['max_latency_seconds']):.3f}s")
    sparse_total_ms = _as_float(sparse_debug.get("total_ms"))
    if expect.get("max_sparse_ms") is not None and sparse_total_ms is not None:
        if sparse_total_ms > float(expect["max_sparse_ms"]):
            failures.append(f"sparse_total_ms {sparse_total_ms:.3f} > {float(expect['max_sparse_ms']):.3f}")
    if expect.get("retrieval_backend_in"):
        backend = sparse_debug.get("lexical_backend")
        if backend not in set(expect["retrieval_backend_in"]):
            failures.append(f"sparse backend expected one of {expect['retrieval_backend_in']!r}, got {backend!r}")

    expected_sources = _as_list(expect.get("sources_any")) + _as_list(expect.get("sources_all"))
    source_rank = _rank_for_expected_source(response, expected_sources)
    source_hit = bool(expected_sources and source_rank is not None)
    code_expected = bool(expect.get("codes_any"))
    code_hit = bool(code_expected and _contains_any(search_text, expect.get("codes_any") or []))
    entity_expected = bool(expect.get("entities_any") or expect.get("entities_all"))
    entity_hit = bool(
        entity_expected
        and _contains_all(search_text, expect.get("entities_all") or [])
        and (not expect.get("entities_any") or _contains_any(search_text, expect.get("entities_any") or []))
    )
    no_evidence = not citations and int(_nested_get(metadata, "context_stats.final") or 0) == 0
    source_limited = answer_state == "source_limited_answer"

    return {
        "id": case.get("id") or case["query"],
        "kb_id": case["kb_id"],
        "query": case["query"],
        "query_class": case.get("query_class") or "unknown",
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "latency_seconds": latency,
            "sparse_total_ms": sparse_total_ms,
            "dense_total_ms": _as_float(dense_debug.get("total_ms")),
            "citation_count": len(citations),
            "expected_source_rank": source_rank,
            "expected_source_mrr": round(1 / source_rank, 3) if source_rank else 0.0,
            "source_expected": bool(expected_sources),
            "source_hit": source_hit,
            "code_expected": code_expected,
            "code_hit": code_hit,
            "entity_expected": entity_expected,
            "entity_hit": entity_hit,
            "file_expected": bool(expected_sources),
            "file_hit": source_hit,
            "no_evidence": no_evidence,
            "source_limited": source_limited,
            "answer_state": answer_state,
            "sparse_backend": sparse_debug.get("lexical_backend"),
            "sparse_match_strategy": sparse_debug.get("match_strategy"),
        },
        "response_preview": {
            "answer": str(response.get("answer") or "")[:500],
            "citations": citations[:5],
            "metadata_present": bool(metadata),
        },
    }


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def build_summary(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    latencies = [row["metrics"]["latency_seconds"] for row in rows]
    sparse_ms = [row["metrics"]["sparse_total_ms"] for row in rows if row["metrics"]["sparse_total_ms"] is not None]
    source_rows = [row for row in rows if row["metrics"]["source_expected"]]
    code_rows = [row for row in rows if row["metrics"]["code_expected"]]
    entity_rows = [row for row in rows if row["metrics"]["entity_expected"]]
    file_rows = [row for row in rows if row["metrics"]["file_expected"]]
    failures = [{"id": row["id"], "failures": row["failures"]} for row in rows if row["failures"]]
    by_class: Dict[str, Dict[str, Any]] = {}
    for query_class in sorted({row["query_class"] for row in rows}):
        class_rows = [row for row in rows if row["query_class"] == query_class]
        class_latencies = [row["metrics"]["latency_seconds"] for row in class_rows]
        by_class[query_class] = {
            "total": len(class_rows),
            "passed": sum(1 for row in class_rows if row["passed"]),
            "failed": sum(1 for row in class_rows if not row["passed"]),
            "latency_p50": _percentile(class_latencies, 0.50),
            "latency_p95": _percentile(class_latencies, 0.95),
            "source_hit_rate": _rate(
                sum(1 for row in class_rows if row["metrics"]["source_hit"]),
                sum(1 for row in class_rows if row["metrics"]["source_expected"]),
            ),
        }

    gate_metrics = {
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": len(failures),
        "latency_p50": _percentile(latencies, 0.50),
        "latency_p95": _percentile(latencies, 0.95),
        "latency_p99": _percentile(latencies, 0.99),
        "sparse_ms_p50": _percentile(sparse_ms, 0.50),
        "sparse_ms_p95": _percentile(sparse_ms, 0.95),
        "citation_source_accuracy": _rate(
            sum(1 for row in source_rows if row["metrics"]["source_hit"]),
            len(source_rows),
        ),
        "file_name_hit_rate": _rate(sum(1 for row in file_rows if row["metrics"]["file_hit"]), len(file_rows)),
        "statute_code_hit_rate": _rate(sum(1 for row in code_rows if row["metrics"]["code_hit"]), len(code_rows)),
        "exact_entity_hit_rate": _rate(sum(1 for row in entity_rows if row["metrics"]["entity_hit"]), len(entity_rows)),
        "mrr": round(statistics.mean([row["metrics"]["expected_source_mrr"] for row in source_rows]), 4)
        if source_rows else None,
        "no_evidence_rate": _rate(sum(1 for row in rows if row["metrics"]["no_evidence"]), len(rows)),
        "source_limited_rate": _rate(sum(1 for row in rows if row["metrics"]["source_limited"]), len(rows)),
    }

    gate_failures = list(failures)
    if gate_metrics["latency_p95"] is not None and gate_metrics["latency_p95"] > args.max_p95_seconds:
        gate_failures.append({
            "id": "gate_latency_p95",
            "failures": [f"latency_p95 {gate_metrics['latency_p95']} > {args.max_p95_seconds}"],
        })
    for metric_name, minimum in [
        ("citation_source_accuracy", args.min_source_accuracy),
        ("file_name_hit_rate", args.min_file_hit_rate),
        ("statute_code_hit_rate", args.min_code_hit_rate),
        ("exact_entity_hit_rate", args.min_entity_hit_rate),
    ]:
        value = gate_metrics.get(metric_name)
        if value is not None and value < minimum:
            gate_failures.append({"id": f"gate_{metric_name}", "failures": [f"{value} < {minimum}"]})

    return {
        "passed": not gate_failures,
        "failures": gate_failures,
        "gate_metrics": gate_metrics,
        "by_query_class": by_class,
        "thresholds": {
            "max_p95_seconds": args.max_p95_seconds,
            "min_source_accuracy": args.min_source_accuracy,
            "min_file_hit_rate": args.min_file_hit_rate,
            "min_code_hit_rate": args.min_code_hit_rate,
            "min_entity_hit_rate": args.min_entity_hit_rate,
        },
    }


def write_outputs(output_dir: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Retrieval Evaluation Gates\n\n")
        handle.write(f"- Passed: {summary['passed']}\n")
        handle.write(f"- Total cases: {summary['gate_metrics']['total']}\n")
        handle.write(f"- Failed cases/gates: {len(summary['failures'])}\n\n")
        handle.write("## Gate Metrics\n\n```json\n")
        handle.write(json.dumps(summary["gate_metrics"], indent=2, sort_keys=True))
        handle.write("\n```\n\n")
        handle.write("## By Query Class\n\n```json\n")
        handle.write(json.dumps(summary["by_query_class"], indent=2, sort_keys=True))
        handle.write("\n```\n")
        if summary["failures"]:
            handle.write("\n## Failures\n\n")
            for failure in summary["failures"]:
                handle.write(f"- `{failure['id']}`: {'; '.join(failure['failures'])}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=Path("regression_artifacts/retrieval_gates"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--max-p95-seconds", type=float, default=20.0)
    parser.add_argument("--min-source-accuracy", type=float, default=0.80)
    parser.add_argument("--min-file-hit-rate", type=float, default=0.80)
    parser.add_argument("--min-code-hit-rate", type=float, default=0.80)
    parser.add_argument("--min-entity-hit-rate", type=float, default=0.80)
    args = parser.parse_args()

    cases = load_cases(args.input_jsonl)
    sessions: Dict[Tuple[str, str], str] = {}
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
            f"class={row['query_class']} latency={row['metrics']['latency_seconds']}s "
            f"sparse={row['metrics']['sparse_total_ms']}ms"
        )

    summary = build_summary(rows, args)
    write_outputs(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
