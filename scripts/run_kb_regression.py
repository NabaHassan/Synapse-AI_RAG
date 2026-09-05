#!/usr/bin/env python3
"""Run KB regression suites against the multi-KB HTTP API.

Input JSONL rows:

{"id":"cafl_training_hours","query":"...","session_group":"cafl_basics","expect":{"answer_state":"grounded_answer"}}

The runner is intentionally dependency-free: deterministic assertions first,
JSONL artifacts always, and CI-friendly exit codes. It also records warning
checks for style drift, latency drift, and dynamic answer-budget behavior.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SUITES = {
    "smoke": [
        {"id": "definition_smoke", "query": "What is this knowledge base about?"},
    ],
    "cafl_reported": [
        {"id": "cafl_training_hours", "query": "How many hours of training are required?"},
    ],
    "epstein_core": [
        {"id": "epstein_files_mentioning_name", "query": "Which files mention Jeffrey Epstein?"},
    ],
    "ownify_core": [
        {"id": "ownify_definition", "query": "What is Ownify?"},
    ],
}


def _words(text: str) -> List[str]:
    return [part for part in text.replace("\n", " ").split(" ") if part.strip()]


def _nested_get(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_cases(path: Optional[Path], suite: str) -> List[Dict[str, Any]]:
    if path is None:
        return list(DEFAULT_SUITES.get(suite) or DEFAULT_SUITES["smoke"])
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                cases.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return cases


def post_query(
    endpoint: str,
    kb_id: str,
    case: Dict[str, Any],
    timeout: float,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = endpoint.rstrip("/") + f"/kb/{kb_id}/query"
    payload = {
        "query": case["query"],
    }
    effective_session_id = case.get("session_id") or session_id
    if effective_session_id:
        payload["session_id"] = effective_session_id
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = {"status": "http_error", "status_code": exc.code, "body": body}
    except Exception as exc:  # pragma: no cover - exercised in deployment scripts
        parsed = {"status": "error", "error": str(exc)}
    parsed["_runner_latency_seconds"] = round(time.time() - started, 3)
    parsed["_runner_session_id"] = effective_session_id
    return parsed


def create_session(endpoint: str, kb_id: str, timeout: float) -> str:
    url = endpoint.rstrip("/") + f"/kb/{kb_id}/session/new"
    request = urllib.request.Request(
        url,
        data=json.dumps({}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    return str(parsed["session_id"])


def _check_expectation_block(
    block: Dict[str, Any],
    response: Dict[str, Any],
    *,
    severity: str,
) -> List[str]:
    failures: List[str] = []
    metadata = response.get("metadata") or {}
    answer = str(response.get("answer") or "")
    answer_lower = answer.lower()
    citations = response.get("citations") or []

    if block.get("status") and response.get("status") != block["status"]:
        failures.append(f"status: expected {block['status']!r}, got {response.get('status')!r}")

    for key in ["answer_state", "grounding_status"]:
        if key in block and metadata.get(key) != block[key]:
            failures.append(f"{key}: expected {block[key]!r}, got {metadata.get(key)!r}")

    if "answer_state_in" in block and metadata.get("answer_state") not in set(block["answer_state_in"]):
        failures.append(f"answer_state: expected one of {block['answer_state_in']!r}, got {metadata.get('answer_state')!r}")

    if "grounding_status_in" in block and metadata.get("grounding_status") not in set(block["grounding_status_in"]):
        failures.append(
            f"grounding_status: expected one of {block['grounding_status_in']!r}, "
            f"got {metadata.get('grounding_status')!r}"
        )

    if "contains" in block:
        for phrase in _as_list(block["contains"]):
            if str(phrase).lower() not in answer_lower:
                failures.append(f"answer missing phrase {phrase!r}")

    if "contains_any" in block:
        phrases = [str(phrase).lower() for phrase in block["contains_any"]]
        if not any(phrase in answer_lower for phrase in phrases):
            failures.append(f"answer missing any of {block['contains_any']!r}")

    if "not_contains" in block:
        for phrase in _as_list(block["not_contains"]):
            if str(phrase).lower() in answer_lower:
                failures.append(f"answer unexpectedly contains {phrase!r}")

    if "min_citations" in block and len(citations) < int(block["min_citations"]):
        failures.append(f"citations: expected at least {block['min_citations']}, got {len(citations)}")

    if "max_citations" in block and len(citations) > int(block["max_citations"]):
        failures.append(f"citations: expected at most {block['max_citations']}, got {len(citations)}")

    if block.get("require_citations") and not citations:
        failures.append("expected at least one citation")

    if block.get("require_no_citations") and citations:
        failures.append(f"expected no citations, got {len(citations)}")

    if block.get("require_evidence_state") and not metadata.get("evidence_conversation_state"):
        failures.append("missing evidence_conversation_state metadata")

    if block.get("require_domain_profile") and not metadata.get("domain_profile"):
        failures.append("missing domain_profile metadata")

    if block.get("require_generation_decision") and not metadata.get("generation_decision"):
        failures.append("missing generation_decision metadata")

    if "max_latency_seconds" in block:
        latency = float(response.get("_runner_latency_seconds") or 0.0)
        if latency > float(block["max_latency_seconds"]):
            failures.append(f"latency {latency:.3f}s > {float(block['max_latency_seconds']):.3f}s")

    if "min_answer_words" in block:
        count = len(_words(answer))
        if count < int(block["min_answer_words"]):
            failures.append(f"answer words {count} < {int(block['min_answer_words'])}")

    if "max_answer_words" in block:
        count = len(_words(answer))
        if count > int(block["max_answer_words"]):
            failures.append(f"answer words {count} > {int(block['max_answer_words'])}")

    if "max_generation_tokens" in block:
        tokens = _nested_get(metadata, "generation_decision.max_tokens")
        if tokens is None:
            failures.append("generation_decision.max_tokens missing")
        elif int(tokens) > int(block["max_generation_tokens"]):
            failures.append(f"generation max_tokens {tokens} > {int(block['max_generation_tokens'])}")

    if "min_generation_tokens" in block:
        tokens = _nested_get(metadata, "generation_decision.max_tokens")
        if tokens is None:
            failures.append("generation_decision.max_tokens missing")
        elif int(tokens) < int(block["min_generation_tokens"]):
            failures.append(f"generation max_tokens {tokens} < {int(block['min_generation_tokens'])}")

    if "evidence_admission_status" in block:
        got = (metadata.get("evidence_admission") or {}).get("admission_status")
        if got != block["evidence_admission_status"]:
            failures.append(f"evidence_admission_status: expected {block['evidence_admission_status']!r}, got {got!r}")

    if "evidence_admission_status_in" in block:
        got = (metadata.get("evidence_admission") or {}).get("admission_status")
        if got not in set(block["evidence_admission_status_in"]):
            failures.append(f"evidence_admission_status: expected one of {block['evidence_admission_status_in']!r}, got {got!r}")

    for item in _as_list(block.get("metadata_equals")):
        path = str(item.get("path") or "")
        expected = item.get("value")
        got = _nested_get(metadata, path)
        if got != expected:
            failures.append(f"metadata {path}: expected {expected!r}, got {got!r}")

    for item in _as_list(block.get("metadata_in")):
        path = str(item.get("path") or "")
        expected = set(item.get("values") or [])
        got = _nested_get(metadata, path)
        if got not in expected:
            failures.append(f"metadata {path}: expected one of {sorted(expected)!r}, got {got!r}")

    prefix = "warning" if severity == "warning" else "failure"
    return [f"{prefix}: {message}" if severity == "warning" else message for message in failures]


def check_expectations(case: Dict[str, Any], response: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    expect = case.get("expect") or {}
    warn = case.get("warn") or {}
    failures = _check_expectation_block(expect, response, severity="failure")
    warnings = _check_expectation_block(warn, response, severity="warning")
    return not failures, failures, warnings


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return round(ordered[index], 3)


def build_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [
        float((row.get("response") or {}).get("_runner_latency_seconds"))
        for row in rows
        if (row.get("response") or {}).get("_runner_latency_seconds") is not None
    ]
    answer_states: Dict[str, int] = {}
    grounding_statuses: Dict[str, int] = {}
    evidence_statuses: Dict[str, int] = {}
    citation_counts: List[int] = []
    word_counts: List[int] = []
    generation_tokens: List[int] = []

    for row in rows:
        response = row.get("response") or {}
        metadata = response.get("metadata") or {}
        answer_state = metadata.get("answer_state") or "missing"
        grounding_status = metadata.get("grounding_status") or "missing"
        evidence_status = (metadata.get("evidence_admission") or {}).get("admission_status") or "missing"
        answer_states[answer_state] = answer_states.get(answer_state, 0) + 1
        grounding_statuses[grounding_status] = grounding_statuses.get(grounding_status, 0) + 1
        evidence_statuses[evidence_status] = evidence_statuses.get(evidence_status, 0) + 1
        citation_counts.append(len(response.get("citations") or []))
        word_counts.append(len(_words(str(response.get("answer") or ""))))
        tokens = _nested_get(metadata, "generation_decision.max_tokens")
        if tokens is not None:
            generation_tokens.append(int(tokens))

    return {
        "latency": {
            "count": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "answer_states": answer_states,
        "grounding_statuses": grounding_statuses,
        "evidence_admission_statuses": evidence_statuses,
        "citation_count_avg": round(statistics.mean(citation_counts), 3) if citation_counts else 0.0,
        "answer_words_avg": round(statistics.mean(word_counts), 3) if word_counts else 0.0,
        "generation_max_tokens": {
            "min": min(generation_tokens) if generation_tokens else None,
            "max": max(generation_tokens) if generation_tokens else None,
            "avg": round(statistics.mean(generation_tokens), 3) if generation_tokens else None,
        },
    }


def write_outputs(output_dir: Path, rows: Iterable[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    summary_path = output_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# KB Regression Summary\n\n")
        handle.write(f"- KB: `{summary['kb_id']}`\n")
        handle.write(f"- Suite: `{summary['suite']}`\n")
        handle.write(f"- Total: {summary['total']}\n")
        handle.write(f"- Passed: {summary['passed']}\n")
        handle.write(f"- Failed: {summary['failed']}\n")
        handle.write(f"- Warnings: {summary['warning_count']}\n")
        handle.write("\n## Metrics\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(summary.get("metrics") or {}, indent=2, sort_keys=True))
        handle.write("\n```\n")
        if summary["failures"]:
            handle.write("\n## Failures\n\n")
            for item in summary["failures"]:
                handle.write(f"- `{item['id']}`: {'; '.join(item['failures'])}\n")
        if summary["warnings"]:
            handle.write("\n## Warnings\n\n")
            for item in summary["warnings"]:
                handle.write(f"- `{item['id']}`: {'; '.join(item['warnings'])}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("regression_artifacts"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--session-prefix", default=None)
    args = parser.parse_args()

    cases = load_cases(args.input_jsonl, args.suite)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    session_prefix = args.session_prefix or f"reg_{args.suite}_{uuid.uuid4().hex[:10]}"
    session_groups: Dict[str, str] = {}

    for case in cases:
        group = case.get("session_group")
        session_id = None
        if group:
            if str(group) not in session_groups:
                try:
                    session_groups[str(group)] = create_session(args.endpoint, args.kb_id, min(args.timeout, 60.0))
                except Exception:
                    session_groups[str(group)] = f"{session_prefix}_{group}"
            session_id = session_groups[str(group)]
        response = post_query(args.endpoint, args.kb_id, case, timeout=args.timeout, session_id=session_id)
        passed, case_failures, case_warnings = check_expectations(case, response)
        row = {
            "id": case.get("id") or case.get("query"),
            "query": case.get("query"),
            "session_group": group,
            "session_id": response.get("_runner_session_id"),
            "passed": passed,
            "failures": case_failures,
            "warnings": case_warnings,
            "response": response,
        }
        rows.append(row)
        if not passed:
            failures.append({"id": row["id"], "failures": case_failures})
        if case_warnings:
            warnings.append({"id": row["id"], "warnings": case_warnings})
        warning_suffix = f" WARN={len(case_warnings)}" if case_warnings else ""
        print(f"{'PASS' if passed else 'FAIL'} {row['id']} ({response.get('_runner_latency_seconds')}s){warning_suffix}")

    summary = {
        "kb_id": args.kb_id,
        "suite": args.suite,
        "total": len(rows),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "warning_count": sum(len(item["warnings"]) for item in warnings),
        "warnings": warnings,
        "metrics": build_metrics(rows),
    }
    write_outputs(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
