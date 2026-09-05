#!/usr/bin/env python3
"""Run Phase 11 regression gates for CAFL, Epstein, and an Ownify tenant.

The Ownify leg provisions a temporary tenant, indexes the provided document,
runs the tenant suite, and deletes the tenant unless --keep-ownify-tenant is set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ownify_tenant_smoke import delete_tenant, provision, upload_doc  # noqa: E402


def run_command(command: List[str], label: str) -> Dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 3),
        "output": completed.stdout,
        "passed": completed.returncode == 0,
    }


def load_summary(output_dir: Path) -> Dict[str, Any]:
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.md"
    rows: List[Dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    return {
        "summary_md": str(summary_path),
        "results_jsonl": str(results_path),
        "rows": len(rows),
        "failed": sum(1 for row in rows if not row.get("passed")),
        "warnings": sum(len(row.get("warnings") or []) for row in rows),
    }


def run_regression(
    *,
    endpoint: str,
    kb_id: str,
    suite_name: str,
    input_jsonl: Path,
    output_dir: Path,
    timeout: float,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_kb_regression.py"),
        "--endpoint",
        endpoint,
        "--kb-id",
        kb_id,
        "--suite",
        suite_name,
        "--input-jsonl",
        str(input_jsonl),
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(timeout),
    ]
    result = run_command(command, suite_name)
    result["artifacts"] = load_summary(output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("regression_artifacts/phase11"))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--cafl-kb-id", default="client_cafl_kb")
    parser.add_argument("--epstein-kb-id", default="client_epstein_kb_09_03_2025")
    parser.add_argument("--ownify-tenant-id", default=f"ownify_phase11_eval_{int(time.time())}")
    parser.add_argument("--ownify-doc-path", type=Path, default=REPO_ROOT / "test_document.md")
    parser.add_argument("--keep-ownify-tenant", action="store_true")
    parser.add_argument("--skip-cafl", action="store_true")
    parser.add_argument("--skip-epstein", action="store_true")
    parser.add_argument("--skip-ownify", action="store_true")
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suites_dir = REPO_ROOT / "tests" / "regression_suites"
    runs: List[Dict[str, Any]] = []
    setup_steps: List[Dict[str, Any]] = []
    failures: List[str] = []

    if not args.skip_cafl:
        runs.append(
            run_regression(
                endpoint=endpoint,
                kb_id=args.cafl_kb_id,
                suite_name="cafl_phase11",
                input_jsonl=suites_dir / "cafl_phase11.jsonl",
                output_dir=output_dir / "cafl_phase11",
                timeout=args.timeout,
            )
        )
    if not args.skip_epstein:
        runs.append(
            run_regression(
                endpoint=endpoint,
                kb_id=args.epstein_kb_id,
                suite_name="epstein_phase11",
                input_jsonl=suites_dir / "epstein_phase11.jsonl",
                output_dir=output_dir / "epstein_phase11",
                timeout=args.timeout,
            )
        )

    ownify_deleted = False
    try:
        if args.skip_ownify:
            raise StopIteration
        provision_result = provision(endpoint, args.ownify_tenant_id, args.timeout)
        setup_steps.append({"step": "ownify_provision", "result": provision_result})
        if provision_result.get("job_status") != "succeeded":
            failures.append(f"Ownify provision failed: {provision_result.get('job_status')}")
        upload_result = upload_doc(endpoint, args.ownify_tenant_id, args.ownify_doc_path, args.timeout)
        setup_steps.append({"step": "ownify_upload", "result": upload_result})
        if upload_result.get("job_status") != "succeeded":
            failures.append(f"Ownify upload failed: {upload_result.get('job_status')}")
        else:
            runs.append(
                run_regression(
                    endpoint=endpoint,
                    kb_id=args.ownify_tenant_id,
                    suite_name="ownify_test_document_phase11",
                    input_jsonl=suites_dir / "ownify_test_document_phase11.jsonl",
                    output_dir=output_dir / "ownify_test_document_phase11",
                    timeout=args.timeout,
                )
            )
    except StopIteration:
        setup_steps.append({"step": "ownify_skipped", "result": {"skipped": True}})
    finally:
        if not args.keep_ownify_tenant and not args.skip_ownify:
            try:
                delete_result = delete_tenant(endpoint, args.ownify_tenant_id, args.timeout)
                setup_steps.append({"step": "ownify_delete", "result": delete_result})
                ownify_deleted = True
            except Exception as exc:  # pragma: no cover - cleanup best effort
                setup_steps.append({"step": "ownify_delete", "error": str(exc)})
                failures.append(f"Ownify cleanup failed: {exc}")

    for run in runs:
        if not run.get("passed"):
            failures.append(f"{run['label']} failed with exit code {run['returncode']}")

    summary = {
        "passed": not failures,
        "failures": failures,
        "endpoint": endpoint,
        "output_dir": str(output_dir),
        "ownify_tenant_id": args.ownify_tenant_id,
        "ownify_deleted": ownify_deleted,
        "setup_steps": setup_steps,
        "runs": runs,
    }
    summary_path = output_dir / "phase11_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = output_dir / "phase11_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Phase 11 Evaluation Summary\n\n")
        handle.write(f"- Passed: {summary['passed']}\n")
        handle.write(f"- Ownify tenant deleted: {ownify_deleted}\n")
        handle.write(f"- JSON summary: `{summary_path}`\n\n")
        handle.write("## Runs\n\n")
        for run in runs:
            artifacts = run.get("artifacts") or {}
            handle.write(
                f"- `{run['label']}`: {'PASS' if run.get('passed') else 'FAIL'}; "
                f"rows={artifacts.get('rows')}; failed={artifacts.get('failed')}; "
                f"warnings={artifacts.get('warnings')}; seconds={run.get('seconds')}\n"
            )
        if failures:
            handle.write("\n## Failures\n\n")
            for failure in failures:
                handle.write(f"- {failure}\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
