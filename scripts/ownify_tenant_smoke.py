#!/usr/bin/env python3
"""Provision, index, query, and optionally delete a temporary Ownify tenant.

This is intentionally generic. It validates the automated `/ownify/...` path
with uploaded tenant documents rather than relying on a hand-created KB.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_DOC = """# Tenant Product Guide

Acme Tenant Assistant answers questions from uploaded tenant documents.

The Bronze plan includes document search, grounded answers, basic chat sessions,
and a branded customer portal for a small team. Bronze is intended for tenants
who need a simple document assistant that answers from uploaded source material.

The Gold plan includes everything in Bronze plus team administration, advanced
workspace controls, priority document processing, saved conversation history,
and configurable public-facing assistant settings. Gold is intended for tenants
who need a managed customer support assistant connected to their own knowledge
base.

Support is available through the tenant portal.

This guide intentionally does not disclose exact monthly prices. It also does
not disclose refund deadlines, support phone numbers, or physical mailing
addresses. If a user asks for those exact missing details, the assistant should
say that the uploaded sources do not specify the detail.
"""


def request_json(method: str, url: str, payload: Dict[str, Any] | None = None, timeout: float = 240.0) -> Dict[str, Any]:
    data = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_job(endpoint: str, tenant_id: str, job_id: str, timeout: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    url = f"{endpoint}/ownify/tenants/{tenant_id}/ai/jobs/{job_id}"
    while time.time() < deadline:
        result = request_json("GET", url, None, timeout=30)
        if result.get("is_terminal"):
            return result
        time.sleep(1.0)
    raise TimeoutError(f"Ownify provisioning job did not finish within {timeout}s: {job_id}")


def provision(endpoint: str, tenant_id: str, timeout: float) -> Dict[str, Any]:
    result = request_json(
        "POST",
        f"{endpoint}/ownify/tenants/{tenant_id}/ai/provision",
        {
            "display_name": "Phase 10 Automation Smoke",
            "description": "Temporary tenant used to validate automated Ownify RAG defaults.",
            "system_prompt": "Answer only from the uploaded tenant documents.",
            "replace_existing": True,
            "idempotency_key": f"{tenant_id}-provision-v1",
            "timeout_seconds": timeout,
        },
        timeout=60,
    )
    return wait_job(endpoint, tenant_id, result["job_id"], timeout)


def upload_doc(endpoint: str, tenant_id: str, doc_path: Path, timeout: float) -> Dict[str, Any]:
    return request_json(
        "POST",
        f"{endpoint}/ownify/tenants/{tenant_id}/ai/documents",
        {
            "idempotency_key": f"{tenant_id}-documents-v1",
            "timeout_seconds": timeout,
            "documents": [
                {
                    "file_id": "phase10-product-guide",
                    "file_name": "phase10_product_guide.md",
                    "local_path": str(doc_path),
                }
            ],
        },
        timeout=timeout + 30,
    )


def query(endpoint: str, tenant_id: str, session_id: str, text: str, timeout: float) -> Dict[str, Any]:
    started = time.time()
    result = request_json(
        "POST",
        f"{endpoint}/kb/{tenant_id}/query",
        {"query": text, "session_id": session_id},
        timeout=timeout,
    )
    result["_latency_seconds"] = round(time.time() - started, 3)
    return result


def create_session(endpoint: str, tenant_id: str) -> str:
    result = request_json("POST", f"{endpoint}/ownify/tenants/{tenant_id}/ai/session/new", {}, timeout=60)
    return result["session_id"]


def delete_tenant(endpoint: str, tenant_id: str, timeout: float) -> Dict[str, Any]:
    return request_json(
        "DELETE",
        f"{endpoint}/ownify/tenants/{tenant_id}/ai?idempotency_key={tenant_id}-delete-v1&timeout_seconds={timeout}",
        None,
        timeout=timeout + 30,
    )


def validate_response(label: str, result: Dict[str, Any], *, expected_state: str | None = None) -> List[str]:
    failures: List[str] = []
    metadata = result.get("metadata") or {}
    if result.get("status") != "success":
        failures.append(f"{label}: status={result.get('status')!r}")
    if expected_state and metadata.get("answer_state") != expected_state:
        failures.append(f"{label}: answer_state expected {expected_state!r}, got {metadata.get('answer_state')!r}")
    if not metadata.get("evidence_conversation_state"):
        failures.append(f"{label}: missing evidence_conversation_state metadata")
    domain_profile = metadata.get("domain_profile") or {}
    if domain_profile.get("type") != "automated_tenant":
        failures.append(f"{label}: expected automated_tenant domain profile, got {domain_profile.get('type')!r}")
    if domain_profile.get("risk_level") not in {"elevated", "strict"}:
        failures.append(f"{label}: expected elevated/strict risk level, got {domain_profile.get('risk_level')!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--tenant-id", default=f"ownify_phase10_{int(time.time())}")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--keep-tenant", action="store_true")
    parser.add_argument("--doc-path", type=Path)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    tenant_id = args.tenant_id
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    temp_path = None

    try:
        doc_path = args.doc_path
        if doc_path is None:
            handle = tempfile.NamedTemporaryFile("w", suffix=".md", prefix="ownify_phase10_", delete=False)
            handle.write(DEFAULT_DOC)
            handle.close()
            temp_path = Path(handle.name)
            doc_path = temp_path

        provision_result = provision(endpoint, tenant_id, args.timeout)
        rows.append({"step": "provision", "result": provision_result})
        if provision_result.get("job_status") != "succeeded":
            failures.append(f"provision failed: {provision_result.get('job_status')}")

        upload_result = upload_doc(endpoint, tenant_id, doc_path, args.timeout)
        rows.append({"step": "upload", "result": upload_result})
        if upload_result.get("job_status") != "succeeded":
            failures.append(f"upload failed: {upload_result.get('job_status')}")
            raise RuntimeError("document upload failed; skipping query assertions")

        session_id = create_session(endpoint, tenant_id)
        supported = query(endpoint, tenant_id, session_id, "What does the Gold plan include?", args.timeout)
        missing = query(endpoint, tenant_id, session_id, "What is the exact monthly price?", args.timeout)
        rows.append({"step": "supported_query", "result": supported})
        rows.append({"step": "missing_detail_query", "result": missing})

        failures.extend(validate_response("supported_query", supported, expected_state="grounded_answer"))
        failures.extend(validate_response("missing_detail_query", missing, expected_state="source_limited_answer"))

        answer = str(missing.get("answer") or "").lower()
        if "not specify" not in answer and "not disclose" not in answer:
            failures.append("missing_detail_query: answer did not abstain with a missing-detail message")

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        failures.append(f"http {exc.code}: {body}")
    except Exception as exc:
        failures.append(str(exc))
    finally:
        if not args.keep_tenant:
            try:
                rows.append({"step": "delete", "result": delete_tenant(endpoint, tenant_id, args.timeout)})
            except Exception as exc:  # pragma: no cover - cleanup best effort
                rows.append({"step": "delete", "error": str(exc)})
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    summary = {
        "tenant_id": tenant_id,
        "passed": not failures,
        "failures": failures,
        "steps": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
