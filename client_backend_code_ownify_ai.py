import time
import requests


def _base_url(vm_ip="localhost", port=8000):
    return f"http://{vm_ip}:{port}"


def _headers(api_key=None):
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def provision_tenant_ai(
        tenant_id,
        display_name,
        description=None,
        system_prompt=None,
        ai_config=None,
        idempotency_key=None,
        replace_existing=False,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    payload = {
        "display_name": display_name,
        "replace_existing": replace_existing,
    }
    if description is not None:
        payload["description"] = description
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if ai_config is not None:
        payload["ai_config"] = ai_config
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds

    response = requests.post(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/provision",
        json=payload,
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_ownify_job_status(
        tenant_id,
        job_id,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    response = requests.get(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/jobs/{job_id}",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def wait_for_ownify_job(
        tenant_id,
        job_id,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        poll_timeout=900,
        poll_interval=1.5,
        request_timeout=30,
):
    deadline = time.time() + poll_timeout

    while time.time() < deadline:
        status = get_ownify_job_status(
            tenant_id=tenant_id,
            job_id=job_id,
            api_key=api_key,
            vm_ip=vm_ip,
            port=port,
            timeout=request_timeout,
        )
        if status.get("is_terminal"):
            job_status = status.get("job_status")
            if job_status in {"succeeded", "succeeded_with_errors"}:
                return status
            raise RuntimeError(
                f"Ownify job did not succeed. job_status={job_status}, error={status.get('error')}"
            )

        time.sleep(poll_interval)

    raise TimeoutError(f"Ownify job polling timed out after {poll_timeout} seconds")


def provision_tenant_ai_and_wait(
        tenant_id,
        display_name,
        description=None,
        system_prompt=None,
        ai_config=None,
        idempotency_key=None,
        replace_existing=False,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        submit_timeout=30,
        poll_timeout=900,
        poll_interval=1.5,
):
    accepted = provision_tenant_ai(
        tenant_id=tenant_id,
        display_name=display_name,
        description=description,
        system_prompt=system_prompt,
        ai_config=ai_config,
        idempotency_key=idempotency_key,
        replace_existing=replace_existing,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        vm_ip=vm_ip,
        port=port,
        timeout=submit_timeout,
    )
    return wait_for_ownify_job(
        tenant_id=tenant_id,
        job_id=accepted["job_id"],
        api_key=api_key,
        vm_ip=vm_ip,
        port=port,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        request_timeout=submit_timeout,
    )


def update_tenant_ai_config(
        tenant_id,
        display_name=None,
        description=None,
        system_prompt=None,
        ai_config=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=60,
):
    payload = {}
    if display_name is not None:
        payload["display_name"] = display_name
    if description is not None:
        payload["description"] = description
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if ai_config is not None:
        payload["ai_config"] = ai_config

    response = requests.post(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/config",
        json=payload,
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def add_documents(
        tenant_id,
        documents,
        idempotency_key=None,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=900,
):
    """
    documents example:
    [
        {
            "file_id": "doc-uuid-123",
            "file_name": "faq.pdf",
            "sas_url": "https://storage.blob.core.windows.net/..."
        }
    ]

    The API returns only after indexing finishes, but indexing still runs
    through the Ownify async job queue and indexing executor.
    """
    payload = {
        "documents": documents,
    }
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds

    response = requests.post(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/documents",
        json=payload,
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def list_documents(
        tenant_id,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    response = requests.get(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/documents",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def delete_document(
        tenant_id,
        file_id,
        file_name=None,
        idempotency_key=None,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=900,
):
    """
    The API returns only after deletion finishes, but deletion still runs
    through the Ownify async job queue and indexing executor.
    """
    params = {}
    if file_name is not None:
        params["file_name"] = file_name
    if idempotency_key is not None:
        params["idempotency_key"] = idempotency_key
    if timeout_seconds is not None:
        params["timeout_seconds"] = timeout_seconds

    response = requests.delete(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/documents/{file_id}",
        params=params,
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_tenant_ai_status(
        tenant_id,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    response = requests.get(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/status",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def delete_tenant_ai(
        tenant_id,
        idempotency_key=None,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=900,
):
    """
    Fully deletes the tenant from the AI system
    """
    params = {}
    if idempotency_key is not None:
        params["idempotency_key"] = idempotency_key
    if timeout_seconds is not None:
        params["timeout_seconds"] = timeout_seconds

    response = requests.delete(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai",
        params=params,
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def create_session(
        tenant_id,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    response = requests.post(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/session/new",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def submit_question_job(
        tenant_id,
        question,
        session_id=None,
        user_id=None,
        top_k=None,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    payload = {
        "query": question,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if user_id is not None:
        payload["user_id"] = user_id
    if top_k is not None:
        payload["top_k"] = top_k
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds

    response = requests.post(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/query/jobs",
        json=payload,
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_query_job_status(
        tenant_id,
        job_id,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        timeout=30,
):
    response = requests.get(
        f"{_base_url(vm_ip, port)}/ownify/tenants/{tenant_id}/ai/query/jobs/{job_id}",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def ask_question(
        tenant_id,
        question,
        session_id=None,
        user_id=None,
        top_k=None,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        submit_timeout=30,
        poll_timeout=300,
        poll_interval=1.5,
):
    job = submit_question_job(
        tenant_id=tenant_id,
        question=question,
        session_id=session_id,
        user_id=user_id,
        top_k=top_k,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        vm_ip=vm_ip,
        port=port,
        timeout=submit_timeout,
    )

    job_id = job["job_id"]
    deadline = time.time() + poll_timeout

    while time.time() < deadline:
        status = get_query_job_status(
            tenant_id=tenant_id,
            job_id=job_id,
            api_key=api_key,
            vm_ip=vm_ip,
            port=port,
            timeout=submit_timeout,
        )

        if status.get("is_terminal"):
            job_status = status.get("job_status")
            if job_status == "succeeded":
                return status.get("result", {})
            raise RuntimeError(
                f"Query job did not succeed. job_status={job_status}, error={status.get('error')}"
            )

        time.sleep(poll_interval)

    raise TimeoutError(f"Query job polling timed out after {poll_timeout} seconds")


def ask_question_with_new_session(
        tenant_id,
        question,
        user_id=None,
        top_k=None,
        timeout_seconds=None,
        api_key=None,
        vm_ip="localhost",
        port=8000,
        submit_timeout=30,
        poll_timeout=300,
        poll_interval=1.5,
):
    session = create_session(
        tenant_id=tenant_id,
        api_key=api_key,
        vm_ip=vm_ip,
        port=port,
        timeout=submit_timeout,
    )
    session_id = session["session_id"]
    result = ask_question(
        tenant_id=tenant_id,
        question=question,
        session_id=session_id,
        user_id=user_id,
        top_k=top_k,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        vm_ip=vm_ip,
        port=port,
        submit_timeout=submit_timeout,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )
    return {
        "session_id": session_id,
        "result": result,
    }


if __name__ == "__main__":
    tenant_id = "acme_ai"
    api_key = None

    status = get_tenant_ai_status(
        tenant_id=tenant_id,
        api_key=api_key,
        vm_ip="localhost",
        port=8000,
    )
    print(status)


# example usage:
#
# from client_backend_code_ownify_ai import (
#     provision_tenant_ai_and_wait,
#     add_documents,
#     create_session,
#     ask_question,
# )
#
# tenant_id = "client_abdullah_ai"
#
# idempotency_key = tenant_id+id-key+provision+rand(3)
# provision_tenant_ai_and_wait(
#     tenant_id=tenant_id,
#     display_name="Abdullah AI",
#     system_prompt="You are Abdullah AI. Answer using the knowledge base content first.",
#     idempotency_key="abdullah-provision-v1",
#     api_key="our-own-api-key",
#     vm_ip="172.190.29.113",
#     port=8000,
# )
#
# idempotency_key = tenant_id+id-key+documents+rand(3)
# add_documents(
#     tenant_id=tenant_id,
#     documents=[
#         {
#             "file_id": "doc-uuid-123",
#             "file_name": "faq.pdf",
#             "sas_url": "https://storage.blob.core.windows.net/...",
#         }
#     ],
#     idempotency_key="abdullah-docs-batch-001",
#     api_key="our-own-api-key",
#     vm_ip="172.190.29.113",
#     port=8000,
# )
#
# session = create_session(
#     tenant_id=tenant_id,
#     api_key="our-own-api-key",
#     vm_ip="172.190.29.113",
#     port=8000,
# )
# session_id = session["session_id"]
#
# answer = ask_question(
#     tenant_id=tenant_id,
#     question="What does Acme offer?",
#     session_id=session_id,
#     api_key="our-own-api-key",
#     vm_ip="172.190.29.113",
#     port=8000,
# )
