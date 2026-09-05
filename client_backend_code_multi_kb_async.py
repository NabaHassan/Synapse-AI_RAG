import time
import requests


def _base_url(vm_ip="localhost", port=8004):
    return f"http://{vm_ip}:{port}"


def create_session(kb_id, vm_ip="localhost", port=8004, timeout=30):
    response = requests.post(
        f"{_base_url(vm_ip, port)}/kb/{kb_id}/session/new",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def add_document(
        kb_id,
        file_id,
        file_name,
        sas_url,
        vm_ip="localhost",
        port=8004,
        timeout=60,
):
    payload = {
        "file_id": file_id,
        "file_name": file_name,
        "sas_url": sas_url,
    }

    response = requests.post(
        f"{_base_url(vm_ip, port)}/kb/{kb_id}/documents/add",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def delete_document(
        kb_id,
        file_id,
        file_name=None,
        vm_ip="localhost",
        port=8004,
        timeout=60,
):
    params = {}
    if file_name:
        params["file_name"] = file_name

    response = requests.delete(
        f"{_base_url(vm_ip, port)}/kb/{kb_id}/documents/{file_id}",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def submit_question_job(kb_id, question, session_id=None, user_id=None, vm_ip="localhost", port=8004, timeout=30):
    payload = {"query": question}
    if session_id:
        payload["session_id"] = session_id
    if user_id:
        payload["user_id"] = user_id

    response = requests.post(
        f"{_base_url(vm_ip, port)}/kb/{kb_id}/query/jobs",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_job_status(kb_id, job_id, vm_ip="localhost", port=8004, timeout=30):
    response = requests.get(
        f"{_base_url(vm_ip, port)}/kb/{kb_id}/query/jobs/{job_id}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def cancel_job(kb_id, job_id, vm_ip="localhost", port=8004, timeout=30):
    response = requests.post(
        f"{_base_url(vm_ip, port)}/kb/{kb_id}/query/jobs/{job_id}/cancel",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def ask_question(
        kb_id,
        question,
        session_id=None,
        user_id=None,
        vm_ip="localhost",
        port=8004,
        submit_timeout=15,
        poll_timeout=140,
        poll_interval=1.5,
):
    job = submit_question_job(
        kb_id=kb_id,
        question=question,
        session_id=session_id,
        user_id=user_id,
        vm_ip=vm_ip,
        port=port,
        timeout=submit_timeout,
    )

    job_id = job["job_id"]
    deadline = time.time() + poll_timeout

    while time.time() < deadline:
        status = get_job_status(
            kb_id=kb_id,
            job_id=job_id,
            vm_ip=vm_ip,
            port=port,
            timeout=submit_timeout,
        )

        if status.get("is_terminal"):
            job_status = status.get("job_status")
            if job_status == "succeeded":
                return status.get("result", {})
            raise RuntimeError(
                f"Async job did not succeed. job_status={job_status}, error={status.get('error')}"
            )

        time.sleep(poll_interval)

    raise TimeoutError(f"Async job polling timed out after {poll_timeout} seconds")


if __name__ == "__main__":
    kb_id = "client_epstein_kb_09_03_2025"
    session = create_session(kb_id=kb_id)
    session_id = session["session_id"]
    # session_id = "session_2cf9e167c26c"
    result = ask_question(
        kb_id=kb_id,
        question="Who is Jeffrey Epstein?",
        session_id=session_id,
        vm_ip="localhost",
        port=8004,
    )
    print(result)
