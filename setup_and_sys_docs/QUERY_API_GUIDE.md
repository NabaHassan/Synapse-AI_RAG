# Query API Guide: Web and KB Toggles

This guide explains how to interact with the Ownify AI conversational endpoints, specifically focusing on how to control the retrieval sources (Web Search vs. Internal Knowledge Base) for each query.

## Prerequisites

All API calls require authentication using a bearer token.
- **Header:** `Authorization: Bearer <your-internal-secret>`
- **Tenant ID:** You need to know the specific tenant ID you are querying (e.g., `acme_ai`).

---

## Step 1: Create a New Conversation Session

Before sending a query, you should create a new conversation session. This session ID will be used to track the conversation history and context.

**Endpoint:** `POST /ownify/tenants/{tenant_id}/ai/session/new`

**Example Request:**
```bash
# Store the session ID in a variable for later use
SESSION_ID=$(curl -s -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/session/new \
  -H "Authorization: Bearer your-internal-secret" | jq -r '.session_id')

echo "Started new session: $SESSION_ID"
```

---

## Step 2: Send a Query with Custom Retrieval Flags

When sending a query, you can explicitly control where the AI searches for context using the `web` and `kb` parameters in the JSON payload. 

**Endpoint:** `POST /ownify/tenants/{tenant_id}/ai/query/jobs`

### Configuration Options:
- **`web`**: `"on"` or `"off"` (Enables/Disables DuckDuckGo web search)
- **`kb`**: `"on"` or `"off"` (Enables/Disables searching the internal vector database)

### Scenario A: Web Search Only
Useful when you want the AI to search the internet for current events or public information, bypassing the internal knowledge base.

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/query/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d "{
    \"query\": \"Who is the current CEO of Tesla?\",
    \"session_id\": \"$SESSION_ID\",
    \"web\": \"on\",
    \"kb\": \"off\"
  }"
```

### Scenario B: Knowledge Base Only (Default Behavior)
Strictly restricts the AI to only use the ingested documents from your internal knowledge base. It will not search the public internet.

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/query/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d "{
    \"query\": \"What is our company policy on remote work?\",
    \"session_id\": \"$SESSION_ID\",
    \"web\": \"off\",
    \"kb\": \"on\"
  }"
```

### Scenario C: Hybrid Retrieval (Web + KB)
The pipeline will execute searches against both the internal knowledge base AND the public internet in parallel, fusing the results together to answer the user's query.

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/query/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d "{
    \"query\": \"How does our internal product compare to the latest features of Microsoft Word?\",
    \"session_id\": \"$SESSION_ID\",
    \"web\": \"on\",
    \"kb\": \"on\"
  }"
```

---

## Response Format

The query endpoint is asynchronous. It returns a `job_id` which you can then use to poll for the final result or stream the response.

**Sample Successful Response:**
```json
{
  "status": "accepted",
  "job_id": "cc6a434e644246d698cf5028456dffd0",
  "kb_id": "acme_ai",
  "request_id": "9d0876d5b10e4b398874b70ff9604865",
  "queued_at": "2026-05-08T13:53:53.932820Z",
  "timeout_seconds": 300.0
}
```

## changes made

1. src/retrival/web_retriever.py, mcp tool to fetch web results.

2. src/indexing/anchor_term_extractor.py, this file reads the kb and uses an llm to generate a kb scope, this step is done when the files are indexing. 

3. src/config/profile_updater, injects a paragraph into the general prompt template, this tells the llm to answer questions based on the kb if the question is related to the kb. The results are fetched from anchor_term_extractor.py file.



## Problems

1. With web endpoint on, general queries are answered.

2. There is no control on the web seraches, beacuse the prompt mentions 
that if there is document/context provided use it to answer the query, the queries are always answered. 
