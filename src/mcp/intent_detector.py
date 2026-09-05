"""
Google Workspace MCP Intent Detector using Qwen2.5-1.5B-Instruct.
Classifies query into service, tool, and extracts clean search parameters.
"""

import json
import logging
import re
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("transformers/torch not available. MCPIntentDetector requires transformers.")


class MCPIntentDetector:
    _instance: Optional["MCPIntentDetector"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.is_shared = False
        self._inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> "MCPIntentDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_name=model_name)
            return cls._instance

    @classmethod
    def set_shared_resources(cls, model: Any, tokenizer: Any):
        """Set preloaded model and tokenizer to avoid double loading and reduce latency."""
        instance = cls.get_instance()
        with instance._inference_lock:
            instance.model = model
            instance.tokenizer = tokenizer
            instance.is_shared = True
            logger.info("MCPIntentDetector configured to use shared preloaded LLM resources")

    def _ensure_model_loaded(self):
        """Lazily load model if shared resources were not configured."""
        if self.model is not None and self.tokenizer is not None:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers/torch not available for intent detection.")

        with self._inference_lock:
            if self.model is not None and self.tokenizer is not None:
                return

            logger.info(f"Lazily loading intent detection model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float32
                )
                self.model = self.model.to(self.device)
            logger.info(f"Intent detection model loaded on {self.device}")

    @staticmethod
    def check_explicit_read_command(query: str, connector_toggle: Optional[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Fast-path evaluation to intercept deterministic slash commands that map to direct MCP tools.
        """
        from src.mcp.slash_commands import parse_slash_command

        result = parse_slash_command(query, connector_toggle)
        if result and result.kind == "direct_mcp" and result.tool and result.params is not None:
            logger.info(
                "Slash direct MCP: /%s -> tool=%s params=%r",
                result.command,
                result.tool,
                result.params,
            )
            return result.tool, result.params
        return None

    def _sanitize_drive_query(self, query: str) -> str:
        """
        Sanitize the extracted query for Google Drive search.
        Converts conversational queries into proper Google Drive search format.
        """
        if not query:
            return ""
        
        # Remove common conversational prefixes
        prefixes_to_remove = [
            "find file", "find document", "find doc", "search file", "search document", "search doc",
            "file about", "document about", "doc about", "prompts what is in the file",
            "show me", "show", "get", "list", "find", "search", "look for", "what is in"
        ]
        
        query_lower = query.lower()
        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                query = query[len(prefix):].strip()
                break
        
        # Strip quotes and extra whitespace
        query = query.strip().strip("'\"")
        
        # If the query is empty after sanitization, return empty string
        if not query:
            return ""
        
        # Escape single quotes for Google Drive query
        query_escaped = query.replace("'", "\\'")
        
        # Return in proper Google Drive search format
        return f"name contains '{query_escaped}' or fullText contains '{query_escaped}'"
    
        return f"name contains '{query_escaped}' or fullText contains '{query_escaped}'"

    def _sanitize_gmail_query(self, query: str) -> str:
        """Compile natural language into a Gmail search q= string (#1, #14)."""
        from src.mcp.gmail_search import parse_gmail_query_struct, build_gmail_search_query

        params = parse_gmail_query_struct(query)
        compiled = build_gmail_search_query(params, stage="strict")
        logger.debug("Sanitized Gmail query %r -> %r", query, compiled)
        return compiled

    async def detect_gmail_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Called when the Email connector is on.
        Returns (tool, params) with structured Gmail search fields (#2, #6).
        """
        from src.mcp.gmail_search import (
            parse_gmail_query_struct,
            resolve_gmail_tool,
            prepare_gmail_call_params,
        )

        rule_params = parse_gmail_query_struct(query)
        default_tool = resolve_gmail_tool(query, rule_params)
        default_params = {
            "query": query,
            "gmail_struct": rule_params.to_dict(),
            "timezone": rule_params.timezone,
        }

        try:
            self._ensure_model_loaded()
        except Exception as e:
            logger.warning("Gmail intent: model unavailable (%s), using rule-based routing", e)
            return default_tool, default_params

        prompt = f"""The user has Gmail/Email enabled. Classify their query into exactly one Gmail action and extract structured search parameters.

Available tools:
1. "list_recent_inbox" — list recent inbox mail (today, yesterday, this week, check my emails, recent mail).
2. "search_threads" — search inbox by sender, subject, keywords, unread, attachment, date range.
3. "search_sent" — search sent mail (did I email X, emails I sent).
4. "get_thread" — read a specific thread when thread_id is known or user refers to a prior email/thread.

Users may narrow queries with:
- @sender or @name — restrict to emails from that person
- #inbox #sent #spam #drafts #trash #snoozed #starred #primary #promotions etc.

Return JSON with:
- "tool": one of the four tools above.
- "extracted_query": cleaned keyword string for search (not full sentence). Use "" for list_recent_inbox when only asking for recent/today mail.
- "sender": email or name if mentioned, else null.
- "subject_keywords": array of subject phrases, else [].
- "keywords": free-text keywords not covered by sender/subject, else "".
- "date_range": {{"after": "YYYY-MM-DD or null", "before": "YYYY-MM-DD or null"}} for today/yesterday/this week/last week.
- "is_unread": boolean.
- "is_starred": boolean.
- "has_attachment": boolean.
- "in_sent": boolean if asking about sent mail.
- "thread_id": Gmail thread id if present, else null.

Example 1:
Query: "what are my emails today"
Response:
{{"tool": "list_recent_inbox", "extracted_query": "", "sender": null, "subject_keywords": [], "keywords": "", "date_range": {{"after": "today", "before": "today"}}, "is_unread": false, "is_starred": false, "has_attachment": false, "in_sent": false, "thread_id": null}}

Example 2:
Query: "email from Sarah about the invoice last week"
Response:
{{"tool": "search_threads", "extracted_query": "invoice", "sender": "Sarah", "subject_keywords": ["invoice"], "keywords": "invoice", "date_range": {{"after": "last week", "before": null}}, "is_unread": false, "is_starred": false, "has_attachment": false, "in_sent": false, "thread_id": null}}

Example 3:
Query: "unread emails about project alpha"
Response:
{{"tool": "search_threads", "extracted_query": "project alpha", "sender": null, "subject_keywords": [], "keywords": "project alpha", "date_range": {{"after": null, "before": null}}, "is_unread": true, "is_starred": false, "has_attachment": false, "in_sent": false, "thread_id": null}}

Example 4:
Query: "did I email John about the contract"
Response:
{{"tool": "search_sent", "extracted_query": "contract", "sender": "John", "subject_keywords": ["contract"], "keywords": "contract", "date_range": {{"after": null, "before": null}}, "is_unread": false, "is_starred": false, "has_attachment": false, "in_sent": true, "thread_id": null}}

Query: "{query}"
Response:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured JSON output."},
            {"role": "user", "content": prompt},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            with self._inference_lock:
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                import asyncio
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()

            result = self._extract_json(response)
            valid_tools = {"list_recent_inbox", "search_threads", "search_sent", "get_thread"}
            tool = result.get("tool") if result else None
            if tool not in valid_tools:
                logger.warning("Invalid Gmail tool %r from LLM, using rule-based %s", tool, default_tool)
                tool = default_tool

            gmail_struct = {
                "tool": tool,
                "sender": result.get("sender") if result else None,
                "subject_keywords": result.get("subject_keywords") or [],
                "keywords": (result.get("keywords") or "") if result else "",
                "date_range": result.get("date_range") or {},
                "is_unread": bool(result.get("is_unread")) if result else False,
                "is_starred": bool(result.get("is_starred")) if result else False,
                "has_attachment": bool(result.get("has_attachment")) if result else False,
                "in_sent": bool(result.get("in_sent")) if result else False,
                "thread_id": result.get("thread_id") if result else None,
            }

            rule_dict = rule_params.to_dict()
            gmail_struct["location"] = rule_dict.get("location") or gmail_struct.get("location")
            gmail_struct["category"] = rule_dict.get("category") or gmail_struct.get("category")
            merged_struct = {**rule_dict, **gmail_struct}
            merged = parse_gmail_query_struct(query, llm_struct=merged_struct)
            tool = resolve_gmail_tool(query, merged)
            params = {
                "query": query,
                "gmail_struct": merged_struct,
                "timezone": merged.timezone,
            }
            if merged.thread_id:
                params["thread_id"] = merged.thread_id
                tool = "get_thread"

            logger.info("Gmail intent: query=%r -> tool=%s params=%s", query, tool, params)
            return tool, params

        except Exception as e:
            logger.warning("Gmail intent detection failed (%s), using rule-based routing", e)
            return default_tool, default_params
    
    async def detect_intent(self, query: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Detect if the user query has a Google Workspace intent (gmail, calendar, drive).
        Returns (service, tool, params) or (None, None, {}) if no match.
        """
        try:
            self._ensure_model_loaded()
        except Exception as e:
            logger.warning(f"Failed to load/initialize model for intent detection: {e}")
            return None, None, {}

        prompt = f"""Analyze the user's input query and determine if they want to interact with Google Workspace tools (Gmail, Google Calendar, or Google Drive). 

We support the following tools:
1. Service: "gmail", Tools: "list_recent_inbox", "search_threads", "search_sent", "get_thread". Used for: searching, checking, listing, or reading emails, inbox messages, threads, mail, sent mail, etc.
2. Service: "calendar", Tool: "list_events". Used for: listing, checking, viewing, scheduling, or finding calendar events, meetings, schedules, appointments, syncs, invitations, etc.
3. Service: "drive", Tool: "search_files". Used for: searching, checking, listing, finding, or reading files, documents, sheets, spreadsheets, pdfs, reports, folders, docs, etc.

Return a JSON object with the following fields:
- "is_workspace_intent": true if the query is asking to look up or interact with Gmail, Calendar, or Drive. Otherwise false.
- "service": The service name ("gmail", "calendar", "drive", or null if not a workspace intent).
- "tool": The tool name ("search_threads", "list_recent_inbox", "search_sent", "get_thread", "list_events", "search_files", or null if not a workspace intent).
- "extracted_query": A cleaned search query for the Google service search engine (e.g. if the user says "find my emails about sync meetings", "sync meetings" is the search query. If the user says "check my calendar for tomorrow", "tomorrow" is the search query. If the user says "show my drive files about design", "design" is the search query. If no specific search query is mentioned, use a blank string "").

Example 1:
Query: "find my emails about sync meetings"
Response:
{{
  "is_workspace_intent": true,
  "service": "gmail",
  "tool": "search_threads",
  "extracted_query": "sync meetings"
}}

Example 2:
Query: "what meetings do I have tomorrow?"
Response:
{{
  "is_workspace_intent": true,
  "service": "calendar",
  "tool": "list_events",
  "extracted_query": "tomorrow"
}}

Example 3:
Query: "Do we have a document about design systems?"
Response:
{{
  "is_workspace_intent": true,
  "service": "drive",
  "tool": "search_files",
  "extracted_query": "design systems"
}}

Example 4:
Query: "what is the refund policy?"
Response:
{{
  "is_workspace_intent": false,
  "service": null,
  "tool": null,
  "extracted_query": ""
}}

Query: "{query}"
Response:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured JSON output."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Format with chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Thread-safe tokenization and execution offloaded to background thread
            with self._inference_lock:
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                import asyncio
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,  # greedy decoding for speed and determinism
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            result_dict = self._extract_json(response)
            if result_dict and result_dict.get("is_workspace_intent"):
                service = result_dict.get("service")
                tool = result_dict.get("tool")
                extracted_query = result_dict.get("extracted_query") or query
                if service in ("gmail", "calendar", "drive") and tool:
                    if service == "drive":
                        extracted_query = self._sanitize_drive_query(extracted_query)
                    elif service == "gmail":
                        from src.mcp.gmail_search import parse_gmail_query_struct, resolve_gmail_tool
                        merged = parse_gmail_query_struct(extracted_query or query)
                        tool = resolve_gmail_tool(query, merged)
                        params = {
                            "query": extracted_query or query,
                            "gmail_struct": merged.to_dict(),
                            "timezone": merged.timezone,
                        }
                        logger.info(
                            "LLM Intent Detection: query=%s -> service=%s, tool=%s, params=%s",
                            query,
                            service,
                            tool,
                            params,
                        )
                        return service, tool, params
                    params = {"query": extracted_query}
                    logger.info(f"LLM Intent Detection: query={query} -> service={service}, tool={tool}, params={params}")
                    return service, tool, params

        except Exception as e:
            logger.warning(f"Error during intent detection inference: {e}")

        return None, None, {}

    async def detect_slack_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Called only when the Slack toggle is already on.
        Returns (tool, params) — never returns None, defaults to search_messages.
        """
        try:
            self._ensure_model_loaded()
        except Exception as e:
            logger.warning(f"Model load failed, defaulting to search_messages: {e}")
            return "search_messages", {"query": query, "channel_name": None}

        prompt = f"""The user has Slack enabled. Classify their query into exactly one Slack action and extract clean parameters.

Available actions:
1. Tool: "list_channels" — user wants to see or list their channels/rooms/spaces.
2. Tool: "list_dms" — user wants to see or list their DM contacts or people they message.
3. Tool: "get_channel_history" — user wants to read recent messages inside a specific channel (or the current chat/channel/conversation if referred to as "this chat", "this channel", "this conversation").
4. Tool: "get_channel_members" — user wants to know who is in a specific channel, or list the members of a channel.
5. Tool: "search_messages" — user wants to search for a topic, keyword, or piece of information across Slack.

Return a JSON object with:
- "tool": one of "list_channels", "list_dms", "get_channel_history", "get_channel_members", "search_messages".
- "extracted_query": keyword-only search string (strip all filler words like "can you", "show me", "find me", "give me", "what is", "please", "the", "a", "for" — keep only nouns, names, topics). Use "" for list_channels and list_dms.
- "channel_name": the channel name (without #) if tool is "get_channel_history" or "get_channel_members". If a specific channel is mentioned in the query (e.g. "reporting-ai", "general"), you MUST extract it as the channel_name. If the query uses @PersonName (e.g. "@Arshiya S. last message"), treat the person name as channel_name for a DM history lookup — do NOT substitute a random public channel. ONLY if NO specific channel or @person is mentioned AND they refer to "this chat/channel/conversation", set "channel_name" to "current". Otherwise null.

Example 1:
Query: "what channels do I have?"
Response:
{{"tool": "list_channels", "extracted_query": "", "channel_name": null}}

Example 2:
Query: "who are the people in my direct messages"
Response:
{{"tool": "list_dms", "extracted_query": "", "channel_name": null}}

Example 3:
Query: "list all the direct messages"
Response:
{{"tool": "list_dms", "extracted_query": "", "channel_name": null}}

Example 4:
Query: "show me recent messages in the general channel"
Response:
{{"tool": "get_channel_history", "extracted_query": "", "channel_name": "general"}}

Example 5:
Query: "tell me about this chat"
Response:
{{"tool": "get_channel_history", "extracted_query": "", "channel_name": "current"}}

Example 6:
Query: "who is in the channel"
Response:
{{"tool": "get_channel_members", "extracted_query": "", "channel_name": "current"}}

Example 7:
Query: "reporting-ai what is happening here"
Response:
{{"tool": "get_channel_history", "extracted_query": "", "channel_name": "reporting-ai"}}

Example 8:
Query: "reporting-ai who is in this channel"
Response:
{{"tool": "get_channel_members", "extracted_query": "", "channel_name": "reporting-ai"}}

Example 9:
Query: "companywide last message sent"
Response:
{{"tool": "get_channel_history", "extracted_query": "", "channel_name": "companywide"}}

Example 10:
Query: "@Arshiya S. last message"
Response:
{{"tool": "get_channel_history", "extracted_query": "", "channel_name": "Arshiya S."}}

Example 11:
Query: "@Muhammad Hasnain last message"
Response:
{{"tool": "get_channel_history", "extracted_query": "", "channel_name": "Muhammad Hasnain"}}

Example 12:
Query: "any messages about the Q3 launch deadline?"
Response:
{{"tool": "search_messages", "extracted_query": "Q3 launch deadline", "channel_name": null}}

Query: "{query}"
Response:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured JSON output."},
            {"role": "user", "content": prompt}
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            with self._inference_lock:
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                import asyncio
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=64,   # shorter — simpler output than Google routing
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            result = self._extract_json(response)

            valid_tools = {"list_channels", "list_dms", "get_channel_history", "get_channel_members", "search_messages"}
            tool = result.get("tool") if result else None

            if tool not in valid_tools:
                logger.warning(f"Invalid tool '{tool}' from LLM, defaulting to search_messages")
                tool = "search_messages"

            params = {
                "query": result.get("extracted_query") if result and result.get("extracted_query") is not None else query,
                "channel_name": result.get("channel_name") if result else None,
            }

            logger.info(f"Slack intent: query={query!r} -> tool={tool}, params={params}")
            return tool, params

        except Exception as e:
            logger.warning(f"Slack intent detection failed ({e}), defaulting to search_messages")
            return "search_messages", {"query": query, "channel_name": None}

    async def detect_notion_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Called only when the Notion toggle is already on.
        Returns (tool, params) — never returns None, defaults to search_pages.
        """
        try:
            self._ensure_model_loaded()
        except Exception as e:
            logger.warning(f"Model load failed, defaulting to search_pages: {e}")
            return "search_pages", {"query": query, "page_name": None}

        prompt = f"""The user has Notion enabled. Classify their query into exactly one Notion action and extract clean parameters.

Available actions:
1. Tool: "search_pages" — user wants to find pages, notes, tasks, or documents in Notion (Note: database rows/tasks are considered pages).
2. Tool: "search_databases" — user wants to find the databases/tables themselves, not the items within them.
3. Tool: "get_page_content" — user wants to read the full content/body of a specific named page.

Return a JSON object with:
- "tool": one of "search_pages", "search_databases", "get_page_content".
- "extracted_query": keyword-only search string (strip filler words). Use "" for get_page_content.
- "page_name": the page name/ID if tool is "get_page_content", otherwise null.

Example 1:
Query: "find notes about Q3 planning"
Response:
{{"tool": "search_pages", "extracted_query": "Q3 planning", "page_name": null}}

Example 2:
Query: "show me the tasks database"
Response:
{{"tool": "search_databases", "extracted_query": "tasks", "page_name": null}}

Example 3:
Query: "read the onboarding page"
Response:
{{"tool": "get_page_content", "extracted_query": "", "page_name": "onboarding"}}

Example 4:
Query: "what documents do we have about sales?"
Response:
{{"tool": "search_pages", "extracted_query": "sales", "page_name": null}}

Example 5:
Query: "list the tasks which are done in Synapse Task Testing"
Response:
{{"tool": "search_pages", "extracted_query": "Done Synapse Task Testing", "page_name": null}}

Example 6:
Query: "what is in my notes"
Response:
{{"tool": "search_pages", "extracted_query": "notes", "page_name": null}}

Query: "{query}"
Response:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured JSON output."},
            {"role": "user", "content": prompt}
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            with self._inference_lock:
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                import asyncio
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            result = self._extract_json(response)

            valid_tools = {"search_pages", "search_databases", "get_page_content"}
            tool = result.get("tool") if result else None

            if tool not in valid_tools:
                logger.warning(f"Invalid Notion tool '{tool}' from LLM, defaulting to search_pages")
                tool = "search_pages"

            params = {
                "query":     result.get("extracted_query") if result and result.get("extracted_query") is not None else query,
                "page_name": result.get("page_name") if result else None,
            }

            logger.info(f"Notion intent: query={query!r} -> tool={tool}, params={params}")
            return tool, params

        except Exception as e:
            logger.warning(f"Notion intent detection failed ({e}), defaulting to search_pages")
            return "search_pages", {"query": query, "page_name": None}


    @staticmethod
    def _extract_json(response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        response = response.strip()

        # Remove Markdown code blocks
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except:
                    continue

        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            try:
                return json.loads(response[start:end])
            except:
                return None

        return None
