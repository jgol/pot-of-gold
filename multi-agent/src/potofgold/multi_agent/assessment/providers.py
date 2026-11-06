"""Run tool-calling conversations through OpenAI and Gemini REST endpoints.

Both clients return the same response and usage metadata as the Anthropic
client in agents.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from potofgold.multi_agent.assessment.agents import FINAL_ANSWER_TOOLS, _preserve_partial_usage, _validate_terminal_submission

_log = logging.getLogger(__name__)

# Endpoint and API-key environment variable for each REST provider.
_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
}


def provider_of(model: str) -> str:
    """Route Claude IDs to Anthropic, Gemini IDs to Gemini, and other IDs to OpenAI."""
    m = (model or "").lower()
    if m.startswith("claude") or "claude" in m:
        return "anthropic"
    if m.startswith("gemini") or "gemini" in m:
        return "gemini"
    return "openai"


def _cache_route_key(system_prompt: str) -> str:
    """Hash the role and system prompt to group requests with a shared cache prefix."""
    return hashlib.sha256((system_prompt or "").encode()).hexdigest()[:40]


def _rest_omit_temperature(model: str) -> bool:
    """Identify models that use reasoning request parameters."""
    m = (model or "").lower()
    return (m.startswith(("o1", "o3", "o4")) or bool(re.match(r"gpt-([5-9]|\d\d)", m))
            or "reasoning" in m or "thinking" in m)


_RESPONSES_URL = "https://api.openai.com/v1/responses"
RESPONSES_REASONING_EFFORT = "medium"


def _is_openai_responses_model(model: str) -> bool:
    """Select the Responses endpoint for the OpenAI reasoning models handled here."""
    return provider_of(model) == "openai" and _rest_omit_temperature(model)


def _retry_after_seconds(e: urllib.error.HTTPError, cap: float = 60.0) -> Optional[float]:
    """Parse numeric Retry-After seconds, capped at cap; otherwise return None."""
    try:
        ra = e.headers.get("Retry-After") if e.headers else None
    except Exception:                                        # noqa: BLE001
        return None
    if not ra:
        return None
    try:
        v = float(ra)
    except (TypeError, ValueError):
        return None                                          # Retry HTTP-date headers using the regular exponential backoff.
    return min(max(v, 0.0), cap) if math.isfinite(v) else None   # Nonfinite delays cannot be passed to sleep().


def _post_json(url: str, headers: Dict[str, str], body: dict, timeout: int = 120, retries: int = 3) -> dict:
    """POST JSON, retrying rate limits, server errors and timeouts with backoff."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = _retry_after_seconds(e) if e.code in (429, 503) else None   # Respect a numeric server delay on rate-limit and service-unavailable responses.
                time.sleep(wait if wait is not None else 2 ** attempt)
                continue
            raise RuntimeError(f"{url} HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{url}: {type(e).__name__}: {e}") from e


def _to_openai_tools(tools: List[dict]) -> List[dict]:
    """Convert Anthropic tool definitions to Chat Completions function tools."""
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema", {"type": "object"})}}
            for t in tools]


def _to_responses_tools(tools: List[dict]) -> List[dict]:
    """Convert Anthropic tool definitions to Responses function tools."""
    return [{"type": "function", "name": t["name"], "description": t.get("description", ""),
             "parameters": t.get("input_schema", {"type": "object"})} for t in tools]


def _responses_final_text(outputs: List[dict]) -> Optional[str]:
    """Collect assistant text from Responses message items."""
    parts = [c.get("text") for o in outputs if o.get("type") == "message"
             for c in (o.get("content") or []) if c.get("type") == "output_text" and c.get("text")]
    return "\n".join(parts) if parts else None


def _responses_reasoning_summary(outputs: List[dict]) -> Optional[str]:
    """Collect the provider's reasoning summaries for the audit record."""
    parts = [s.get("text") for o in outputs if o.get("type") == "reasoning"
             for s in (o.get("summary") or []) if isinstance(s, dict) and s.get("text")]
    return "\n".join(parts) if parts else None


def _preview(x, n: int = 800) -> str:
    """Limit the length of a value included in an audit preview."""
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"


def run_rest(role: str, user_message: str, system_prompt: str, tools: List[dict],
             tool_handlers: Dict[str, Callable], model: str = "", max_turns: int = 10,
             temperature: Optional[float] = 0.0, cache_prefix: Optional[str] = None) -> Tuple[Optional[dict], dict]:
    """Run the shared tool loop through a Chat Completions endpoint.

    Prepend cache_prefix to the user message and return the structured final
    response with token usage and per-turn transcripts.
    """
    if _is_openai_responses_model(model):
        return run_responses(role, user_message, system_prompt, tools, tool_handlers,
                             model=model, max_turns=max_turns, temperature=temperature, cache_prefix=cache_prefix)
    provider = provider_of(model)
    if provider not in _ENDPOINTS:
        raise ValueError(f"run_rest: model {model!r} is not an OpenAI/Gemini model (provider={provider})")
    url, key_env = _ENDPOINTS[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"{key_env} not set (needed for {provider} model {model})")
    headers = {"Authorization": f"Bearer {api_key}"}
    oai_tools = _to_openai_tools(tools)
    _terminal = next((t.get("name") for t in tools if t.get("name") in FINAL_ANSWER_TOOLS), None)  # Reserve the last turn for the final response.
    full_user = f"{cache_prefix}\n\n{user_message}" if cache_prefix else user_message
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": full_user}]

    in_tok = out_tok = cached_tok = 0
    t0 = time.time()
    structured_output: Optional[dict] = None
    final_text: Optional[str] = None
    transcript: List[dict] = []
    turn = 0
    with _preserve_partial_usage(lambda: {
        "role": role, "model": model, "turns": len(transcript),
        "input_tokens": in_tok, "output_tokens": out_tok, "cached_input_tokens": cached_tok,
        "transcript": transcript, "wall_time_s": round(time.time() - t0, 2),
    }):
        for turn in range(max_turns):
            body: Dict[str, Any] = {"model": model, "messages": messages}
            if provider == "openai":            # OpenAI accepts an explicit cache-routing key; omit it for other providers.
                body["prompt_cache_key"] = _cache_route_key(system_prompt)
            if oai_tools:
                body["tools"] = oai_tools
                if _terminal is not None and max_turns > 1 and turn == max_turns - 1:
                    body["tool_choice"] = {"type": "function", "function": {"name": _terminal}}
                else:
                    body["tool_choice"] = "auto"
            if temperature is not None and not _rest_omit_temperature(model):
                body["temperature"] = temperature
            if not _rest_omit_temperature(model):
                body["max_tokens"] = 4096
            else:
                body["max_completion_tokens"] = 8192
            resp = _post_json(url, headers, body)
            usage = resp.get("usage") or {}
            in_tok += usage.get("prompt_tokens", 0) or 0
            out_tok += usage.get("completion_tokens", 0) or 0
            cached_tok += ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0

            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            transcript.append({"turn": turn + 1, "model": model,
                               "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                                               "arguments": _preview((tc.get("function") or {}).get("arguments"))} for tc in tool_calls],
                               "reasoning_summary": None, "text": msg.get("content")})
            if not tool_calls:
                final_text = msg.get("content")
                break

            finals = [tc for tc in tool_calls if (tc.get("function") or {}).get("name") in FINAL_ANSWER_TOOLS]
            if len(finals) > 1:                                  # Retain the first final submission and record additional ones in the audit.
                _log.warning("run_rest[%s/%s] emitted %d final-answer tools in one turn; using the first",
                             role, model, len(finals))
            terminal_feedback = None
            if finals:
                fn = finals[0].get("function") or {}
                raw = _parse_args(fn.get("arguments"), ctx=f"{role}/{model} {fn.get('name')}")
                structured_output, terminal_feedback = _validate_terminal_submission(
                    fn.get("name"), raw, tool_handlers.get(fn.get("name")), transcript[-1],
                    raw_arguments=fn.get("arguments"))
                if terminal_feedback is None:
                    break

            # Echo supported tool-call fields, including Gemini's required thought signature.
            echoed_tcs: List[dict] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                e: Dict[str, Any] = {"id": tc.get("id"), "type": "function",
                                     "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")}}
                if tc.get("extra_content") is not None:
                    e["extra_content"] = tc["extra_content"]
                echoed_tcs.append(e)
            asst: Dict[str, Any] = {"role": "assistant", "tool_calls": echoed_tcs}
            if msg.get("content") is not None:                   # Omit null content while preserving an explicitly empty string.
                asst["content"] = msg["content"]
            messages.append(asst)
            for tc in tool_calls:
                fn = tc.get("function") or {}
                handler = tool_handlers.get(fn.get("name"))
                if fn.get("name") in FINAL_ANSWER_TOOLS and terminal_feedback is not None:
                    out = terminal_feedback
                elif handler is None:
                    out = json.dumps({"error": f"Unknown tool: {fn.get('name')}"})
                else:
                    try:
                        r = handler(_parse_args(fn.get("arguments"), ctx=f"{role}/{model} {fn.get('name')}"))
                        out = r if isinstance(r, str) else json.dumps(r, default=str)
                    except Exception as e:
                        _log.warning("tool handler %s [%s/%s] raised %s: %s", fn.get("name"), role, model,
                                     type(e).__name__, e)         # Record handler failures before returning an error to the model.
                        out = json.dumps({"error": f"{type(e).__name__}: {e}"})
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": out})

        if structured_output is None and final_text is None:
            _log.warning("run_rest[%s/%s] exhausted max_turns=%d without a final-answer tool or a text reply "
                         "(%d paid call(s), structured_output=None)", role, model, max_turns, turn + 1)
        if in_tok == 0 and out_tok == 0:                         # Mark missing usage so cost summaries do not imply complete accounting.
            _log.warning("run_rest[%s/%s] reported zero tokens across %d turn(s) — usage block missing?",
                         role, model, turn + 1)
        meta = {"role": role, "model": model, "provider": provider, "turns": turn + 1,
                "input_tokens": in_tok, "output_tokens": out_tok, "cached_input_tokens": cached_tok,
                "wall_time_s": round(time.time() - t0, 2),
                "final_text": final_text, "has_structured_output": structured_output is not None,
                "transcript": transcript}
    return structured_output, meta


def run_responses(role: str, user_message: str, system_prompt: str, tools: List[dict],
                  tool_handlers: Dict[str, Callable], model: str = "", max_turns: int = 10,
                  temperature: Optional[float] = None, cache_prefix: Optional[str] = None) -> Tuple[Optional[dict], dict]:
    """Run the shared tool loop through the OpenAI Responses endpoint.

    Use previous_response_id to continue the conversation with new tool outputs.
    Return the same response and metadata structure as run_rest.
    """
    provider = provider_of(model)
    if provider != "openai":
        raise ValueError(f"run_responses: {model!r} is not an OpenAI model (provider={provider})")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not set (needed for OpenAI Responses model {model})")
    headers = {"Authorization": f"Bearer {api_key}"}
    resp_tools = _to_responses_tools(tools)
    _terminal = next((t.get("name") for t in tools if t.get("name") in FINAL_ANSWER_TOOLS), None)
    # Continuations send new tool outputs with the previous response ID.
    full_user = f"{cache_prefix}\n\n{user_message}" if cache_prefix else user_message
    input_items: List[dict] = [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": full_user}]
    prev_id: Optional[str] = None
    in_tok = out_tok = cached_tok = 0
    t0 = time.time()
    structured_output: Optional[dict] = None
    final_text: Optional[str] = None
    transcript: List[dict] = []
    turn = 0
    with _preserve_partial_usage(lambda: {
        "role": role, "model": model, "turns": len(transcript),
        "input_tokens": in_tok, "output_tokens": out_tok, "cached_input_tokens": cached_tok,
        "transcript": transcript, "wall_time_s": round(time.time() - t0, 2),
    }):
        for turn in range(max_turns):
            body: Dict[str, Any] = {"model": model, "input": input_items, "store": True,   # Store responses so later turns can refer to their conversation state.
                                    "reasoning": {"effort": RESPONSES_REASONING_EFFORT, "summary": "auto"},
                                    "max_output_tokens": 8192,
                                    "prompt_cache_key": _cache_route_key(system_prompt)}
            if prev_id:
                body["previous_response_id"] = prev_id
            if resp_tools:
                body["tools"] = resp_tools
                if _terminal is not None and max_turns > 1 and turn == max_turns - 1:
                    body["tool_choice"] = {"type": "function", "name": _terminal}
                else:
                    body["tool_choice"] = "auto"
            resp = _post_json(_RESPONSES_URL, headers, body)
            usage = resp.get("usage") or {}
            in_tok += usage.get("input_tokens", 0) or 0     # Responses uses different token-usage field names from Chat Completions.
            out_tok += usage.get("output_tokens", 0) or 0
            cached_tok += ((usage.get("input_tokens_details") or {}).get("cached_tokens", 0)) or 0
            prev_id = resp.get("id")
            outputs = resp.get("output") or []
            calls = [o for o in outputs if o.get("type") == "function_call"]
            transcript.append({"turn": turn + 1, "model": model,
                               "tool_calls": [{"name": c.get("name"), "arguments": _preview(c.get("arguments"))} for c in calls],
                               "reasoning_summary": _responses_reasoning_summary(outputs),
                               "text": _responses_final_text(outputs), "status": resp.get("status")})
            if not calls:
                final_text = _responses_final_text(outputs)
                if final_text is None and resp.get("status") != "completed":
                    _log.warning("run_responses[%s/%s] no tool call + no text (status=%s, %s) — likely max_output_tokens "
                                 "truncation before the final tool; a court-drop here is a BUDGET issue, not the model",
                                 role, model, resp.get("status"), resp.get("incomplete_details"))
                break
            finals = [c for c in calls if c.get("name") in FINAL_ANSWER_TOOLS]
            if len(finals) > 1:
                _log.warning("run_responses[%s/%s] emitted %d final-answer tools in one turn; using the first",
                             role, model, len(finals))
            terminal_feedback = None
            if finals:
                call = finals[0]
                raw = _parse_args(call.get("arguments"), ctx=f"{role}/{model} {call.get('name')}")
                structured_output, terminal_feedback = _validate_terminal_submission(
                    call.get("name"), raw, tool_handlers.get(call.get("name")), transcript[-1],
                    raw_arguments=call.get("arguments"))
                if terminal_feedback is None:
                    break
            new_items: List[dict] = []
            for c in calls:
                name, call_id = c.get("name"), c.get("call_id")
                handler = tool_handlers.get(name)
                if name in FINAL_ANSWER_TOOLS and terminal_feedback is not None:
                    out = terminal_feedback
                elif handler is None:
                    out = json.dumps({"error": f"Unknown tool: {name}"})
                else:
                    try:
                        r = handler(_parse_args(c.get("arguments"), ctx=f"{role}/{model} {name}"))
                        out = r if isinstance(r, str) else json.dumps(r, default=str)
                    except Exception as e:
                        _log.warning("tool handler %s [%s/%s] raised %s: %s", name, role, model, type(e).__name__, e)
                        out = json.dumps({"error": f"{type(e).__name__}: {e}"})   # Record handler failures in addition to returning tool feedback.
                new_items.append({"type": "function_call_output", "call_id": call_id, "output": out})
            input_items = new_items

        if structured_output is None and final_text is None:
            _log.warning("run_responses[%s/%s] exhausted max_turns=%d without a final-answer tool or a text reply "
                         "(%d paid call(s), structured_output=None)", role, model, max_turns, turn + 1)
        if in_tok == 0 and out_tok == 0:
            _log.warning("run_responses[%s/%s] reported zero tokens across %d turn(s) — usage block missing?",
                         role, model, turn + 1)
        meta = {"role": role, "model": model, "provider": "openai", "turns": turn + 1,
                "input_tokens": in_tok, "output_tokens": out_tok, "cached_input_tokens": cached_tok,
                "wall_time_s": round(time.time() - t0, 2),
                "final_text": final_text, "has_structured_output": structured_output is not None,
                "transcript": transcript}
    return structured_output, meta


def _parse_args(arguments, ctx: str = "") -> dict:
    """Decode tool arguments from a JSON string or mapping.

    Log malformed arguments with the role, model and tool name, then return an
    empty mapping so validation can report the missing fields.
    """
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments) if arguments else {}
    except (TypeError, ValueError):
        _log.warning("tool-call arguments not valid JSON%s (coercing to {}): %r",
                     f" [{ctx}]" if ctx else "", str(arguments)[:200])
        return {}
