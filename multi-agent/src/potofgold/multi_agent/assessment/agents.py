"""Shared agent loop: send instructions and evidence, execute tools, validate the final response.

Each role has its own conversation. Provider dispatch, usage accounting and audit
records are shared by the court and the single-model comparator.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_MODEL = "claude-sonnet-4-6"

import anthropic

# Append one audit record for each completed agent call.
AUDIT_LOG_DIR = Path(os.environ.get("POTOFGOLD_COURT_AUDIT_DIR", "runs/agent_audit"))

# These tools submit the final structured response.
FINAL_ANSWER_TOOLS = frozenset({
    "submit_assessment",
    "report_result",
    "submit_prosecution",
    "submit_defense",
    "submit_verdict",
})


def _get_client() -> anthropic.Anthropic:
    """Get or create Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _omit_temperature(model: str) -> bool:
    """Identify model IDs for which this client omits the temperature parameter."""
    m = (model or "").lower()
    return "opus-4" in m or "opus4" in m


def _audit_log(record: dict) -> None:
    """Append one record to the JSONL audit log."""
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_LOG_DIR / "agent_calls.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _validate_terminal_submission(name, raw, handler, turn_record, *, raw_arguments=None):
    """Validate a final tool response before ending the conversation.

    Return validation errors as tool feedback within the remaining turn budget.
    Keep the submitted payload in the audit alongside the normalized response.
    """
    validate = getattr(handler, "validate_submission", None)
    if not callable(validate):
        return raw, None
    audit = {"name": name, "submitted_output": deepcopy(raw)}
    if raw_arguments is not None:
        audit["raw_arguments"] = deepcopy(raw_arguments)
    turn_record.setdefault("terminal_submissions", []).append(audit)
    try:
        accepted = validate(deepcopy(raw))
        if not isinstance(accepted, dict):
            raise ValueError("Terminal validator must return an object")
    except ValueError as exc:
        audit.update(accepted=False, validation_error={"error_type": type(exc).__name__, "message": str(exc)})
        feedback = json.dumps({"error": "invalid_terminal_submission", "message": str(exc)[:2400],
            "instruction": (
                "Submit a COMPLETE replacement object to the same final tool, including ALL required fields "
                "and unchanged fields from your prior submission. Never submit a partial patch: prior fields "
                "are not merged into the replacement. Correct the reported errors, keep the full object "
                "concise, and do not invent evidence."
            )})
        return None, feedback
    audit.update(accepted=True, normalized_output=deepcopy(accepted))
    return accepted, None


@contextmanager
def _preserve_partial_usage(snapshot):
    """Attach recorded usage when a later API call or response-processing step fails."""
    try:
        yield
    except Exception as exc:
        exc._agent_metadata = deepcopy({**snapshot(), "usage_incomplete": True})
        raise


async def task(
    role: str,
    user_message: str,
    system_prompt: str,
    tools: List[dict],
    tool_handlers: Dict[str, Callable],
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
    temperature: Optional[float] = 0.0,
    cache_prefix: Optional[str] = None,
) -> Tuple[Optional[dict], dict]:
    """Run one agent conversation using the Anthropic client.

    Dispatch requested tools through tool_handlers until the model submits a
    final response, returns text without a tool call, or reaches max_turns.
    Return the structured response and metadata containing usage and transcripts.
    A None temperature leaves sampling settings to the provider.
    """
    client = _get_client()
    # Cache the judge's shared first message; other roles use a plain user message.
    first_user = f"{cache_prefix}\n\n{user_message}" if cache_prefix else user_message
    _reuse_prefix = str(role).lower() == "judge" and bool(cache_prefix)
    messages = [{"role": "user",
                 "content": ([{"type": "text", "text": first_user, "cache_control": {"type": "ephemeral"}}]
                             if _reuse_prefix else first_user)}]

    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_creation = 0
    t_start = time.time()
    structured_output = None
    final_text = None
    transcript = []
    _terminal = next((t["name"] for t in tools if t.get("name") in FINAL_ANSWER_TOOLS), None) if tools else None
    # Request a final submission on the last available turn.

    with _preserve_partial_usage(lambda: {
        "role": role, "model": model, "turns": len(transcript),
        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
        "cache_read_input_tokens": total_cache_read, "cache_creation_input_tokens": total_cache_creation,
        "transcript": transcript, "wall_time_s": round(time.time() - t_start, 2),
    }):
        for turn in range(max_turns):
            max_output_tokens = 4096
            response = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                temperature=(temperature if (temperature is not None and not _omit_temperature(model))
                             else anthropic.NOT_GIVEN),
                system=([{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
                        if system_prompt else system_prompt),   # The role's tools and system instructions are shared across strategies.
                tools=tools if tools else anthropic.NOT_GIVEN,
                tool_choice=({"type": "tool", "name": _terminal}
                             if (_terminal is not None and max_turns > 1 and turn == max_turns - 1)
                             else anthropic.NOT_GIVEN),
                messages=messages,
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            total_cache_read += getattr(response.usage, "cache_read_input_tokens", 0) or 0
            total_cache_creation += getattr(response.usage, "cache_creation_input_tokens", 0) or 0

            _tu = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            _txt = next((b.text for b in response.content
                         if getattr(b, "type", None) == "text" and getattr(b, "text", None)), None)
            # Retain per-turn usage as well as totals to diagnose interrupted responses.
            usage_dump = getattr(response.usage, "model_dump", None)
            raw_usage = (usage_dump(mode="json") if callable(usage_dump)
                         else vars(response.usage))
            transcript.append({"turn": turn + 1, "model": model, "reasoning_summary": None, "text": _txt,
                               "stop_reason": response.stop_reason,
                               "max_tokens": max_output_tokens, "usage": deepcopy(raw_usage),
                               "tool_calls": [{"name": b.name,
                                               "arguments": (json.dumps(b.input, default=str)[:800])} for b in _tu]})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                        break
                break

            tool_use_blocks = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_blocks.append(block)
                elif hasattr(block, "text") and block.text:
                    final_text = block.text

            if not tool_use_blocks:
                break

            # An invalid final submission receives feedback within the remaining turn budget.
            terminal_feedback = None
            for tb in tool_use_blocks:
                if tb.name in FINAL_ANSWER_TOOLS:
                    structured_output, terminal_feedback = _validate_terminal_submission(
                        tb.name, tb.input, tool_handlers.get(tb.name), transcript[-1])
                    break

            if structured_output is not None:
                break

            messages.append({
                "role": "assistant",
                "content": [
                    {"type": b.type, "id": b.id, "name": b.name, "input": b.input}
                    if b.type == "tool_use"
                    else {"type": "text", "text": b.text}
                    for b in response.content
                ],
            })

            tool_results = []
            for tb in tool_use_blocks:
                handler = tool_handlers.get(tb.name)
                if tb.name in FINAL_ANSWER_TOOLS and terminal_feedback is not None:
                    result_text = terminal_feedback
                elif handler is None:
                    result_text = json.dumps({"error": f"Unknown tool: {tb.name}"})
                else:
                    try:
                        result = handler(tb.input)
                        result_text = result if isinstance(result, str) else json.dumps(result, default=str)
                    except Exception as e:
                        result_text = json.dumps({"error": f"{type(e).__name__}: {e}"})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        wall_time = time.time() - t_start

        metadata = {
            "role": role,
            "model": model,
            "turns": turn + 1,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": total_cache_read,
            "cache_creation_input_tokens": total_cache_creation,
            "wall_time_s": round(wall_time, 2),
            "final_text": final_text,
            "has_structured_output": structured_output is not None,
            "transcript": transcript,
        }

        _audit_log({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **metadata,
            "structured_output_keys": list(structured_output.keys()) if structured_output else None,
        })

    return structured_output, metadata


def task_sync(
    role: str,
    user_message: str,
    system_prompt: str,
    tools: List[dict],
    tool_handlers: Dict[str, Callable],
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
    temperature: Optional[float] = 0.0,
    cache_prefix: Optional[str] = None,
) -> Tuple[Optional[dict], dict]:
    """Run an agent synchronously through the client selected by its model ID.

    Anthropic uses task(); OpenAI and Gemini use the REST provider loop.
    """
    from potofgold.multi_agent.assessment import providers
    if providers.provider_of(model) != "anthropic":
        return providers.run_rest(role, user_message, system_prompt, tools, tool_handlers,
                                  model, max_turns, temperature, cache_prefix=cache_prefix)
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            task(role, user_message, system_prompt, tools,
                 tool_handlers, model, max_turns, temperature, cache_prefix),
        )
    # Propagate worker errors without retrying a conversation that already used tokens.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(
            asyncio.run,
            task(role, user_message, system_prompt, tools,
                 tool_handlers, model, max_turns, temperature, cache_prefix),
        )
        return future.result()
