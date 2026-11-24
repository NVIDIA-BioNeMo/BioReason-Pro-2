# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Transport for an LLM-judge reward: one rendered prompt in, the judge's text out.

What the judge is asked lives in `bioreason2.rl.judge_reward`. Everything here is about getting
an answer from an OpenAI-compatible endpoint inside a GRPO step, whose judge time is the MAX of
its 512 call latencies:

- a semaphore on in-flight calls: an endpoint can return HTTP 200 with unusable content under a burst;
- hedged retries: a duplicate after `hedge_after_s`, first usable answer wins, within a deadline;
- a circuit breaker and an in-flight shed, so an outage fails fast instead of wedging the step;
- optional prefix-cache warm-up: one call per group leads, the other 15 follow and route to the
  same replica (`user` field), since all 16 share the rubric + ground truth + question prefix.

Point it at an OpenAI-compatible endpoint with SKYRL_JUDGE_URL (the full `/chat/completions` URL),
SKYRL_JUDGE_MODEL and the key in JUDGE_API_KEY. The defaults are the training judge of the
released RL model: Kimi K3 at reasoning effort low, temperature 0. For another API, change
`_payload` and the response readers (`_content`, `_meta`, `reasoning_chain`).
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

URL = os.environ.get("SKYRL_JUDGE_URL", "")
MODEL = os.environ.get("SKYRL_JUDGE_MODEL", "")
EFFORT = os.environ.get("SKYRL_JUDGE_EFFORT", "low")
# Latency is decode of the hidden reasoning, ~3k tokens a call at effort low; a 3000-token cap
# truncated 19 of 32 verdicts.
MAX_TOKENS = int(os.environ.get("SKYRL_JUDGE_MAX_TOKENS", "24000"))
# Temperature 0: against 1.0 it took the noise share of the reward 0.33 -> 0.13 and rank stability
# 0.63 -> 0.78. Empty sends none and the endpoint uses its default.
TEMPERATURE = os.environ.get("SKYRL_JUDGE_TEMPERATURE", "0")
# Only a reply that reasoned (reasoning tokens or a reasoning chain) is a verdict; one that did not
# is retried. Some deployments never report reasoning tokens, so the chain's length counts too.
REQUIRE_REASONING = os.environ.get("SKYRL_JUDGE_REQUIRE_REASONING", "1") == "1"
# Extra JSON merged into every request body, for endpoint-specific switches
# (e.g. '{"chat_template_kwargs": {"enable_thinking": true}}').
EXTRA = json.loads(os.environ.get("SKYRL_JUDGE_EXTRA_JSON", "") or "{}")

# The slow calls are the ones that never answer: with every answered call under 115 s, a duplicate
# at 120 s targets only those (~1% of calls), and the deadline caps what one can hold a step for.
ATTEMPT_TIMEOUT_S = float(os.environ.get("SKYRL_JUDGE_ATTEMPT_TIMEOUT_S", "300"))
HEDGE_AFTER_S = float(os.environ.get("SKYRL_JUDGE_HEDGE_AFTER_S", "120"))
# Adaptive hedge: once HEDGE_WINDOW_MIN attempts have succeeded, hedge at HEDGE_MULT x their median
# (floor HEDGE_FLOOR_S) instead of the fixed HEDGE_AFTER_S. 0 turns it off.
HEDGE_MULT = float(os.environ.get("SKYRL_JUDGE_HEDGE_MULT", "0"))
HEDGE_FLOOR_S = float(os.environ.get("SKYRL_JUDGE_HEDGE_FLOOR_S", "120"))
HEDGE_WINDOW = 256
HEDGE_WINDOW_MIN = 32
DEADLINE_S = float(os.environ.get("SKYRL_JUDGE_DEADLINE_S", "300"))
ATTEMPTS = int(os.environ.get("SKYRL_JUDGE_ATTEMPTS", "3"))
FAST_FAIL_S = 2.0
# A fast failure is retried after a random wait up to this, so a burst of them does not re-fire in step.
FAST_FAIL_JITTER_S = 1.5
BACKOFF_MAX_S = 60.0
MAX_CONCURRENT = int(os.environ.get("SKYRL_JUDGE_MAX_CONCURRENT", "512"))

# The prefix cache is written at prefill, so a follower only needs to lag its leader. Off by
# default: followers waited ~38 s for their leader to save input tokens, while cost and time are
# dominated by output.
WARM_PREFIX = os.environ.get("SKYRL_JUDGE_WARM_PREFIX", "0") == "1"
WARM_WAIT_S = float(os.environ.get("SKYRL_JUDGE_WARM_WAIT_S", "30"))
# A group's key recurs every epoch; an event older than this is stale, or epoch 2 would have no
# leader and all 16 calls would fire cold at once.
WARM_TTL_S = float(os.environ.get("SKYRL_JUDGE_WARM_TTL_S", "300"))
WARM_MAX_KEYS = 4096
ROUTE_BY_GROUP = os.environ.get("SKYRL_JUDGE_ROUTE_BY_GROUP", "1") == "1"
PHASE_EVERY = int(os.environ.get("SKYRL_JUDGE_PHASE_EVERY", "512"))

# The reward mean cannot show whether judgements are sane (it can hold steady through a complete
# format collapse), so a sample of full judgements is written when a path is set.
LOG_PATH = os.environ.get("SKYRL_JUDGE_LOG_PATH", "")
LOG_EVERY = int(os.environ.get("SKYRL_JUDGE_LOG_EVERY", "128"))
LOG_MAX_REJECTS = int(os.environ.get("SKYRL_JUDGE_LOG_MAX_REJECTS", "50"))
# The judge's own text (per-axis rationales and the box), so a logged score can be read, not only seen.
LOG_VERDICT_CHARS = 6000
REJECT_HEAD_CHARS = 2000
ERROR_DETAIL_CHARS = 300

# One named key and no fallback: a silent fall-through to a second key could bill the wrong budget.
KEY_VARS = (os.environ.get("SKYRL_JUDGE_KEY_VAR", "JUDGE_API_KEY"),)
# Or a KEY=VALUE file read here, so the key is never in a launcher or a job's environment listing.
KEY_FILE = os.environ.get("SKYRL_JUDGE_KEY_FILE", "")

_POOL_SIZE = int(os.environ.get("SKYRL_JUDGE_POOL", "2048"))
# Abandoned threads never decrement the in-flight count, so a rising count is the leak; shed then.
MAX_INFLIGHT = int(os.environ.get("SKYRL_JUDGE_MAX_INFLIGHT", str(_POOL_SIZE - 256)))
BREAKER_FAILS = int(os.environ.get("SKYRL_JUDGE_BREAKER_FAILS", "64"))
BREAKER_COOLDOWN_S = float(os.environ.get("SKYRL_JUDGE_BREAKER_COOLDOWN_S", "120"))
BREAKER_429 = int(os.environ.get("SKYRL_JUDGE_BREAKER_429", "16"))
BACKOFF_429_S = float(os.environ.get("SKYRL_JUDGE_BACKOFF_429_S", "5"))
# Error-body fragments marking a 4xx as a proxy's transient overload rather than a refusal.
TRANSIENT_4XX = ("too many clients",)

_GATE = threading.Semaphore(MAX_CONCURRENT)
_LOCK = threading.Lock()
_KEY: dict[str, str] = {}
_STATS = {"calls": 0, "ok": 0, "fallback": 0, "hedges": 0, "retries": 0, "shed": 0, "rejected": 0,
          "no_reasoning": 0}
_PHASE = {"warm_s": 0.0, "gate_s": 0.0, "net_s": 0.0, "in_tok": 0, "cached_tok": 0,
          "n": 0, "reported": 0}
_POOL: dict[str, cf.ThreadPoolExecutor] = {}
_BREAKER = {"fails": 0, "open_until": 0.0}
_IDENTITY = {"logged": False}
_WARM: dict[str, tuple[threading.Event, float]] = {}
_INFLIGHT = {"n": 0, "sent": 0}
# Run totals for stats(); _PHASE is reset by every phase report.
_TOTAL = {"in_tok": 0, "cached_tok": 0}
_LATENCIES: deque[float] = deque(maxlen=HEDGE_WINDOW)
_LOG = {"n": 0, "rejects": 0, "prompt_written": False}


def hedge_after_s() -> float:
    """When to hedge a call still in flight.

    Returns:
        HEDGE_MULT x the median of recent successful attempts (at least HEDGE_FLOOR_S) once the
        window has HEDGE_WINDOW_MIN of them, else the fixed HEDGE_AFTER_S.
    """
    with _LOCK:
        recent = list(_LATENCIES)
    if HEDGE_MULT <= 0 or len(recent) < HEDGE_WINDOW_MIN:
        return HEDGE_AFTER_S
    return max(HEDGE_FLOOR_S, HEDGE_MULT * statistics.median(recent))


def _shed() -> bool:
    """Whether to fail straight away without touching the network.

    Returns:
        True while the breaker is open or the in-flight count has leaked past the cap.
    """
    with _LOCK:
        if time.monotonic() < _BREAKER["open_until"]:
            return True
        return _INFLIGHT["n"] >= MAX_INFLIGHT


def _record(ok: bool, meta: dict) -> None:
    """Feed the breaker; one success closes it, so an outage that ended costs one cooldown.

    Calls that END on 429 (their own backoff retries exhausted the deadline) open it after
    BREAKER_429 in a row, for at least the server's Retry-After: one 429 is a transient saturation
    the per-call backoff absorbs, a run of them is a limit that more requests only deepen.

    Args:
        ok: Whether the call produced usable content.
        meta: The deciding reply's metadata.
    """
    with _LOCK:
        if ok:
            _BREAKER.update(fails=0, r429=0, open_until=0.0)
            return
        if meta.get("status") == "http:429":
            _BREAKER["r429"] = _BREAKER.get("r429", 0) + 1
            if _BREAKER["r429"] < BREAKER_429:
                return
            try:
                wait = max(float(meta.get("retry_after") or 0.0), BREAKER_COOLDOWN_S)
            except ValueError:
                wait = BREAKER_COOLDOWN_S
            _BREAKER.update(fails=0, r429=0, open_until=max(_BREAKER["open_until"], time.monotonic() + wait))
            return
        _BREAKER["fails"] += 1
        if _BREAKER["fails"] >= BREAKER_FAILS:
            _BREAKER.update(fails=0, open_until=time.monotonic() + BREAKER_COOLDOWN_S)


def _pool() -> cf.ThreadPoolExecutor:
    """The shared thread pool, built on first use so importing costs no threads.

    Returns:
        The executor.
    """
    with _LOCK:
        if "pool" not in _POOL:
            _POOL["pool"] = cf.ThreadPoolExecutor(max_workers=_POOL_SIZE, thread_name_prefix="judge")
        return _POOL["pool"]


def log_identity() -> None:
    """Print the judge's identity once per process, so a changed grader cannot go unnoticed."""
    with _LOCK:
        if _IDENTITY["logged"]:
            return
        _IDENTITY["logged"] = True
    print(f"[judge] model={MODEL} url={URL} effort={EFFORT or 'none'} "
          f"temperature={TEMPERATURE or 'default'} require_reasoning={REQUIRE_REASONING} "
          f"extra={json.dumps(EXTRA)} "
          f"max_tokens={MAX_TOKENS} max_concurrent={MAX_CONCURRENT} attempts={ATTEMPTS} "
          f"attempt_timeout={ATTEMPT_TIMEOUT_S}s hedge_after={HEDGE_AFTER_S}s "
          f"hedge_adaptive={HEDGE_MULT}x/floor {HEDGE_FLOOR_S}s deadline={DEADLINE_S}s "
          f"warm_prefix={WARM_PREFIX}/{WARM_WAIT_S}s route_by_group={ROUTE_BY_GROUP} "
          f"sample_log={LOG_PATH or 'off'}/1-in-{LOG_EVERY}", flush=True)


def _phase_report() -> None:
    """Every PHASE_EVERY calls, print where the time went (warm lead, gate, endpoint)."""
    with _LOCK:
        _PHASE["n"] += 1
        if PHASE_EVERY <= 0 or _PHASE["n"] - _PHASE["reported"] < PHASE_EVERY:
            return
        n = _PHASE["n"] - _PHASE["reported"]
        warm, gate, net = _PHASE["warm_s"], _PHASE["gate_s"], _PHASE["net_s"]
        tin, cached = _PHASE["in_tok"], _PHASE["cached_tok"]
        _PHASE.update(reported=_PHASE["n"], warm_s=0.0, gate_s=0.0, net_s=0.0, in_tok=0, cached_tok=0)
        inflight, rejected, fallback = _INFLIGHT["n"], _STATS["rejected"], _STATS["fallback"]
    print(f"[judge] phase over {n} calls: warm {warm / n:5.1f}s  gate {gate / n:5.1f}s  net {net / n:5.1f}s  "
          f"cached {cached / max(tin, 1):5.1%}  "
          f"inflight={inflight} rejected={rejected} fallback={fallback}", flush=True)


def log_sample(record: dict, prompt: str, result: CallResult) -> None:
    """Append one sampled judgement to SKYRL_JUDGE_LOG_PATH as a JSON line; never raises.

    Logged: the first call (with its rendered prompt), every LOG_EVERY-th call, every call slower
    than the hedge point (the one that sets the step), and calls whose judge text was refused, up
    to a cap, with that text's head: the only way to see why a judge fails to box.

    Args:
        record: Caller's fields (generation, score, length, ...).
        prompt: The rendered judge prompt.
        result: The call's outcome.
    """
    if not LOG_PATH:
        return
    try:
        straggler = result.seconds >= hedge_after_s()
        rejected = result.rejected is not None
        with _LOCK:
            _LOG["n"] += 1
            n = _LOG["n"]
            first = not _LOG["prompt_written"]
            _LOG["prompt_written"] = True
            if rejected and _LOG["rejects"] < LOG_MAX_REJECTS:
                _LOG["rejects"] += 1
            else:
                rejected = False
        if not (first or rejected or straggler or (LOG_EVERY > 0 and (n - 1) % LOG_EVERY == 0)):
            return
        line = {"ts": round(time.time(), 3), "call": n, "model": MODEL, "latency_s": round(result.seconds, 2),
                "straggler": straggler, "rejected": rejected, "reply": result.meta,
                "verdict": (result.content or "")[:LOG_VERDICT_CHARS], **record}
        if rejected:
            line["raw_head"] = result.rejected[:REJECT_HEAD_CHARS]
        if first:
            line["rendered_prompt"] = prompt
        directory = os.path.dirname(LOG_PATH)
        with _LOCK:
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _key_from_file(path: str) -> str:
    """The first of KEY_VARS defined in a KEY=VALUE file.

    Args:
        path: The file, or "".

    Returns:
        The key, or "".
    """
    if not path or not os.path.isfile(path):
        return ""
    found = {}
    with open(path) as handle:
        for line in handle:
            name, _, value = line.strip().partition("=")
            if name in KEY_VARS and value:
                found[name] = value.strip().strip("'\"")
    return next((found[n] for n in KEY_VARS if n in found), "")


def _api_key() -> str:
    """The judge key from the environment or KEY_FILE, cached.

    Returns:
        The key.

    Raises:
        RuntimeError: Neither KEY_VARS nor KEY_FILE gives one.
    """
    if "key" not in _KEY:
        key = next((os.environ[n] for n in KEY_VARS if os.environ.get(n)), "") or _key_from_file(KEY_FILE)
        if not key:
            raise RuntimeError(f"no judge API key: set one of {KEY_VARS} or SKYRL_JUDGE_KEY_FILE")
        _KEY["key"] = key
    return _KEY["key"]


def _check_endpoint() -> None:
    """Fail clearly when no endpoint is configured.

    Raises:
        RuntimeError: SKYRL_JUDGE_URL or SKYRL_JUDGE_MODEL is unset.
    """
    if not URL or not MODEL:
        raise RuntimeError("set SKYRL_JUDGE_URL (the full /chat/completions URL) and SKYRL_JUDGE_MODEL")


def _content(payload: dict) -> str | None:
    """The final answer text of a chat completion.

    Args:
        payload: Decoded response body.

    Returns:
        The message text, or None.
    """
    if "choices" not in payload:
        return None
    return payload["choices"][0]["message"].get("content") or None


def _meta(payload: dict) -> dict:
    """How a reply ended: truncation and a missing box both read as "no box" otherwise.

    Args:
        payload: Decoded response body.

    Returns:
        status (the finish reason), token counts and the reasoning chain's length.
    """
    # Endpoints differ in which usage names they report; read either.
    usage = payload.get("usage") or {}
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    cache = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    finish = (payload.get("choices") or [{}])[0].get("finish_reason")
    return {"status": payload.get("status") or finish,
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "cached_tokens": cache.get("cached_tokens"),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "reasoning_chars": len(reasoning_chain(payload))}


def reasoning_chain(payload: dict) -> str:
    """The judge's hidden reasoning, as the endpoint returns it.

    Args:
        payload: Decoded response body.

    Returns:
        The chain's text, or "".
    """
    if "choices" not in payload:
        return ""
    message = payload["choices"][0].get("message") or {}
    return message.get("reasoning_content") or message.get("reasoning") or ""


def _count_tokens(payload: dict) -> None:
    """Accumulate input tokens and prefix-cache hits for the phase report.

    Args:
        payload: Decoded response body.
    """
    usage = payload.get("usage") or {}
    tin = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    with _LOCK:
        _PHASE["in_tok"] += tin
        _PHASE["cached_tok"] += cached
        _TOTAL["in_tok"] += tin
        _TOTAL["cached_tok"] += cached


def _once(body: bytes, headers: dict[str, str], gate_timeout: float) -> tuple[str | None, float, dict]:
    """One attempt; never raises. A thread stuck in urlopen keeps the in-flight count up on purpose.

    Args:
        body: Encoded request.
        headers: Request headers.
        gate_timeout: Longest wait for a concurrency slot: the call's remaining deadline, not the
            attempt timeout, which a queue of slow calls exhausts before any request is sent.

    Returns:
        (content or None, seconds taken, reply metadata).
    """
    with _LOCK:
        _INFLIGHT["n"] += 1
    t0 = time.monotonic()
    acquired = _GATE.acquire(timeout=max(gate_timeout, 0.0))
    gated = time.monotonic()
    sent = False
    try:
        if not acquired:
            return None, gated - t0, {"status": "gate_timeout"}
        # Re-checked after the gate: calls queued behind it before a 429 opened the breaker would
        # otherwise all still be sent.
        if _shed():
            return None, time.monotonic() - t0, {"status": "shed"}
        with _LOCK:
            _INFLIGHT["sent"] += 1
        sent = True
        request = urllib.request.Request(URL, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=ATTEMPT_TIMEOUT_S) as response:
            payload = json.load(response)
            _count_tokens(payload)
        return _content(payload), time.monotonic() - t0, _meta(payload)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read()[:ERROR_DETAIL_CHARS].decode(errors="replace")
        except Exception:
            detail = ""
        headers = dict(exc.headers.items()) if exc.headers else {}
        limits = {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}
        return None, time.monotonic() - t0, {"status": f"http:{exc.code}", "error": detail,
                                             "retry_after": headers.get("Retry-After"), "limits": limits}
    except (urllib.error.URLError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return None, time.monotonic() - t0, {"status": f"error:{type(exc).__name__}"}
    finally:
        if acquired:
            _GATE.release()
        with _LOCK:
            _INFLIGHT["n"] -= 1
            _INFLIGHT["sent"] -= int(sent)
            _PHASE["gate_s"] += gated - t0
            _PHASE["net_s"] += time.monotonic() - gated


def _warm_leader(key: str) -> tuple[bool, threading.Event]:
    """Claim the first call for a shared prefix, or get the leader's event.

    Args:
        key: The group identity.

    Returns:
        (is leader, the event the leader sets once its call is under way or done).
    """
    now = time.monotonic()
    with _LOCK:
        if len(_WARM) > WARM_MAX_KEYS:
            _WARM.clear()
        entry = _WARM.get(key)
        if entry is None or now - entry[1] > WARM_TTL_S:
            event = threading.Event()
            _WARM[key] = (event, now)
            return True, event
    return False, entry[0]


def _route_key(group: str | None) -> str:
    """A stable per-group id for the `user` field, pinning a group to one replica's prefix cache.

    Args:
        group: The group identity, or None.

    Returns:
        A short hash, or "" without a group.
    """
    return hashlib.md5(group.encode()).hexdigest()[:16] if group else ""


def _payload(prompt: str, route: str) -> bytes:
    """The chat-completions request body.

    Args:
        prompt: The rendered judge prompt.
        route: The `user` routing id, or "".

    Returns:
        Encoded JSON.
    """
    payload: dict = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": MAX_TOKENS}
    # Both fields, as the released run sent them: deployments serving one model name can
    # each read only one of them, and an ignored effort silently runs at the default.
    if EFFORT and EFFORT != "none":
        payload["reasoning_effort"] = EFFORT
        payload["reasoning"] = {"effort": EFFORT}
    if TEMPERATURE:
        payload["temperature"] = float(TEMPERATURE)
    payload.update(EXTRA)
    if route and ROUTE_BY_GROUP:
        payload["user"] = route
    return json.dumps(payload).encode()


def _race(prompt: str, accept: Callable[[str], bool], route: str) -> tuple[str | None, str | None, dict]:
    """Hedge a duplicate against a slow original; the first usable answer wins.

    Args:
        prompt: The rendered judge prompt.
        accept: Whether returned content is usable; unusable content spends the next attempt.
        route: The `user` routing id.

    Returns:
        (usable content or None, the last unusable content or None, metadata of the reply that
        decided the outcome).
    """
    body = _payload(prompt, route)
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    pool = _pool()
    start = time.monotonic()
    futures = [pool.submit(_once, body, headers, DEADLINE_S)]
    launched = 1
    rejected, meta = None, {}
    backoff = BACKOFF_429_S
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= DEADLINE_S:
            return None, rejected, {**meta, "status": meta.get("status") or "deadline"}
        # Hedge only with spare capacity: a call still queued at a saturated gate is slow because
        # of the queue, and its duplicate only lengthens it.
        with _LOCK:
            spare = _INFLIGHT["sent"] < MAX_CONCURRENT
        may_hedge = launched < ATTEMPTS and spare
        # The deadline bounds the wait whether or not a hedge is due.
        budget = min(hedge_after_s(), DEADLINE_S - elapsed)
        done, _ = cf.wait(futures, timeout=max(0.0, budget), return_when=cf.FIRST_COMPLETED)
        fast_fail, backed_off = False, False
        for future in done:
            content, took, meta = future.result()
            if content and REQUIRE_REASONING and not (meta.get("reasoning_tokens") or meta.get("reasoning_chars")):
                meta = {**meta, "status": "no_reasoning"}
                content = None
                with _LOCK:
                    _STATS["no_reasoning"] += 1
            if content and accept(content):
                with _LOCK:
                    _LATENCIES.append(took)
                return content, rejected, meta
            status = str(meta.get("status", ""))
            # A 429 is "try later" (a shared model pool saturates under other users' load): back
            # off with jitter and retry within the deadline, so a blip costs latency, not the
            # step's signal. Any other 4xx is a refusal a retry cannot fix, and a hedge of it only
            # adds load. The exception is a 4xx whose body is a proxy's own transient overload,
            # retried like a 429.
            if status == "http:429" or any(t in str(meta.get("error", "")) for t in TRANSIENT_4XX):
                wait = backoff * (0.5 + random.random())
                if time.monotonic() - start + wait >= DEADLINE_S:
                    return None, rejected, meta
                time.sleep(wait)
                backoff = min(backoff * 2, BACKOFF_MAX_S)
                futures.append(pool.submit(_once, body, headers, DEADLINE_S - (time.monotonic() - start)))
                backed_off = True
                continue
            if status.startswith("http:4"):
                return None, rejected, meta
            if content:
                rejected = content
                with _LOCK:
                    _STATS["rejected"] += 1
            fast_fail = fast_fail or took < FAST_FAIL_S
        futures = [f for f in futures if not f.done()]
        # A failed attempt with nothing else in flight is RETRIED, gate or no gate: the retry queues
        # at the gate like any call. Spare capacity gates only hedging a call still in flight; at
        # step shape the gate is always full, so gating this too would turn a no-box reply into NaN.
        if not futures and not backed_off:
            if launched >= ATTEMPTS or time.monotonic() - start >= DEADLINE_S:
                return None, rejected, meta
            if fast_fail:
                time.sleep(random.random() * FAST_FAIL_JITTER_S)
            futures.append(pool.submit(_once, body, headers, DEADLINE_S - (time.monotonic() - start)))
            launched += 1
            with _LOCK:
                _STATS["retries"] += 1
            continue
        if may_hedge and not backed_off and time.monotonic() - start < DEADLINE_S:
            if fast_fail:
                time.sleep(random.random() * FAST_FAIL_JITTER_S)
            futures.append(pool.submit(_once, body, headers, DEADLINE_S - (time.monotonic() - start)))
            launched += 1
            with _LOCK:
                _STATS["hedges"] += 1


@dataclass
class CallResult:
    """The outcome of one judge call.

    Attributes:
        content: Usable judge text, or None (outage, deadline, breaker, or nothing usable).
        seconds: Wall time of the call, warm wait included.
        rejected: The last text the judge returned that `accept` refused, for the log.
        meta: How the deciding reply ended (status, incomplete reason, token counts).
    """

    content: str | None
    seconds: float
    rejected: str | None = None
    meta: dict = field(default_factory=dict)


def call(prompt: str, accept: Callable[[str], bool], group: str | None = None) -> CallResult:
    """Ask the judge.

    Args:
        prompt: The rendered judge prompt, its group-shared part first.
        accept: Whether returned content is usable (e.g. it carries a parseable score).
        group: Identity of the GRPO group (the protein), for prefix-cache warm-up and routing.

    Returns:
        The call's outcome; `content` is None when no usable answer arrived.

    Raises:
        RuntimeError: No endpoint is configured.
    """
    _check_endpoint()
    log_identity()
    t0 = time.monotonic()
    if _shed():
        with _LOCK:
            _STATS["shed"] += 1
            _STATS["calls"] += 1
            _STATS["fallback"] += 1
        return CallResult(None, 0.0, meta={"status": "shed"})
    leader, event = _warm_leader(group) if (WARM_PREFIX and group) else (True, None)
    if not leader:
        waited = time.monotonic()
        event.wait(timeout=WARM_WAIT_S)
        with _LOCK:
            _PHASE["warm_s"] += time.monotonic() - waited
    try:
        content, rejected, meta = _race(prompt, accept, _route_key(group))
    finally:
        if leader and event is not None:
            event.set()
    _record(content is not None, meta)
    with _LOCK:
        _STATS["calls"] += 1
        _STATS["ok" if content is not None else "fallback"] += 1
    _phase_report()
    return CallResult(content, time.monotonic() - t0, rejected, {**meta, "leader": leader})


def stats() -> dict[str, float]:
    """Process-wide judge counters, for the env's metrics.

    Returns:
        Calls, fallbacks, hedges, sheds and their rates.
    """
    # Outside the lock: hedge_after_s takes _LOCK itself, and threading.Lock is not reentrant.
    hedge_after = hedge_after_s()
    with _LOCK:
        n = max(_STATS["calls"], 1)
        return {
            "judge_calls": float(_STATS["calls"]),
            "judge_ok_rate": _STATS["ok"] / n,
            "judge_fallback_rate": _STATS["fallback"] / n,
            "judge_hedge_rate": _STATS["hedges"] / n,
            "judge_retry_rate": _STATS["retries"] / n,
            "judge_no_reasoning_rate": _STATS["no_reasoning"] / n,
            "judge_rejected": float(_STATS["rejected"]),
            "judge_shed": float(_STATS["shed"]),
            "judge_breaker_open": float(time.monotonic() < _BREAKER["open_until"]),
            "judge_hedge_after_s": hedge_after,
            "judge_cached_frac": _TOTAL["cached_tok"] / max(_TOTAL["in_tok"], 1),
        }
