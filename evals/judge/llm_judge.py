#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""LLM-as-a-judge scoring of BioReason-Pro 2 generations. Manual: evals/judge/README.md.

Two steps. `--export-rows` joins a directory of eval.py result JSONs with the test corpus's
ground truth into one JSONL; `--rows-jsonl` then has the judge score each row on five axes
(judge_prompt.py) and writes one verdict JSON per protein. `--summarize` reports on a
finished run.

The judge is any OpenAI-compatible /chat/completions endpoint: JUDGE_BASE_URL (ending in /v1),
JUDGE_API_KEY and --model (or JUDGE_MODEL). The published judge is Nemotron 3.5 Super VL at
`--effort max`; scores from another model or effort are not comparable. For another API, replace
`call_judge`.
"""

import argparse
import ast
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import shutil
import sys
import time

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)          # so `python evals/judge/llm_judge.py` finds bioreason2

from bioreason2 import paths as _paths  # noqa: E402
from bioreason2.dataset.protein_fields import MiniGoDag, prepare_fields  # noqa: E402
from evals.judge.judge_prompt import AXES, PROMPT_TEMPLATE, SCHEMA, SEARCH_POLICY  # noqa: E402

DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "")
DEFAULT_EFFORT = "max"
BASE_URL_VAR = "JUDGE_BASE_URL"
API_KEY_VAR = "JUDGE_API_KEY"

# Validated here because some endpoints accept an invalid reasoning_effort with HTTP 200 and
# silently run at their default, so a typo would change the judge without a sign.
EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh", "max")

# The generation's reasoning tags are Nemotron's own reasoning delimiters, which its tokenizer
# encodes as such even inside user text: ~5% of verdicts then took the generation for the judge's
# reasoning and scored an "empty" answer. The judge sees them under a neutral tag.
THINK_TAGS = (("<think>", "<reasoning>"), ("</think>", "</reasoning>"))

# A verdict can be perfectly well-formed and still look internally odd. Those are
# ADVISORY -- the record is usable and must still count in the aggregate. Only a
# missing/unparseable verdict or an out-of-range score disqualifies one.
SOFT_PREFIX = "reliability "

SESSION = None


class Retryable(RuntimeError):
    """Transport error, timeout, 429 or 5xx -- worth another attempt.

    A 4xx other than 429 is our bug, not the endpoint's, and is not retried.
    """


# ------------------------------------------------------------------- render
def _parse_go_list(v):
    """The dataset stores GO sets as the repr of a python list, or None/''."""
    if not v:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    s = str(v).strip()
    if not s or s.lower() == "none":
        return []
    return re.findall(r"GO:\d{7}", s)


def _gt_terms(row, col):
    """Ground truth for one aspect: tier 1 UNION tier 2.

    Presence stays a tier-1 question -- an aspect whose only annotation is
    tier-2 was never scorable -- but once an aspect counts, the label set is
    everything the teacher was handed.
    """
    a = row.get(col) or []
    if isinstance(a, str):
        a = ast.literal_eval(a) if a else []
    if not len(a):
        return a
    b = row.get(col + "_tier2") or []
    if isinstance(b, str):
        b = ast.literal_eval(b) if b else []
    seen, out = set(), []
    for t in list(a) + list(b):
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def render_go(ids, dag, max_terms=60):
    """Ground truth is DAG-propagated, so a 200-term MF list is mostly ancestors of a
    handful of real annotations. Showing all of them buries the signal and costs
    input tokens; showing the LEAVES -- the terms no other term in the set descends
    from -- is what a curator would actually read. Ancestors are still summarised by
    count so nothing looks hidden.
    """
    ids = [dag.resolve(g) for g in _parse_go_list(ids)] if dag else _parse_go_list(ids)
    ids = [g for g in dict.fromkeys(ids)]
    if not ids:
        return "None"
    if dag is None:
        return ", ".join(ids[:max_terms])
    inset = set(ids)
    covered = set()
    for g in ids:
        covered |= dag.ancestors(g) & inset
    leaves = [g for g in ids if g not in covered]
    leaves.sort()
    shown = leaves[:max_terms]
    out = ", ".join("%s (%s)" % (g, dag.name.get(g, "?")) for g in shown)
    extra = []
    if len(leaves) > max_terms:
        extra.append("%d more specific terms omitted" % (len(leaves) - max_terms))
    if len(covered):
        extra.append("%d ancestor terms implied by DAG propagation" % len(covered))
    if extra:
        out += "  [%s]" % "; ".join(extra)
    return out


def _field(row, key, default="None"):
    """One row value as stripped text, or `default` when empty."""
    v = row.get(key)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def build_prompt(row, dag, max_terms=60):
    """The judge prompt for one row.

    Args:
        row: an exported row (ground truth, generation, model input).
        dag: a MiniGoDag, or None to render bare GO ids.
        max_terms: most GO leaves shown per aspect.

    Returns:
        The prompt.
    """
    return PROMPT_TEMPLATE.format(
        search_policy=SEARCH_POLICY,
        protein_names=_field(row, "protein_names"),
        organism=_field(row, "organism"),
        protein_function=_field(row, "protein_function"),
        interpro_formatted=_field(row, "interpro_formatted"),
        ppi_formatted=_field(row, "ppi_formatted"),
        subcellular_location=_field(row, "subcellular_location"),
        features_formatted=_field(row, "features_formatted"),
        additional_context=_field(row, "additional_context"),
        length=_field(row, "length"),
        # The accession and the numbered sequence, so residue identities can be checked.
        protein_id=_field(row, "protein_id"),
        sequence_numbered=_field(row, "sequence_numbered"),
        # TIER 1 UNION TIER 2, to match both the teacher prompt and cafa_evals. The
        # teacher was supplied tier 1 and tier 2 as one unmarked list, so the traces
        # assert tier-2 terms as confidently as tier-1. A judge shown only tier-1
        # marks those as unsupported: of the extra confident terms per protein, ALL
        # were tier-2 and none were invented.
        go_mf_x=render_go(_gt_terms(row, "go_mf"), dag, max_terms),
        go_bp_x=render_go(_gt_terms(row, "go_bp"), dag, max_terms),
        go_cc_x=render_go(_gt_terms(row, "go_cc"), dag, max_terms),
        **({"model_input": _field(row, "model_input", "(prompt not recorded)")}
           if "{model_input}" in PROMPT_TEMPLATE else {}),
        generation=rename_think_tags(_field(row, "generation", "(the model produced no output)")))


def rename_think_tags(text: str) -> str:
    """The generation with its reasoning tags renamed, so no judge tokenizer reads them as its own.

    Args:
        text: The generation.

    Returns:
        The text with `THINK_TAGS` replaced.
    """
    for tag, neutral in THINK_TAGS:
        text = text.replace(tag, neutral)
    return text


# ------------------------------------------------------------------ provider
def init_session(workers):
    """One pooled HTTP session sized for `workers` concurrent calls."""
    global SESSION
    n = max(16, workers + 4)
    SESSION = requests.Session()
    ad = HTTPAdapter(pool_connections=n, pool_maxsize=n, max_retries=0)
    SESSION.mount("https://", ad)
    SESSION.mount("http://", ad)


def _endpoint(base_url):
    """The base URL and key for a call.

    Args:
        base_url: --base-url, or None to read JUDGE_BASE_URL.

    Returns:
        (base URL without a trailing slash, API key).

    Raises:
        RuntimeError: either is unset.
    """
    base = (base_url or os.environ.get(BASE_URL_VAR) or "").rstrip("/")
    key = os.environ.get(API_KEY_VAR)
    if not base:
        raise RuntimeError("%s not set: the judge endpoint's base URL, ending in /v1" % BASE_URL_VAR)
    if not key:
        raise RuntimeError("%s not set (keep it in the environment, never in a job spec)" % API_KEY_VAR)
    return base, key


def _post(url, key, payload, timeout):
    """POST one request; raise Retryable on what a retry can fix.

    Args:
        url: the full endpoint URL.
        key: the API key.
        payload: the JSON body.
        timeout: seconds.

    Returns:
        The decoded response body.

    Raises:
        Retryable: a transport error, 408, 409, 429 or 5xx.
        RuntimeError: any other non-200 reply.
    """
    try:
        r = (SESSION or requests).post(
            url, timeout=timeout,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"},
            json=payload)
    except requests.RequestException as e:
        raise Retryable("%s: %s" % (type(e).__name__, str(e)[:200])) from e
    if r.status_code != 200:
        msg = "HTTP %d: %s" % (r.status_code, r.text[:400])   # never echo the header
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise Retryable(msg)
        raise RuntimeError(msg)
    return r.json()


def extract_json(text):
    """Pull the verdict object out of a completion.

    A chat endpoint may accept response_format and still ignore it, so the object can
    arrive fenced or with a preamble. Try bare, then a fenced block, then the outermost
    balanced {...}; give up rather than guess.
    """
    t = (text or "").strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except ValueError:
            pass
    i = t.find("{")
    if i < 0:
        return None
    depth, instr, esc = 0, False, False
    for j in range(i, len(t)):
        c = t[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[i:j + 1])
                except ValueError:
                    return None
    return None


def call_judge(prompt, model, effort, base_url=None, timeout=900, max_output_tokens=16000):
    """One judge call through an OpenAI-compatible /chat/completions endpoint.

    The one function to replace for another API. response_format is sent but may be
    ignored, so the JSON is extracted leniently rather than trusted. Neither the effort
    field nor anything else GUARANTEES reasoning: the stored verdict records
    usage.reasoning_tokens to check it.

    Args:
        prompt: the rendered judge prompt.
        model: the judge model's name at the endpoint.
        effort: reasoning effort, or "" to send none.
        base_url: the endpoint's base URL, or None for JUDGE_BASE_URL.
        timeout: seconds.
        max_output_tokens: the completion cap, reasoning included.

    Returns:
        (text, usage, the judge's reasoning text).

    Raises:
        Retryable: a transport error, a retryable status, or a truncated reply.
    """
    base, key = _endpoint(base_url)
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_output_tokens,
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "protein_judgement",
                                                   "strict": True, "schema": SCHEMA}}}
    if effort:
        payload["reasoning_effort"] = effort
    b = _post(base + "/chat/completions", key, payload, timeout)
    ch = (b.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    thinking = msg.get("reasoning_content") or ""
    if ch.get("finish_reason") == "length":
        raise Retryable("hit max_tokens before finishing; raise --max-output-tokens")
    u = b.get("usage") or {}
    usage = {"input_tokens": u.get("prompt_tokens"),
             "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
             "output_tokens": u.get("completion_tokens"),
             "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
             "total_tokens": u.get("total_tokens")}
    return text, usage, thinking


# --------------------------------------------------------------- validation
def hard_problems(probs):
    """The problems that disqualify a verdict (everything but the advisory ones)."""
    return [p for p in (probs or []) if not p.startswith(SOFT_PREFIX)]


def validate(verdict):
    """Every axis present with a 1-10 or -1 score, plus one advisory consistency check.

    Args:
        verdict: the parsed verdict object.

    Returns:
        Problem strings; empty when the verdict is clean.
    """
    probs = []
    if not isinstance(verdict, dict):
        return ["not an object"]
    for a in AXES:
        v = verdict.get(a)
        if not isinstance(v, dict):
            probs.append("axis %s: missing" % a)
            continue
        s = v.get("score")
        if not isinstance(s, int) or (s != -1 and not 1 <= s <= 10):
            probs.append("axis %s: bad score %r" % (a, s))
    # A 10 for "grounded" alongside a list of things nothing supports is the one
    # internally inconsistent verdict worth surfacing; strict mode cannot catch it.
    rel = (verdict.get("reliability") or {}).get("score")
    n_unsup = len(verdict.get("unsupported_claims") or [])
    if isinstance(rel, int) and rel >= 9 and n_unsup >= 3:
        probs.append("reliability %d but %d unsupported claims" % (rel, n_unsup))
    return probs


# ------------------------------------------------------------------- corpus
def load_eval_results(evals_dir):
    """protein_id -> (generation, model input), from a directory of eval.py result JSONs."""
    out = {}
    for fn in sorted(os.listdir(evals_dir)):
        if not fn.endswith(".json") or fn.startswith("evaluation_errors"):
            continue
        try:
            with open(os.path.join(evals_dir, fn), encoding="utf-8") as f:
                rec = json.load(f)
        except (ValueError, OSError) as e:
            print("  skipping %s: %s" % (fn, e), file=sys.stderr)
            continue
        pid = rec.get("protein_id")
        if not pid:
            continue
        # eval.py marks a protein that raised during generation; judging its empty
        # string would score the harness, not the model.
        if not rec.get("success", True):
            continue
        gen = rec.get("generated_response") or ""
        if not gen.strip():
            continue
        out[pid] = (gen, rec.get("input_prompt") or "")
    return out


def do_export(args):
    """Join eval generations against the test corpus and freeze the pair.

    One file instead of thousands, and the sample is pinned so a second checkpoint
    can be judged on exactly the same proteins.

    Args:
        args: the parsed command line.

    Returns:
        The process exit code.
    """
    gens = load_eval_results(args.evals_dir)
    print("loaded %d generations from %s" % (len(gens), args.evals_dir), file=sys.stderr)
    if not gens:
        sys.exit("no usable generations in %s" % args.evals_dir)

    # THE GROUND TRUTH COMES FROM THE CORPUS, NOT THE SFT TABLE.
    #
    # The judge scores a generation against what is TRUE of the protein, so it
    # needs `protein_names`, `protein_function`, `subcellular_location`,
    # `interpro_formatted` and `ppi_formatted`. Those are LABEL columns: the SFT
    # table deliberately excludes every one of them, because handing them to the
    # student is the leak the whole design is built to prevent. Pointing the judge
    # at the SFT table would therefore silently score against a reference missing
    # five of its eleven fields.
    #
    # A local directory of parquet shards is read directly; a Hub id still goes
    # through load_dataset.
    if os.path.isdir(args.test_dataset):
        shards = sorted(glob.glob(os.path.join(args.test_dataset, "%s-*.parquet" % args.split)))
        if not shards:
            shards = sorted(glob.glob(os.path.join(args.test_dataset, "*.parquet")))
        if not shards:
            sys.exit("no parquet shards under %s" % args.test_dataset)
        by_id = {}
        for sh in shards:
            t = pq.read_table(sh)
            cols = {c: t.column(c).to_pylist() for c in t.column_names}
            for i in range(t.num_rows):
                r = {c: cols[c][i] for c in t.column_names}
                by_id[r["protein_id"]] = r
        print("read %d rows from %d shard(s) under %s"
              % (len(by_id), len(shards), args.test_dataset), file=sys.stderr)
    else:
        from datasets import load_dataset
        ds = load_dataset(args.test_dataset, split=args.split)
        by_id = {r["protein_id"]: r for r in ds}

    ids = sorted(set(gens) & set(by_id))
    missing = sorted(set(gens) - set(by_id))
    if missing:
        print("WARNING: %d generated proteins are not in %s (e.g. %s)"
              % (len(missing), args.test_dataset, ", ".join(missing[:5])), file=sys.stderr)

    # A PINNED LIST BEATS A SEEDED SAMPLE. --ids-file names the proteins outright,
    # so the subset is a reviewable artifact rather than a function of a seed and
    # whatever `ids` happened to contain. It also survives a change in the eval
    # set: a seeded sample silently picks different proteins the moment the
    # candidate pool changes. data/eval/judge_subset.txt is the published 1,000:
    # scorable, multi-aspect, IA-weighted and organism-capped.
    if args.ids_file:
        with open(args.ids_file) as f:
            want = [l.strip() for l in f if l.strip()]
        have = set(ids)
        ids = [i for i in want if i in have]
        gone = [i for i in want if i not in have]
        print("pinned subset: %d of %d ids present in this eval%s"
              % (len(ids), len(want),
                 ("; %d absent e.g. %s" % (len(gone), ", ".join(gone[:5]))) if gone else ""),
              file=sys.stderr)
        if not ids:
            sys.exit("none of the pinned ids are in this eval directory")
    elif args.sample and args.sample < len(ids):
        ids = sorted(random.Random(args.sample_seed).sample(ids, args.sample))

    keep = ("protein_id", "protein_names", "protein_function", "organism",
            "subcellular_location", "go_mf", "go_bp", "go_cc",
            "go_mf_tier2", "go_bp_tier2", "go_cc_tier2",
            "interpro_formatted", "ppi_formatted", "length",
            # The judge scores against the evidence the TEACHER had. The
            # residue-feature block is the authoritative source for every positional
            # claim, and without it a judge flags correctly-grounded coordinates as
            # invented -- measured at reliability 6.56 on the corpus's own reference
            # traces.
            "features_formatted", "additional_context",
            # Residue identities are checked against the numbered sequence.
            "sequence_numbered")
    # Say which reference fields are missing rather than emitting silent nulls --
    # a judge scoring against a reference with holes in it produces numbers that
    # look ordinary and mean less. Probe the SOURCE only for fields it is supposed
    # to carry as columns: the rendered ones are checked after the loop instead.
    _rendered = {"features_formatted", "additional_context", "sequence_numbered"}
    _probe = next(iter(by_id.values()), {})
    _absent = [k for k in keep if k not in _probe and k not in _rendered]
    if _absent:
        print("WARNING: %s carries none of %s; the judge will score against an "
              "incomplete reference" % (args.test_dataset, ", ".join(_absent)),
              file=sys.stderr)
    dest = os.path.abspath(args.export_rows)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        _n_feat = _n_ctx = _n_seq = _n_prep_failed = 0
        for pid in ids:
            row = {k: by_id[pid].get(k) for k in keep}
            # Render with the TEACHER'S OWN function rather than re-implementing it,
            # so the judge's Input cannot drift from what was actually shown.
            # Render whenever ANY field is missing: a source that ships
            # `features_formatted` as a column has no `additional_context` column.
            # A column value takes precedence over the rendered one.
            if not all(row.get(k) for k in _rendered):
                try:
                    fields = prepare_fields(by_id[pid])
                    for k in _rendered:
                        if fields.get(k) and not row.get(k):
                            row[k] = fields[k]
                except Exception:
                    _n_prep_failed += 1
            _n_feat += 1 if row.get("features_formatted") else 0
            _n_ctx += 1 if row.get("additional_context") else 0
            _n_seq += 1 if row.get("sequence_numbered") else 0
            row["generation"], row["model_input"] = gens[pid]
            # Provenance travels with the data; a judged verdict must be traceable
            # to the checkpoint that produced the generation.
            row["evals_dir"] = os.path.abspath(args.evals_dir)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %d rows -> %s" % (len(ids), dest), file=sys.stderr)
    # The rubric scores positional claims against the feature block; if the
    # rendering silently produced nothing, reliability drops ~2 points with no
    # other sign.
    if _n_feat == 0 and ids:
        print("WARNING: features_formatted is EMPTY on every row -- the judge will "
              "score positional claims blind.", file=sys.stderr)
    else:
        print("  features_formatted on %d/%d rows, additional_context on %d, sequence_numbered on %d"
              % (_n_feat, len(ids), _n_ctx, _n_seq), file=sys.stderr)
    if _n_prep_failed:
        print("WARNING: rendering the feature fields raised on %d/%d rows; those rows keep "
              "only their column values" % (_n_prep_failed, len(ids)), file=sys.stderr)
    return 0


def load_rows(args):
    """The exported rows to judge, cut to --limit."""
    with open(args.rows_jsonl, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print("loaded %d rows from %s" % (len(rows), args.rows_jsonl), file=sys.stderr)
    if args.limit:
        rows = rows[: args.limit]
    return rows


# --------------------------------------------------------------------- main
def process(row, args, dag):
    """Judge one row and write its verdict JSON atomically.

    Args:
        row: an exported row.
        args: the parsed command line.
        dag: a MiniGoDag, or None.

    Returns:
        (protein id, status, problems).
    """
    pid = row["protein_id"]
    dest = os.path.join(args.out, "%s.json" % pid)
    if os.path.exists(dest) and not args.overwrite:
        return pid, "skipped", []
    prompt = build_prompt(row, dag, args.max_go_terms)
    if args.dry_run:
        return pid, "dry-run:%dchars" % len(prompt), []
    t0 = time.time()
    attempts = 0
    while True:
        attempts += 1
        try:
            text, usage, thinking = call_judge(
                prompt, args.model, args.effort, args.base_url, args.timeout,
                args.max_output_tokens)
            break
        except Retryable as e:
            if attempts > args.retries:
                raise
            print("  %s: attempt %d failed (%s), retrying"
                  % (pid, attempts, str(e)[:110]), file=sys.stderr)
            time.sleep(min(30, 10 * attempts))
    verdict = extract_json(text)
    probs = validate(verdict) if verdict is not None else ["unparseable JSON"]
    rec = {"protein_id": pid,
           "model": args.model, "effort": args.effort,
           "elapsed_s": round(time.time() - t0, 1), "attempts": attempts,
           "problems": probs,
           "verdict": verdict,
           # The full model output, always -- not just on failure. It is the only
           # way to re-audit a verdict later.
           "raw": text,
           "model_thinking": thinking,
           "model_thinking_chars": len(thinking),
           "usage": usage,
           "prompt_chars": len(prompt),
           "generation_chars": len(row.get("generation") or ""),
           "evals_dir": row.get("evals_dir")}
    tmp = dest + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    os.replace(tmp, dest)      # atomic: a killed run never leaves half a file
    hard = hard_problems(probs)
    return pid, ("ok" if not hard else "invalid:%d" % len(hard)), probs


def judge_rows(rows, args, dag):
    """Judge rows concurrently, printing progress.

    Args:
        rows: the exported rows to judge.
        args: the parsed command line.
        dag: a MiniGoDag, or None.

    Returns:
        (status -> count, problem strings).
    """
    counts, problems, t_start = {}, [], time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, r, args, dag): r["protein_id"] for r in rows}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            pid = futs[fut]
            try:
                pid, status, probs = fut.result()
            except Exception as e:
                status, probs = "error", ["%s: %s" % (type(e).__name__, str(e)[:200])]
                print("  %s: %s" % (pid, probs[0]), file=sys.stderr)
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            problems += ["%s: %s" % (pid, p) for p in probs]
            if i % 25 == 0 or i == len(rows):
                eta = (time.time() - t_start) / i * (len(rows) - i) / 60.0
                print("  %d/%d  %s  eta %.0f min" % (i, len(rows), counts, eta), file=sys.stderr)
    return counts, problems


def scores_every_axis(path):
    """Whether a verdict record carries a numeric score on every axis.

    Args:
        path: the record's JSON.

    Returns:
        False for a missing or unreadable record, or one missing any axis score.
    """
    try:
        with open(path, encoding="utf-8") as f:
            verdict = json.load(f).get("verdict") or {}
    except (OSError, ValueError):
        return False
    return all(isinstance(verdict.get(a), dict) and isinstance(verdict[a].get("score"), (int, float))
               for a in AXES)


def set_aside_incomplete(out, pid, done_passes):
    """Move a protein's incomplete verdict to `<out>/_retry/`, so the next pass judges it again.

    Args:
        out: the run directory.
        pid: the protein.
        done_passes: passes finished so far, recorded in the moved file's name.

    Returns:
        True when the protein needs judging again.
    """
    dest = os.path.join(out, "%s.json" % pid)
    if not os.path.exists(dest):
        return True
    if scores_every_axis(dest):
        return False
    retry = os.path.join(out, "_retry")
    os.makedirs(retry, exist_ok=True)
    os.replace(dest, os.path.join(retry, "%s.pass%d.json" % (pid, done_passes)))
    return True


def do_summarize(args):
    """Print per-axis means and distributions for a finished run.

    Args:
        args: the parsed command line.

    Returns:
        The process exit code.
    """
    files = sorted(f for f in os.listdir(args.out)
                   if f.endswith(".json") and not f.startswith("_"))
    if not files:
        sys.exit("no results in %s" % args.out)
    recs = []
    for fn in files:
        with open(os.path.join(args.out, fn), encoding="utf-8") as f:
            recs.append(json.load(f))
    ok = [r for r in recs if r.get("verdict") and not hard_problems(r.get("problems"))]
    # A verdict the judge wrote without reasoning is not the protocol's; an endpoint that reports
    # no reasoning count at all is given the benefit of the doubt.
    unreasoned = [r for r in ok if (r.get("usage") or {}).get("reasoning_tokens") == 0]
    ok = [r for r in ok if r not in unreasoned]
    soft = sum(1 for r in ok if r.get("problems"))
    print("results     : %d" % len(recs))
    print("usable      : %d" % len(ok))
    if unreasoned:
        print("  excluded, judged without reasoning : %d" % len(unreasoned))
    if soft:
        print("  of which carry an advisory flag : %d" % soft)
    bad = len(recs) - len(ok) - len(unreasoned)
    if bad or soft:
        seen = {}
        for r in recs:
            for p in r.get("problems") or []:
                k = re.sub(r"[-0-9].*", "", p).strip()
                seen[k] = seen.get(k, 0) + 1
        print("unusable    : %d" % bad)
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1]):
            mark = "  (advisory)" if k.startswith(SOFT_PREFIX.strip()) else ""
            print("    %5d  %s%s" % (v, k, mark))
    if not ok:
        return 2
    print()
    print("%-14s %7s %7s %7s %7s   %s" % ("axis", "n", "mean", "median", "n(-1)", "distribution 1..10"))
    for a in AXES:
        vals = [r["verdict"][a]["score"] for r in ok if isinstance(
            (r["verdict"].get(a) or {}).get("score"), int)]
        scored = sorted(v for v in vals if v != -1)
        na = sum(1 for v in vals if v == -1)
        if not scored:
            print("%-14s %7d %7s %7s %7d" % (a, len(vals), "-", "-", na))
            continue
        mean = sum(scored) / float(len(scored))
        med = scored[len(scored) // 2]
        hist = "".join("%3d" % sum(1 for v in scored if v == k) for k in range(1, 11))
        print("%-14s %7d %7.2f %7d %7d   %s" % (a, len(vals), mean, med, na, hist))
    unsup = [len(r["verdict"].get("unsupported_claims") or []) for r in ok]
    print()
    print("unsupported claims : %.2f mean, %d with none, %d with 3+"
          % (sum(unsup) / float(len(unsup)),
             sum(1 for u in unsup if u == 0), sum(1 for u in unsup if u >= 3)))
    el = [r.get("elapsed_s") or 0 for r in recs]
    tin = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in recs)
    tout = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in recs)
    print("wall clock         : %.1f s mean per protein" % (sum(el) / float(len(el))))
    print("tokens             : %d in, %d out  (%.0f in / %.0f out per protein)"
          % (tin, tout, tin / float(len(recs)), tout / float(len(recs))))
    return 0


def main():
    """Export rows, judge them, or summarize a run.

    Returns:
        The process exit code.
    """
    ap = argparse.ArgumentParser(
        description="LLM-as-a-judge scoring for BioReason-Pro 2 generations. "
                    "See evals/judge/README.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-jsonl", help="judge these rows (from --export-rows)")
    ap.add_argument("--out", default=os.path.join(REPO, "judge", "run"))
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="the judge model's name at the endpoint (default: $JUDGE_MODEL)")
    ap.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORT_CHOICES)
    ap.add_argument("--base-url", default=None,
                    help="the endpoint's base URL, ending in /v1 (default: $%s)" % BASE_URL_VAR)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-output-tokens", type=int, default=32000)
    ap.add_argument("--max-go-terms", type=int, default=60)
    # Per-call latency is 30-55 s, so throughput is almost pure concurrency:
    # 8 workers gives ~4 s/protein, ~192 gives ~0.3 s.
    ap.add_argument("--workers", type=int, default=64,
                    help="concurrent calls")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3,
                    help="judging passes; each pass after the first re-judges the verdicts that miss "
                         "an axis score (the published runs were complete within 3)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the first prompt in full and exit")
    ap.add_argument("--summarize", action="store_true", help="report on an existing --out")
    ap.add_argument("--go-obo", default=_paths.GO_OBO)
    ap.add_argument("--export-rows", help="write judgeable rows to this JSONL")
    ap.add_argument("--evals-dir", help="directory of eval.py result JSONs")
    ap.add_argument("--test-dataset", default=_paths.CORPUS,
                    help="Hub id or local parquet directory. The SOURCE corpus, "
                         "never the SFT table -- that drops the label columns the "
                         "judge scores against.")
    ap.add_argument("--ids-file", default=None,
                    help="file of protein ids, one per line; overrides --sample")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--sample-seed", type=int, default=23)
    args = ap.parse_args()

    if args.summarize:
        return do_summarize(args)
    if args.export_rows:
        if not args.evals_dir:
            sys.exit("--export-rows needs --evals-dir")
        return do_export(args)
    if not args.rows_jsonl:
        sys.exit("need --rows-jsonl (or --export-rows / --summarize)")
    if not args.model and not args.dry_run and not args.print_prompt:
        sys.exit("need --model (or JUDGE_MODEL): the judge model's name at the endpoint")

    rows = load_rows(args)
    if not rows:
        sys.exit("no rows")
    # A missing sequence renders as "None" and the judge would score residue claims blind.
    n_missing = sum(1 for r in rows if not r.get("sequence_numbered"))
    if n_missing:
        sys.exit("%d/%d rows lack sequence_numbered, which the prompt needs; "
                 "re-export them with --export-rows" % (n_missing, len(rows)))

    dag = None
    if os.path.exists(args.go_obo):
        dag = MiniGoDag(args.go_obo)
    else:
        print("WARNING: %s missing; GO terms will be rendered as bare ids"
              % args.go_obo, file=sys.stderr)

    if args.print_prompt:
        print(build_prompt(rows[0], dag, args.max_go_terms))
        return 0

    os.makedirs(args.out, exist_ok=True)
    # Keep the frozen input WITH its results, so a run folder is readable later
    # without hunting for the jsonl it was fed -- copied, not moved, because one
    # rows file legitimately feeds several runs.
    kept = os.path.join(args.out, "rows.jsonl")
    if os.path.abspath(args.rows_jsonl) != os.path.abspath(kept):
        if not os.path.exists(kept):
            shutil.copy2(args.rows_jsonl, kept)
    # Make the output directory self-describing. Scores from different models or
    # efforts are NOT poolable.
    with open(os.path.join(args.out, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "effort": args.effort,
                   "max_output_tokens": args.max_output_tokens,
                   "max_go_terms": args.max_go_terms,
                   "rows_jsonl": "rows.jsonl",
                   "rows_jsonl_source": os.path.abspath(args.rows_jsonl),
                   "n_rows": len(rows),
                   "evals_dir": rows[0].get("evals_dir")}, f, indent=2)
    print("judging %d proteins  model=%s effort=%s -> %s"
          % (len(rows), args.model, args.effort, args.out), file=sys.stderr)
    if not args.dry_run:
        init_session(args.workers)

    counts, problems = {}, []
    todo = rows
    for done_passes in range(1 if args.dry_run else args.passes):
        if done_passes:
            # The protocol's retry: a verdict missing an axis score is moved aside and judged again.
            todo = [r for r in rows if set_aside_incomplete(args.out, r["protein_id"], done_passes)]
            if not todo:
                break
            print("pass %d: retrying %d incomplete verdict(s)" % (done_passes + 1, len(todo)),
                  file=sys.stderr)
        counts, problems = judge_rows(todo, args, dag)
    print("\ndone: %s" % counts, file=sys.stderr)
    if problems:
        print("%d problem(s); first 10:" % len(problems), file=sys.stderr)
        for p in problems[:10]:
            print("  " + p, file=sys.stderr)
    return 0 if not counts.get("error") else 2


if __name__ == "__main__":
    sys.exit(main())
