# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Random per-sample dropout of the prompt's optional context: partners, GO-GPT, features.

WHY THIS IS NOT A `.map()` STEP
-------------------------------
The loader's `.map()` steps are content-addressed and cached to arrow. Randomising there
would freeze ONE realisation for the whole run and bake it into the cache, so every epoch
would see the identical mask. Dropout therefore happens after the cache, per `__getitem__`,
which is what `ContextDropoutDataset` below does.

HOW A SECTION IS DROPPED
------------------------
A prompt table stores the finished `question` beside the `ctx_*` pieces it was rendered
from. Dropping a section replaces one piece with the template's absence string and
re-formats the template; it never edits the finished string. With every probability at 0.0
this module is a no-op and the rendered prompt is byte-identical to the table's.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, fields as _dc_fields
from random import Random
from typing import Any, Dict

import bioreason2.utils.torchmetrics_guard  # noqa: F401  (before the Lightning import below)
from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm
from bioreason2.dataset.cafa5.load import _format_reasoning_prompt


@dataclass
class ContextDropoutConfig:
    """Per-source dropout probabilities. All default to 0.0 => exact no-op.

    Each context field is drawn once from none / complete / partial: `*_section` is
    P(none), `*_complete` is P(the block exactly as the corpus rendered it), and partial
    is the remainder, which drops each line at `*_item`. The shares are absolute fractions
    of all accesses, so none + complete must be at most 1.

    The published settings, and why (measured over all 133,114 v2 train rows):
      features  0.50 / 0.10 / 0.40 @ item 0.25
        -> no features on 50.8% of accesses, 13.1 of 32.7 bullets shown (40%
           of the evidence), partial removes 8.2 and the at-least-one guard
           fires 3.2%. 0.25 rather than 0.10 because at 0.10 the guard fires
           14.3% of the time and only 3.3 bullets go, so the label shouts
           "subset" over a list that is 90% intact.
      partners  0.80 / 0.00 / 0.20 @ item 0.50
        -> no partners on 84.9% (22.3pp of that structural: those rows have
           none in the data at all), mean 1.25 of 8.3 shown.

    `structure_section` withholds the structure from a row that has one; it reaches ESM3,
    not the text.
    """

    ppi_section: float = 0.0
    ppi_item: float = 0.0
    ppi_complete: float = 0.0
    gogpt_section: float = 0.0
    gogpt_item: float = 0.0
    features_section: float = 0.0
    features_complete: float = 0.0
    features_item: float = 0.0
    structure_section: float = 0.0

    def __post_init__(self):
        """Reject a mode split that cannot be a probability distribution.

        `partial` is the remainder, so none + complete > 1 does not raise anywhere
        downstream -- `_draw_mode` simply never reaches the partial branch and the run
        trains with a silently different recipe than its config states.

        Raises:
            ValueError: when a field's none + complete exceeds 1.
        """
        for none_k, comp_k in (("features_section", "features_complete"),
                               ("ppi_section", "ppi_complete")):
            total = float(getattr(self, none_k)) + float(getattr(self, comp_k))
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"{none_k} + {comp_k} = {total:.3f} > 1. The partial branch "
                    f"is the remainder, so this would silently be 0 rather than "
                    f"the split the config names.")

    def enabled(self) -> bool:
        """True if any knob is active.

        Returns:
            Whether any probability is above zero.
        """
        return any(float(getattr(self, f.name)) > 0.0 for f in _dc_fields(self))

    def describe(self) -> str:
        """Every knob as `name=value`.

        Returns:
            One line for the run log.
        """
        return " ".join(f"{f.name}={getattr(self, f.name):g}" for f in _dc_fields(self))


#: Which `ctx_*` column each set of knobs governs, as
#: `column -> (p_none_knob, p_complete_knob, p_item_knob)`.
#:
#: A field is drawn ONCE from a three-way categorical -- none / complete /
#: partial -- rather than from a chain of independent coin flips, because the
#: shares were chosen as absolute fractions of all accesses (the published recipe:
#: features are 50% none, 10% complete, 40% partial) and a chain would make each one
#: conditional on the last. `partial` is the remainder and has no knob of its
#: own; a config where none + complete > 1 is rejected at construction.
#:
#: `ctx_organism` has no knobs and never varies: it is one token of context the
#: model needs to know which compartments and pathways exist at all; removing it
#: is a distinct condition, not a dropout.
CTX_FIELDS = {
    "ctx_partners":     ("ppi_section", "ppi_complete", "ppi_item"),
    "ctx_speculations": ("gogpt_section", None, "gogpt_item"),
    "ctx_features":     ("features_section", "features_complete", "features_item"),
    "ctx_organism":     (None, None, None),
}

#: Kept as an alias: `verify_no_op` and the tests iterate the column names.
CTX_TO_DROP = {k: v[0] for k, v in CTX_FIELDS.items()}

#: The cap notice `render_features` appends -- "- ... 412 further features
#: omitted". It is a statement ABOUT the list rather than an item in it, so it is
#: never a dropout candidate and never counts toward "at least one was removed".
#: Dropping it would delete the only signal that the block is short.
_CAP_NOTICE = "- ..."


def _drop_items_at_least_one(block, p_item, rng, absent):
    """Drop each line independently, but guarantee at least one goes.

    The guarantee is what makes the `a subset, others withheld` label true. Without
    it the label lies whenever the sampling happens to keep everything, which is
    not rare: at p=0.10 over the real length distribution it happens **14.3%** of
    the time, and the override would then be setting the distribution rather than
    the rate you configured. At the chosen 0.25 it fires 3.2% of the time, so it
    is a guard rather than a mechanism -- which is the reason 0.25 was picked over
    0.10.

    A one-item list is returned UNCHANGED and reported as complete: forcing its
    only line out would empty the block, and a "partial" list of nothing is just
    the none branch wearing a different label. 0.47% of rows.

    Args:
        block: the rendered section, one item per line.
        p_item: per-line drop probability.
        rng: the access's random generator.
        absent: the template's absence string.

    Returns:
        (the surviving block, whether anything was dropped).
    """
    lines = [ln for ln in str(block).splitlines() if ln.strip()]
    items = [ln for ln in lines if not ln.startswith(_CAP_NOTICE)]
    notice = [ln for ln in lines if ln.startswith(_CAP_NOTICE)]
    if len(items) <= 1:
        return block, False
    kept = [ln for ln in items if rng.random() >= p_item]
    if len(kept) == len(items):                     # nothing fell out -- force one
        kept.pop(rng.randrange(len(kept)))
    if not kept:
        return absent, True
    return "\n".join(kept + notice), True


def _draw_mode(rng, p_none, p_complete):
    """One draw from none / complete / partial. `partial` is the remainder.

    Args:
        rng: the access's random generator.
        p_none: P(none).
        p_complete: P(complete).

    Returns:
        "none", "complete" or "partial".
    """
    r = rng.random()
    if r < p_none:
        return "none"
    if r < p_none + p_complete:
        return "complete"
    return "partial"


def apply_context_dropout_prerendered(example, cfg, rng, template):
    """Drop sections of a pre-rendered question by rebuilding it from its pieces.

    Dropping a section means replacing one `ctx_*` value and re-formatting, not editing
    the finished string -- a text edit would have to parse a prompt to find a section
    boundary, a class of repair that measured 4.9% precision where it was tried.

    THE SCOPE LABEL. `features_scope` states which of the three modes produced the
    block the model is looking at, so a model handed the full list is told so and
    has no reason to invent a span. Without it the dropout would teach one blurred
    behaviour -- "features may be missing, so guess" -- which at inference, where
    the list is always whole, is fabrication. Partners carry the modes but NOT a
    label: the risk it guards is verbatim spans and coordinates, and a partner list is
    prose names the trace mentions in passing.

    Args:
        example: one row of a prompt table.
        cfg: the dropout probabilities.
        rng: the access's random generator.
        template: (PROMPT, ABSENT, (SCOPE_ALL, SCOPE_SUBSET, SCOPE_NONE)) from
            scripts/data/question_template.py.

    Returns:
        The row with its `question` re-rendered.
    """
    prompt, absent, scopes = template
    ctx = {k[4:]: example.get(k) or absent for k in CTX_FIELDS}
    scope_all, scope_subset, scope_none = scopes

    for col, (k_none, k_complete, k_item) in CTX_FIELDS.items():
        if k_none is None:
            continue
        field = col[4:]
        value = ctx[field]
        p_none = float(getattr(cfg, k_none, 0.0) or 0.0)
        p_complete = float(getattr(cfg, k_complete, 0.0) or 0.0) if k_complete else 0.0
        p_item = float(getattr(cfg, k_item, 0.0) or 0.0) if k_item else 0.0
        if not (p_none or p_complete or p_item):
            continue

        mode = _draw_mode(rng, p_none, p_complete)
        if mode == "none":
            ctx[field] = absent
        elif mode == "partial" and p_item > 0.0 and value != absent:
            ctx[field], _ = _drop_items_at_least_one(value, p_item, rng, absent)
        # `complete` leaves the stored value exactly as the corpus rendered it.

        if field == "features":
            ctx["features_scope"] = (
                scope_none if ctx[field] == absent
                else scope_all if ctx[field] == value
                else scope_subset)

    ctx.setdefault("features_scope",
                   example.get("ctx_features_scope") or scope_none)
    return {**example, "question": prompt.format(**ctx)}


def load_question_template(path: str) -> tuple:
    """The prompt pieces of a question_template.py.

    The prompt is not this module's, so it is loaded from the path the run names rather
    than imported. The file is pure constants -- no imports, no code -- so this cannot pull
    anything in.

    Args:
        path: a question template, normally scripts/data/question_template.py.

    Returns:
        (PROMPT, ABSENT, (SCOPE_ALL, SCOPE_SUBSET, SCOPE_NONE)).
    """
    spec = importlib.util.spec_from_file_location("question_template", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # SCOPE_ALL falls back to SCOPE_SUBSET, not to "": a template with no "complete"
    # label wants the subset wording, and "" renders `Residue features, :`. Reading the
    # scopes with getattr keeps a template that predates the labelled feature block
    # loadable -- it simply never produces a `features_scope` slot for them to fill.
    return (mod.PROMPT, mod.ABSENT,
            (getattr(mod, "SCOPE_ALL", None) or getattr(mod, "SCOPE_SUBSET", ""),
             getattr(mod, "SCOPE_SUBSET", ""),
             getattr(mod, "SCOPE_NONE", "")))


class ContextDropoutDataset:
    """Wraps a processed prompt-table split and re-renders the prompt per access.

    Epoch handling: `persistent_workers=True` forks the dataloader workers once and
    they keep their OWN copy of this object, so assigning `self.epoch = n` in the
    main process would never reach them. The epoch is therefore held in a
    `multiprocessing.Value` created before the fork, which every worker shares.
    `ContextDropoutEpochCallback` bumps it.

    The mask is derived from (base_seed, epoch, index), so it is different every
    epoch but reproducible across restarts and identical on every DDP rank for a
    given (epoch, index) -- which matters because ranks must not disagree about what
    sample `i` looks like when resuming.
    """

    def __init__(self, dataset, cfg: ContextDropoutConfig, base_seed: int = 23,
                 epoch_value=None, question_template_py: str = None):
        """Wrap a split.

        Args:
            dataset: the processed split (rows carry `question` and `ctx_*`).
            cfg: the dropout probabilities.
            base_seed: seed the per-access masks derive from.
            epoch_value: shared `multiprocessing.Value` holding the epoch, or None for 0.
            question_template_py: scripts/data/question_template.py; required when any
                text knob is active.

        Raises:
            RuntimeError: when dropout is enabled and no template was named, which would
                leave the question untouched while the config says otherwise.
        """
        self.dataset = dataset
        self.cfg = cfg
        self.base_seed = int(base_seed)
        self.epoch_value = epoch_value
        self.template = None
        if question_template_py:
            self.template = load_question_template(question_template_py)
            print(f"context dropout: question template = {question_template_py}")
        elif cfg.enabled():
            raise RuntimeError(
                "context dropout is enabled but no question_template_py was given, so the "
                "question could not be rebuilt and the dropout would silently do nothing. "
                "Set QUESTION_TEMPLATE_PY (or --question_template_py) to "
                "scripts/data/question_template.py.")

    def verify_no_op(self, index: int = 0) -> None:
        """Assert that rebuilding row `index` with NOTHING dropped reproduces its question.

        If the `ctx_*` pieces and the stored question disagree, every dropped sample
        would differ from the corpus in ways nothing else would surface.

        Args:
            index: the row to check.

        Raises:
            RuntimeError: when the rebuilt question differs from the stored one.
        """
        row = self.dataset[int(index)]
        prompt, absent, scopes = self.template
        pieces = {k[4:]: row.get(k) or absent for k in CTX_FIELDS}
        # The stored scope, NOT a recomputed one: the corpus decided what the
        # un-dropped block is called, and this assertion exists to catch the
        # table and the template disagreeing.
        pieces["features_scope"] = row.get("ctx_features_scope") or scopes[2]
        rebuilt = prompt.format(**pieces)
        if rebuilt != row["question"]:
            raise RuntimeError(
                "ContextDropoutDataset: rebuilding the question from its "
                "ctx_* columns does not reproduce `question`. The template "
                "and the SFT table were built from different revisions.\n"
                f"  stored : {row['question'][:200]!r}\n"
                f"  rebuilt: {rebuilt[:200]!r}")

    def __len__(self) -> int:
        """Rows in the wrapped split.

        Returns:
            The row count.
        """
        return len(self.dataset)

    @property
    def _epoch(self) -> int:
        """The current epoch, shared across dataloader workers."""
        return int(self.epoch_value.value) if self.epoch_value is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Row `idx` with this epoch's mask applied.

        Args:
            idx: row index.

        Returns:
            The row; when dropout is on, its question, user turn and structure path
            reflect the mask and the assistant turn is the stored one.
        """
        row = self.dataset[int(idx)]
        if not self.cfg.enabled():
            return row

        seed = (self.base_seed * 1000003 + self._epoch * 10007 + int(idx)) & 0x7FFFFFFF
        rng = Random(seed)

        ablated = apply_context_dropout_prerendered(row, self.cfg, rng, self.template)
        rendered = format_cafa5_for_protein_llm(_format_reasoning_prompt(ablated))

        out = dict(row)
        # Carry the ablated question on the row as well as inside `prompt`.
        # Without this the column and the turn disagree: the model trains on the
        # dropped prompt while anything reading `question` -- a check, a dump, a
        # test -- sees the full one and reports the dropout as not firing.
        out["question"] = ablated["question"]
        new_prompt = rendered["prompt"]
        cached_prompt = row.get("prompt")
        # Swap ONLY the user turn. Splicing the cached assistant turn back in makes
        # "targets are untouched" structural rather than incidental -- nothing about
        # the reasoning trace or the answer can drift when the context changes.
        if isinstance(cached_prompt, list) and len(cached_prompt) == len(new_prompt):
            out["prompt"] = [
                new_prompt[i] if m.get("role") == "user" else m
                for i, m in enumerate(cached_prompt)
            ]
        else:
            out["prompt"] = new_prompt
        # Nulling the path degrades to sequence-only, as a row with no file already does.
        if out.get("structure_path") and rng.random() < self.cfg.structure_section:
            out["structure_path"] = None

        return out


try:  # pragma: no cover - lightning is present wherever training runs
    from pytorch_lightning import Callback as _PLCallback
except Exception:  # keep this module importable for data-only tests
    _PLCallback = object


class ContextDropoutEpochCallback(_PLCallback):
    """Lightning callback that advances the shared epoch counter.

    MUST subclass lightning's Callback rather than duck-type it: the trainer's
    `_validate_callbacks_list` reads `callback.state_key`, which only the base class
    provides, and a plain object fails there with an AttributeError after model setup
    -- i.e. minutes into a run, not at submit time.
    """

    def __init__(self, epoch_value):
        """Hold the shared counter.

        Args:
            epoch_value: the `multiprocessing.Value` the datasets read.
        """
        self.epoch_value = epoch_value

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: D401
        """Publish the trainer's epoch to the workers.

        Args:
            trainer: the Lightning trainer.
            pl_module: unused.
        """
        if self.epoch_value is not None:
            self.epoch_value.value = int(trainer.current_epoch)

    def state_dict(self):
        """Checkpointed state: the epoch.

        Returns:
            {"epoch": n}.
        """
        return {"epoch": int(self.epoch_value.value) if self.epoch_value else 0}

    def load_state_dict(self, state):
        """Restore the epoch on resume.

        Args:
            state: what `state_dict` saved.
        """
        if self.epoch_value is not None:
            self.epoch_value.value = int(state.get("epoch", 0))
