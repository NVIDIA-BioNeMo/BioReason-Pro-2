# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Render a protein's annotation fields the way the reasoning-trace prompt showed them.

The corpus traces were written from the templates in `synth/` with these fields substituted
for their placeholders. Three readers need the same bytes:

- `scripts/data/build_question.py` renders the student's residue-feature block with
  `render_features`, so the block the student reads is the block the teacher read;
- the RL judge reward (`bioreason2/rl/judge_reward.py`) and the eval judge
  (`evals/judge/llm_judge.py`) show the judge the teacher's inputs via `prepare_fields`.

`MiniGoDag` is a minimal is_a / part_of reader of go-basic.obo, so leaf filtering runs without
goatools.
"""
import os
import re
import threading

from bioreason2.dataset.utils import filter_go_terms_to_leaf_terms, format_go_terms_with_names

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The pinned ontology, and the one every other tool here already defaults to.
DEFAULT_OBO = os.path.join(REPO, "bioreason2", "dataset", "go-basic.obo")
_DAG_LOCK = threading.Lock()
_DEFAULT_DAG = None

ABSENT = "None annotated"
ECO_CLASS = {
    "0000269": "experimental",
    "0000305": "curator inference",
    "0000250": "by similarity",
    "0000255": "sequence model",
    "0000256": "automatic",
    "0000303": "author statement",
    # Rare -- 7 rows of 140,694 -- but an unmapped code renders as the raw
    # accession, which is the one thing this table exists to keep out of a prompt.
    "0000312": "imported",
    "0007744": "combinatorial evidence",
}
_ECO_RE = re.compile(r"\s*\{ECO:[^}]*\}")

#: Literature apparatus that is noise to a reader who cannot follow a citation.
#: `{ECO:0000269|PubMed:36048923}` says a curator read a paper, which changes no
#: inference the trace can draw, and `Xref=Rhea:RHEA:75299` identifies a reaction
#: the same sentence already spells out. Stripped from EVERY prose field, not just
#: comments: `subcellular_location` alone drops 52%, 36 -> 17 tokens.
#:
#: NOT stripped: `(Probable)`, `(By similarity)` and `(Potential)` are the
#: curator's confidence in the claim and belong in the trace's own hedging, and
#: `(residues 134-267)` is content.
_RE_ECO = re.compile(r"\s*\{ECO:[^}]*\}")
_RE_CITE = re.compile(r"\s*\((?:PubMed|Ref\.)[^)]*\)", re.I)
_RE_EVID = re.compile(r"\s*Evidence=\{[^}]*\};?")
_RE_XREF = re.compile(r"\s*Xref=[^;]*;?")
_RE_IACT = re.compile(r"\s*(?:IntAct=[^;]*;?|NbExp=\d+;?|EBI-\d+)")

#: Does this text contain any of the apparatus `strip_refs` removes? Used to
#: leave every other line ALONE. See `strip_refs_block`.
_RE_HAS_REF = re.compile(
    r"\{ECO:|\(PubMed:|\(Ref\.|Evidence=\{|Xref=|IntAct=|NbExp=\d|EBI-\d", re.I)

#: UniProt CC topics that reach the prompt, in the order they are rendered, with
#: the label the reader sees. Everything else is dropped: INTERACTION duplicates
#: `ppi_formatted`, TISSUE SPECIFICITY / DEVELOPMENTAL STAGE / INDUCTION are
#: where it is expressed rather than what it does, and the remaining twelve are
#: bookkeeping -- sequence caveats, web links, masses, LD50.
#:
#: Ordered from the molecular outward: what reaction it runs, what it needs, how
#: it is controlled, what it assembles into, where it sits, what it feeds, and
#: what breaks without it.
_CC_KEEP = (
    ("SIMILARITY", "Family", None),
    ("CATALYTIC ACTIVITY", "Catalytic activity", "catalytic"),
    ("COFACTOR", "Cofactor", "cofactor"),
    ("ACTIVITY REGULATION", "Activity regulation", None),
    ("BIOPHYSICOCHEMICAL PROPERTIES", "Kinetics and optima", None),
    ("SUBUNIT", "Subunit structure", None),
    ("DOMAIN", "Domain function", None),
    ("PTM", "Post-translational modification", None),
    #: Not a CC topic -- `subcellular_location` is its own column, but it is the
    #: same KIND of thing and sits naturally between what the protein assembles
    #: into and what pathway it feeds.
    ("__LOCATION__", "Subcellular location", None),
    ("PATHWAY", "Pathway", None),
    ("DISRUPTION PHENOTYPE", "Disruption phenotype", None),
    ("DISEASE", "Disease", None),
)

#: Feature classes, by how much they constrain a mechanism. Used ONLY to decide
#: what survives the cap -- never to order what is printed.
#: MUTAGEN sits here, not in _F_BOOK: 90% of its records state a functional
#: outcome at a named residue ("G->A: Loss of NADase activity."), which is direct
#: experimental evidence for a catalytic residue and strictly stronger than a
#: family assignment. Corpus-wide: 89,162 records on 12.9% of proteins, 87% of
#: them stating an outcome.
_F_CHEM = frozenset(("ACT_SITE", "BINDING", "DISULFID", "CROSSLNK", "SITE",
                     "MOD_RES", "CARBOHYD", "LIPID", "ZN_FING", "DNA_BIND",
                     "MOTIF", "MUTAGEN"))
_F_TOPO = frozenset(("TRANSMEM", "TOPO_DOM", "SIGNAL", "TRANSIT", "PROPEP",
                     "INTRAMEM", "COILED", "CHAIN", "PEPTIDE"))
#: Not rendered at all. HELIX/STRAND/TURN arrive on 7.1% of proteins at a mean of
#: 28 records each. Everything they imply is already stated by something better --
#: `homologous_superfamily` names the fold, mobidblt marks disorder -- and the
#: exact boundaries are the one thing here that cannot ground a functional claim:
#: "a helix runs 143-159" licenses nothing about what the protein does, unlike an
#: active site or a domain span. Rendered as three span lines they cost a median
#: 94 tokens but reach 2,211.
#:
#: `structure_*` is dropped outright. Those records duplicate the `pdb_formatted`
#: column, and a coverage summary built from either said only what the domain
#: calls and the SUBUNIT comment already say -- which parts are a folded unit, and
#: what it assembles into.
_F_SKIP = frozenset(("HELIX", "STRAND", "TURN", "structure_xray", "structure_em",
                     "structure_nmr", "structure_neutron", "structure_fiber",
                     "structure_other"))
_F_REGN = frozenset(("DOMAIN", "REPEAT", "REGION", "COMPBIAS"))
#: Dropped outright. CONFLICT is two sequence submissions disagreeing about what
#: the protein IS -- "T -> A (in Ref. 1; ABQ01214)" -- which says nothing about
#: function and invites reasoning about a residue that is a submission artefact.
#: Measured over all 133,114 train rows: 1.23% of records, on 22.6% of proteins.
#:
#: VARIANT is kept: 46.6% of it is dbSNP polymorphism and 10.5% strain variation,
#: but 23.4% carries a disease or phenotype note, which is functional evidence
#: about a named residue and is exactly what the block exists to supply.
_F_DROP = frozenset(("CONFLICT",))
#: Sequence bookkeeping. Kept, but first to go when a protein exceeds the cap.
_F_BOOK = frozenset(("VAR_SEQ", "NON_TER", "NON_CONS", "NON_STD",
                     "UNSURE", "INIT_MET"))
#: Databases whose `member_acc` is a label, not an accession. Their records carry
#: no `desc` at all -- all 11,304 phobius rows -- so without this promotion the
#: bullet would say nothing.
_F_LABEL_DB = frozenset(("phobius", "tmhmm", "signalp_e", "signalp_g+",
                         "signalp_g-", "mobidblt", "coils"))
#: Bullets per protein, after deduplication. It buys tail control rather than
#: money: titin renders 2,716 rows, ~60,300 tokens uncapped.
FEATURE_CAP = 200

#: How far two spans of one entry may differ, at each end, and still be treated
#: as the same finding. Five residues is boundary fuzz between member databases;
#: anything larger is a different call and is kept.
_SPAN_FUZZ = 5

#: Corpus `go_*_leaf` strings and go_leaf_block both store `GO:id name`.
_RE_GO_ID_FIRST = re.compile(r"^(GO:\d{7})\s+(.+)$")


def default_dag(path=None):
    """One shared MiniGoDag, so re-leafing never depends on the caller.

    Locked because the judges build prompts on many threads.

    Args:
        path: go-basic.obo to read on the first call; defaults to the pinned copy.

    Returns:
        The shared MiniGoDag.
    """
    global _DEFAULT_DAG
    with _DAG_LOCK:
        if _DEFAULT_DAG is None:
            _DEFAULT_DAG = MiniGoDag(path or DEFAULT_OBO)
        return _DEFAULT_DAG


def _text(v, default="Not known"):
    if v is None:
        return default
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "[]") else default


def _leaf_fields(row, go_dag=None):
    """Leaf GO blocks for a row when no MiniGoDag is given.

    If the row already carries go_*_leaf they are used as-is; otherwise they are
    derived from go_bp/go_mf/go_cc, which requires go_dag (a goatools DAG).

    Args:
        row: one corpus row.
        go_dag: a goatools GO DAG, or None.

    Returns:
        `go_{mf,bp,cc}_leaf` -> `GO:id name` lines, or "Not known".

    Raises:
        ValueError: a leaf block is missing and no go_dag was given to derive it.
    """
    leaf = {}
    for asp in ("mf", "bp", "cc"):
        key = f"go_{asp}_leaf"
        if row.get(key) not in (None, ""):
            leaf[key] = _text(row[key])
            continue
        raw = row.get(f"go_{asp}")
        if not raw:
            leaf[key] = "Not known"
            continue
        if go_dag is None:
            raise ValueError(
                f"{key} missing and no go_dag given; pass --go-obo to derive it")
        terms = filter_go_terms_to_leaf_terms(list(raw), go_dag)
        leaf[key] = "\n".join(format_go_terms_with_names(terms, go_dag)) if terms \
            else "Not known"
    return leaf


def _blank(v):
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in ("nan", "none", "[]", "not known") else s


def render_sequence_numbered(seq):
    """One residue per line with its position number."""
    seq = _blank(seq)
    if not seq:
        return ABSENT
    return "\n".join(f"{i} {aa}" for i, aa in enumerate(seq, 1))


def strip_refs(text):
    """Drop citations and evidence codes, then repair what removing them leaves.

    The punctuation matters: `... binding. {ECO:...}.` becomes `binding..` and
    `Optimum pH is 5.5. {ECO:...};` becomes `5.5.;` unless the trailing stops are
    collapsed afterwards.
    """
    if not text:
        return ""
    s = _RE_ECO.sub("", text)
    s = _RE_CITE.sub("", s)
    s = _RE_EVID.sub("", s)
    s = _RE_XREF.sub("", s)
    s = _RE_IACT.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([;.,])", r"\1", s)
    s = re.sub(r"[;,]\s*([.;])", r"\1", s)
    s = re.sub(r"\.\s*\.+", ".", s)
    s = re.sub(r"\.\s*;", ";", s)
    s = re.sub(r";\s*$", "", s)
    return s.strip(" ;,")


def strip_refs_block(text):
    """`strip_refs`, per line, and ONLY on lines that carry a reference.

    Two reasons it is not just `strip_refs(text)`.

    The patterns are unanchored and their character classes match a newline, so
    `\\(PubMed:[^)]*\\)` on a description whose closing paren is missing would run
    to the next `)` several bullets down and delete everything between. Working a
    line at a time makes that impossible.

    And the punctuation repair afterwards is written for PROSE -- it collapses
    `binding..` and `5.5.;`, which is what removing an `{ECO:...}` leaves behind.
    Run over a line that never had a reference it is not a no-op: it turned the
    cap marker `- ... 2442 further features omitted` into `-. 2442 further
    features omitted`, because `\\s+([;.,])` closed the gap after the dash and
    `\\.\\s*\\.+` then collapsed the ellipsis. So a line with nothing to strip is
    returned untouched, which is also the honest contract: this removes citations
    and changes nothing else.
    """
    if not text:
        return ""
    if not _RE_HAS_REF.search(text):
        return text
    return "\n".join(strip_refs(ln) if _RE_HAS_REF.search(ln) else ln
                     for ln in text.split("\n"))


def _cc_catalytic(text):
    """The reaction and the EC number, and nothing else.

    A CATALYTIC ACTIVITY block is 126 median tokens of which the chemistry is a
    fraction: the rest is Rhea and ChEBI accessions for the same reaction the
    sentence spells out. Keeping `Reaction=`, `EC=` and `PhysiologicalDirection=`
    is an 84% cut that loses no biology -- and the direction is real biology,
    since it says which way the reaction runs in vivo.
    """
    out = []
    m = re.search(r"Reaction=(.+?)(?:;\s*(?:Xref|Evidence|EC|PhysiologicalDirection)=|$)",
                  text, re.S)
    if m:
        out.append(strip_refs(m.group(1)))
    e = re.search(r"EC=([0-9.\-]+)", text)
    if e:
        out.append(f"EC {e.group(1)}")
    d = re.search(r"PhysiologicalDirection=([a-z-]+)", text)
    if d:
        out.append(d.group(1).replace("-", " "))
    return "; ".join(x for x in out if x)


def _cc_cofactor(text):
    """`Name=` only. The `Note=` is dropped with the ChEBI accessions."""
    names = re.findall(r"Name=([^;]+);", text)
    return ", ".join(x.strip() for x in names) if names else strip_refs(text)


_CC_EXTRACT = {"catalytic": lambda t: _cc_catalytic(t),
               "cofactor": lambda t: _cc_cofactor(t)}


def render_additional_context(row):
    """The prose-conclusion fields, one labelled line each, in _CC_KEEP order.

    `subcellular_location` joins the comments here rather than keeping its own
    line. It is the same kind of thing -- a conclusion drawn from published
    work rather than a reading of the chain -- so grouping them lets ONE template
    rule govern all of them instead of one rule per field.
    """
    blocks = {}
    for c in (row.get("comments") or []):
        topic = (c.get("topic") or "").strip()
        text = c.get("text") or ""
        if topic and text:
            blocks.setdefault(topic, []).append(text)
    out = []
    for topic, label, extract in _CC_KEEP:
        if topic == "__LOCATION__":
            loc = render_location(row.get("subcellular_location"))
            if loc and loc != ABSENT:
                loc = " ".join(strip_refs(loc).split())
                if loc:
                    out.append(f"- {label}: {loc}")
            continue
        fn = _CC_EXTRACT.get(extract, strip_refs)
        for text in blocks.get(topic, []):
            rendered = " ".join(fn(text).split())
            if rendered:
                out.append(f"- {label}: {rendered}")
    return "\n".join(out) if out else ABSENT


def _feature_tier(f):
    """Cap priority. Lower survives."""
    src, typ, db = f.get("src"), f.get("type"), f.get("member_db")
    if src == "uniprot" and typ in _F_CHEM:
        return 1
    if src == "uniprot" and typ in _F_TOPO:
        return 2
    if src == "interpro_site" and db in ("pirsr", "sfld"):
        return 3
    if src == "uniprot" and typ in _F_REGN:
        return 4
    if src == "interpro_site":
        return 5
    if src == "interpro_match":
        return 6
    if typ in _F_BOOK:
        return 8
    return 7


def _feature_rows(feats):
    """Feature records -> bullet rows, grouping repeated single residues.

    Only `_F_DROP` and `_F_SKIP` are filtered. Everything else reaches a row,
    including matches that restate an InterPro entry, because the block is the
    complete record of what is annotated on the chain and it is not its job to be
    minimal.

    The one transform is grouping. A dimer interface is ONE annotation written
    once per residue, not N facts, so records sharing (type, desc, member_db)
    collapse to a single row -- a span when the positions are consecutive, which
    also spells the motif out of the residue letters, and the full comma list
    when they are scattered. That merges 37.6% of records.
    """
    points, rows = {}, []
    for f in (feats or []):
        if f.get("type") in _F_DROP or f.get("type") in _F_SKIP:
            continue
        desc = f.get("desc") or (f.get("member_acc")
                                 if f.get("member_db") in _F_LABEL_DB else None)
        if f.get("start") == f.get("end") and not f.get("end_unknown"):
            points.setdefault((f.get("type"), desc, f.get("member_db")), []).append(f)
            continue
        text = desc or ""
        if f.get("pair"):
            text = (text + " (pair)").strip()
        rows.append(dict(start=f["start"],
                         end="?" if f.get("end_unknown") else f.get("end"),
                         type=f.get("type"), residue=f.get("residue"),
                         desc=text or None, acc=f.get("acc"),
                         tier=_feature_tier(f), at=f["start"], point=False,
                         span=(f.get("end") or f["start"]) - f["start"]))
    for (typ, desc, _db), group in points.items():
        group.sort(key=lambda x: x["start"])
        text = desc or ""
        if any(x.get("pair") for x in group):
            text = (text + " (pair)").strip()
        # The group is keyed on (type, desc, member_db) and so spans `src`, and
        # two sources routinely annotate the SAME residue -- so a position can
        # appear many times. Left in, the bullet reads
        # `74,74,74,... [residue: KKKKKKKKKKKKKKKD]`: 16 letters for 4 residues,
        # and the trace template licenses quoting a motif exactly WHEN the
        # features list carries it, so we would be supplying a fabricated one.
        # Dedupe on position, keep the first, and the residue string then has
        # one letter per residue. It also repairs the consecutive-run test
        # below, which a repeated position could only ever fail.
        # `pair`, `tier` and `acc` are read from the FULL group above and below,
        # so deduping cannot drop a pair flag or promote a tier.
        seen_pos, unique = set(), []
        for item in group:
            if item["start"] in seen_pos:
                continue
            seen_pos.add(item["start"])
            unique.append(item)
        pos = [x["start"] for x in unique]
        if len(pos) > 1 and pos == list(range(pos[0], pos[-1] + 1)):
            start, end, span = pos[0], pos[-1], pos[-1] - pos[0]
        elif len(pos) == 1:
            start, end, span = pos[0], None, 0
        else:
            start, end, span = ",".join(str(p) for p in pos), None, 0
        rows.append(dict(start=start, end=end, type=typ,
                         residue="".join(x.get("residue") or "" for x in unique) or None,
                         desc=text or None, acc=group[0].get("acc"),
                         # min, not group[0]: the group is keyed on
                         # (type, desc, member_db) and so spans `src` values,
                         # and taking the first member's tier would make cap
                         # priority depend on input order.
                         tier=min(_feature_tier(x) for x in group),
                         at=pos[0], point=True, span=span))
    return rows


def _dedupe_rows(rows):
    """Collapse rows that render to the same bullet.

    `member_db` is not printed, so one site matched by several signatures --
    PIRSR, CDD and SFLD all calling residues 233-241 an ATP-binding P-loop --
    would reach the reader as the same sentence over and over: 76,337 duplicate
    bullets on 13.89% of rows, the worst one line repeated 23 times. They only
    become indistinguishable once the fields that separate them are dropped at
    render time, so this is the right boundary to fix it at.

    The survivor keeps the LOWEST tier of the group, so a chemistry row is not
    evicted by the cap because a member-DB copy of it was the one retained.
    First-seen order is preserved; the caller sorts afterwards anyway.
    """
    best = {}
    for r in rows:
        line = _feature_line(r)
        if line not in best or r["tier"] < best[line]["tier"]:
            best[line] = r
    return list(best.values())


def _collapse_near_duplicate_spans(rows):
    """One entry matched several times at nearly the same place is one finding.

    Member databases disagree by a residue or two about where an entry sits, so
    IPR001916 arrives as [19-147] [19-146] [19-145] -- three bullets saying the
    same thing, the redundancy the trace's own output format forbids. Measured
    over all 133,114 train rows: 187,970 removable rows, 14.1% of interpro rows,
    on 54.9% of proteins.

    The rule is deliberately the simplest thing that works: both ends within
    _SPAN_FUZZ residues of the cluster's first span. Nothing else -- no overlap
    ratio, no containment test, no representative flag.

    It is tuned for RECALL of real domains, not precision of collapsing. Missing
    a redundant pair costs one extra bullet; merging two real repeat copies
    deletes a domain from the chain, so every ambiguous case is left alone. A
    PROSITE motif nested in a family span ([21-31] inside [5-146]) and two
    adjacent zinc fingers ([130-157] and [158-185]) are both far outside the
    fuzz and are never touched.

    Only rows carrying an InterPro accession are grouped, so two UniProt features
    that happen to sit at the same place are never merged. The survivor spans the
    union of its cluster and inherits the lowest tier in it, so no residue that
    had a domain call loses it and a cap decision is never made worse by which
    copy happened to win.
    """
    grouped, out = {}, []
    for r in rows:
        if (not r.get("acc") or r.get("point")
                or not isinstance(r.get("start"), int)
                or not isinstance(r.get("end"), int)):
            out.append(r)
            continue
        grouped.setdefault((r["acc"], r.get("type"), r.get("desc")), []).append(r)
    for group in grouped.values():
        # Longest first, ties by start, so the result cannot depend on input order.
        group.sort(key=lambda r: (-(r["end"] - r["start"]), r["start"]))
        clusters = []
        for r in group:
            for c in clusters:
                seed = c[0]
                if (abs(r["start"] - seed["start"]) <= _SPAN_FUZZ
                        and abs(r["end"] - seed["end"]) <= _SPAN_FUZZ):
                    c.append(r)
                    break
            else:
                clusters.append([r])
        for c in clusters:
            # The survivor spans the UNION of its cluster, not its longest member:
            # the union is the honest extent of "every signature that agrees found
            # it here". Membership is tested against the cluster SEED, never the
            # growing union, so a chain of slightly-shifted spans cannot walk the
            # cluster across the chain and swallow a neighbouring repeat copy.
            survivor = c[0]
            survivor["start"] = min(x["start"] for x in c)
            survivor["end"] = max(x["end"] for x in c)
            survivor["at"] = survivor["start"]
            survivor["span"] = survivor["end"] - survivor["start"]
            survivor["tier"] = min(x["tier"] for x in c)
            out.append(survivor)
    return out


def _feature_line(r):
    """`- 74-79 (SITE): description [residue: MQT] [IPR003265]`"""
    end = r.get("end")
    # A malformed record with end < start would otherwise render "7-1".
    if isinstance(end, int) and isinstance(r["start"], int) and end < r["start"]:
        end = None
    span = str(r["start"]) if end in (None, "", r["start"]) else f"{r['start']}-{end}"
    out = f"- {span} ({str(r['type']).upper()})"
    if r.get("desc"):
        out += f": {r['desc']}"
    if r.get("residue"):
        out += f" [residue: {r['residue']}]"
    if r.get("acc"):
        out += f" [{r['acc']}]"
    return out


def render_features(feats, cap=FEATURE_CAP):
    """The residue-feature block: one bullet per feature, ordered N to C.

    Coordinates are 1-based inclusive on this protein's own sequence, so a span
    here is a fact about the chain the model is reading rather than something it
    has to count out.

    Order is total and deterministic, so the same input always renders
    identically:

        1. start ascending                 reads N to C
        2. named residues before regions   a residue-level record is a fact
                                           about that position, not a region
                                           the ones after it nest inside, so it
                                           does not join the nesting at all
        3. longer span first               among regions, the containing one
                                           before what sits inside it
        4. tier ascending                  ACT_SITE 265 before site 265
        5. type, then desc, alphabetically breaks the remaining ties

    Above the cap, tier decides membership and position still decides order, so
    a truncated block still reads down the chain. What was dropped is stated,
    because a silently truncated block reads as a complete one.

    Args:
        feats: the row's `features` records.
        cap: the most bullets to render.

    Returns:
        The block, or ABSENT when nothing is annotated.
    """
    rows = _collapse_near_duplicate_spans(_dedupe_rows(_feature_rows(feats)))
    if not rows:
        return ABSENT
    dropped = 0
    if len(rows) > cap:
        dropped = len(rows) - cap
        rows = sorted(rows, key=lambda r: (r["tier"], r["at"]))[:cap]
    rows.sort(key=lambda r: (r["at"], 0 if r["point"] else 1, -r["span"],
                             r["tier"], str(r["type"]),
                             str(r.get("desc") or "")))
    out = [_feature_line(r) for r in rows]
    if dropped:
        out.append(f"- ... {dropped} further features omitted, "
                   f"lowest-priority first")
    return "\n".join(out)


def render_ppi(s):
    """Deduplicate partner names, preserving first-seen order."""
    s = _blank(s)
    if not s:
        return ABSENT
    seen = []
    for line in s.split("\n"):
        line = line.rstrip()
        if line.strip() and line not in seen:
            seen.append(line)
    return "\n".join(seen)


def _eco_words(m):
    """The evidence classes of one `{ECO:...}` blob, in words."""
    out = []
    for code in re.findall(r"ECO:(\d+)", m.group(0)):
        word = ECO_CLASS.get(code, f"ECO:{code}")
        if word not in out:
            out.append(word)
    return f" ({', '.join(out)})" if out else ""


def render_location(s):
    """Replace each {ECO:...} blob with its evidence class in words."""
    s = _blank(s)
    if not s:
        return ABSENT
    return _ECO_RE.sub(_eco_words, s).strip()


def render_go(s):
    """Mark up a leaf block so it sits under its aspect key, each term as `name GO:id`.

    Each leaf is marked `  - ` under an aspect line `- Molecular function:`, which is
    exactly the Final Answer's own shape, so the input, the derivation and the output
    all say hierarchy in the same alphabet. The `GO:id name` lines are flipped here
    rather than in the store because the corpus stores identifier-first.
    """
    lead = "  - "
    s = _blank(s)
    if not s:
        return lead + ABSENT
    out = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _RE_GO_ID_FIRST.match(line)
        if m:
            line = f"{m.group(2)} {m.group(1)}"
        out.append(lead + line)
    return "\n".join(out)


def merge_go_leaves(*blocks, dag=None):
    """One LEAF list from several `GO:id name` blocks, first spelling wins.

    Deduplicated on the resolved identifier, not the line: the same term can
    arrive from two tiers under two spellings or under an `alt_id`, and a
    line-level dedupe would keep both, reading as two findings.

    Then re-filtered to leaves. Each tier's column is leaf-clean on its own, but
    their union is not: 24,342 terms (2.82%) end up an ancestor of another term in
    the same aspect, on 13.5% of proteins. Every one of those parents comes from
    tier 1 and every child from tier 2:

        tier1 methyltransferase activity -> tier2 uroporphyrin-III C-methyltransferase activity
        tier1 plasma membrane            -> tier2 basolateral plasma membrane

    Dropping the parent costs nothing at scoring time, since `cafa_eval`
    propagates predictions up the DAG and the child credits every ancestor.

    Order is tier order, so what the protein is experimentally known to do comes
    first.
    """
    dag = dag if isinstance(dag, MiniGoDag) else default_dag()
    seen, out = set(), []
    for block in blocks:
        for line in _blank(block).split("\n"):
            line = line.strip()
            if not line:
                continue
            m = _RE_GO_ID_FIRST.match(line)
            key = dag.resolve(m.group(1)) if m else line
            if key in seen:
                continue
            seen.add(key)
            out.append((key if m else None, line))
    ids = {g for g, _ in out if g}
    return "\n".join(
        line for g, line in out
        if not (g and any(g in dag.ancestors(o) for o in ids if o != g)))


def prepare_fields(row, go_dag=None, sequence=None):
    """Row -> every placeholder of the trace templates (`synth/v5/`, `synth/v6/`) but `examples`.

    The GO blocks are tier 1 and tier 2 merged into one leaf list per aspect.
    Subcellular location is folded into Additional Context, and InterPro entries
    reach the prompt only through the residue-feature block, which carries every
    accession, name and span of them.

    Args:
        row: one corpus row.
        go_dag: a MiniGoDag, a goatools DAG, or None (then the row must carry
            `go_*_leaf`).
        sequence: overrides the row's `sequence`.

    Returns:
        Placeholder name -> rendered text.
    """
    # MiniGoDag has its own leaf filter; goatools goes through _leaf_fields.
    if isinstance(go_dag, MiniGoDag):
        leaf = {}
        for asp in ("mf", "bp", "cc"):
            key = f"go_{asp}_leaf"
            have = _blank(row.get(key))
            leaf[key] = have or go_leaf_block(row.get(f"go_{asp}") or [], go_dag)
    else:
        leaf = _leaf_fields(row, go_dag)
    for asp in ("mf", "bp", "cc"):
        key = f"go_{asp}_leaf"
        leaf[key] = merge_go_leaves(leaf[key], row.get(f"go_{asp}_tier2_leaf"),
                                    dag=go_dag)
    seq = _blank(sequence if sequence is not None else row.get("sequence"))
    fields = {
        "protein_names": _blank(row.get("protein_names")) or ABSENT,
        "protein_id": _blank(row.get("protein_id")) or ABSENT,
        "organism": _blank(row.get("organism")) or ABSENT,
        "protein_function": _blank(row.get("protein_function")) or ABSENT,
        "sequence_numbered": render_sequence_numbered(seq),
        "ppi_formatted": render_ppi(row.get("ppi_formatted")),
        "go_mf_leaf": render_go(leaf["go_mf_leaf"]),
        "go_bp_leaf": render_go(leaf["go_bp_leaf"]),
        "go_cc_leaf": render_go(leaf["go_cc_leaf"]),
        "additional_context": render_additional_context(row),
        "features_formatted": render_features(row.get("features")),
    }
    # LAST, after every field exists, and over every prose field rather than only
    # the ones known to carry references, so a data refresh cannot leak citations
    # silently: MUTAGEN descriptions carry `(PubMed:...)` and `(Ref. N)`, e.g.
    # A5JTM5 90: `H->Q: Complete loss of catalytic activity (PubMed:8718880)`.
    for key in ("protein_names", "organism", "protein_function",
                "ppi_formatted", "features_formatted"):
        if fields[key] and fields[key] != ABSENT:
            fields[key] = strip_refs_block(fields[key]) or ABSENT
    return fields


class MiniGoDag:
    """id -> name, and id -> immediate is_a + part_of parents, from go-basic.obo."""

    def __init__(self, obo_path):
        """Read the ontology.

        Args:
            obo_path: go-basic.obo.
        """
        self.name, self.parents, self.alt, self.ns = {}, {}, {}, {}
        cur = None
        with open(obo_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                # ANY stanza header closes the previous one, not just [Term].
                # go-basic.obo ends with a run of [Typedef] blocks whose `id:`
                # is not a GO identifier, so a reader that only reset on [Term]
                # kept writing into the LAST term parsed.
                if line.startswith("["):
                    cur = None
                elif line.startswith("id: GO:"):
                    cur = line[4:]
                    self.parents.setdefault(cur, set())
                elif cur is None:
                    continue
                elif line.startswith("name: "):
                    self.name[cur] = line[6:]
                elif line.startswith("namespace: "):
                    self.ns[cur] = line[11:]
                elif line.startswith("alt_id: GO:"):
                    self.alt[line[8:]] = cur
                elif line.startswith("is_a: GO:"):
                    self.parents[cur].add(line[6:16])
                elif line.startswith("relationship: part_of GO:"):
                    self.parents[cur].add(line[22:32])
                elif line == "is_obsolete: true":
                    self.name.pop(cur, None)
                    self.parents.pop(cur, None)
        self._anc = {}

    def __contains__(self, gid):
        return gid in self.parents

    def resolve(self, gid):
        """The primary id of `gid`, following alt_ids."""
        return gid if gid in self.parents else self.alt.get(gid, gid)

    def ancestors(self, gid):
        """Every is_a / part_of ancestor of `gid`, cached."""
        hit = self._anc.get(gid)
        if hit is not None:
            return hit
        out, stack = set(), list(self.parents.get(gid, ()))
        while stack:
            p = stack.pop()
            if p not in out and p in self.parents:
                out.add(p)
                stack.extend(self.parents[p])
        self._anc[gid] = out
        return out


def leaf_terms(go_ids, dag):
    """Drop any term that is an ancestor of another in the same list (is_a u part_of).

    Mirrors bioreason2.dataset.utils.filter_go_terms_to_leaf_terms: in-order
    dedup, unknown ids dropped, input order preserved.
    """
    seen, terms = set(), []
    for gid in go_ids or ():
        gid = dag.resolve(gid)
        if gid in dag and gid not in seen:
            seen.add(gid)
            terms.append(gid)
    if len(terms) <= 1:
        return terms
    ancestral = set()
    for gid in terms:
        ancestral |= dag.ancestors(gid)
    return [g for g in terms if g not in ancestral]


def go_leaf_block(go_ids, dag):
    """`GO:id name` lines for the leaves of `go_ids`."""
    return "\n".join(f"{g} {dag.name.get(g, '(not found in GO DAG)')}"
                     for g in leaf_terms(go_ids, dag))
