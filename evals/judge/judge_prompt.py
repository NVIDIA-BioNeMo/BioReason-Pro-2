# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The eval judge's prompt: the evidence the teacher had, the sequence included.

This module is data: the prompt, the verdict schema and the axes. `llm_judge.py` runs it against an
OpenAI-compatible endpoint; any other client can render `PROMPT_TEMPLATE` and send `SCHEMA` itself.

The judge is shown every input the teacher had when it wrote the corpus traces: the UniProt
accession, the numbered sequence (one residue per line with its position, rendered by the
teacher's own `render_sequence_numbered`), the residue features, the additional context, the
interaction partners and the GO terms. It scores five axes: molecular function, biological
process, cellular component, specificity and depth, reliability and grounding.

The sequence is what lets Axis 5 check residue claims. Without it a judge cannot tell a correct
residue claim ("Cys123 is the catalytic nucleophile", where residue 123 is a Cys) from an invented
one, so it flags correct claims as invented: 85% of the SFT residue claims a sequence-blind judge
called invented are correct by the sequence. A model that deletes its residue claims, right or
wrong, would then gain reliability.

  * Axis 5: a residue's IDENTITY is checked against the sequence. A claim the sequence confirms is
    supported even when no feature names it; one it contradicts (another amino acid at that
    position, or a position past the end) is invented. A ROLE at a confirmed residue (active
    site, binding) is still checked against the Residue Features, and is a licensed inference
    when a domain or motif in the evidence implies it.

Scores from a different prompt or judge model are not comparable with the published ones.
"""

AXES = ("mf", "bp", "cc", "specificity", "reliability")

PROMPT_TEMPLATE = """\
ROLE: You are an expert Biocurator and Senior Evaluator for UniProt. You are judging
whether an AI model has correctly described, IN WORDS, what a protein does.

HOW TO DO THIS -- two steps, in order

STEP 1. Before you look at the model's output, read the Composite Ground Truth and
write for yourself a short statement of what this protein does: its molecular
activity, the process it serves, and where in the cell it acts. Use every source.
The UniProt Summary, when there is one, is that statement already. When there is
not, build it from the Residue Features (domains, active sites, topology), the
Additional Context, the GO term names and the interaction partners -- a GO term is
a phrase of biology
("methionine synthase activity", "mitochondrial inner membrane"), and you should
read it as biology, exactly as you would read a sentence from the summary.

STEP 2. Compare the model's description against your statement, in words, and score
the axes below.

WHAT YOU ARE NOT DOING
You are not checking which of the supplied GO terms the model reproduced. Identifier
accuracy is measured separately and completely. If you catch yourself matching the
model's term list against the supplied term list, stop -- you are answering the
wrong question. Two descriptions of the same enzyme that pick different levels of
the ontology are the same answer. A correct identifier attached to a wrong
description is not credit, and the right biology stated in plain prose with no
identifier at all is full credit.

The supplied GO terms are what a curator happened to record, not a specification of
the protein. Read them for what they say; do not treat the edge of the list as the
edge of the protein's function.

The -1 Rule: return -1 for an axis only if, after STEP 1, you could write nothing at
all about that aspect -- no summary, no informative domain, no GO term, no location,
no partner speaks to it.
- If GO_MF is empty but the Residue Features show a kinase domain, or the summary
  or Additional Context implies an activity, YOU MUST SCORE IT.
- If nothing anywhere speaks to localisation, CC is -1.

SCORING (1-10)

AXIS 1: Molecular Function -- does the activity the model describes match the
activity in your STEP 1 statement?
- 1: Contradicts it (your statement says kinase, the model says phosphatase).
- 5: Right superfamily or general class, missing the specific activity your
     statement carries.
- 10: The specific activity, described in words that mean the same thing as your
     statement -- whatever vocabulary or ontology level the model chose.

AXIS 2: Biological Process -- does the process the model describes match yours?
- 1: Wrong pathway.
- 5: Broadly correct ("a metabolic process").
- 10: The specific process, and how this protein contributes to it.

AXIS 3: Cellular Component -- does the localisation the model describes match yours?
- 1: A location your statement rules out.
- 5: Right general compartment ("membrane").
- 10: The specific sub-compartment or topology.

AXIS 4: Specificity & Depth -- how much useful detail is in the model's prose?
- 1 (Generic): "It is a protein."
- 5 (Average): "It is a kinase."
- 10 (Deep): "A kinase that phosphorylates [substrate] via the [domain] domain,
     regulated by [mechanism]." Rate this high for detail even where a specific
     guess is slightly wrong -- correctness belongs to Axes 1-3, depth belongs here.

AXIS 5: Reliability & Grounding -- is the model's confidence earned by the evidence?

Separate three things. Only the third is a failure.

- SUPPORTED: carried by any evidence source -- the Summary, Additional Context, the
  Residue Features block, the GO terms, the partners, or the organism/length line.
  ANYTHING POSITIONAL OR STRUCTURAL IS CHECKED AGAINST THE RESIDUE FEATURES BLOCK
  AND NOWHERE ELSE: residue numbers, spans, disulfide pairings, signal-peptide and
  transmembrane boundaries, domain extents, InterPro matches and their coordinates.
  If a claim matches that block it IS supported, even where the Summary and the GO
  terms are silent -- those are prose and annotation summaries and are not expected
  to restate coordinates.
  THE ONE EXCEPTION IS A RESIDUE'S IDENTITY, which is checked against the numbered
  Sequence: "Cys123", "a histidine at position 57", "the conserved Gly-X-Gly at
  15-17". Look the position up. If the Sequence has that amino acid there, the
  identity is SUPPORTED even when no feature names the residue.
- LICENSED INFERENCE: follows from supported evidence by ordinary domain reasoning
  -- DNA binding from a winged-helix match, a mechanism from an ACT_SITE triad,
  secretion from a SIGNAL peptide, an obligate dimer from a dimerisation domain, a
  catalytic or binding role for a residue the Sequence confirms inside a domain or
  motif that implies it. Models are expected to reason. THIS IS NOT A RELIABILITY
  FAILURE AND NOT AN UNSUPPORTED CLAIM.
- INVENTED: contradicts the evidence, or asserts a specific structural fact the
  evidence does not carry -- a residue the Sequence contradicts (another amino acid
  at that position, or a position past the end), a span or domain absent from the
  Residue Features block, a partner absent from the PPI list, a localisation the
  evidence excludes.

- 1 (Hallucination/Contradiction): explicit contradiction, or fabricated structural
     features -- residues the Sequence contradicts, or spans or domains not in the
     Residue Features block.
- 5 (Unsubstantiated Speculation): specific claims disconnected from the evidence
     and not reachable by domain reasoning (a disease link with no pathway or
     partner behind it).
- 10 (Plausible/Grounded): every key claim is supported or is a licensed inference.
     Reasoning beyond the ground truth is expected and is NOT penalised when it
     follows from the input.

{search_policy}

READING THE MODEL GENERATION
The generation may contain a <reasoning> block: the model's reasoning, followed by its
final answer. Judge BOTH. Axes 4 and 5 are largely about the reasoning -- that is
where depth and fabrication live. Do not penalise the model for the presence of the
reasoning block, its length, or its format.

Judge the prose. If the model states the right biology in a sentence and never
writes a single GO identifier, Axes 1-3 should be high.

The final answer may separate "Confident GO Terms" from "Speculated GO Terms". That
split is a calibration signal and a good one: do not penalise a model for placing an
uncertain term under Speculated, and do not reward one for asserting everything as
Confident.

OUTPUT
Return JSON only, matching the supplied schema. For each axis give a score and a
rationale of at most 280 characters. **Quote or paraphrase the model's own words**
in the rationale -- the phrase you scored -- rather than citing which GO terms
matched. `unsupported_claims` lists ONLY assertions you judge INVENTED under Axis 5 --
contradicted by the evidence (the Sequence included), or asserting structural
specifics the Residue Features block does not carry. A residue identity the Sequence
confirms is never listed. DO NOT list licensed inferences there. It is the evidence
behind your Axis 5 score and must be consistent with it: a 10 has none, a 9 has at
most one borderline item.

COMPOSITE GROUND TRUTH
- Protein Name: {protein_names}
- UniProt Accession: {protein_id}
- Organism: {organism}
- Length: {length}
- UniProt Summary: {protein_function}
- Additional Context: {additional_context}
- Residue Features: {features_formatted}
- Sequence (residue number, amino acid):
{sequence_numbered}
- Known Protein Protein Interaction Partners: {ppi_formatted}
- GO (Molecular Function): {go_mf_x}
- GO (Biological Process): {go_bp_x}
- GO (Cellular Component): {go_cc_x}

MODEL GENERATION
{generation}
"""

SEARCH_POLICY = """NO EXTERNAL LOOKUP
You have NO search tool and no way to look anything up. Build your STEP 1 statement
from the Composite Ground Truth plus your own knowledge, and be explicit about the
difference: if a model claim is one you cannot confirm from the supplied evidence,
that is Axis 5's middle ground (unsubstantiated speculation), NOT its floor.
Reserve a 1 for a claim that CONTRADICTS the evidence, or for a structural feature
-- a domain, motif or InterPro entry -- asserted despite not appearing in the Input
at all. Do not guess that something is fabricated merely because you have not heard
of it, and never state or imply that you verified anything externally."""


def _axis_schema(desc):
    """The schema of one axis's verdict.

    Args:
        desc: the axis description shown to the judge.

    Returns:
        A JSON schema for {score, rationale}.
    """
    return {"type": "object", "additionalProperties": False,
            "required": ["score", "rationale"],
            "properties": {
                "score": {"type": "integer", "enum": [-1] + list(range(1, 11)),
                          "description": desc},
                "rationale": {"type": "string"}}}


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": list(AXES) + ["unsupported_claims", "summary"],
    "properties": {
        "mf": _axis_schema("Axis 1 Molecular Function, as described in words"),
        "bp": _axis_schema("Axis 2 Biological Process, as described in words"),
        "cc": _axis_schema("Axis 3 Cellular Component, as described in words"),
        "specificity": _axis_schema("Axis 4 Specificity & Depth"),
        "reliability": _axis_schema("Axis 5 Reliability & Plausibility"),
        "unsupported_claims": {
            "type": "array", "items": {"type": "string"},
            "description": "Model assertions neither in the Ground Truth nor "
                           "confirmed by search. Evidence for the Axis 5 score."},
        "summary": {"type": "string", "description": "<=400 chars, overall verdict"},
    },
}
