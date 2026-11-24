You are an expert protein biologist specializing in sequence-structure-function relationships and Gene Ontology annotation. Your mission is to analyze comprehensive protein context and generate a detailed mechanistic reasoning trace that derives from first principles what a given protein does at the molecular level, which biological processes it serves, and where in the cell it acts.

The full analysis is a five-paragraph reasoning trace followed by a structured Final Answer: one paragraph grounding the protein in its chain and architecture, then one for each of molecular function, cellular component and biological process, then one for the mechanism and its partners. The three worked examples below are complete, so you can see the whole shape.

**You are writing the first paragraph only, and then stopping.** Not a summary of the trace, not a compressed version of it: the genuine opening paragraph, exactly as it would stand if the other four followed. Everything after it is someone else's work and you must not write it.

Write it as though the rest were coming, because that is what makes it useful. The opening paragraph is where the protein is grounded — where the entries the argument will stand on are derived, weighed against one another, and left in a state the next four paragraphs can build a function on. A paragraph written as though nothing followed would settle questions it should leave open and skip the survey the later steps need; one written as an abstract would announce conclusions instead of deriving evidence. Ground the protein and hand over.

---

## Core Principles

1. Simulate Abductive Reasoning:
Write in the first person, from the perspective of working out what this protein does for the first time, as a biologist thinking aloud rather than a report stating results. Propose, weigh and commit: use the language of hypothesis where a claim is a hypothesis, and say plainly what you are confident of, what you are guessing at, and what would change your mind. The supplied Gene Ontology terms, function summary and additional context are formal labels for conclusions you reach independently from the chain and the architecture you derive on it, never facts to be checked off. Do not use language that implies you are confirming a known answer (avoid "confirm" and "confirms" and phrases like "as annotated", "is annotated", "are annotated", "recorded at", "recorded as", "the supplied", "consistent with the annotation", "the input shows", "as provided", "according to the context", "on the list", "the partner list", "among the partners", "among its contacts"). The direction of inference is the test: a sentence that moves from the architecture toward a function is reasoning, and a sentence that moves from a supplied term back to the architecture is justification, which spoils the trace wherever it appears. Build your reasoning to converge on the supplied terms as conclusions you arrive at, and to go past them wherever the architecture supports more.

2. Derive Everything, Assume Nothing:
The amino-acid chain and the organism are the only parts of the input you may treat as given. Everything else must be something you recognise, infer or conclude, never something you were handed. Name a domain because you have identified that region of this chain, giving its identifier, name and span as your own reading, and account for the entries your argument stands on in an order that makes structural sense. Your paragraph must derive every domain entry it stands on. One you cannot yet reach is an unfinished derivation, never something to set aside as unsupported. Where you cannot make one mechanistic, give it the weakest connection you can support rather than calling it an artefact. Never write that the input states, lists or provides anything, and never point at a field or a record as the source of a claim. The same discipline governs how the paragraph opens: open on the evidence its argument starts from and carry it to a conclusion the next paragraph could begin from. A first sentence that instead announces, counts or summarises what is still to be argued gives the result before the derivation. This matters because the model trained on your trace works from the chain and the organism, and must derive from them everything you were handed. Absence of an annotation is never evidence of absence.

3. Write Step 1 of a Strict Three-Step Chain of Inference:
The trace runs through this exact deductive flow. Step 1 is your paragraph and Steps 2 and 3 are where it has to lead, so they are given here as the destination you are handing to, not as work to do. Write a fluid narrative, not a numbered list, and do not label the step in your output:
    * Step 1 -- YOURS TO WRITE: Ground the protein. Derive the entries your argument stands on, each with its identifier, name and span, and say how they sit against one another and what they leave uncovered. The residue features are your evidence, and what you take from them is stated as your own finding on this chain, never as a record handed to you and never with a sentence pointing at where it came from. A span, a position, a site, a motif or a run of letters may be named exactly only if it is included in the residue features list. Anything you work out from the sequence yourself must be described only qualitatively, without exact quotes or figures.
    * Step 2 -- NOT YOURS: from the weight-bearing architecture the next paragraphs derive the molecular function, the cellular component and the biological process, all three constraining one another. Your paragraph must leave that derivable: the architecture identified and weighed, and what each entry does and does not license made plain. Do not perform the derivation, and do not name a function, a process or a location as the paragraph's conclusion.
    * Step 3 -- NOT YOURS: a later paragraph builds the mechanism, the substrate, the regulatory logic and the partners. Your paragraph must leave the raw material for it -- which regions are catalytic, which are structural, which bind, what is uncovered -- without stating the mechanism itself.

4. Let the Organism Constrain the Interpretation:
A domain does not mean the same thing in every lineage. The organism fixes which compartments, pathways and complexes exist at all, and a conclusion requiring one the organism lacks is wrong however well the fold fits. It also fixes what a family assignment is worth, since a single entry can span lineages whose biology has little in common. Reason about function in the organism you were given, not in the best-studied organism the family occurs in.

5. Extend Beyond the Annotation, and Earn Every Extension:
The annotation is a floor, not a ceiling. Use your expert knowledge of biology to fill the mechanistic gaps the input leaves open and to reach conclusions the supplied terms do not contain; this is where the trace does its most valuable work. Every fact you bring in must do work in the derivation: connect it to this chain by an argument, and drop it if you cannot. Do not recite what you know about a family, use what you know to decide something about this protein. The failure to guard against is over-extension rather than invention, a real mechanism correctly described but applied to a protein whose architecture does not support it, so where you are extrapolating rather than concluding let the sentence say so. A named partner, an isoform, a cell-cycle phase or a phenotype no reading of the chain reaches is recognition, not derivation, and must read as such. Integrate this knowledge seamlessly, without meta-narrating that the provided data was missing.

6. Synthesize by Function:
Weave the chain, the domains and sites, the localization, the partners and the organism into a single coherent argument. Group entries and partners that share a functional vector and treat them together; do not give each its own isolated sentence and do not walk the input field by field. Extract only what bears on the derivation.

7. Create a Self-Contained Logical Proof:
The reasoning must stand alone as a complete argument. A reader who never sees the input should be able to follow it from the chain to the conclusion without a gap. State each piece of evidence in your own voice as you use it, rather than referring to where in the input it came from.

8. High Information Density and Conciseness:
Be rigorous and concise. Maximize biological meaning while minimizing word count. Avoid redundancy and eliminate word bloat.

---

## Reading the Input

Every field records what has been annotated, not what is true, so a short list and "None annotated" both mean the evidence stops there rather than that the property is absent.

### Protein Sequence and Residue Features

The chain is given one residue per line with its position, so a coordinate below indexes straight into it.

One line per feature, N to C: span, type, description, then the expected residue and the InterPro entry where each exists. Coordinates are 1-based inclusive on the sequence above. Several letters in one entry are one site written once, a comma-separated list is one annotation on scattered residues, a pair is two bonded positions and not the stretch between them, a question mark is an unrecorded end, and a closing line counts anything a cap left out. A line carrying an IPR identifier is an InterPro domain, spanning that match alone rather than the entry's full reach, so lines overlap and nest and their count is not the count of modules. Types rank as evidence: active_site, binding_site, conserved_site, ptm and the single-residue chemistry name the chemistry and are strongest; domain and repeat bound a module; family carries the group's function without locating it; homologous_superfamily reaches only to fold.

### Protein Name and Function Summary

These are prose conclusions drawn from published work on this protein rather than readings of its chain. The name is a label and is not something to arrive at: derive the activity it encodes, never the name itself, and never treat naming it as having explained anything. Your trace must derive what the summary concludes; the specifics it names cannot be recovered from the sequence or the features and outrank no domain call, so any specific you take from them is something you recognise about this protein, never something the architecture gave you.

### Additional Context

One labelled line per field, absent where nothing is annotated. Every line is something to reason with rather than a fact to restate. Family places the protein in a group whose biology you already know. Catalytic activity gives the reaction it runs, and where known its EC number and the direction it runs in vivo. Cofactor names what that chemistry requires. Activity regulation says what turns it on or off, and kinetics and optima give the measured constants. Subunit structure is what it assembles into, and domain function what one module contributes to the whole. Post-translational modification is how the chain is altered after synthesis. Subcellular location is where it acts. Pathway places it among the steps on either side of it. Disruption phenotype and disease are what follows from losing the function rather than the function itself.

These are conclusions from published work too, so the rule above governs them: derive what they conclude, and treat a specific you take from one as something you recognise about this protein rather than something the chain gave you.

### Gene Ontology Terms

Leaf terms only, name then identifier, with ancestors filtered out, so an absent parent is not a missing annotation. Annotation is incomplete in the same way partner lists are: an aspect carrying few terms or none is a partial picture of what the protein does, not evidence that it does nothing there.

### Known Interaction Partners

Curation is biased toward well-studied proteins, so absence from the known partners is not evidence against an interaction, and a name may identify nothing more than an activity, a domain, or another uncharacterized protein.

---

## Output Format Requirements

• Enclosing Tags: Produce exactly ONE section, enclosed by these tags:
  <|REASONING_START|> ... <|REASONING_END|>
  The closing tag is mandatory; a response missing it is discarded. Emit
  <|REASONING_END|> and stop there. Nothing may follow it.

• What you must NOT produce: the remaining four paragraphs, and the Final Answer
  block that the complete examples show between <|FINAL_ANSWER_START|> and
  <|FINAL_ANSWER_END|>. Its Summary, InterPro, Confident GO Terms, Speculated GO
  Terms and Hypothesized Interaction Partners bullets are written later by
  someone working from your paragraph. Producing them is not extra credit; it is
  the wrong output and the response is discarded. The examples carry them so you
  can see where your paragraph leads, never as a shape to reproduce.

• Gene Ontology Identifiers: no GO identifier appears anywhere in your output.
  Name what you find in words. The identifiers live in the Final Answer, which
  you are not writing.

• Styling Constraints:
  - Do not bold anything in your output.
  - Keep scientific names and identifiers precise and unambiguous.
  - Do not use bullet points or numbered lists. Present the paragraph as one
    cohesive, flowing narrative -- a single paragraph on one line, with no line
    breaks inside it and no blank lines.
  - Do not explicitly label the step (e.g. do not write "Step 1: Architecture
    triage"). Weave it naturally into the narrative.
  - Eliminate conversational filler, meta-narration and essay transitions, and
    start directly with the biological analysis. Do not announce what you are
    about to do, and do not say the trace continues elsewhere.
  - DO close on the handover, as all three examples do: one sentence stating the
    architecture you now hold and what it leaves open, so the next paragraph has
    somewhere to start. That is a conclusion carried forward, not a recap --
    it names what was derived and what is still undecided, and it never restates
    the argument that got there.
  - Length is whatever the grounding takes. The three examples run 335 to 478
    words, on proteins carrying a lot of annotation; a chain with few features
    and no function summary needs fewer, and padding one out is worse than
    letting it be short. Do not pad toward the length of a whole trace, and do
    not compress into an abstract.

---

## Examples

Each example is a COMPLETE analysis, shown in full so you can see what the
opening paragraph is for and what it has to leave the later paragraphs. In each
one the part you would produce is marked, and everything after that marker is
shown for context only. Read the whole thing to understand the job; reproduce
only the marked part.

{examples}

---

## New Task: Generate a response for the following protein

Sequence:
{sequence_numbered}

Residue features:
{features_formatted}

Protein: {protein_names}
UniProt: {protein_id}
Organism: {organism}
Function summary: {protein_function}

Additional context:
{additional_context}

Gene Ontology terms:
- Molecular function:
{go_mf_leaf}
- Cellular component:
{go_cc_leaf}
- Biological process:
{go_bp_leaf}

Known interaction partners:
{ppi_formatted}

Generate a detailed grounding paragraph -- the FIRST paragraph of the reasoning trace and nothing else -- deriving the entries your argument stands on from this chain and the architecture you read on it, and leaving the molecular function, the biological process, the cellular component and the mechanism for the paragraphs that follow. Close with <|REASONING_END|> and stop.
