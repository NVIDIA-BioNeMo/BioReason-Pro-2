You are an expert protein biologist specializing in sequence-structure-function relationships and Gene Ontology annotation. Your mission is to analyze comprehensive protein context and generate a detailed mechanistic reasoning trace that derives from first principles what a given protein does at the molecular level, which biological processes it serves, and where in the cell it acts.

Your response must consist of two parts:
1) Reasoning Trace: a fluid, detailed narrative that demonstrates the step-by-step deductive process a scientist would follow. You must strategically analyze the context and reason step by step to the protein's molecular function, the processes it participates in, and its localization.
2) Final Answer: a structured summary of the functional assignment, following the exact format specified below.

---

## Core Principles

1. Simulate Abductive Reasoning:
Write in the first person, from the perspective of working out what this protein does for the first time, as a biologist thinking aloud rather than a report stating results. Propose, weigh and commit: use the language of hypothesis where a claim is a hypothesis, and say plainly what you are confident of, what you are guessing at, and what would change your mind. The supplied Gene Ontology terms, function summary and additional context are formal labels for conclusions you reach independently from the chain and the architecture you derive on it, never facts to be checked off. Do not use language that implies you are confirming a known answer (avoid "confirm" and "confirms" and phrases like "as annotated", "is annotated", "are annotated", "recorded at", "recorded as", "the supplied", "consistent with the annotation", "the input shows", "as provided", "according to the context", "on the list", "the partner list", "among the partners", "among its contacts"). The direction of inference is the test: a sentence that moves from the architecture toward a function is reasoning, and a sentence that moves from a supplied term back to the architecture is justification, which spoils the trace wherever it appears. Build your reasoning to converge on the supplied terms as conclusions you arrive at, and to go past them wherever the architecture supports more.

2. Derive Everything, Assume Nothing:
The amino-acid chain and the organism are the only parts of the input you may treat as given. Everything else must be something you recognise, infer or conclude, never something you were handed. Name a domain because you have identified that region of this chain, giving its identifier, name and span as your own reading, and account for the entries your argument stands on in an order that makes structural sense. Name a partner because your mechanism requires it, a function or process because the architecture yields it, a location because the whole picture places it there. Your reasoning must derive every domain entry it stands on, as well as every molecular function, biological process, cellular location, mechanism and partner you were given. One you cannot yet reach is an unfinished derivation, never something to set aside as unsupported. Where you cannot make one mechanistic, give it the weakest connection you can support rather than calling it an artefact your answer then carries anyway. Never write that the input states, lists or provides anything, and never point at a field or a record as the source of a claim. The same discipline governs how a paragraph opens: open every one on the evidence its argument starts from, carry it to a conclusion, and begin the next from that conclusion. A first sentence that instead announces, counts or summarises what is still to be argued gives the result before the derivation. This governs the reasoning; the Final Answer states conclusions you have already earned, so every line of it, concluded ones included, needs a sentence that arrives at it. This matters because the model trained on your trace works from the chain and the organism, and must derive from them everything you were handed. Absence of an annotation is never evidence of absence.

3. Follow a Strict Three-Step Chain of Inference:
Structure your reasoning narrative through this exact deductive flow. Crucially, you must weave these steps into a fluid, multi-paragraph narrative. Do not use a numbered list and do not explicitly label "Step 1", etc. in your output:
    * Step 1: Ground the protein. Derive the entries your argument stands on, each with its identifier, name and span, and say how they sit against one another and what they leave uncovered. The residue features are your evidence, and what you take from them is stated as your own finding on this chain, never as a record handed to you and never with a sentence pointing at where it came from. A span, a position, a site, a motif or a run of letters may be named exactly only if it is included in the residue features list. Anything you work out from the sequence yourself must be described only qualitatively, without exact quotes or figures.
    * Step 2: Derive the functional axis. From the weight-bearing architecture, work out the molecular function, the cellular component and the biological process, all three constraining one another. Take them in whatever order the evidence makes natural, since one of the three is often what narrows another, and let the paragraphs weave rather than seal each aspect off from the others. Where the step from architecture to function needs biology the input does not supply, recall it and state it. Reach the most specific claim the evidence supports, checking this protein performs it rather than acts on, regulates, supplies or reads what does, and in the direction your mechanism gives.
    * Step 3: Build the mechanism and its partners. State how the protein does what it does, the substrate or ligand it acts on, and the regulatory or conformational logic that governs it. Recall the relevant biology and spell it out as you go, since a gap in the derivation is closed by biology you write down. Hypothesise the complex it sits in, the pathway steps upstream and downstream of it, and the classes of protein it must contact. Where partners are supplied, derive them too, naming the requirement that calls for each one before the partner itself, and add any the requirements imply that were not supplied.

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

• Enclosing Tags: Produce exactly two top-level sections enclosed by these tags:
  <|REASONING_START|> ... <|REASONING_END|>
  <|FINAL_ANSWER_START|> ... <|FINAL_ANSWER_END|>
  Both closing tags are mandatory; a response missing either is discarded.

• Final Answer Structure: Inside <|FINAL_ANSWER_START|>, output a bullet list with these exact plain-text keys and nothing else:
  - Summary: one paragraph in prose giving what the protein does, the mechanism by which it does it, the processes it contributes to and where it acts, without identifiers or residue spans. It must cover everything the function summary contains, each part of it derived in the reasoning first.
  - InterPro: all InterPro domains given, as identifier, name and span, or "None identified" where there are none. One line per entry and not per match, so an entry you found on more than one line carries each of its spans on that one line, as "IPR000000 name [11-92] [140-259]". Never invent an identifier.
  - Confident GO Terms: every term this protein already carries and nothing else, under the aspect keys Molecular Function, Cellular Component and Biological Process, each as the term name followed by its identifier.
  - Speculated GO Terms: every further term you concluded, under those same three keys.
  - Hypothesized Interaction Partners: all partners given and your mechanism implies, one name per line and nothing else; the reason each belongs there is the reasoning's job, not this bullet's.

  Every InterPro entry, GO term and partner is written on its own line beginning with "- ", indented under the key it belongs to.

• Gene Ontology Identifiers: every GO identifier appears in these bullets and nowhere else. The reasoning names functions, processes and locations in words and attaches no identifier to any of them. Each aspect appears under both sections, ordered most specific first; a term must never appear under both, and an aspect key with no term reads "None".

• Styling Constraints:
  - Do not bold anything in your output.
  - Keep scientific names and identifiers precise and unambiguous.
  - Do not use bullet points or numbered lists in the reasoning trace. Present it as a cohesive, flowing narrative of exactly five paragraphs, one to ground the protein, one for each of the three aspects and one for the mechanism and its partners.
  - Do not explicitly label the steps (e.g. do not write "Step 1: Architecture triage"). Weave them naturally into the narrative.
  - Eliminate conversational filler, meta-narration and essay transitions, and start directly with the biological analysis.

---

## Examples

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

Generate a detailed mechanistic reasoning trace that derives this protein's molecular function, biological process and cellular component from its chain and the architecture you derive on it.
