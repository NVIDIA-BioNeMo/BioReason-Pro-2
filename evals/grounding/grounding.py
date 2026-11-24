#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Score the four audited grounding metrics the way a careful reader of each generation would: cited residue accuracy,
feature presence, span placement and InterPro precision. Builds on the claim patterns of patterns.py, with the fixes
of a claim-by-claim audit of the published models' generations and of the rounds that followed the verifiers'
re-readings. Every fix applies identically to every model and context; none was tuned on a model's score. It
reproduces the published grounding values exactly.

    export JUDGE_MODEL=<reader model at your endpoint>     # both steps; JUDGE_BASE_URL and JUDGE_API_KEY for `read`
    python evals/grounding/grounding.py read <results> --name <run> --cache-dir <cache> --workers 192  # 1. readings
    python evals/grounding/grounding.py score <results> --name <run> --cache-dir <cache> --out <dir>   # 2. score
    python evals/grounding/grounding.py read <results> --print-prompt --limit 1

<results> is an eval.py results directory (one <protein_id>_all_k00.json per protein) or a .jsonl(.gz) export of one,
on the v2 test set. The four metrics, each pooled over the run's proteins:

- cited residue accuracy: residue-identity claims about this protein's own residues that are right at the exact
  position, over all such claims; below MIN_RESIDUE_CLAIMS (100) claims it is printed but not reported;
- InterPro precision: named accessions that are the protein's entries, an ancestor of one, or a retired entry whose
  member signature still matches, over every named accession; proteins with no entry count, degenerate accession runs
  do not;
- feature presence F1: per type 2 TP / (said + true), averaged over F1_TYPES (transmembrane, signal peptide, coiled
  coil), "said" read by the LLM reader of record (`read`, from its cache only) and, beside it, by this module's regex
  reader; active and binding sites are reported per type only, binding as recall;
- span placement: the share of claimed spans within WITHIN (5) residues of the nearest true feature of their type, with
  the InterProScan site records in the truth and without them.

`score` writes <out>/<name>.csv.gz (per protein: rip_n, rip_hit, ipr_said, ipr_correct, ipr_*_noloop with degenerate
runs scored, presence_read, and said / true / tp per type from the LLM reader, exist_*, and from the regex reader,
regex_exist_*, blank where a record is unusable or has no reading), <out>/<name>.spans.csv.gz (one row per claimed span:
distance and within5 with the site records and without, blank where the protein has no true feature of the type) and
<out>/<name>.summary.json (every printed value with a 95% percentile bootstrap interval over proteins).

The LLM reader (`read`) decides, for one generation, which of the five tool-annotated feature types the text asserts
for the protein it is about, with a short evidence quote per type. It sees only the generation (reasoning and answer),
never the prompt's feature list or the truth, so the claim side is read blind and identically for every model and
context. One call per generation, strict JSON, temperature 0, cached per (prompt version, effort, run name, protein)
and resumable. Prompt p3 was tuned on 450 hand-labelled decisions, frozen, and tested once on 675 held-out ones: 0.950
agreement, against 0.815 for the regex reader. The reader is any OpenAI-compatible /chat/completions endpoint, set in
the environment as for the LLM judge (evals/judge/llm_judge.py): JUDGE_BASE_URL (ending in /v1), JUDGE_API_KEY and
JUDGE_MODEL. The published values were read by Nemotron 3.5 Super VL at effort max; readings from another model are
not comparable. The key is never printed.

As a library:

    from evals.grounding import grounding
    reference = grounding.load_reference()
    truth = grounding.load_truth(reference)                # the v2 test split, TEST_SET by default
    grounding.residue_claims(record, truth[pid])           # residue-identity claims with verdicts
    grounding.presence(record, truth[pid])                 # five tool-annotated feature types, said vs true
    grounding.spans(record, truth[pid])                    # claimed spans, distance to the nearest true feature
    grounding.interpro(record, truth[pid], reference)      # named InterPro accessions with verdicts

A record is one eval.py result (protein_id, generated_response, protein_sequence, sequence_length, input_prompt);
load_records reads a results directory or a JSON-lines export of one. Needs pyarrow, numpy and
grounding_reference.json beside this file, fetched once from UniProt and InterPro so that scoring never touches the
network: the domain of life of every v2 test protein's organism, and the retired InterPro accessions named by the
generations the audit read (an accession retired before InterPro 109.0 that none of them named gets no retired-entry
credit).

Fixes, by metric:

1. Cited residue accuracy (high-tier residue-identity claims at the exact residue, pooled over claims).
   - A claim about another molecule is not this protein's residue: histone marks, Pol II CTD heptad positions,
     ubiquitin linkages, named substrates and partners ("Phosphorylates CsrA on 'Ser-171'").
   - Locants inside names are not claims: chemical and enzyme names ("aspartate-4-semialdehyde"), zinc-site
     nomenclature ("Cys2-His2"), gene names ("MET10"), interaction-partner list items.
   - A degenerate counting run ("Gly13, Gly14, Gly15, ...", fifteen or more claims of one residue, each within two
     positions of the last) is not a run of citations.
   - The Pol II CTD context is read over the claim's whole paragraph ("Dephosphorylating the CTD reshapes the
     transcription cycle. ..., while Ser-2 dephosphorylation modulates elongation"), and a modified substrate is
     recognised whatever the case of its verb ("Phosphorylation of host STAT1 at Tyr-70").
   - Claims written "<residue> at N" ("the catalytic aspartate at 619") are read, guarded against respective pairing
     ("aspartate, glutamate and asparagine at 241, 244 and 260") and mutant naming ("arginine to lysine at 219").
   - A non-Met residue cited at position 1 that is the first residue of a processed UniProt chain is right in
     mature-chain numbering (proteasome "Thr1", MIF "Pro-1"); at position 1 the precursor is almost always Met.
   - Each claim carries in_prompt, whether its coordinate appears in the prompt, so relay can be split out.
   - Round 5: a counting run may also advance by one constant step ("Serines 100, 104, 108, ..., 344"); an enzyme acting
     on a named substrate ("cleaving fibrinogen at Arg-178 and Met-180", "acetylates enolase at lysine 214"), a bond in
     one ("between Gly-132 and Lys-133 in type I collagen"), a named target ("the target CsrA on Ser-171") and ubiquitin
     named before the lysine ("specific linkages (notably Lys11 and Lys48)") are another molecule's; a residue
     followed by a window ("His43-50") names a stretch.
   - Final round (the round-5 verifier's residue residuals):
     - Names: a token the text introduces as a protein ("a protein called Lys1", then "it binds Lys1"), one joined to a
       gene alias by a slash ("His6/HISN3", "EFG1/Asp3", "CD58/Leu-11") and one followed by a class noun ("the
       catalytically dead Lys49 homologues", "the group I, Asp49 class", "the mycobacterial Asp32 proteases").
     - Metal-site notation: plurals and further nouns ("two Cys4 clusters", "Cys4 knuckles", "a Cys4 GATA finger",
       "a Cys4 Dof finger", "a Cys6 binuclear cluster", "His2-ligated", "a classic Cys4 2Fe-2S cluster", "Zn2-Cys6"),
       and "a 2-His-1-Asp facial triad" as "2-His-1-carboxylate" already was.
     - Ubiquitin: its Gly76 ("the isopeptide bond between ubiquitin Gly76 and a substrate lysine"), "Lys63 linkages",
       "Lys-63 polyubiquitination", "Lys-48-trimming".
     - Another molecule named as such ("cysteine 77 of a beta subunit", "lysine 112 of a partner protein"); a
       homodimer's "partner subunit" or "opposite subunit" stays this protein's.
     - The hyphenated mutant form is guarded like the spaced one ("a proline-to-serine at 3"), and a list headed by two
       plural residue words ("the zinc site at cysteines and histidines 33, 36, 50 and 52") is no claim of either.
     - The substrate rule keeps the protein's own residues: its other copy or itself as the object ("the two kinase
       domains phosphorylate each other on the activation-loop tyrosines 661 and 662", "phosphorylate the enzyme itself
       at histidine 213", "cleaves the chain at glycine 1091"), a residue or segment of it ("phosphorylates the tail
       tyrosines at 1279", "the activation segment at serine 337"), the enzyme acting on it ("phosphorylation by GSK3 at
       serine 215"), a second modification of it ("SUMOylation at lysine 425"), its own substrate pocket ("The arginine
       at 397 in the substrate pocket"), its C1 and C2 domains ("aspartates at 382 and 426 in C1"), and a residue past a
       new clause ("phosphorylates substrates on tyrosine, using the catalytic aspartate at 390").
2. Feature presence on the five tool-annotated types (transmembrane, signal_peptide, coiled_coil, active_site,
   binding_site), per type said vs has any true feature.
   - "rather than" and "instead of" negate within their scope (patterns.py treats both as clause breaks, so they
     never negated); "non-" and "nowhere" negate; "without interruption" does not.
   - Not assertions: mentions in the Hypothesized Interaction Partners section, GO-term names (in either the
     "name GO:id" or the "GO:id name" list style), "transmembrane transport(er activity)", "coiled bodies",
     "heptadiene", the RRM name "Nucleotide-binding alpha-beta plait", the G-protein subunit names ("guanine
     nucleotide-binding protein G"), and a tool's call the same sentence reports and then dismisses ("One predictor
     calls 5 to 23 a transmembrane helix, but that is the hydrophobic core of the signal"; "what the predictor mistakes
     for a membrane anchor"), what the model would need to see ("If this protein had ... a cofactor pocket ..., I would
     reconsider", "a clear substrate pocket would change my mind", "too extensive ... to be a ligand pocket"; a
     condition ends at a main clause, ", and the basic residues ... are the positive-inside signature of an uncleaved
     signal anchor"), and a client GTPase's nucleotide pocket that an exchange or activating factor opens.
   - One definition of a binding site for claims and truth (NOT_LIGAND): a site whose object is a macromolecule or a
     reader of a modified residue is not a ligand site, so DNA-, RNA-, peptide-, protein-, arrestin-, acetyllysine- and
     the IQ motif's EF-hand (calmodulin) binding sites are out of both; "EF-hand calcium-binding site" names calcium.
   - A signal anchor is an uncleaved transmembrane helix (UniProt TRANSMEM), so "signal-anchor" moves from the
     signal-peptide pattern to the transmembrane one.
   - Added phrasings: proton acceptor, nucleophile, covalent and 4-aspartylphosphate intermediates, charge relay
     (active site); metal, cofactor and substrate sites, "ATP-contacting residues" and "binds Fe(II)" forms, a metal
     site named by its coordination or ligands ("zinc coordination", "Cys/His-coordinated", "copper ligands at 551"),
     a ligand's pocket ("a heme b pocket is lined by"), "NAD+ binds at", "NAD(+) binds at", "UDP-alpha-D-glucose
     contacts at" (a ligand's own name; not interdomain, partner, histone or DNA-base contacts), "the glycine-rich ATP
     grip", "heme c attachment sites" (binding site; the metal forms, "calcium ions bind at" among them, and heme
     attachment also for metal binding); "a catalytic quartet" (active site); "a coil at 79-99" (coiled coil);
     "crossings at 113 to 134" and "a single helix crosses at 719 to 739" (transmembrane); "cysteines 99 and 191 are
     paired" (disulfide); "cytosolic loops at" (topological domain). "a homeobox protein" names a family, not a
     DNA-binding domain. Round 4 adds the heads lists are written under: "a carbohydrate-binding set at", "the
     TPP-binding pair at", "substrate-contact residues at", "a glycine-rich cofactor region at", "the GAGGTS nucleotide
     motif at", "magnesium-coordinating residues at", "the metal-chelating pair at", "AMP-binding" (not "AMP-binding
     enzyme"), "iron-sulphur binding" (binding site); "histidines ... bind a single zinc ion" (metal and binding);
     "a nucleophilic elbow at", "a catalytic network at", "catalytic segment" (active site); "a single hydrophobic helix
     at", "Three hydrophobic crossings organize this chain: 38 to 57", "crossings and loops through", "The membrane
     part begins at" (transmembrane); "coil 2 at" (coiled coil); "short extracellular turns at", "a cytoplasmic amino
     end at" (topological domain); "DNA-recognition module" (DNA binding). "a zinc-coordinating fold" with no
     coordinate names no site.
   - Truth: a TMHMM helix lying on a signal peptide (UniProt or SignalP in the organism's mode) with no Phobius helix
     over it is the h-region, not a transmembrane segment; SignalP's eukaryotic mode counts for eukaryotes and their
     viruses and the Gram-positive and Gram-negative modes for prokaryotes and phages; a Phobius helix counts on a
     protein UniProt files as single- or multi-pass; InterProScan's per-residue site records (CDD, PIRSR, SFLD) for
     active sites and chemical-ligand binding join the active- and binding-site truth (sites=False leaves them out);
     PROSITE and Pfam binding-site matches whose object is a protein (the IQ motif) leave it, as their claims do.
   - Round 5, said side:
     - Another molecule's feature is not the protein's: a keyword whose noun is a partner, a class in the plural or a
       membrane machine's part ("transmembrane cargo", "type I transmembrane serine/threonine kinase receptors",
       "the membrane-embedded Vo sector", "coiled-coil proteins"), one a partner or a plural owns, one in a partner's
       name ("Transmembrane protein 231"), one the protein binds or docks onto ("docked onto the transmembrane
       permease"), and one predicated of a partner ("The one contact I have to work with is a small integral membrane
       protein of the ER, SMIM12"); family membership keeps it ("one of the multi-pass membrane proteins").
     - A "membrane anchor" a lipid, an amphipathic helix or a C2 or PH domain makes, or one the sentence calls
       peripheral, is no transmembrane segment.
     - A tool's call the paragraph's next mention of the type rejects ("the predictors split between a cleavable signal
       and a transmembrane anchor ... rather than a transmembrane one"), one identified with the signal's core or
       called a misprediction, and the alternative the model sets aside ("better read as a buried hydrophobic helix
       than as a true transmembrane anchor", "the one reading I reject", "mistaken for", "mimic") are not claims.
     - Binding objects that are proteins, peptides or nucleic acids are out on both sides, also with a token between
       ("histone H3-binding", "calmodulin (CaM)-binding"; E2, PDZ, SH3, ESCRT-III, NLS, "-motif"); a partner's
       nucleotide or substrate pocket the protein inserts into or binds ("binds the Hsp70 nucleotide-binding domain")
       is the partner's.
     - Added phrasings: "coordinate" and "chelate" with any metal, "Zn2+", "binds ferrous iron", ligand verbs and
       subjects ("binds acetyl-CoA", "acetyl-CoA binds first"), "<ligand>-binding cavity/tunnel/surface", "binding
       cavity", "binding positions", "substrate-contact field", "cluster ligands", "assigned to manganese binding";
       "proton-acceptor", "a catalytic set at"; a receiver's aspartylphosphate is a modification, not an active site.
     - Negation ends at a new main clause with a verb of its own (", and the decisive chemistry is") or at the comma
       closing "with no ...", unless the new clause is negated; "argue against", "incapable", "too ...", "wrong ...
       for", "have lost" and "excluding a" negate.
   - Round 5, truth: in a reviewed entry a tool helix on a curated signal peptide is its h-region, and a SignalP call on
     a curated helix with no curated signal is an uncleaved signal anchor; the InterProScan labels the claim side reads
     as active sites ("Active site cavity", "catalytic network", "catalytic core", "stabilizes intermediate") join the
     active-site truth.
   - Final round, truth: a tool helix on the C-terminal propeptide of a curated GPI-anchored protein is its GPI signal;
     a TMHMM helix on a Phobius signal peptide Phobius calls no helix is the h-region, as on a UniProt or SignalP
     signal, but a single-pass membrane protein other than type I that either signal rule would leave with no helix
     keeps it as its uncleaved amino-terminal anchor; a UniProt BINDING record for a macromolecule ("RNA", "ssDNA", "a
     protein", "a peptide") leaves the binding truth by the NOT_LIGAND test the claims and the InterProScan records
     pass; a UniProt SIGNAL ending before residue 10 ("SIGNAL 1-1", "1-3") is the remnant of a signal the fragment cuts
     off.
   - Final round, said side (shared with spans): "not merely metal binding but ..." affirms the type, a denial does not
     reach past ", so" or "and for" into a new statement, and only the capitalised COILS names the coil predictor
     ("resolved more tightly into coils" is no tool call). How presence claims are read is otherwise unchanged;
     the LLM reader replaces that reading.
   - Final round, the reader of record: the LLM reader (prompt p3 at effort max, frozen on a development set of hand
     labels and tested once on a held-out set, 0.950 agreement against 0.815 for this regex reader) decides "said";
     `score` scores it against presence_truth and keeps this module's reading beside it for comparison.
   - Final round, truth completeness (presence only): an enzyme by EC number (not the
     RING-type E3 2.3.2.27) or, with none, by a chemical CATALYTIC ACTIVITY reaction has an active site; a single- or
     multi-pass membrane protein has a transmembrane segment; an entry, domain or region named for a coiled coil, SNARE
     motif, bZIP or leucine zipper has a coiled coil (presence_evidence). Binding sites are reported as recall alone
     (RECALL_ONLY_TYPES): a third of their false positives stay true assertions on proteins no source records. The
     headline presence F1 runs over transmembrane, signal peptide and coiled coil (F1_TYPES).
3. Span placement (share of claimed spans within 5 residues of the nearest true feature of their type).
   - The same hit filter and patterns as presence, so the two metrics read the text alike.
   - A coordinate binds to a keyword only within its line and sentence, and the nearest head wins:
     - another InterPro entry's citation takes its own coordinates, read from its first cited coordinate (digits in its
       name do not count, "IPR049328 (CD39, conserved site, residues 206-210)") through whatever its statement runs on
       to before a new head or clause ("IPR016024, across 111 to 442, leaving 33 to 110"), sought back to the
       sentence's start; the keyword's own accession in apposition, before or after its range ("a copper-binding
       motif at 711 to 724 (IPR049947) with copper ligands at 551", "residues 85-178 form IPR002487 (...), a coiled-coil
       module", "IPR018076 (..., residues 167-176) pinpoints the metal-binding signature"), does not, nor one whose
       name holds the keyword ("IPR001440 Coiled-coil vimentin, conserved site, residues 1568-1584"); an accession
       followed by a sequence phrase ("carries IPR040920 (...), immediately followed by a trio of signal peptide
       annotations") or sitting where the keyword is placed ("sitting inside IPR043441") is not the keyword's own;
     - another feature type's keyword between them takes the coordinate (binding, metal and zinc-finger sites
       excepted), as do a rival noun (disordered, linker, interface, interaction site, activation loop, ...), a
       "while/whereas/but" clause and a nearer keyword of the same type ("short extracellular loops, larger
       cytoplasmic loops at 1-112");
     - named elements of no scored type take what they head (relay loop, SH1 helix, converter, sterile alpha motif,
       the RGD triplet), and a coordinate a helix leaves in the cytosol or on one side of the membrane is no helix;
     - going forward, the phrase that last heads coordinates decides: it heads its own list when it names another type
       and not this one ("a catalytic set at 111" after a magnesium site, "prenylated FMN is contacted at" after one),
       when a domain-level noun or a named element (conserved site, signature, attachment site, scaffold) heads the
       coordinates directly ("and the kinase domain at 200", "and a cell attachment site at 1698"), or when it names a
       residue the feature cannot sit on ("and two asparagines at 38 and 192" after a disulfide), until a later phrase
       names the keyword's type again ("six crossings: S1 at"); parts, generic nouns, quantity words and the type's
       own nouns head nothing ("a hydrophobic h-region at 4 to 15", "with sub-segments at"), nor does a place
       ("between the fifth and sixth crossings", "in the first domain"), nor the keyword's own items after a colon
       ("two tandem helix-turn-helix modules rather than one: a first SANT/Myb domain over roughly 8 to 57");
     - two keywords joined into one head ("active-site and nucleotide-binding residues at") both own its list;
     - a list attached loosely (after a break) goes to a later head of another type it is the subject of ("two
       asparagines at 38 and 192 carry N-linked glycans", "with aspartate 63 to activate water");
     - going backward, a hard boundary (";", ", and", "and the", "while") separates; a keyword that heads
       coordinates of its own takes the list before it only through a labelling verb ("the first eighteen residues
       ... read as a classic cleavable signal peptide: a short basic n-region at 1 to 2"), never by adjacency ("the
       ATP-contacting residues at 569, ..., 971 with magnesium ligands at 569, 571") or after a soft boundary ("an
       ATP-binding loop at 412 to 420, the catalytic lysine at 442"); one that heads none labels the list ("histidine
       303, histidine 349 and aspartate 404, a catalytic triad"), unless it sits in a locative phrase deep in a
       relative clause ("which I read as family-level echoes across the transmembrane modules"); a relative clause
       inside the claim is skipped ("around 304 to 313, where the pocket residues 304, 307 and 308 sit, is the
       oxygen-binding groove"); a landmark keyword ("just past the SAM-binding loop", "two residues from the catalytic
       aspartate") owns nothing unless a new clause intervenes ("..., so I read two ThDP-fold regions flanking an
       iron-sulfur relay"); a preceding keyword or typed word of another type ("the proton-accepting aspartate at
       1251"), or rival noun, that holds the list directly keeps it, sought back to the sentence's start, unless the
       coordinate is the subject of a predicate naming the keyword ("(captured by the NTYK motif at residues 81, 104,
       134, 138) protonates the carbonyl oxygen and coordinates the nucleotide-sugar"; not respectively, "... and
       IPR008271 (..., residues 686-698) define the nucleotide-binding pocket and catalytic base"); a colon after a list
       opens a new statement about it;
     - round 4: a list whose own head names the keyword's type and no other stays the keyword's where the rules above
       would give it to that head or to another type's keyword between ("a single helix at 170 to 194, to a type I
       single-pass topology", "a short cytosolic amino end at 1 to 6, a first helix at 7 to 27"), except a non-ligand
       head or a helix that does not cross the membrane ("intramembrane helix 225 to 237"); the head is read back to the
       previous break or coordinate; the keyword's own accession may be led by a passive verb ("is indicated by
       IPR011054") or a dash, and accessions joined to it by "and" are its own too; a participle or "adds" sets an
       entry in apposition ("(..., residues 428-437) corresponding to the cysteine heme-iron ligand motif"), as does
       "which are the first intracellular loop, the second extracellular loop and ..."; the bracketed name of the
       keyword's own accession ("IPR006130 (3-dehydroquinate synthase, conserved site, residues 159-169)") and a
       possessive element ("its conserved site at 265 to 287") head nothing, but a family name in that bracket does,
       and so does a conserved site named with words between ("a conserved terpene synthase site at 601-615") except as
       the first item after the keyword's own colon ("the heme pocket: a conserved P450 site at 435 to 444"); an entry
       name may hold digit-led words and the keyword glued to the word before it ("IPR003015 (2-oxo acid
       dehydrogenase, lipoyl-binding site, ..., residues 94-103)"); a residue named after an uncited entry ("under
       IPR017972, with cysteine 451") is not on its list; contacts with DNA bases or grooves are DNA binding.
     Amino-acid names are part of a residue list; the cue words of single positions must be whole words; a number cut
     by the window edge is dropped; an InterPro citation's range is no longer re-read as a single position at its
     start; a number glued to letters other than a residue name ("CD39", "P450") is no coordinate; a single position
     is dropped only when its digits belong to a range, or it restates a range the keyword holds: for a site type a
     range earlier in the same list item ("at 120 to 130, with Gly125"; a separate list, "SAM-contacting positions at
     ..., 96, 124 and 125, with a tighter site across 74 to 80, 96 to 97 and 123 to 125", is its own claim), for a
     region type any held range, for a disulfide a listed pair's cysteine; a residue token the residue classifier
     rejects ("L-aspartate 4-semialdehyde", "histone H3 lysine 14") is not a position. Round 4: another molecule's
     numbering ("the D/T loops that house position 26", "position 37 of tRNA") and a helix count ("a 10-12
     transmembrane-helix bundle") are no coordinates; a family or superfamily the model calls broader or dismisses
     ("the broader RING/FYVE/PHD zinc-finger superfamily at 197 to 269"; "IPR027417 (P-loop containing ... homologous
     superfamily, residues 1-25), but ... rather than an NTPase function") places nothing, wherever the entry is
     listed; residues the positive-inside rule names flank a helix rather than sit in it; for a topological domain a
     coordinate placed in a helix ("at 136 in the third helix", "at 92 at the far end of the third") is dropped and
     one whose own list item names a side takes that side ("a long stretch at 117 to 180 faces out again").
   - Round 5: a number inside a degenerate counting run (the residue metric's run rule, counting_runs) is no
     placement; topology nouns after a crossing list head their own items ("with the amino end 1-20 and the long
     carboxyl end 203-319 on the cytoplasmic side") and "returns across at" names helices; a side predicate is a
     topological domain ("54 to 455 lie on the non-cytoplasmic side", "1 to 32 are cytoplasmic", "returns to the
     cytosol") and the sides of the membrane no longer rival that type; a coordinate's own predicate beats a rival
     noun before it ("residues 1 to 39 are disordered and non-cytoplasmic, 40 to 60 form a single hydrophobic
     transmembrane helix"); "positions, <list>" and "residues including <list>" are list cues; a coordinate whose own
     item names the keyword's type is not taken by a rival head ("and at 473 a cysteine named as the axial ligand to the
     heme iron"); the last residue is a chain length only in a length phrase; a family entry's name places nothing
     ("IPR050463 (Distal-less homeobox transcription factors family) spanning residues 12-286").
   - Final round (the round-5 span verifier's four regressions): a metal or ligand verb's object ends its phrase, so a
     coordinate past a clause break is never the verb's when its clause gives it another role ("the glutamate at 131
     sits between the histidines as the general base", "and an arginine finger at 703") or, for a metal, another
     ligand ("and lysine 252 binds 2-oxoglutarate"), and an item added by "with" or (for a metal) a clause with a verb
     of its own is the verb's only when it names the type again ("and histidine 394 binds copper B", "and the second, at
     249 to 253, does the same"), while a verbless continuation shares the verb ("ATP binds at 213, ..., and again at
     790"); a side predicate whose subject is a new terminus, tail or "both" does not read back over a list another
     head holds ("seven crossings, at 2 to 22, ..., with the amino terminus on the periplasmic side"), unless the list's
     own words name its side; a primed digit is a nucleic-acid or sugar locant ("binds ssDNA at 5′-terminated
     regions"); and the later_denial and COILS fixes above.
   - A position governed by a boundary word states where a feature ends: for a signal or transit peptide "preceding
     residue 10" claims the peptide 1-9 and "cleaved after Ala-22" 1-22, scored by the end; for a region type a
     boundary word opening the clause states its edge ("After residue 172 the chain becomes coiled": the coil starts at
     173), scored against the same edge of the nearest true feature; otherwise the position is dropped.
   - "The first twenty-three residues" and "the last five residues" are read as spans; disulfide pairs written
     "129 with 570", "between 34 and 78", "Cys34-Cys78", "A disulfide bond pairs cysteines 313 and 346" (with any
     further pairs listed, "pair cysteines at 31 and 110, 64 and 69") or "cysteines at 28 and 44 bonded to each other"
     are read, and their cysteines are not claimed again as singles. A disulfide range headed directly by a
     domain-level noun or an entry's citation ("IPR003582 (ShKT domain, residues 276-320), a compact
     disulfide-stabilized module", "SRCR domains at 70 to 170", "IPR054717 (..., disulphide-bonds motif, residues
     64-75)") or said to hold bonds ("the extracellular loop from 112 to 280 carries two disulfide bonds") names a
     region and places no bond, so it is dropped.
   - Distance: a disulfide claim is scored by its cysteines against the nearest bond; a signal or transit peptide
     claimed from residue 1 (at least 12 residues, or by a boundary word) by its end, the cleavage site; a labelled
     n-, h- or c-region by how far it overhangs the true peptide; a coiled-coil claim spanning several COILS segments
     against their merged extent; a topological domain against the nearest true domain on the side its keyword names
     ("non-cytoplasmic" names the outside), unless the sentence names both sides and the coordinate does not hang
     directly on that keyword, when either side counts.
   - Not changed: a range claim near a dense InterProScan site set (a CDD "active site" of 30 residues) is scored by
     the gap to the nearest site residue like any other point truth; the audit counted the claims placed within 5 only
     that way.
   - Truth: as presence, plus metal-binding records the patterns.py description regex misses ("[4Fe-4S] cluster",
     "a divalent metal cation", K+, Na+, molybdopterin), and InterProScan metal sites when sites=True. The site
     records raise the chance level of a randomly placed claim by 4 to 7 points and the observed share by 5 to 15, so
     the lift above chance grows.
4. InterPro precision (accessions named anywhere in the generation against the protein's true accessions, pooled).
   - Proteins with no InterPro entry are scored: InterPro covers every protein, so any accession named for them is
     wrong.
   - A named ancestor of a true entry counts, walking the parent links the test set's own features carry.
   - A retired accession counts when a member signature it held still matches the protein (right in its release).
   - Each accession carries in_prompt, so copying from the +feat prompt can be reported apart.
   - Final round: a degenerate accession run ("IPR003650, IPR003651, IPR003652, ...", fifteen or more mentions in a row,
     each at most three numbers from the last) names no entries, as a counting run cites no residues and places no span.

Declined: counting an EC-numbered enzyme as having an active site for the span metric (it places no residue; the
final round counts it for presence alone, where 60% of active-site false positives were such enzymes); LLM adjudication
of partner and substrate mentions in the residue and span metrics; excluding every keyword inside an InterPro entry
name (most such names are real ligand-binding domains, "ABC transporter-like, ATP-binding domain"; only non-ligand
objects and the RRM misnomer are excluded); excluding class III signal peptides
and mature-chain toxin entries (no reliable automatic signal); a hedge rule (the audit measured it as negligible);
per-protein instead of pooled averaging (a reporting choice, not a correctness fix; the audit printed it alongside).
MIN_RESIDUE_CLAIMS is a reporting floor: below it a cell's residue accuracy is too few citations to plot.
"""

import argparse
import collections
import concurrent.futures as cf
import csv
import functools
import gzip
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bioreason2 import paths as _paths  # noqa: E402
from evals.grounding import patterns as rc  # noqa: E402

REFERENCE = Path(__file__).resolve().parent / "grounding_reference.json"
# The benchmark test split (scripts/data/fetch_corpus.py): a parquet file, or a directory whose TEST_SHARDS are read.
TEST_SET = Path(_paths.CORPUS)
TEST_SHARDS = "test-*.parquet"
# eval.py writes one <protein_id>_<aspect>_k<sample>.json per protein; the unified-aspect first sample is scored.
RESULT_PATTERN = "*_all_k00.json"
IPR_RE = re.compile(r"IPR\d{6}")

TOOL_TYPES = ["transmembrane", "signal_peptide", "coiled_coil", "active_site", "binding_site"]
# The headline presence F1 runs over the three types whose annotation is complete. Active- and
# binding-site annotation misses about half the real sites, and the presence reader
# agrees less well on them, so they are reported per type only; binding without a precision, since no source fills
# its negatives.
RECALL_ONLY_TYPES = ("binding_site",)
F1_TYPES = ["transmembrane", "signal_peptide", "coiled_coil"]
WITHIN = 5
# Below this many counted claims a cell's residue accuracy is reported as too few citations rather than plotted.
MIN_RESIDUE_CLAIMS = 100
CONTEXT = 160

# --- residue claims (rules validated on 53 of 54 fresh claims) ---
# A counting run needs fifteen claims: shorter same-residue lists are compositional claims the model means ("lysines
# at 61, 62, 63, 65, 66 and 67 packed into nineteen residues") and stay scored; the degenerate enumerations ("Gly13,
# Gly14, ..., Gly208") run to hundreds and hold 98% of the claims a six-claim threshold flags.
LOOP_RUN = 15
LOOP_STEP = 2
LOOP_GAP = 40
LEAD = 90
TRAIL = 70
COUNTED_CATEGORIES = ("self", "homolog")
AA_WORDS = "|".join(sorted([c.title() for c in rc.AA3] + list(rc.AA_FULL), key=len, reverse=True))
RE_SENT_START = re.compile(r"(?:[.;!?]\s|\n)")
RE_CHEM = re.compile(
    r"^['’]?\s*(?:,\s*\d+\s*)?[-–—]\s*(?:semialdehyde|mono-?oxygenase|di-?oxygenase|hydroxylases?|"
    r"hydroxylation|oxoglutarate|oxo\w*|phosphate|carboxylate|decarboxylase|dehydrogenase|dehydratase|kinases?|"
    r"oxidase|aminotransferase|aminomutase|diaminopimelate|carboxypropylamine|reductase|synthase|synthetase|"
    r"transferase|lyase|isomerase|mutase|epimerase|hydrolase|ligase|methyltransferase|monophosphate|diphosphate|"
    r"triphosphate|amino\w*|hydroxy\w*)", re.IGNORECASE)
RE_LIKE = re.compile(r"^\s*[-–]\s*like\b")
RE_NAME_LEAD = re.compile(r"(?:protease|factor|activator|repressor|receptor|transporter|regulator|Rec2p/)\s*$")
# Final round: a residue token joined to a gene alias by a slash ("His6/HISN3", "EFG1/Asp3", "CD58/Leu-11") or
# followed by a class noun ("the catalytically dead Lys49 homologues", "the group I, Asp49 class", "the mycobacterial
# Asp32 proteases") names a gene, family or class; residues joined by slashes ("Asp157/Asp288", "Ser/Thr") stay claims.
RESIDUE_TOKEN = r"(?:%s)(?:[\s-]?\d|\b)" % "|".join([c.title() for c in rc.AA3] + [c.upper() for c in rc.AA3])
NOT_ALIAS = r"(?:ATP|ADP|GTP|GDP|NADP?H?|FAD|FMN|SAM|CoA|PLP|DNA|RNA|Zn|Mg|Mn|Ca|Fe|Cu|Co|Ni)\b"
RE_ALIAS_LEAD = re.compile(r"(?<![\w/-])(?!%s|%s)[A-Z][A-Za-z0-9]{1,9}/\s*$" % (RESIDUE_TOKEN, NOT_ALIAS))
RE_ALIAS_TRAIL = re.compile(r"^/(?!%s|%s)[A-Z][A-Za-z0-9]{1,9}\b" % (RESIDUE_TOKEN, NOT_ALIAS))
RE_CLASS_TRAIL = re.compile(r"^(?:[-–]type)?\s+(?:famil(?:y|ies)|group|class|subfamily|clade|homolog(?:ue)?s|orthologs|"
                            r"paralogs|myotoxins?|factors|proteases|proteinases)\b", re.IGNORECASE)
# A residue token the text introduces as a protein's name ("a protein called Lys1", "the peroxisomal protein Lys1",
# "whose partner Lys1") names that protein wherever it recurs ("it binds Lys1", "the half-life of Lys1").
RE_COMPACT_TOKEN = re.compile(r"\b(?:%s)\d{1,4}\b" % "|".join(c.title() for c in rc.AA3))
RE_NAME_INTRO = re.compile(r"\b(?:proteins?|called|named|termed|partners?|factor|regulator|gene)\s+$", re.IGNORECASE)
NAME_INTRO_REACH = 30
NAME_CACHE = 8
RE_UB_AFTER = re.compile(r"^['’]?\s*[-–—]?\s*(?:linked|linkages?|specific|preferring|trimming|selective|chains?|"
                         r"scaffolds?|(?:poly|de)?ubiquitin\w*)\b")
# Plural "chains" only: "an internal aldimine at Lys48 and the rest of the chain" is this protein's lysine.
RE_UB_NEAR = re.compile(r"^[^.;]{0,30}?(?<!cross)\b(?:linked|linkages?|chains)\b", re.IGNORECASE)
UB_POSITIONS = {1, 6, 11, 27, 29, 33, 48, 63}
# Ubiquitin's own C-terminal glycine ("the isopeptide bond between ubiquitin Gly76 and a substrate lysine").
UB_GLYCINES = {75, 76}
RE_UB_GLY_LEAD = re.compile(r"\bubiquitin(?:'s|’s)?\b[^.;]{0,40}$", re.IGNORECASE)
# Round 5: ubiquitin named before the lysine ("specific linkages (notably Lys11 and Lys48)", "Polyubiquitin chains of
# Lys11/Lys48 types", "presents the acceptor ubiquitin so that lysine 11 attacks"); not a modification of this protein
# ("ubiquitinated at Lys48").
RE_UB_LEAD = re.compile(r"(?:\b(?:poly)?ubiquitin(?!at|yl)\w*|\blinkages?\b|\bchains\b)[^.;]{0,40}$", re.IGNORECASE)
# A residue followed by a window ("The catalytic dyad (His43-50 and Asp85-95)") names a stretch, not one position.
RE_WINDOW_TRAIL = re.compile(r"^\s*[-–—]\s*\d")
# A bare mark abbreviation ("H3K9me3") precedes the protein's own reader-cage residues, so it is not a histone mention.
RE_HISTONE_LEAD = re.compile(r"\bhistones?\b|\bH[1-4](?:\.\d)?[ABXZ]?\b|\bH2A\b|\bH2B\b", re.IGNORECASE)
RE_HISTONE_TRAIL = re.compile(r"^['’]?\s*(?:\w+\s+){0,2}(?:on|of|in|at)\s+(?:the\s+)?(?:histone|H[1-4])\b")
RE_CTD = re.compile(r"\bCTD\b|heptad|heptapeptide|YSPTSPS|RNA polymerase II|\bPol II\b|\bRNAP ?II\b|\bRPB1\b|"
                    r"\bRpb1\b", re.IGNORECASE)
RE_ZINC_ADJ_LEAD = re.compile(r"(?:(?:Cys|His|C|H)\s?\d|Zn\d?)\s*[-–/]?\s*$")
RE_ZINC_ADJ_TRAIL = re.compile(r"^\s*[-–/]?\s*(?:Cys|His)\s?\d|^\s*[-–/]\s*(?:Cys|His)\b|^\s+pairs?\b")
# Final round: plurals ("two Cys4 clusters") and the nouns of other metal-site notations ("Cys4 knuckles", "a Cys4
# tetrad", "a Cys4 GATA finger", "a Cys4 Dof finger", "a Cys6 binuclear cluster", "His2-ligated ferredoxins", "Cys4
# ligation", "the Cys4 spacing", "a classic Cys4 2Fe-2S cluster"); "Zn2-Cys6" leads with the metal.
RE_ZINC_WORD = re.compile(r"^\s*[-–]?\s*(?:types?|zinc|Zn|metal|sites?|modules?|motifs?|clusters?|fingers?|ligands?|"
                          r"centres?|centers?|arrangements?|configurations?|patterns?|coordination|domains?|"
                          r"class(?:es)?|tetrads?|ribbons?|knuckles?|GATA|Dof|binuclear|ligated|ligation|spacing|"
                          r"\[?\d?Fe-\d?S\]?)(?![A-Za-z])", re.IGNORECASE)
# A 2-His-1-carboxylate facial triad written with the residue ("a 2-His-1-Asp facial triad", "2-His-1-Glu/Asp") is
# iron-site notation, like "2-His-1-carboxylate".
RE_TRIAD_LEAD = re.compile(r"\b2[-–]$")
RE_TRIAD_TRAIL = re.compile(r"^[-–](?:Asp|Glu)(?:/(?:Asp|Glu))?\b")
# The modified protein has to be named gene-style ("eIF2", "CsrA", "cdc20"); "the sumoylation site at lysine 445" is
# this protein's own residue. Only the verb ignores case ("Phosphorylation of host STAT1 at Tyr-70").
# Final round: the enzyme acting on this protein ("phosphorylation by GSK3 at serine 215", "phosphorylated by host
# casein kinase II at threonine 11") and a second modification of it ("Autophosphorylation and SUMOylation at lysine
# 425") are not a named substrate.
RE_SUBSTRATE_LEAD = re.compile(
    r"(?i:phosphorylat|methylat|acetylat|ubiquitinat|deacetylat|demethylat|dephosphorylat|sumoylat|hydroxylat|"
    r"citrullinat)\w*\s+(?:of\s+)?(?:the\s+)?(?:(?!(?i:by)\b)[\w-]+\s+){0,4}?(?!\w*(?i:ylation|ylated|ylates)\b)"
    r"(?:[A-Za-z]*[A-Z][A-Za-z]*\d+[A-Za-z0-9]*|[A-Z][a-z]*[A-Z][A-Za-z0-9]*|[a-z]{2,}\d+[a-z]?)"
    r"\s+(?:at|on)\s+['‘]?$")
# Round 5: an enzyme acting on a named substrate at a residue ("cleaving fibrinogen at Arg-178 and Met-180",
# "acetylates enolase at lysine 214", "It modifies connexin-43 at serine 44"), and a bond in a named substrate
# ("between Gly-132 and Lys-133 in type I collagen").
# The residue may be a later item of the substrate's list ("cleaving fibrinogen at Arg-178 and Met-180", "Cleaves
# fibrinogen at '178-Arg-|-Met-179' and '180-Arg-|-Lys-181'"), and a named target leads to it ("the target CsrA on
# Ser-171").
# Final round: the object is no named substrate when it is the protein itself or its other copy ("the two kinase
# domains phosphorylate each other on the activation-loop tyrosines", "phosphorylate the enzyme itself at histidine
# 213", "cleaves the chain at glycine 1091") or a residue or segment of it ("phosphorylates the tail tyrosines at 1279",
# "phosphorylate the activation segment at serine 337"), nor when a participle ends it ("the modifying system acting on
# lysine 64"); and the residue must be on the substrate's list, not past a new clause ("phosphorylates substrates on
# tyrosine, using the catalytic aspartate at 390").
SELF_OBJECT = (r"(?:itself|its|themselves|each\s+other|one\s+another|the\s+other|this\s+(?:enzyme|protein|chain|kinase|"
               r"subunit)|the\s+(?:enzyme|protein|chain|kinase)\s+itself|the\s+chain)\b")
OWN_OBJECT_END = (r"(?:(?:%s)s?|loops?|segments?|tails?|motifs?|domains?|regions?|termin\w+|ends?|chains?|residues?|"
                  r"sites?|other|itself|\w+ing)\s+(?:at|on|after)\b" % AA_WORDS)
SUBSTRATE_TAIL = r"(?:[^.;,]|,(?=\s*(?:(?:and|or)\s+)?(?:['‘\d]|(?:%s)s?\b)))" % AA_WORDS
RE_SUBSTRATE_ACT = re.compile(
    r"\b(?:(?:cleav|acetylat|phosphorylat|methylat|ubiquitinat|hydroxylat|sumoylat|deacetylat|dephosphorylat)"
    r"(?:es|ing|e)|modif(?:ies|ying|y))\s+(?!%s)(?:the\s+|host\s+)*(?:[\w-]+\s+){0,3}?(?!%s)[\w-]+\s+"
    r"(?:at|on|after)\s+%s{0,40}$|\btarget\s+(?:protein\s+)?(?-i:[A-Z][\w-]*)\s+(?:at|on)\s+['‘]?$"
    % (SELF_OBJECT, OWN_OBJECT_END, SUBSTRATE_TAIL), re.IGNORECASE)
# "substrate" as a noun, not as a modifier of the protein's own pocket ("The arginine at 397 in the substrate pocket",
# "an arginine at 100 in the substrate-binding region").
RE_SUBSTRATE_TRAIL = re.compile(
    r"^['’]?\s*(?:(?:and|/)\s*(?:%s)[\s-]?\d{1,4}\s*)?(?:in|of|within)\s+(?:the\s+)?(?:host\s+|human\s+)?"
    r"(?:type\s+[IVX]+\s+)?(?:collagens?|fibrinogen|fibronectin|elastin|laminin|tropoelastin|prothrombin|"
    r"plasminogen|kininogen|substrates?(?![\w-]|\s+(?!(?:and|or|but|which|that|is|are|was|were|by|to|for|with|in|on|"
    r"at|of|as|from|via|during|while|when|where)\b)[A-Za-z]))\b" % "|".join(c.title() for c in rc.AA3))
RE_PARTNER_TRAIL = re.compile(r"^['’]?\s+(?:of|on|in|from)\s+(?!the\b|this\b|its\b|a\b|an\b|each\b|both\b|"
                              r"that\b|IPR)([A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9-]*)")
# Not partners: ions, membrane segments, a protein kinase C's C1 and C2 domains ("aspartates at 382 and 426 in C1") and
# motifs spelled in one-letter code ("GHSLG").
RE_PARTNER_NOT = re.compile(r"^(?:Mg|Zn|Ca|Fe|Mn|Cu|Co|Ni|Na|K)\d?$|^(?:M|TM|S|H|TMH)\d|^C[12][AB]?$|"
                            r"^[ACDEFGHIKLMNPQRSTVWYX]{4,}$")
# Final round: a residue of another molecule named as such ("cysteine 77 of a beta subunit", "lysine 112 of a partner
# protein"); "the partner subunit" and "the opposite subunit" of a homodimer are this protein's own copy.
RE_PARTNER_SUBUNIT = re.compile(r"^['’]?\s+(?:of|on|from|in)\s+(?:a|an)\s+(?:partner|beta|alpha|gamma|delta|β|α|γ|"
                                r"heterologous)\s+(?:subunit|chain|protein|polypeptide)s?\b", re.IGNORECASE)
# Final round: a list headed by two plural residue words ("the zinc site at cysteines and histidines 33, 36, 50 and 52")
# does not say which residue sits where.
RE_MIXED_HEAD = re.compile(r"\b(?:%s)s\s*(?:,|and|or|/|&)\s*(?:the\s+)?$" % AA_WORDS, re.IGNORECASE)
PARTNER_NOT = {"HRD", "DFG", "TXY", "DNA", "RNA", "ATP", "GTP", "NAD", "NADP", "FAD", "FMN", "SAM", "CoA", "PLP",
               "TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7", "CXXC", "HXXXH", "SH2", "SH3", "PDZ", "EF", "GXGXXG",
               "PKC", "ER", "ECL1", "ECL2", "ICL1", "ICL2", "ICL3", "NPA", "DRY", "PIP2", "HEXXH", "HXH", "CC1", "CC2"}
RE_HOMOLOG = re.compile(r"numbering|equivalent to|correspond(?:s|ing)? to|homolog|ortholog|paralog|"
                        r"\bin (?:human|mouse|yeast|bovine|rat|E\. coli|Escherichia)\b", re.IGNORECASE)
RE_PARTNER_HEADER = re.compile(r"Interaction Partners")
RE_LIST_HEADER = re.compile(r"^\s*-?\s*[^\n]*:\s*$", re.MULTILINE)
RE_MISSED = re.compile(
    r"\b(%s)(s?)\s+(?:residues?\s+|at\s+(?:positions?\s+|residues?\s+)?)(\d{1,4})\b(?!\s*(?:-|–|to)\s*\d)"
    % AA_WORDS, re.IGNORECASE)
# Final round: the hyphenated mutant form too ("a proline-to-serine at 3").
RE_MULTI_AA = re.compile(r"\b(?:%s)s?\s*[-–]?\s*(?:,|and|or|/|to)\s*[-–]?\s*(?:the\s+)?"
                         r"(?:(?:two|three|four|five|six|both)\s+)?$" % AA_WORDS, re.IGNORECASE)
RE_PROMPT_COORD = re.compile(r"\b\d{1,4}\b")

# --- presence: the five tool-annotated types, patterns.py's patterns with the audit's changes ---
# One definition of a binding site for claims and truth: what a non-ligand site binds is a macromolecule or a
# modified residue of one, so "acetyllysine binding site" is out on both sides, and so is the IQ motif's
# "EF-hand binding site" (it binds calmodulin, an EF-hand protein); "EF-hand calcium-binding site" names calcium.
NOT_LIGAND = (r"(?:ss|ds|t|m|r|sn|mi)?(?:DNA|RNA)|nucleic[- ]acids?|(?:poly|phospho)?peptides?|proteins?|histones?|"
              r"chromatin|ubiquitin|actin|"
              r"calmodulin|CaM|receptors?|membranes?|partners?|antibod(?:y|ies)|acetyl-?lysines?|acetylated lysines?|"
              r"methyl-?lysines?|phospho-?(?:tyrosine|serine|threonine)|arrestins?|tubulin|microtubules?|clathrin|"
              r"ankyrin(?:-G)?|Aurora(?:-A)?|Skp1|dimer|interface|interdomain|IQ motif|EF-hand(?=[- ]binding)|"
              # Round 5: protein domains and adaptors named as the bound object ("E2-binding site", "PDZ-binding
              # motif", "SH3-binding", "ESCRT-III binding site", importin's "NLS-binding site").
              r"E2|E3|PDZ|SH2|SH3|WW|EVH1|14-3-3|ESCRT(?:-[IV]+)?|NLS|importins?|scaffold|"
              # A sequence motif of a partner ("YPXnL-motif binding site").
              r"[\w]+-motif")
SIGNAL_ANCHOR = r"signal[- ]anchor"
# A range said to lie on one side of the membrane is a topological domain ("residues 1 to 32 are cytoplasmic", "54 to
# 455 lie on the non-cytoplasmic side", "740 to 797 returns to the cytosol", "203 to 319 on the cytoplasmic side").
SIDES = r"cytoplasmic|cytosolic|non-cytoplasmic|extracellular|lumenal|luminal|periplasmic|exoplasmic"
SIDE_PREDICATE = (r"\b(?:is|are|lies|lie|sits?|faces?|remains?|stays?)\s+(?:(?:on|in)\s+the\s+)?(?:%s)"
                  r"(?:\s+(?:side|face))?\b(?![- ](?:domains?|loops?|tails?|segments?|turns?|amino|carboxyl|N|C)\b)|"
                  r"\b(?:returns?\s+to|faces?|lies\s+in|lie\s+in|sits?\s+in|remains?\s+in)\s+the\s+"
                  r"(?:cytosol|cytoplasm|lumen|periplasm|extracellular space)\b|"
                  r"\b(?:on|in)\s+the\s+(?:%s)\s+side\b" % (SIDES, SIDES))
RE_SIDE_PREDICATE = re.compile(SIDE_PREDICATE, re.IGNORECASE)
RE_NON_INSIDE = re.compile(r"non[-‐‑]?cyto", re.IGNORECASE)
PATTERNS = {name: pattern for name, (_, pattern) in rc.FEATURE_TYPES.items()}
# "repeated membrane crossings at 113 to 134, 136 to 158" and "a single helix crosses at 719 to 739" state the helices
# as plainly as "crosses the membrane"; only the forms followed by coordinates, since "a membrane-crossing thioether" is
# a property of a small molecule. So do "a single hydrophobic helix at 205 to 228" (every such phrase in the
# generations is the one membrane helix), "Three hydrophobic crossings organize this chain: 38 to 57, 64 to 81" and
# "The membrane part begins at 298 to 320".
PATTERNS["transmembrane"] = re.compile(
    rc.FEATURE_TYPES["transmembrane"][1].pattern + "|" + SIGNAL_ANCHOR
    + r"|\b(?:crossings?(?: and loops)?|hel(?:ix|ices) (?:crosses|cross|spans|span|traverses|traverse)) "
      r"(?:at|from|over|through) "
      r"(?:about |roughly |approximately |around )?(?=\d)|"
      r"\bsingle (?:hydrophobic )?helix (?:at|from|over|spanning|across) (?:about |roughly |approximately |around )?"
      r"(?=\d)|\bcrossings?\b(?:(?!\b(?:loops?|turns?|separated)\b)(?:\d+-residues?\b|[^.;:\d])){0,50}:\s*(?=\d)|"
      r"\bmembrane part (?:begins|starts)\b|"
      # Round 5: "the chain returns across at 249 to 270, 340 to 360" and "crosses back at" name the next helices.
      r"\b(?:returns?|passes|goes|crosses) (?:back )?across (?:the (?:membrane|bilayer) )?(?:at|from|over|through) "
      r"(?:about |roughly |approximately |around )?(?=\d)|\bcross(?:es)? back (?:at|from|over|through) (?=\d)",
      re.IGNORECASE)
PATTERNS["signal_peptide"] = re.compile(
    r"signal[- ]peptide|signal sequence|SignalP|secreto?ry signal|secretion signal|cleavable signal|"
    r"leader (?:peptide|sequence)|N[- ]terminal signal", re.IGNORECASE)
# An intermediate filament's rod segments are coils by name ("coil 1A at 25 to 63, ..., coil 2 at 242 to 403").
PATTERNS["coiled_coil"] = re.compile(
    r"coiled[- ]?coil|\bcoiled\b(?!\s+bod)|\bheptads?\b|leucine zipper|(?:alpha|α)[- ]helical coil|"
    r"\bcoils? (?:at|from|over|around) (?=\d)|"
    r"\bcoils? (?:\d[AB]?|[IV]{1,3}[AB]?) (?:at|from|over|across|spanning) (?:about |roughly |around )?(?=\d)",
    re.IGNORECASE)
# The serine hydrolase's "nucleophilic elbow at 169 to 173" carries the nucleophile, and "a catalytic network at 29, 47,
# 48" or "the catalytic segment (470-481)" names catalytic residues as plainly as "catalytic core" does.
PATTERNS["active_site"] = re.compile(
    rc.FEATURE_TYPES["active_site"][1].pattern
    + r"|proton[- ]acceptor|proton-donor|\bnucleophile\b|aspartyl[- ]?phosphate|phospho-?enzyme intermediate|"
      r"covalent (?:\w+[- ])?intermediate|acyl[- ]enzyme|charge[- ]relay|catalytic quartet|nucleophilic elbow|"
      r"catalytic (?:network|segment)s?\b|"
      # Round 5: "the DEDYD catalytic set at 76, 78, 137" lists residues; "a catalytic SET module" is a domain.
      r"(?-i:catalytic sets?)(?=\s+(?:at|over|across|of|formed|built|comprising|made)\b|\s*[(:])", re.IGNORECASE)
# A metal site stated by its coordination or its ligands ("requires zinc coordination", "Cys/His-coordinated modules",
# "copper ligands at 551, 553 and 716", "calcium ions bind at 201", "magnesium-coordinating residues at 638 and 651",
# "the metal-chelating pair at 141 and 157"); "a zinc-coordinating fold" names no site.
METAL_SITE = (r"\b(?:zinc|metal|iron|copper|calcium|magnesium|manganese|cobalt|nickel|Zn|Fe|Cu|Ca|Mg|Mn)"
              r"(?:\(?2\+\)?)?[- ](?:ion[- ])?(?:coordination|ligands?)\b|"
              r"\b(?:zinc|metal|iron|copper|calcium|magnesium|manganese|cobalt|nickel|Zn|Fe|Cu|Ca|Mg|Mn)(?:\(?2\+\)?)?-"
              r"(?:coordinating|chelating|liganding)\b(?=(?:\s+[A-Za-z][\w-]*){0,3}\s+\d)|"
              r"\b(?:Cys|His|cysteine|histidine)(?:[/-](?:Cys|His|cysteine|histidine))*[- ]coordinated\b|"
              r"\b(?:zinc|metal|iron|copper|calcium|magnesium|manganese)(?:\(?2\+\)?)?(?:\s+ions?)?"
              r"\s+(?:binds?|is bound|are bound)(?=\s+at\b)|"
              # Round 5: "the cluster ligands at cysteine 86, histidine 88" are an iron-sulfur cluster's; residues
              # "assigned to manganese binding" and "zinc binding assigned to 233, 235" are a metal site's.
              r"(?:\bcluster|(?<![\w-])\[\d?Fe-\d?S\](?:\s+cluster)?|\bFe-S(?:\s+cluster)?|"
              r"\biron[- ]sul(?:f|ph)ur(?:\s+cluster)?)[- ]ligands?\b|"
              r"\bassigned\s+to\s+(?:the\s+)?(?:metal|zinc|iron|copper|calcium|magnesium|manganese|cobalt|nickel)"
              r"[- ]binding\b|\b(?:metal|zinc|iron|copper|calcium|magnesium|manganese|cobalt|nickel)[- ]binding\s+"
              r"(?:is\s+|are\s+)?assigned\s+to\b")
# A metal or cluster a verb coordinates, chelates or binds, whatever its count and adjectives ("six cysteines ...
# coordinate two zinc ions", "coordinates two Zn2+ ions", "coordinating two divalent cations", "to chelate zinc", "243,
# 245 and 299 bind a ferrous iron", "must bind an iron-sulfur cluster").
# The metal word must end the bound object: "bind calcium transporters" binds proteins, "a zinc finger" is a domain.
NOT_BOUND_NOUN = (r"(?!\s+(?:fingers?|channels?|transporters?|pumps?|exchangers?|sensors?|chaperones?|homeostasis|"
                  r"transport|uptake|efflux|import|export|signal\w*)\b)")
METAL_OBJECT = (r"(?:(?:a|an|the|one|two|three|four|both|each|its)\s+)?(?:(?:divalent|dinuclear|binuclear|mononuclear|"
                r"catalytic|structural|single|essential|bridging|non-?heme|ferrous|ferric|metal)\s+)*"
                r"(?:zinc|calcium|magnesium|iron|copper|manganese|cobalt|nickel|(?-i:Zn|Ca|Mg|Fe|Cu|Mn|Ni)|metals?|"
                r"cations?|\[\d?Fe-\d?S\](?:\s+clusters?)?|iron[- ]sul(?:f|ph)ur(?:\s+clusters?)?|"
                r"(?-i:Fe-S)(?:\s+clusters?)?)(?:\(?\d?\+\)?)?(?![\w-])" + NOT_BOUND_NOUN)
METAL_VERBS = r"\b(?:coordinat(?:e|es|ing|ed)|chelat(?:e|es|ing|ed)|bind(?:s|ing)?)\s+" + METAL_OBJECT
# Small-molecule ligands a "-binding" site noun names ("the CoA-binding tunnel", "a quinone-binding chamber",
# "phosphoinositide-binding surface at 383"); "lipid binding" alone is the activity, not a site.
SITE_NOUNS = (r"sites?|residues?|positions?|pockets?|cavit(?:y|ies)|chambers?|cups?|tunnels?|channels?|clefts?|"
              r"grooves?|surfaces?|loops?|motifs?")
SMALL_LIGANDS = (r"quinone|ubiquinone|menaquinone|plastoquinone|lipid|phospholipid|phosphoinositide|PI3P|PI4P|PIP2|"
                 r"PIP3|PI\(\d(?:,\d)*\)P\d?|PtdIns(?:\(\d(?:,\d)*\)P\d?|\dP)?|phosphatidylinositol[\w(),-]*|sterol|"
                 r"steroid|cholesterol|acetyl-CoA|acyl-CoA|odorant|retinal|chlorophyll|carotenoid|fatty[- ]acid|"
                 r"nucleoside|pyrimidine|purine|lipoyl|molybdopterin|cobalamin|siderophore|(?:2-)?oxoglutarate|"
                 r"NAD\(P\)H?|NAD\+|NADH|NADPH|PLP|TPP|ThDP|flavin|biotin|CoA")
# The ligands a "binds" verb may name besides the metals ("It binds ferrous iron and 2-oxoglutarate").
BOUND_LIGANDS = (r"heme|haem|FAD|FMN|NADP?H?|NAD\(P\)H?|NAD\+|ATP|ADP|GTP|GDP|PLP|TPP|ThDP|SAM|S-adenosyl[\w-]*"
                 r"methionine|CoA|[\w-]+-CoA|biotin|lipoate|flavin|cofactor|substrate|nucleotides?|Na\+|K\+|"
                 r"(?:2-)?oxoglutarate|(?:alpha|α)-ketoglutarate|ubiquinone|quinones?|menaquinone|plastoquinone|"
                 r"sterols?|cholesterol|phosphoinositides?|PI3P|PI4P|PIP2|PIP3|fatty acids?|molybdopterin|cobalamin|"
                 r"retinal")
# "NAD+ binds at 308 to 312, 318 to 320, ..." and "NAD(+) binds at 275 to 294" place the ligand's site.
# Round 5: "acetyl-CoA binds first" in an ordered mechanism; residues "assigned to manganese binding" and "zinc binding
# assigned to 233, 235"; "coordinates phosphate".
LIGAND_BINDS = (r"\b(?:ATP|ADP|GTP|NADP?H?|NAD\(P\)H?|FAD|FMN|SAM|CoA|TPP|ThDP|heme|haem|substrate|cofactor)"
                r"(?:\+|\(\+\)|\(2\+\))?\s+(?:binds|is bound)(?=\s+(?:at|first|second|in|within|to)\b)|"
                r"\bcoordinat(?:e|es|ing|ed)\s+(?:the\s+|a\s+|an\s+)?(?:phosphates?|substrate|nucleotide|ligand|"
                r"cofactor|ATP|GTP|NADP?H?|FAD)\b")
# A ligand's residue set named by its own head ("a carbohydrate-binding set at 28, 97, 98", "the TPP-binding pair at 29
# and 64", "substrate-contact residues at 56, 103", "a glycine-rich cofactor region at 19 to 46"); a macromolecule's
# set ("DNA-contact residues", "a protein-binding cluster") is no ligand site.
LIGAND_SETS = (r"(?<![\w-])(?!(?:%s)-)[\w()+/]+-binding (?:sets?|pairs?|clusters?|constellations?)\b|"
               r"\b(?:substrate|ligand|cofactor|nucleotide|ATP|ADP|GTP|GDP|NADP?H?|NAD\(P\)H?|FAD|FMN|SAM|CoA|TPP|"
               r"ThDP|heme|haem|sugar|phosphate|metal)[- ]contact (?:residues?|positions?|sites?|sets?|fields?|shells?|"
               r"networks?|surfaces?|clusters?)\b|"
               r"\bglycine-rich (?:cofactor|dinucleotide|nucleotide|phosphate|NADP?H?|NAD\(P\)H?|FAD|ATP|GTP)[- ]"
               r"(?:regions?|edges?|motifs?|segments?)\b" % NOT_LIGAND)
# "UDP-alpha-D-glucose contacts at 291, 350 to 352" names the ligand by its own name; contacts with a macromolecule,
# a partner protein or another part of the chain are not a ligand site.
NOT_CONTACT = (r"the|its|their|these|those|further|other|additional|and|or|with|plus|interdomain|inter-domain|"
               r"intersubunit|inter-subunit|subunit|intramolecular|crystal|lattice|backbone|helix|helical|domain|lobe|"
               r"loop|GEF|GDI|GAP|effector|Ig|H[1-4]|H2A|H2B|base|base-specific|groove|major-groove|minor-groove")
LIGAND_CONTACTS = (r"(?<![\w-])(?!(?:%s|%s)\s)[\w()+/',-]*[\w)+]\s+contacts?\s+(?:fall\s+|sit\s+|lie\s+)?"
                   r"(?:at|across|map to)(?=\s+\d)" % (NOT_LIGAND, NOT_CONTACT))
# "the glycine-rich ATP grip at 19 to 27" and "a glycine-rich phosphate loop" name the nucleotide-binding loop.
NUCLEOTIDE_LOOP = r"(?<![\w-])(?:ATP|ADP|GTP|GDP|nucleotide|phosphate|NADP?H?|FAD|FMN|SAM|CoA)[- ](?:loops?|grips?)\b"
# A placed cofactor motif names the cofactor's site ("the GAGGTS nucleotide motif at 1395-1400", "a dinucleotide
# fingerprint at 38 to 43", "an HFA FAD motif at 335-338").
NUCLEOTIDE_MOTIF = (r"(?<![\w-])(?:(?:di)?nucleotide|ATP|GTP|NADP?H?|NAD\(P\)H?|FAD|FMN|SAM|CoA)[- ]"
                    r"(?:motifs?|fingerprints?)(?=\s+(?:at|over|from|spanning|across)\s+\d)")
# "eight covalent heme c attachment sites at 118-119" bind heme; heme counts as a metal site on both sides.
HEME_ATTACHMENT = r"\b(?:heme|haem)(?: c)?[- ]attachment (?:sites?|motifs?|cysteines?|residues?)\b"
# "histidine 177, glutamate 180 and histidine 304 bind a single zinc ion" states a metal site.
METAL_BINDS = (r"\bbinds? (?:a |the |two |one )?(?:divalent |catalytic |structural |single )?(?:Fe|iron|calcium|Ca2\+|"
               r"zinc|Zn2\+|Mg2\+|magnesium|Mn2\+|manganese|copper|cobalt|nickel|metal)(?![\w-])" + NOT_BOUND_NOUN)
PATTERNS["binding_site"] = re.compile(
    rc.FEATURE_TYPES["binding_site"][1].pattern
    + r"|(?:(?<![A-Za-z])c?AMP|iron[-–]sul(?:f|ph)ur(?: cluster)?)[- ]binding\b(?!\s*GO:)(?![- ](?:enzyme|protein))|"
      r"(?:calcium|Ca2\+|Ca\(2\+\)|zinc|Zn2\+|iron|magnesium|Mg2\+|manganese|copper|cobalt|nickel|PLP|"
      r"pyridoxal[- ]phosphate|phosphate|sugar|carbohydrate|CoA|biotin|thiamine|TPP|ThDP|flavin|cation|anion)"
      r"[- ]binding (?:sites?|residues?|pockets?|loops?|motifs?|region|surface)|"
      r"(?:calcium|metal|zinc|iron|magnesium|cofactor|substrate|ligand|heme|haem|nucleotide|PLP|ATP|ADP|GTP|"
      r"NADP?H?|FAD|FMN|SAM|CoA|TPP|ThDP) (?:sites?|contacts?)\b|"
      # "a heme b pocket is lined by residues 45, 65, ..." names the ligand the pocket holds.
      r"(?:metal|zinc|iron|calcium|magnesium|cofactor|substrate|ligand|heme|haem|nucleotide|PLP|ATP|ADP|GTP|"
      r"NADP?H?|FAD|FMN|SAM|CoA|TPP|ThDP)(?: [a-z])? pockets?\b|"
      r"(?:ATP|ADP|GTP|NADP?H?|FAD|FMN|SAM|CoA|TPP|ThDP|heme|haem|metal|cofactor|substrate|ligand)[- ]contacting\b|"
      # "binds a GTP-loaded Rho-family GTPase" binds a protein, so the ligand word must end the noun.
      r"\bbinds? (?:a |the |two |one )?(?:divalent |catalytic |structural |single )?(?:Fe|iron|calcium|Ca2\+|zinc|"
      r"Zn2\+|Mg2\+|magnesium|Mn2\+|manganese|copper|heme|haem|FAD|NADP?H?|ATP|GTP|PLP|TPP|ThDP|cofactor|substrate|"
      r"metal)(?![\w-])" + NOT_BOUND_NOUN + "|"
      r"metal[- ]coordinat|coordinated (?:to|by) (?:the )?(?:calcium|zinc|metal|iron|magnesium)|" + METAL_SITE
    + "|" + LIGAND_BINDS + "|" + LIGAND_CONTACTS + "|" + NUCLEOTIDE_LOOP + "|" + HEME_ATTACHMENT + "|" + LIGAND_SETS
    + "|" + NUCLEOTIDE_MOTIF + "|" + METAL_VERBS
    # Round 5: binding nouns patterns.py lacks ("the binding cavity", "the binding positions scattered across 179 to
    # 419"); a bare one still takes its object from the word before it.
    + r"|\bbinding (?:cavit(?:y|ies)|chambers?|cups?|tunnels?|crevices?)\b|"
      # "positions" as a noun, not the verb ("This multivalent binding positions ubiquitinated substrates").
      r"\bbinding positions?(?=\s+(?:at|across|scattered|spread|from|over|include|including|lie|sit)\b|\s+\d|"
      r"\s*[,(:])|"
      r"(?<![\w-])(?:" + SMALL_LIGANDS + r")[- ]binding (?:" + SITE_NOUNS + r")\b|"
      r"\bbind(?:s|ing)?\s+(?:a |an |the |one |two |both )?(?:" + BOUND_LIGANDS + r")(?![\w-])",
    re.IGNORECASE)
# "iron-sulphur binding" is the British spelling of patterns.py's iron-sulfur.
PATTERNS["metal_binding"] = re.compile(rc.FEATURE_TYPES["metal_binding"][1].pattern + "|" + METAL_SITE + "|"
                                       + HEME_ATTACHMENT + "|" + METAL_BINDS + r"|iron[-–]sulphur|" + METAL_VERBS,
                                       re.IGNORECASE)
# "a homeobox protein (cdc4)" and "DUF3528, a homeobox-protein region" name a family, not a DNA-binding domain; "the
# core DNA-recognition module" is the DNA-binding domain.
PATTERNS["dna_binding"] = re.compile(
    rc.FEATURE_TYPES["dna_binding"][1].pattern.replace("homeobox", r"homeobox(?![- ]proteins?\b)")
    + r"|DNA[- ]recognition (?:modules?|domains?|regions?|motifs?)\b", re.IGNORECASE)
# "cytosolic loops at 156 to 179, 242 to 260" names the cytoplasmic side as plainly as "cytoplasmic loops", and so do
# "short extracellular turns at 90 to 113" and "a short cytoplasmic amino end at 1 to 11".
PATTERNS["topological_domain"] = re.compile(
    rc.FEATURE_TYPES["topological_domain"][1].pattern + r"|cytosolic (?:domain|loop|tail|segment)|"
    r"(?:extracellular|cytoplasmic|cytosolic|lumenal|luminal|periplasmic) turns?\b|"
    r"(?:extracellular|cytoplasmic|cytosolic|lumenal|luminal|periplasmic) (?:amino|carboxyl|N|C)[- ]"
    r"(?:terminal )?(?:ends?|terminus|termini|tails?)\b|" + SIDE_PREDICATE, re.IGNORECASE)
# "cysteines 99 and 191 are paired" states a disulfide without naming it.
PATTERNS["disulfide"] = re.compile(
    rc.FEATURE_TYPES["disulfide"][1].pattern
    + r"|\bcysteines?\b(?=(?:\s+at)?\s+\d{1,4}(?:\s*(?:,|and|or)\s*\d{1,4})*,?\s+(?:are|is|were)\s+paired\b)",
    re.IGNORECASE)

# What ends the scope of "rather than X" / "instead of X" before a later keyword: a new clause after a comma, or a
# parenthesis or dash. A list ("rather than catalytic triads, metal-binding clusters or ...") stays in scope.
RE_SCOPE_END = re.compile(
    r",\s*(?:and|but|so|because|since|as|which|that|while|whereas|with|where|when|there|then|consistent|meaning|"
    r"making|leaving|giving|I|we|it|its|this|these|those|the)\b|[()\[\]—]|\s[-–]\s", re.IGNORECASE)
RE_UNINTERRUPTED = re.compile(r"without\s+interruption", re.IGNORECASE)
# A negation patterns.RE_NEGATION lacks: "Nowhere do I find a membrane crossing, a signal peptide or ...".
RE_NOWHERE = re.compile(r"\bnowhere\b", re.IGNORECASE)
RE_NON_PREFIX = re.compile(r"\bnon[-‐‑]$", re.IGNORECASE)
# Round 5: "cytoskeletal protein-membrane anchor activity" is a GO term, "signal-peptide peptidase" an enzyme, "cleave
# the backbone in a single pass" an idiom, and "the catalytic his|tone deacetylase" a patterns.py keyword cut out of
# a longer word.
RE_FUNCTION_PHRASE = re.compile(r"transmembrane\s+(?:transport\b|transporter\s+activity|transporters?\s+complex|"
                                r"(?:electrochemical\s+|proton\s+|ion\s+)?gradients?\b|potential\b|"
                                r"receptor\s+(?:protein\s+)?(?:tyrosine|serine/threonine)\s+kinase\s+(?:signal\w*\s+)?"
                                r"pathway|signal(?:l)?ing\s+receptor\s+activity)|membrane anchor activity|"
                                r"signal[- ]peptide[- ]peptidases?", re.IGNORECASE)
# "cytoskeletal protein-membrane anchoring" is the GO term's activity.
RE_PROTEIN_MEMBRANE_LEAD = re.compile(r"protein-$", re.IGNORECASE)
RE_PASS_IDIOM = re.compile(r"\b(?:in|on)\s+(?:a|one)\s+$", re.IGNORECASE)
RE_PASS_HIT = re.compile(r"single[- ]pass", re.IGNORECASE)
RE_CUT_RESIDUE = re.compile(r"catalytic (?:Asp|Glu|His|Ser|Cys|Lys|Tyr|Thr|Arg)$", re.IGNORECASE)
RE_RESIDUE_WORD = re.compile(r"(?:%s)s?$|aspartic$|glutamic$|[a-z]+yl$" % "|".join(rc.AA_FULL), re.IGNORECASE)
RE_LETTERS = re.compile(r"[A-Za-z]+")
RE_GO_ID = re.compile(r"GO:\d{7}")
# The rest of a term name before its id: a few words, none of them a determiner, conjunction or verb of prose
# ("an N-terminal signal peptide and luminal ectodomain tethered at the GO:0005886 plasma membrane" is prose).
PROSE_WORDS = (r"the|a|an|at|and|or|with|via|is|are|was|were|that|which|this|its|for|to|by|from|through|as|so|thus|"
               r"supports?|supporting|implies|implying|suggests|indicates|gives?|places|fits|matches|means|consistent")
RE_GO_AFTER = re.compile(r"(?:[ \t-]+(?!(?:%s)\b)[\w'/+-]+){0,5}?[ \t]*\(?GO:\d{7}" % PROSE_WORDS, re.IGNORECASE)
RE_GO_BEFORE = re.compile(r"GO:\d{7}\)?[ \t:(-]*(?:(?!(?:%s)\b)[\w'/+-]+[ \t-]+){0,6}$" % PROSE_WORDS,
                          re.IGNORECASE)
# RE_GO_AFTER's word and separator characters: its id can only start where a run of them ends in "GO:" or meets
# "(GO:". Checking that first settles most keywords at once; on a hyphen-joined run ("M-S-Q-...") RE_GO_AFTER alone
# tries every split of the run and can take hours.
RE_GO_NAME_RUN = re.compile(r"[\w'/+ \t-]*")
# Round 5: a histone variant or a bracketed abbreviation may stand between the object and "binding" ("histone H3-binding
# residues", "the histone H3 binding site", "calmodulin (CaM)-binding motif").
RE_NON_LIGAND_BEFORE = re.compile(r"\b(?:%s|EF-hand)\b(?:[\s-]+(?:\([\w-]+\)|(?-i:[A-Z][A-Za-z]?\d[A-Za-z0-9]?)|tails?|"
                                  r"class|type))"
                                  r"{0,2}[- ]?$" % NOT_LIGAND, re.IGNORECASE)
# "binding sites in enhancers" and "enhancers carry its binding sites" are DNA sites the protein binds.
RE_DNA_SITES_AFTER = re.compile(r"\s+(?:in|on|within|across)\s+(?:the\s+|its\s+|target\s+|their\s+)?(?:[\w-]+\s+)?"
                                r"(?:enhancers?|promoters?|DNA|genomes?|chromatin|operators?|regulatory regions?)\b",
                                re.IGNORECASE)
RE_DNA_SITES_LEAD = re.compile(r"\b(?:enhancers?|promoters?|operators?|cis-elements?)\b[^.;,]{0,30}$", re.IGNORECASE)
# A hit that names no specific ligand takes its object from the word before it ("an acetyllysine binding site",
# "a peptide-ligand binding surface").
RE_BARE_BINDING = re.compile(r"(?:binding|ligand|substrate)\b", re.IGNORECASE)
RE_RRM_NAME = re.compile(r"nucleotide[- ]binding alpha[- ]beta plait", re.IGNORECASE)
# The heterotrimeric G-protein subunit names ("Guanine nucleotide-binding protein G(i) subunit alpha") are names.
RE_PROTEIN_NAME = re.compile(r"guanine nucleotide[- ]binding proteins?\b", re.IGNORECASE)
PROTEIN_NAME_LEAD = len("guanine ")
RE_SENTENCE_STOP = re.compile(r"[.!?](?:\s|$)|\n")
SENTENCE_REACH = 400
# The coil predictor is named in capitals; the word "coils" is no tool cue ("resolved more tightly into coils at").
RE_TOOL_CUE = re.compile(r"\b(?:predictors?|predicts?|predicted|prediction|calls?|called|flags?|flagged|tempting|"
                         r"TMHMM|Phobius|SignalP|DeepTMHMM|(?-i:COILS))\b", re.IGNORECASE)
RE_TOOL_AFTER = re.compile(r"^\s*(?:helix\s+)?(?:predictors?|caller|calls?|assignment|predictions?)\b", re.IGNORECASE)
RE_BUT = re.compile(r"\bbut\b", re.IGNORECASE)
RE_DISMISSAL = re.compile(r"\bnot\b|n't\b|rather than|instead|misread|mistak\w*|artefact|artifact|spurious|\bfalse\b|"
                          r"over-?call\w*|\bno (?:second|other|real|true|genuine)\b", re.IGNORECASE)
RE_DISMISSAL_TM = re.compile(r"\bh-region\b|hydrophobic core|signal peptide|cleavable signal|the signal\b|"
                             r"secretory signal", re.IGNORECASE)
# "the hydrophobic core is what the predictor mistakes for a membrane anchor" and "the predicted membrane crossings
# at 563 to 586 ... do not persuade me" dismiss without a "but".
RE_MISREAD = re.compile(r"\bmistak\w*|\bmisread\w*|\bmisinterpret\w*|\bconfuses\b", re.IGNORECASE)
RE_UNPERSUADED = re.compile(r"\b(?:do|does|did) not (?:persuade|convince|survive|hold)|\bunconvinc\w*|\bnot convinc\w*|"
                            r"\boverrides?\b|\b(?:is|are) not supported\b|\bunsupported\b|\bmispredict\w*|"
                            r"\bfalse positive\b", re.IGNORECASE)
# A predictor catching a signal peptide's own core ("a hydrophobic h-region over 9 to 24 that a helix predictor also
# catches as a membrane-spanning stretch").
RE_SIGNAL_CAUGHT = re.compile(r"(?:h-region|hydrophobic core|signal peptide|cleavable signal|the signal)\b[^.;]{0,80}?"
                              r"\b(?:that|which)\s+[^.;]{0,60}$", re.IGNORECASE)
# A signal's core taken for a helix by a tool, with or without a tool word ("the hydrophobic core of the signal read as
# a membrane anchor").
RE_SIGNAL_READ = re.compile(r"(?:h-region|hydrophobic core|signal peptide|cleavable signal|the signal)\b[^.;]{0,40}?"
                            r"\b(?:(?:mis)?read|called|flagged|scored|counted|taken|caught)\s+as\s+(?:an?\s+)?"
                            r"(?:[\w-]+\s+){0,2}$", re.IGNORECASE)
# What the model says a protein would need is not what it sees ("A membrane-bound terpene synthase would need a signal
# anchor or transmembrane segment and is not seen here").
RE_WOULD_LEAD = re.compile(r"\bwould\s+(?:need|require|carry|possess|contain|show|have\s+(?:a|an|the|no|any))\b",
                           re.IGNORECASE)
# A tool's helix identified with the signal's own core ("The transmembrane helix call at 7 to 29 is that same
# hydrophobic core seen by a different predictor").
RE_IDENTIFIED = re.compile(r"\b(?:is|are|reflects?|represents?)\s+(?:just\s+|simply\s+|really\s+|only\s+|merely\s+)?"
                           r"(?:that|the|this|its)\s+(?:same\s+)?(?:h-region|hydrophobic core|signal\b|signal peptide|"
                           r"cleavable signal)", re.IGNORECASE)
# What the model would need to see is not what it says it sees: "If this protein had ... a cofactor pocket ..., I would
# reconsider", "a demonstrated catalytic motif or a clear substrate pocket would change my mind", "too extensive ... to
# be a ligand pocket".
RE_IF = re.compile(r"\b(?:if|unless|whether)\b", re.IGNORECASE)
# A main clause after the condition ends it ("Whether it is cleaved matters for topology, and the basic residues ... are
# the positive-inside signature of an uncleaved signal anchor").
RE_MAIN_CLAUSE = re.compile(r",\s*(?:(?:and|but)\s+)?(?:I|we|the|this|it|so|then)\b|"
                            r"\b(?:because|since|although|though|whereas)\b", re.IGNORECASE)
RE_DEMONSTRATED = re.compile(r"\bdemonstrated\s+(?:[\w-]+\s+){0,2}$", re.IGNORECASE)
RE_WOULD_AFTER = re.compile(r"[^.;:,]{0,80}?\bwould\s+(?:(?:change|settle|make me|force|overturn|alter|shift|"
                            r"reverse)\b|(?:also\s+)?be\s+(?:\w+\s+)?(?:odd|unlikely|surprising|strange|implausible|"
                            r"inconsistent)\b)", re.IGNORECASE)
# "What would change my mind is evidence of a catalytic site in the middle domains or a transmembrane anchor".
RE_MIND_LEAD = re.compile(r"\b(?:change|changes|alter)\s+my\s+mind\b", re.IGNORECASE)
RE_TOO_TO_BE = re.compile(r"\btoo\b[^.;:]{0,80}?\bto be\s+(?:a|an|the)\s+(?:[\w-]+\s+){0,2}$", re.IGNORECASE)
# A guanine-nucleotide exchange or activating factor opens its client GTPase's nucleotide pocket ("binds Ras-GDP, pries
# open the nucleotide pocket"); the pocket is the partner's.
RE_NUCLEOTIDE_POCKET = re.compile(r"nucleotide(?: [a-z])? pockets?", re.IGNORECASE)
RE_GTPASE_PARTNER = re.compile(r"\b(?:GTPases?|Rabs?|Ras|Rho|Rac|Arfs?|Ran|G[- ]proteins?|EF-Tu|EF-G)\b|"
                               r"\bGDP[- ]bound\b", re.IGNORECASE)
PARTNER_REACH = 160
# Round 5: a pocket, cleft or domain a named partner owns ("the Hsp70 nucleotide-binding domain", "the protease
# substrate-binding cleft", "the EF-Tu nucleotide pocket").
RE_POCKET_NOUN = re.compile(r"(?:nucleotide|ATP|GTP|substrate)(?:[- ]binding)?(?: [a-z])? (?:pockets?|clefts?|grooves?|"
                            r"domains?)\b", re.IGNORECASE)
POCKET_VIEW = 30
RE_POCKET_OWNER = re.compile(r"\b(?:GTPases?|Rabs?|Ras|Rho|Rac|Arfs?|Ran|EF-Tu(?:\.GDP)?|EF-G|Hsp70|Hsp90|DnaK|"
                             r"chaperones?|proteases?|proteinases?|partners?|clients?|targets?|hosts?)(?:'s|’s)?\b",
                             re.IGNORECASE)
RE_POCKET_OF = re.compile(r"\s+of\s+(?:the\s+|its\s+|a\s+)?(?:partner|client|target|GTPases?|EF-Tu|EF-G|Ras|Rabs?|Rho|"
                          r"Rac|Arfs?|Ran|G[- ]proteins?|Hsp70|Hsp90|DnaK)\b", re.IGNORECASE)
# Only as the object of the protein's own act on the partner ("it inserts into the nucleotide-binding pocket", "binds
# the Hsp70 nucleotide-binding domain", "invades the nucleotide pocket of EF-Tu"), since a GTPase or chaperone says the
# same of its own pocket.
RE_PARTNER_VERB = re.compile(r"\b(?:binds?|binding|contacts?|engages?|docks?\s+(?:on|onto|against)|grips?|"
                             r"stimulates?|inserts?\s+into|inserting\s+into|invades?|invading|opens?|opening|"
                             r"pries?\s+open|enters?|occludes?|plugs?|blocks?|wedges?\s+into|acts?\s+on|recognizes?)\s+"
                             r"(?:the\s+|its\s+|their\s+|a\s+)?(?:[\w.'’-]+\s+){0,2}$", re.IGNORECASE)

# --- presence: what the said side reads as another molecule's, a peripheral anchor or a rejected call (round 5) ---
# The noun a keyword modifies names another molecule when it is a partner, a class in the plural, or a membrane
# machine's part ("type II and type I transmembrane serine/threonine kinase receptors", "transmembrane cargo", "the
# membrane-embedded Vo sector", "coiled-coil proteins"); singular "protein", "enzyme" or "receptor" stays the protein's
# own, since the model says that of itself.
NP_STOP = (r"and|or|of|in|on|at|that|which|with|to|from|is|are|was|were|the|a|an|its|this|by|for|as|but|not|no|than|"
           r"into|onto|through|via|whose|where|near|toward|towards|against|across|along|within|inside|around|beside|"
           r"between|among|over|under|after|before|during|upon|behind")
PREP_STOP = r"to|on|at|in|into|with|by|from|of|onto|through|via|and|or|that|which|for|as"
OTHER_NOUNS = (r"partners?|cargos?|cargoes|clients?|interactors?|substrates?|receptors|proteins|polypeptides|"
               r"precursors|subunits|permeases|transporters|enzymes|kinases|sensors|regulators|adaptors|adapters|"
               r"peroxins|nucleoporins|lipids?|sphingolipids?|phospholipids?|quinones?|intermediates|components")
MACHINE_NOUNS = r"sectors?|c-rings?|rings?|rotors?|motors?|turbines?|platforms?|apparatus|machinery|machines?|arms?"
# The keyword may be one of several coordinated modifiers ("coiled-coil and helical proteins", "crowded with
# coiled-coil, repeat and intrinsically disordered proteins").
# The noun ends the phrase: "a separate substrate-pocket set" has no substrate head. Only a modifier keyword may be
# coordinated with others before its noun; a keyword with a noun or a verb of its own heads no later noun ("binds zinc,
# and binds substrates", "the binding groove and thereby tune substrate selectivity").
COORDINATED = r"(?:\s*,\s*(?!(?:%s)\b)[\w-]+){0,3}(?:\s*,?\s+(?:and|or)\s+(?!(?:%s)\b)[\w-]+)?" % (NP_STOP, NP_STOP)
NOUN_PHRASE = r"(?:[\s/-]+(?!(?:%s)\b)[\w()+'-]+){0,3}?[\s/-]+" % NP_STOP
# A keyword with a noun of its own takes another noun only as a compound ("the catalytic core subunits", "the membrane
# anchor subunits"), not across later words ("the active site once both substrates are bound").
RE_OTHER_HEAD = re.compile(r"^[\s/-]+(?:%s)(?![\w-])" % OTHER_NOUNS, re.IGNORECASE)
RE_OTHER_HEAD_LIST = re.compile(r"^%s%s(?:%s)(?![\w-])" % (COORDINATED, NOUN_PHRASE, OTHER_NOUNS), re.IGNORECASE)
RE_OTHER_MACHINE = re.compile(r"^%s%s(?:%s)(?![\w-])" % (COORDINATED, NOUN_PHRASE, MACHINE_NOUNS), re.IGNORECASE)
RE_MODIFIER_HIT = re.compile(r"(?:transmembrane|embedded|spanning|integral membrane|pass|polytopic|bitopic|coiled-coil|"
                             r"coiled|binding|TMD?s?)$", re.IGNORECASE)
RE_VERB_HIT = re.compile(r"\b(?:binds?|bound|coordinat\w*|chelat\w*)\b", re.IGNORECASE)
# "the transmembrane segments of hydrophobic precursor proteins", "transmembrane helices of a client".
RE_OTHER_OF = re.compile(r"^(?:[\s-]+(?!(?:%s)\b)[\w()+'-]+){0,2}?\s+of\s+"
                         r"(?:the\s+|a\s+|an\s+|its\s+|their\s+|each\s+)?(?:[\w/-]+\s+){0,3}?(?:partners?|clients?|"
                         r"cargos?|cargoes|substrates?|precursors?|targets?|hosts?|receptors|proteins|channels|"
                         r"transporters|subunits|complexes)(?![\w-])" % NP_STOP,
                         re.IGNORECASE)
# A partner owns the keyword ("the client's transmembrane helices"), or a plural does, since the protein is "it"
# ("shields their hydrophobic membrane-spanning segments").
RE_OTHER_POSSESSIVE = re.compile(r"(?:\b(?:partner|client|cargo|host|target|substrate|precursor)(?:'s|’s)|\b\w+s['’]|"
                                 r"\btheir)\s+(?:[\w/-]+\s+){0,2}$", re.IGNORECASE)
# The keyword's noun phrase is said of another molecule ("Because the substrate is membrane-embedded", "a transmembrane
# protein is exactly the kind of partner an ankyrin repeat would grip").
RE_OTHER_SUBJECT = re.compile(r"\b(?:substrates?|cargos?|cargoes|clients?|partners?|interactors?|targets?)\b"
                              r"(?:\s+[\w-]+){0,3}\s+(?:is|are|was|were|remains?)\s+(?:an?\s+|the\s+)?"
                              r"(?:[\w-]+\s+){0,2}$", re.IGNORECASE)
RE_OTHER_PREDICATE = re.compile(r"^(?:[\s/-]+(?!(?:%s)\b)[\w()+'-]+){0,3}?\s+(?:is|are|was|were|would be|could be)\s+"
                                r"(?:(?:exactly|precisely|just|likely|probably)\s+)?"
                                r"(?:(?:the\s+kind\s+of\s+|an?\s+|the\s+)(?:[\w-]+\s+){0,2}?"
                                r"(?:partners?|contacts?|clients?|cargos?|targets?|substrates?)|"
                                r"(?:partners|clients|cargos?|cargoes|targets|substrates|adaptors))(?![\w-])" % NP_STOP,
                                re.IGNORECASE)
# The protein binds, docks onto or engages a molecule the keyword describes ("docked onto the transmembrane permease",
# "binding to a transmembrane transporter", "it engages a transmembrane receptor").
RE_PARTNER_ACT = re.compile(r"\b(?:binds?|binding|bound|docks?|docked|docking|engages?|engaging|contacts?|contacting|"
                            r"recruits?|recruited|tethers?|tethered|attaches?|attached|associates?|associated|"
                            r"interacts?|interacting|grips?|gripping|captures?|capturing|delivers?|delivered|presents?|"
                            r"presented|recognizes?|recognizing|couples?|coupled|acts\s+on|activates?|stimulates?)\s+"
                            r"(?:(?:it|them)\s+)?(?:(?:to|onto|on|with|against)\s+)?(?:(?:a|an|the|its)\s+)?"
                            r"(?:(?:(?!(?:%s|as|is|are|it)\b)[\w/-]+\s+){0,2}of\s+(?:a|an|the)\s+)?"
                            r"(?:(?!(?:%s|as|is|are|it)\b)[\w/-]+\s+){0,2}$" % (PREP_STOP, PREP_STOP), re.IGNORECASE)
RE_MOLECULE_HEAD = re.compile(r"^(?:\s+(?:and|or)\s+(?!(?:%s)\b)[\w-]+)?(?:[\s/-]+(?!(?:%s)\b)[\w()+'-]+){0,3}?"
                              r"[\s/-]+(?:proteins?|permeases?|transporters?|receptors?|kinases?|enzymes?|channels?|"
                              r"pumps?|subunits?|adaptors?|adapters?|complex(?:es)?|partners?)(?![\w-])"
                              % (NP_STOP, NP_STOP), re.IGNORECASE)
# "a transmembrane-like signal" and "a signal/TM-like segment" liken a stretch to the feature.
RE_LIKE_SUFFIX = re.compile(r"[-‐]like\b", re.IGNORECASE)
# "Transmembrane protein 231", "transmembrane protease serine 2" and "transmembrane 9 superfamily member 4" are
# partners' names; "transmembrane helix 1" numbers the protein's own helices.
RE_NAMED_PARTNER = re.compile(r"^\s+(?:[a-z]+\s+){0,2}(?:protein|protease|member|homolog)\s+(?:[a-z]+\s+){0,2}"
                              r"\d+[A-Za-z]?\b|^\s+\d+\s+(?:superfamily|family|homolog)\b", re.IGNORECASE)
# The keyword is the predicate of a partner subject ("The one contact I have to work with is a small integral membrane
# protein of the endoplasmic reticulum, SMIM12", "The partners point the same way: a transmembrane protein and ...").
# A contact is a partner only as such ("The one contact I have to work with", "The named contacts"), not as a ligand's
# ("the heme contacts: a histidine at 93 is the proximal iron ligand").
RE_PARTNER_SUBJECT = re.compile(r"\b(?:partners?|interactors?|(?:the|one|named|listed|known|other|protein|"
                                r"interaction)\s+contacts?)\b[^.;]{0,80}?(?:\b(?:is|are|was|were|include|"
                                r"includes|including|such as)\b|:)\s+(?:(?-i:[A-Z][\w-]*[A-Z0-9][\w-]*),\s+)?"
                                r"(?:(?:a|an|the|other|several|two|three)\s+)?"
                                r"(?:(?!(?:%s|is|are|was|were|has|have)\b)[\w/-]+\s+|(?:and|or)\s+(?:(?:a|an|the)\s+)?)"
                                r"{0,6}$" % PREP_STOP, re.IGNORECASE)
# Family membership keeps the keyword the protein's own ("one of the multi-pass membrane proteins", "like other
# transmembrane receptors", "belongs to the periplasmic substrate-binding proteins").
RE_FAMILY_LEAD = re.compile(r"\b(?:(?:members?|one|famil(?:y|ies)|class(?:es)?|group|subfamily|superfamily|kind|type)"
                            r"\s+of|belongs?\s+to|like|as\s+with|as\s+in|typical\s+of|among)\s+(?:the\s+|a\s+|an\s+|"
                            r"these\s+|other\s+|many\s+|most\s+)?(?:[\w/-]+\s+){0,3}$|"
                            r"\bfamily\s+(?:proteins|members)\s+(?:are|have|share|form)\b[^.;]{0,60}$", re.IGNORECASE)
OTHER_REACH = 120
# A "membrane anchor" a lipid, an amphipathic helix or a lipid-binding domain makes is no transmembrane segment ("a
# fourteen-carbon acyl chain ... is a membrane anchor", "an amino-terminal C2 membrane anchor", "a calcium-dependent
# membrane anchor"), nor one the sentence calls peripheral ("The hydrophobic carboxyl tail is the membrane anchor, ... a
# peripheral membrane protein that inserts a tail into one leaflet rather than crossing the bilayer").
RE_ANCHOR_HIT = re.compile(r"membrane anchor", re.IGNORECASE)
RE_ANCHOR_KIND = re.compile(r"\b(?:C2|PH|ENTH|ANTH|ENTH/ANTH|F-BAR|BAR|FYVE|PX|GRAM|amphipathic|lipid\w*|acyl\w*|"
                            r"myristoyl\w*|palmitoyl\w*|prenyl\w*|farnesyl\w*|geranylgeranyl\w*|isoprenyl\w*|GPI|"
                            r"glycosylphosphatidylinositol|peripheral\w*|calcium-dependent|Ca2\+-dependent|"
                            r"electrostatic|polybasic|monotopic)\b", re.IGNORECASE)
RE_ANCHOR_DENIAL = re.compile(r"\bperipheral\w*|\bmonotopic\b|\bone leaflet\b|\bin-plane\b|\bamphipathic\b|"
                              r"\b(?:rather than|without|not|instead of|never)\s+(?:\w+\s+){0,2}"
                              r"(?:crossing|spanning|traversing)\b", re.IGNORECASE)
ANCHOR_LEAD = 90
# A call the model sets against another reading: the alternative after "than" ("better read as a buried hydrophobic
# helix than as a true transmembrane anchor"; not a count, "more than one transmembrane helix"), what it rejects or says
# is mistaken for the feature ("the one reading I reject is the transmembrane assignment", "long enough to be mistaken
# for a transmembrane helix").
RE_THAN_LEAD = re.compile(r"(?<!broader )(?<!wider )(?<!larger )(?<!smaller )(?<!longer )(?<!shorter )(?<!bigger )"
                          r"(?<!narrower )(?<!greater )\bthan\s+(?:as\s+)?(?!(?:one|two|three|four|five|six|seven|"
                          r"eight|nine|ten|eleven|twelve|\d+|a few|several|half)\b)(?:an?\s+|the\s+|any\s+)?"
                          r"(?:[\w/-]+\s+(?:or\s+)?){0,3}$", re.IGNORECASE)
RE_REJECT_LEAD = re.compile(r"\breject\w*\b[^.;:]{0,60}$|\bmistak\w*\s+(?:for|as)\s+(?:an?\s+|the\s+)?"
                            r"(?:[\w-]+\s+){0,2}$|\bmisread\w*\s+as\s+(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,2}$|"
                            r"\bmimic\w*\s+(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,2}$", re.IGNORECASE)
# Negations patterns.RE_NEGATION lacks ("the disordered charged tails argue against a membrane-embedded fate",
# "compositionally incapable of building an active site", "explicitly excluding a transmembrane anchor", "too short and
# too cysteine-dense to house a catalytic core", "too long and too regular for a small ligand pocket").
RE_EXTRA_NEGATION = re.compile(r"\bargue\s+against\b|\bincapable\b|\bexclud(?:e|es|ed|ing)\s+(?:a|an|any)\b|"
                               r"\btoo\s+(?!many\b|much\b|few\b)[\w-]+|\bwrong\b[^.;,]{0,20}\bfor\b|"
                               r"\b(?:have|has|had)\s+lost\b", re.IGNORECASE)
# The protein said to be or to form the noun keeps it ("it forms a membrane-embedded ring").
RE_SELF_PREDICATE_LEAD = re.compile(r"\b(?:it|this protein|this chain|the chain|the protein|this subunit|this one)\s+"
                                    r"(?:\w+\s+){0,2}?(?:is|forms?|assembles?\s+into|oligomeri[sz]es?\s+into|"
                                    r"serves?\s+as|acts?\s+as|functions?\s+as)\s+(?:a|an|the|one)?\s*"
                                    r"(?:[\w/-]+\s+){0,2}$", re.IGNORECASE)
# A negation ends where a new main clause with its own subject and verb begins (", and the decisive chemistry is the
# basic PtdIns(4,5)P2-binding site"), or where a comma closes the negated phrase before the verb ("rich in charged and
# polar side chains with no hydrophobic core signature, fits a ligand-binding cup"), unless the new clause carries a
# negation of its own or the keyword's predicate is negated after it ("the substrate-binding surface is not marked").
CLAUSE_VERBS = (r"is|are|was|were|has|have|fits?|forms?|makes?|gives?|places?|points?|reads?|sits?|lies|lie|carries|"
                r"carry|holds?|marks?|defines?|builds?|contains?|creates?|provides?|binds?|completes?|matches|"
                r"suits?|explains?|indicates?|suggests?|supports?|implies|fixes|settles?|remains?|becomes?|tells?")
RE_NEW_CLAUSE = re.compile(r",\s+and\s+(?:the|this|its|that|these|those)\b", re.IGNORECASE)
RE_CLAUSE_VERB = re.compile(r"\b(?:%s)\b" % CLAUSE_VERBS, re.IGNORECASE)
# The comma closes a negated phrase only after "with no" or "without"; a negated subject ("Nothing on this chain
# crosses a membrane, carries a signal peptide") governs every verb of its list.
RE_COMMA_VERB = re.compile(r",\s+(?=(?:is|are|was|were|fits|sits|lies|remains|becomes|suggests|indicates|implies)\b)",
                           re.IGNORECASE)
RE_WITH_NEGATION = re.compile(r"\b(?:with\s+(?:no|little)|without)\s*$", re.IGNORECASE)
RE_TRAILING_NEGATION = re.compile(r"^[^.;:,]{0,60}?\b(?:(?:is|are|was|were|does|do)\s+not|cannot|"
                                  r"(?:is|are)\s+absent)\b", re.IGNORECASE)
RE_CLAUSE_STOP = re.compile(r"[.;!?](?:\s|$)|\n")
# A tool's call reported and then rejected later in the paragraph ("the predictors split between a cleavable signal
# and a transmembrane anchor on that same stretch, ... a lipid-anchored protein ... rather than a transmembrane one");
# a limited denial ("no second transmembrane helix") rejects nothing.
DENIAL_REACH = 1500
RE_LIMITED_DENIAL = re.compile(r"\b(?:second|other|another|further|additional|more|extra|separate)\b", re.IGNORECASE)
# Final round: "not merely X but Y" affirms X, and a denial does not reach past ", so" or "and for" into a new statement
# ("no zinc finger and no obvious protease or kinase motif, so the ... most ordered element is a C-terminal coiled
# coil"; "argues against a compact catalytic core and for extended ... coiled-coil contacts").
RE_NOT_MERELY = re.compile(r"(?:(?:is|are|was|were|does|do)\s*)?n(?:o|['’])t\s+(?:merely|only|just|simply)\b",
                           re.IGNORECASE)
RE_DENIAL_RELEASE = re.compile(r",\s*so\b|\band\s+for\b", re.IGNORECASE)
# A receiver domain's phosphorylated aspartate is a modification, not an active site ("an aspartate that can carry a
# 4-aspartylphosphate modification"); a P-type ATPase's covalent intermediate is one.
RE_ASPARTYL = re.compile(r"aspartyl[- ]?phosphate", re.IGNORECASE)
RE_INTERMEDIATE_AFTER = re.compile(r"^[\s-]*(?:[\w-]+\s+){0,1}intermediate", re.IGNORECASE)
RE_RECEIVER = re.compile(r"\breceivers?\b|response regulators?|phospho-?acceptor|phosphorelay|His-to-Asp|"
                         r"histidine kinases?|histidine phosphotransfer|phospho-?histidine", re.IGNORECASE)

# --- spans ---
# Final round: a primed digit is a nucleic-acid or sugar locant ("binds ssDNA at 5′-terminated regions", "3'-to-5'
# polarity", "the 2'-hydroxyl"), not a residue.
NOT_LOCANT = r"(?![1-5][′'’])"
# Round 5: "eight scattered positions, 53, 98, 99, 100" and "scattered residues including 98, 99, 100" are lists too.
RE_POSITION = re.compile(r"\b(?:(?:residues?|positions?)(?:\s*,\s*|\s+including\s+|\s+)|(?:at|is|are)\s+)"
                         r"(%s\d{1,4}(?:\s*,\s*\d{1,4})*(?:\s*,?\s*and\s+\d{1,4})?)" % NOT_LOCANT, re.IGNORECASE)
# A number in a degenerate counting run ("manganese is bound at 10, 11, 12, ..., 367") is no placement, as in the
# residue metric (counting_runs).
RUN_LABEL = ""
# Topology nouns after a crossing list head their own items ("crossings: 21-41, ..., with the amino end 1-20 and the
# long carboxyl end 203-319 on the cytoplasmic side", "with short outside loops at 478-503").
RE_TOPOLOGY_PART = re.compile(r"\b(?:amino|carboxyl|N|C)[- ]?(?:terminal\s+)?(?:ends?|terminus|termini|tails?)\b|"
                              r"\b(?:N|C)-terminus\b|\bloops?\b|\bturns?\b|\btails?\b", re.IGNORECASE)
# A family entry's name does not place the keyword it holds ("IPR050463 (Distal-less homeobox transcription factors
# family) spanning residues 12-286" is the lineage's extent); the name's head decides, so "Glycoside hydrolase, family
# 16, active site" is an active site.
RE_ENTRY_END = re.compile(r"[)\]\[]|,\s*(?:residues?|positions?)\b|\s+\((?:domain|family|homologous_superfamily|repeat|"
                          r"conserved_site|active_site|binding_site|ptm|site)\)", re.IGNORECASE)
RE_FAMILY_END = re.compile(r"\bfamil(?:y|ies)[\s,.;:]*$", re.IGNORECASE)
RE_FAMILY_LABEL = re.compile(r"\s+\(family\)", re.IGNORECASE)
# A chain-length phrase around a position at the last residue ("the chain is 856 residues", "ends at 211", "the
# carboxyl terminus at 856"); the last residue in a site list is a site ("catalytic residues at 845, 1019, 1047 and
# 1049").
RE_LENGTH_AFTER = re.compile(r"\s*(?:residues?|amino[- ]acids?|aa)\b", re.IGNORECASE)
RE_LENGTH_BEFORE = re.compile(r"(?:\b(?:length|long|size)\s+(?:of\s+|is\s+)?|"
                              r"\b(?:ends?|terminat\w+|stops?)\s+(?:at\s+)?|"
                              r"\b(?:carboxyl|C)[- ]?termin(?:us|al\s+(?:end|residue))\s*(?:at|is|,)?\s*|"
                              r"\bto\s+the\s+(?:end|carboxyl\s+terminus)\s+(?:at\s+)?|\bchain\s+(?:is|of|runs\s+to)\s+)"
                              r"(?:residue\s+)?$", re.IGNORECASE)
LENGTH_REACH = 40
RE_ACCESSION = re.compile(r"IPR\d{6}")
# The keyword's own accession in apposition, "a putative metal-binding site (IPR024722, residues 234-245)" or "a
# coiled-coil domain (IPR...": at most a few words of the keyword's noun phrase lie between them, none of them a place
# ("a coiled-coil domain, sitting inside IPR043441"); after a comma the accession must be cited, not the subject of a
# verb of its own ("as P-loop nucleoside-triphosphate hydrolase, IPR001609 marks the myosin head").
# A passive lead names the keyword's entry too ("explicit coiled-coil propensity is indicated by IPR011054", "The
# catalytic metal-binding features are pinpointed by IPR011605").
RE_OWN_ACCESSION_LEAD = re.compile(r"^[\s,(:—–-]*(?:[A-Za-z][\w'-]*[\s,(:—–-]+){0,3}$|"
                                   r"^\s*(?:[A-Za-z][\w'-]*\s+){0,2}(?:is|are|was|were)\s+[a-z]+ed\s+by\s+$")
RE_LEAD_PLACE = re.compile(r"\b(?:inside|within|in|under|sitting|nested|beneath|across)\b", re.IGNORECASE)
RE_CITED_AFTER = re.compile(r"\s*(?:[(),\[\]]|(?:residues?|positions?|at|over|across|from|spanning)\b)", re.IGNORECASE)
# A keyword inside an InterPro entry's name ("IPR001440 Coiled-coil vimentin, conserved site, residues 1568-1584",
# "IPR036335: Signal peptide, alpha-helical domain superfamily [800-820]") owns that entry's citation; name words may
# start with a digit and the keyword may be glued to the word before it ("IPR003015 (2-oxo acid dehydrogenase,
# lipoyl-binding site, conserved site, residues 94-103)").
RE_ENTRY_NAME_LEAD = re.compile(
    r"^\s*[:(\[]?\s*(?:(?!(?:marks?|names?|gives?|claims?|places?|covers?|spans?|is|are|recurs?|bounds?|reaches|"
    r"runs?|sits?|lies|as|at|from|over|across|and|with)\b)[A-Za-z\d][\w'/()+-]*[\s,]+){0,6}"
    r"(?:[A-Za-z\d][\w'/()+]*-)?$", re.IGNORECASE)
# A family or superfamily the keyword modifies and the model calls broader places the family, not the feature
# ("IPR013083 marks the broader RING/FYVE/PHD zinc-finger superfamily at 197 to 269, ..., IPR019787 and IPR001965
# place the PHD finger at 212 to 263"): the family noun heads the coordinate through cues alone.
RE_BROAD_LEAD = re.compile(r"\b(?:broader|wider|generic|umbrella|overarching|fold-level)\b(?:[\s/-]+[\w/()+'-]+){0,3}"
                           r"[\s/-]*$", re.IGNORECASE)
RE_FAMILY_PHRASE = re.compile(
    r"[\s-]*(?:(?!(?:of|in|on|at|from|with|within|for|to|by|as|like|typical|than|the|a|an|and|or|but|which|that|is|"
    r"are)\b)[\w/()+'-]+\s+){0,4}?(?:homologous\s+)?(?:super)?famil(?:y|ies)\b"
    r"(?:[\s,(\[]|\bIPR\d{6}\b|\b(?:at|over|across|from|spanning|residues?|positions?)\b)*$", re.IGNORECASE)
BROAD_REACH = 60
RE_NAMES_NEXT = re.compile(r"^[\s,(:]*$")
# A family named by an entry the keyword sits in, then dismissed in the same sentence, places nothing: "IPR027417
# (P-loop containing nucleoside triphosphate hydrolase homologous superfamily, residues 1-25), but this span is short
# ...; in type II Golgi enzymes this signal reflects a generic P-loop-like motif rather than an NTPase function".
ENTRY_REACH = 90
# An accession right after a coordinate labels that coordinate ("a copper-binding motif at 711 to 724 (IPR049947)
# with copper ligands at 551"), so it does not claim the coordinates after it.
RE_TRAILING_ACCESSION = re.compile(r"\d\s*[(\[,]?\s*$")
# What may join an entry's coordinates to the next one of the same citation ("IPR036640 [252-585] [600-700]",
# "that claims 14 to 127 and again 24 to 99").
RE_ENUM_GLUE = re.compile(r"^[\s,\[\]()–-]*(?:(?:and|or|to|through|residues?|positions?|at|over|from|again|also|then)"
                          r"\b[\s,\[\]()–-]*)*$", re.IGNORECASE)
# Where an entry's citation starts: its first coordinate after a cue or in a range. Digits before it belong to the
# entry's name ("IPR049328 (CD39, conserved site, residues 206-210)", "Coiled-coil domain-containing protein 24/35").
RE_CITATION_START = re.compile(r"(?:\b(?:residues?|positions?|at|over|across|from|spanning|between|claims?|covers?)\s+"
                               r"|[\[(]\s*)(?=\d)|\b\d{1,4}\s*(?:-|–|—|to|through)\s*\d", re.IGNORECASE)
RE_CITATION_BREAK = re.compile(r"[.;!?]\s|\n")
# A new clause after an accession ends its statement before any coordinate of its own ("superfamily IPR036436 beneath
# it, and it is densely cross-linked by disulfide bonds at 304 to 384").
RE_CLAUSE_START = re.compile(r",\s*and\b|\band\s+(?:then\s+)?(?:a|an|the|each|it|its|this|these|those)\b|;\s|"
                             r"\b(?:while|whereas|but)\b", re.IGNORECASE)
# The end of a backward accession's citation: its name in brackets and the rest of its coordinate list.
RE_CITATION_TAIL = re.compile(r"^\s*(?:\([^()]*\)|\[[^\]]*\])?(?:[\s,\[\]()–-]*(?:(?:and|or|to)\s+)?\d{1,4})*"
                              r"[\s)\]—–]*")
# An accession between a coordinate and a later keyword is the keyword's own only in apposition ("residues 85-178 form
# IPR002487 (Transcription factor, K-box), a coiled-coil module"), not across a sequence phrase ("(residues 1-24)
# carries IPR040920 (...), immediately followed by a trio of signal peptide annotations").
# A verb that names what the entry is counts too ("IPR018076 (..., residues 167-176) pinpoints the metal-binding
# signature", "IPR033145 (initiates catalysis, residues 487-492) together mark the catalytic core"), and so do its
# participles ("(..., residues 428-437) corresponding to the cysteine heme-iron ligand motif", "IPR014784 at 213 to 225
# marking the two copper-binding loops").
APPOSITION_VERBS = (r"is|are|forms?|as|marks?|defines?|constitutes?|pinpoints?|identif(?:y|ies)|provides?|names?|"
                    r"represents?|encodes?|corresponds?\s+to|(?:appears?|serves?|acts?|reads?|functions?)\s+as|"
                    r"corresponding\s+to|marking|denot(?:e|es|ing)|pinpointing|defining|forming|representing|"
                    r"encoding|naming|identifying|constituting|adds?")
RE_APPOSITION = re.compile(
    r"^\s*(?:,\s*(?:(?:a|an|the)\s+)?(?:(?!(?:followed|preceded|then|next|after|before|downstream|upstream|beside|"
    r"adjacent|flanking|so|thus|hence|therefore|which|and|but|while|whereas)\b)[\w/'()+-]+\s+){0,5}|"
    r"(?:,\s*)?(?:which\s+)?(?:(?:together|thus|therefore|also|directly|further|clearly|collectively|here)\s+)?"
    r"(?:%s)\s+(?:(?:a|an|the)\s+)?(?:[\w/'()+-]+\s+){0,3}|"
    # "IPR009132 ... at 71 to 81, 104 to 116 and 296 to 309, which are the first intracellular loop, the second
    # extracellular loop and the third extracellular loop".
    r",\s*which\s+(?:is|are)\s+(?:(?!(?:followed|preceded|then|next|after|before|downstream|upstream|beside|adjacent|"
    r"flanking|within|inside|in|on|near|between)\b)[\w/'()+-]+,?\s+){0,12})[\w/'()+-]*$" % APPOSITION_VERBS,
    re.IGNORECASE)
CITATION_REACH = 120
# The keyword's own accession's bracketed entry name at the start of a forward gap ("a metal-binding catalytic motif
# IPR006130 (3-dehydroquinate synthase, conserved site, residues 159-169)"): its words name the entry, not a rival
# element. A family name stays a rival ("enzyme scaffold: IPR045054 (Prolyl 4-hydroxylase family, residues 38-263)").
RE_OWN_BRACKET = re.compile(r"^(\W*(?:[A-Za-z][\w'/-]*\W+){0,3}?IPR\s*[(\[:,]\s*)([^()\[\]]*?)"
                            r"(?=,\s*(?:residues?|positions?)\b|\s*[)\]])")
# A list joiner between two cited accessions ("IPR013110 (..., helix 2, residues 706-774) and IPR057588 (...)").
RE_JOINER = re.compile(r"^[\s,]*(?:and|or)?\s*$", re.IGNORECASE)
# A coordinate written after a residue name ("under IPR017972, with cysteine 451 named as the axial heme iron ligand")
# names that residue, not the start of the entry's citation.
RE_RESIDUE_END = re.compile(r"\b(?:%s)s?[\s-]*$" % AA_WORDS, re.IGNORECASE)
# A number glued to a three-letter residue name is a coordinate ("Asp42"); one glued to other letters or digits is part
# of a name ("CD39", "P450").
RE_COORD = re.compile(r"(?:(?<=\b(?:%s))|(?<![A-Za-z\d]))%s\d{1,4}(?![A-Za-z\d]|-residue|-amino| residues? long|"
                      r" amino acids long)" % ("|".join(c.title() for c in rc.AA3), NOT_LOCANT))
RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
COUNT = r"(\d{1,4}|(?:%s)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)" % "|".join(NUMBER_WORDS)
RE_FIRST_N = re.compile(r"\b(?:first|N-terminal|amino[- ]terminal|leading|initial)\s+(?:~|about |roughly |some )?"
                        + COUNT + r"\s+(?:residues|amino acids)\b", re.IGNORECASE)
RE_LAST_N = re.compile(r"\b(?:last|final|C-terminal|carboxyl[- ]terminal|carboxy[- ]terminal)\s+"
                       r"(?:~|about |roughly |some )?" + COUNT + r"\s+(?:residues|amino acids)\b", re.IGNORECASE)
# "between 34 and 78" is a pair only after a bond word; "ten cysteines between 54 and 163" is a range.
RE_PAIRS = (
    re.compile(r"\b(?:disulfide|disulphide|bond|bridge|cystine|link)s?\s+(?:bonds?\s+)?between\s+"
               r"(?:cysteines?\s+|Cys[- ]?)?(\d{1,4})\s+and\s+(?:cysteine\s+|Cys[- ]?)?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\b(?:cysteines?\s+|Cys[- ]?)?(\d{1,4})\s+with\s+(?:cysteine\s+|Cys[- ]?)?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\bCys[- ]?(\d{1,4})\s*[-–—/]\s*Cys[- ]?(\d{1,4})\b"),
    # "A disulfide bond pairs cysteines 313 and 346", "one joining cysteines 41 and 145", "with the paired cysteines at
    # 37 and 107", "cysteines at 28 and 44 bonded to each other", "cysteines 99 and 191 are paired".
    re.compile(r"\b(?:pairs?|pairing|paired|links?|linking|joins?|joining|connects?|connecting|bridges?|bridging|bonds|"
               r"ties|tying|staples?|stapling|crosslinks?|crosslinking)\s+(?:the\s+)?(?:cysteines?\s+|Cys[- ]?)?"
               r"(?:at\s+)?(\d{1,4})\s*,?\s*and\s+(?:cysteine\s+|Cys[- ]?)?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\bcysteines?\s+(?:at\s+)?(\d{1,4})\s+and\s+(\d{1,4}),?\s+(?:(?:are|is|were)\s+)?(?:disulfide[- ])?"
               r"(?:paired|bonded|linked|joined|bridged)\b", re.IGNORECASE),
)
# Further pairs listed after a pair verb ("four disulfide bonds pair cysteines at 31 and 110, 64 and 69, 116 and 305,
# and 194 and 218").
RE_PAIR_VERB = RE_PAIRS[3]
RE_PAIR_NEXT = re.compile(r"\s*,\s*(?:and\s+)?(\d{1,4})\s+and\s+(\d{1,4})\b(?!\s*(?:-|–|to)\s*\d)", re.IGNORECASE)
LIST_WORDS = {
    "and", "to", "through", "until", "residues", "residue", "positions", "position", "at", "from", "over", "across",
    "spanning", "spans", "between", "or", "with", "plus", "then", "also", "about", "around", "near", "a", "an", "the",
    "its", "is", "are", "as", "of", "in", "on", "by", "each", "both", "pairs", "pair", "bonds", "bond", "this",
    "n-region", "h-region", "c-region", "region", "regions", "core", "cap", "short", "long", "polar", "basic",
    "hydrophobic", "charged", "helix", "helices", "helical", "segment", "segments", "stretch", "stretches", "second",
    "third", "fourth", "another", "further", "more", "finger", "fingers", "repeat", "repeats", "unit", "units",
    "first", "aspartic", "glutamic", "acid", "acids",
} | {word + suffix for word in list(rc.AA3) + list(rc.AA_FULL) for suffix in ("", "s")}
# A residue list naming its amino acids ("a catalytic triad of aspartate 268, glutamate 298 and aspartate 365") is one
# list, so the names are list words above. Metal sites are binding sites too, so the two types never rival each other;
# nor do they rival a zinc finger, whose zinc ligands are both ("four zinc ligands at 179, 182, 201 and 207, so that
# module is a real cross-braced zinc finger").
COMPATIBLE = {"binding_site": {"metal_binding", "zinc_finger"}, "metal_binding": {"binding_site", "zinc_finger"},
              "zinc_finger": {"binding_site", "metal_binding"}}
# Which types a phrase in a list may name without heading a rival list. Every metal site is a binding site but not the
# reverse: "a metal-binding subset at 162, 283, 285 and 434, and a phosphate-binding subset at 244, 285 and 435" ends
# the metal list.
# A helix is a transmembrane word, but also the h-region of a signal, a coiled coil's or a DNA-binding element's own.
# Signal and transit peptides share their part names (n-, h- and c-region).
SEGMENT_ALLOWED = {"binding_site": {"binding_site", "metal_binding"}, "metal_binding": {"metal_binding", "zinc_finger"},
                   "zinc_finger": {"zinc_finger", "metal_binding"}, "coiled_coil": {"coiled_coil", "transmembrane"},
                   "signal_peptide": {"signal_peptide", "transit_peptide", "transmembrane"},
                   "transit_peptide": {"transit_peptide", "signal_peptide", "transmembrane"},
                   "dna_binding": {"dna_binding", "transmembrane"}}
# A list continues past nouns that name no other feature: generic nouns and quantity words ("plus a scattered set
# across the C-terminal lobe at 323 to 326") and the type's own nouns ("with sub-segments at 139 to 166", "closed into
# three pairs, 4 to 45, 6 to 35").
GENERIC_WORDS = {
    "set", "sets", "subset", "subsets", "cluster", "clusters", "lobe", "lobes", "part", "parts", "portion", "portions",
    "half", "halves", "group", "groups", "run", "runs", "block", "blocks", "array", "series", "string",
    "constellation", "collection", "span", "patch", "patches", "end", "ends", "ones", "rest", "scattered", "putative",
    "conserved", "wider", "additional", "several", "few", "multiple", "single", "last", "final", "next", "other",
    "remaining", "shorter", "longer", "smaller", "larger", "sub-segment", "sub-segments", "subsegment", "subsegments",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "within", "including", "inside", "along", "amino", "carboxyl", "terminal",
    "n-terminal", "c-terminal", "amino-terminal", "carboxyl-terminal",
}
# Lower case, as glue compares them.
TYPE_WORDS = {
    "transmembrane": {"crossing", "crossings", "pass", "passes", "tm", "s"},
    "coiled_coil": {"coil", "coils", "coiled", "heptad", "heptads", "zipper"},
    "disulfide": {"paired", "bonded", "bridge", "bridges", "cystine", "cystines", "link", "links", "linked"},
    "binding_site": {"contact", "contacts", "ligand", "ligands", "site", "sites", "pocket", "coordinating"},
    "metal_binding": {"contact", "contacts", "ligand", "ligands", "site", "sites", "coordinating", "liganding"},
    "active_site": {"site", "sites", "triad", "dyad", "tetrad", "nucleophile", "base", "donor", "acceptor"},
    "glycosylation": {"sequon", "sequons", "glycan", "glycans", "site", "sites"},
    "modified_residue": {"site", "sites", "phosphosite", "phosphosites"},
    "zinc_finger": {"knuckle", "knuckles", "module", "modules"},
    "topological_domain": {"loop", "loops", "tail", "tails", "domain", "domains", "terminus"},
}
# Words that make a phrase name a feature of a type without being one of its keywords ("a catalytic set at 111",
# "a phosphate-pocket cluster at", "the topology they impose is asymmetric: only 1 to 11"); a phrase carrying one of
# another type's words, and none of its own, heads a rival list. Non-ligand binding (DNA, protein) rivals every type
# but DNA binding.
# The words before "contact" that make it a contact with DNA rather than with a ligand.
DNA_CONTACT_WORDS = ("DNA", "RNA", "base", "groove", "specific")
LIGAND_WORDS = (r"ATP|ADP|AMP|GTP|GDP|NADP?H?|NAD\(P\)H?|FAD|FMN|SAM|CoA|TPP|ThDP|heme|haem|substrate|cofactor|"
                r"nucleotide|phosphate|sugar|metal|zinc|calcium|magnesium|iron|copper|manganese|Mg|Zn|Ca|Fe|Cu|Mn")
TYPED_WORDS = {
    # The catalytic roles stated as predicates ("uses the histidine at 39 with aspartate 63 to activate water").
    "active_site": re.compile(r"\bcatalytic\b|\bnucleophil\w*|\boxyanion\b|transition[- ]state|\bacid[- ]base\b|"
                              r"\bproton[- ](?:accept|donat)\w*|\bgeneral (?:acid|base)\b|"
                              r"\bactivat\w* (?:the |a )?(?:catalytic |nucleophilic )?water\b|"
                              r"\baccepts? and donates? protons?\b|\bdeprotonat\w+", re.IGNORECASE),
    "metal_binding": re.compile(r"\b(?:zinc|calcium|magnesium|manganese|iron|copper|cobalt|nickel|metals?|"
                                r"(?-i:Zn|Ca|Mg|Mn|Fe|Cu))(?:\(?\d?\+\)?)?(?![\w(])", re.IGNORECASE),
    # A contact with DNA bases or grooves is DNA binding ("the base-contact residues at 297, 300, 301 and 304 place the
    # recognition helix").
    "binding_site": re.compile(
        r"\b(?!(?:%s)\b)[\w()+/]+[- ](?:binding|contacting)\b|%s\bcontacts?\b|\bligands?\b|\bpockets?\b|"
        r"\b(?:%s)(?:\(?\d?\+\)?)?(?:[- ]\w+)?[- ](?:surface|sites?|set|cluster|contacts?|residues|grip|cleft|groove)\b"
        % (NOT_LIGAND, "".join(r"(?<!%s[- ])" % word for word in DNA_CONTACT_WORDS), LIGAND_WORDS), re.IGNORECASE),
    "disulfide": re.compile(r"\bdisulfide|\bdisulphide|\bcystines?\b|\bpaired cysteines\b|\bcysteine pairs?\b",
                            re.IGNORECASE),
    "glycosylation": re.compile(r"\bglycans?\b|\bglycosyl(?!phosphatidyl)\w*|\bsequons?\b|\b[NO][- ]linked\b|"
                                r"\bN-glycan\w*", re.IGNORECASE),
    # Not "hydroxylated": "a serine-rich, hydroxylated stretch" describes a transit peptide's composition.
    "modified_residue": re.compile(r"\bphosphorylat\w*|\bphospho(?:serine|threonine|tyrosine)s?\b|\bacetylat\w*|"
                                   r"\bmethylat\w*|\bsuccinyl\w*|\bubiquitinat\w*|\bsumoylat\w*|\bcitrullinat\w*|"
                                   r"\bmodified\b|\bPTMs?\b|\bcrosslink\w*|\bSUMO\w*", re.IGNORECASE),
    # Not "prenyl": "prenylated FMN is bound at 139 to 141" names a cofactor.
    "lipidation": re.compile(r"\bmyristoyl\w*|\bpalmitoyl\w*|\bfarnesyl\w*|\bgeranylgeranyl\w*|\bGPI\b|\blipidat\w*|"
                             r"glycosylphosphatidylinositol|\bomega[- ]site\b", re.IGNORECASE),
    # S1 to S6 name a channel's helices when they carry a span ("S3 at 250 to 270"), not in "the S1 pocket".
    "transmembrane": re.compile(r"\bcrossings?\b|\bTM\d+\b|membrane[- ]spanning|\bhelix\b|\bhelices\b|"
                                r"\b(?-i:S[1-6])(?=\s+(?:at|from|over|spanning|runs)\b)", re.IGNORECASE),
    "coiled_coil": re.compile(r"\bcoil(?:s|ed)?\b|\bheptads?\b|\bzipper\b", re.IGNORECASE),
    "signal_peptide": re.compile(r"\bsignal\b|\b[nhc]-regions?\b|\bcleavage\b", re.IGNORECASE),
    "transit_peptide": re.compile(r"\btransit\b|\bpresequence\b|\btargeting\b", re.IGNORECASE),
    "dna_binding": re.compile(r"\bDNA\b|\bhomeo\w*|\bHTH\b|\bbZIP\b|recognition helix|"
                              r"\b(?:base|base-specific|groove)[- ]contacts?\b", re.IGNORECASE),
    "topological_domain": re.compile(r"\bcytoplasm\w*|\bcytosol\w*|\bextracellular\b|\blumenal\b|\bluminal\b|"
                                     r"\bperiplasm\w*|\btopolog\w*|\boutward\b|\binward\b", re.IGNORECASE),
}
RE_NOT_LIGAND_TYPED = re.compile(r"\b(?:%s)\b[- ](?:[\w-]+[- ])?(?:binding|contact\w*|recognition)\b" % NOT_LIGAND,
                                 re.IGNORECASE)
# A domain-level noun after a break heads its own range ("a coiled coil at 100 to 150, and the kinase domain at 200"),
# and so does a named sequence element that is no feature ("and a cell attachment site at 1698 to 1700", "a PfkB
# conserved site at 334 to 347", "a C-terminal BCD regulatory scaffold (497-759)").
RE_DOMAIN_LEVEL = re.compile(r"\b(?:domains?|superfamil(?:y|ies)|famil(?:y|ies)|folds?|modules?)\b", re.IGNORECASE)
RE_FAMILY = re.compile(r"\b(?:superfamil(?:y|ies)|famil(?:y|ies))\b", re.IGNORECASE)
# "with its conserved site at 265 to 287" is the keyword's own domain's site.
RE_POSSESSIVE_ELEMENT = re.compile(r"\b(?:its|their)\s+(?:own\s+)?(?:conserved\s+(?:[\w/-]+\s+){0,2}sites?|signatures?|"
                                   r"consensus)\b", re.IGNORECASE)
RE_ELEMENT = re.compile(r"\b(?:conserved (?:[\w/-]+ ){0,2}sites?|signatures?|consensus|attachment sites?|"
                        r"cleavage sites?|scaffolds?|envelopes?|hubs?)\b", re.IGNORECASE)
# Short parentheticals inside a name ("NAD(P)-binding", "nucleotide-binding (G) domain") do not split it.
RE_SHORT_PAREN = re.compile(r"\((?!IPR\))[A-Za-z0-9+]{1,3}\)")
# For a metal site, a binding phrase is a rival only when it names another ligand ("a phosphate-binding subset at");
# "residues 91, 93, 95, 97 and 102 form a second [binding site]" after "calcium-binding" is the same site.
# "glutamate contacts at 242, 297" names its ligand by the word before "contacts".
RE_OTHER_LIGAND = re.compile(
    r"\b(?!(?:zinc|calcium|magnesium|manganese|iron|copper|cobalt|nickel|metal|cation|Zn|Ca|Mg|Mn|Fe|Cu)\b)"
    r"[\w()+/]+-(?:binding|contacting)\b|\b(?:ATP|ADP|AMP|GTP|GDP|NADP?H?|NAD\(P\)H?|FAD|FMN|SAM|CoA|TPP|ThDP|heme|"
    r"haem|substrate|cofactor|nucleotide|phosphate|sugar)\b|"
    r"(?<![\w-])(?!(?:zinc|calcium|magnesium|manganese|iron|copper|cobalt|nickel|metal|divalent-metal|cation|Zn|Ca|Mg|"
    r"Mn|Fe|Cu|%s|%s)\s)[\w()+/',-]*[\w)+]\s+contacts?\b" % (NOT_LIGAND, NOT_CONTACT), re.IGNORECASE)
# Final round: a metal or ligand verb's object ends its phrase ("the two histidines and the downstream glutamate
# coordinate a single zinc ion, the glutamate at 131 sits between the histidines as the general base"). A coordinate
# past a clause break is the verb's when its own clause names the type or refers back to it ("and histidine 394 binds
# copper B", "with aspartate 177 contributing to a second metal site", "and the second, at 249 to 253, does the same"),
# never when that clause gives it another role ("and an arginine finger at 703", "glutamate 149 shuttles the proton")
# or, for a metal, another ligand ("and lysine 252 binds 2-oxoglutarate").
VERB_HEAD_TYPES = ("metal_binding", "binding_site")
# A colon after the verb's object opens its own items ("coordinate two zinc ions: cysteines 45, 48, 62 and 65"), so it
# is no break.
RE_VERB_BREAK = re.compile(r";|,\s*(?:(?:and|with|while|whereas|but)\s+)?(?:the|a|an|its|this|these|those)\s|"
                           r",\s*(?:and|with|while|whereas|but)\b(?!\s+\d)", re.IGNORECASE)
RE_VERB_CLAUSE_END = re.compile(r"[.;:](?:\s|$)|,\s*(?:and|but|while|whereas|so)\b(?!\s+\d)|\n", re.IGNORECASE)
RE_ITEM_STOP = re.compile(r"[,;.]")
RE_OTHER_ROLE = re.compile(r"arginine[- ]finger|\bgeneral (?:acid|base)\b|\bacid[- ]base\b|\bnucleophil\w*|"
                           r"\bproton[- ]?(?:accept|donat|donor|shuttl)\w*|\bshuttl\w+\s+(?:the\s+|a\s+)?protons?\b|"
                           r"\bdeprotonat\w+|\bpolari[sz]\w*|\boxyanion\b|\btransition[- ]state\b|"
                           r"\bdimeri[sz]ation\s+(?:surface|interface|site|motif|region)", re.IGNORECASE)
METAL_NOUNS = r"zinc|calcium|magnesium|iron|copper|manganese|cobalt|nickel|metals?|cations?|Zn|Ca|Mg|Fe|Cu|Mn|Co|Ni"
RE_OTHER_BOUND = re.compile(r"\b(?:binds?|binding|bound)\s+(?:a\s+|an\s+|the\s+|one\s+|two\s+)?"
                            r"(?!(?:%s)(?![A-Za-z]))[\w(]" % METAL_NOUNS, re.IGNORECASE)
RE_SAME_REFERENCE = re.compile(r"\b(?:does|do)\s+the\s+same\b|\bthat\s+(?:residue|ligand|position)\b", re.IGNORECASE)
RE_VERB_OWN = {
    "metal_binding": re.compile(r"\bligands?\b|\bcoordinat\w*|\bchelat\w*|\bzincins?\b", re.IGNORECASE),
    "binding_site": re.compile(r"\bbind\w*|\bbound\b|\bcoordinat\w*|\bchelat\w*|\bcontacts?\b|\bligands?\b|"
                               r"\bzincins?\b|\b(?:guanine|adenine|guanosine|adenosine|nucleobases?|nucleosides?|%s)\b"
                               % LIGAND_WORDS, re.IGNORECASE),
}
# The coordinate's clause states something of its own when it is an item added by "with" ("with an arginine at 1145
# immediately after the fourth cysteine") or, after a metal verb, carries a finite verb ("the mutagenesis record is
# decisive: replacing histidine 409 ... abolishes deacetylation"); a verbless continuation shares the verb ("ATP binds
# at 213, ..., and again at 790, 796", "ADP at 868 to 873, ..., and shikimate at 890").
RE_WITH_BREAK = re.compile(r",\s*with\b", re.IGNORECASE)
RE_CLAUSE_PREDICATE = re.compile(
    r"\b(?:%s|sit|opens?|closes?|acts?|serves?|specif(?:y|ies)|extends?|contributes?|abolish\w*|abstracts?|donates?|"
    r"accepts?|activates?|attacks?|inserts?|shuttles?|positions?|orients?|stabili[sz]es?|neutrali[sz]es?|aligns?|"
    r"drives?|tunes?|recognizes?|grips?|reaches|reach|faces?|deprotonates?|protonates?|polari[sz]es?|cataly[sz]es?|"
    r"hydroly[sz]es?|forming|making|lining|lines?)\b" % CLAUSE_VERBS, re.IGNORECASE)
# For a metal, a verbless continuation naming another object takes that object ("Magnesium binds at 117, 118 and 250,
# ..., and dimethylallyl diphosphate at 126, 211").
GAPPING_WORDS = {"again", "also", "likewise", "similarly", "further", "more"}
# The residues a feature can sit on: a disulfide joins cysteines; glycans sit on Asn, Ser, Thr, Trp or hydroxylysine.
RESIDUE_TYPES = {"disulfide": "C", "glycosylation": "NSTWK"}
RE_SEGMENT_SPLIT = re.compile(r"([,;:()\[\]]|\b(?:and|plus|with|then)\b)", re.IGNORECASE)
JOINERS = {"and", "plus", "with", "then"}
# A phrase naming where something lies, and the phrases joined to it by "and", is a place, not a head.
PLACE_WORDS = r"between|within|inside|across|along|throughout|flanking|near|around|beside|behind|among"
# A phrase opening with "in" or "on" is a place too ("191 to 210 in the first domain and 175 to 210").
RE_PLACE_START = re.compile(r"^\s*(?:%s|in|on)\b" % PLACE_WORDS, re.IGNORECASE)
RE_PLACE = re.compile(r"\b(?:%s)\b" % PLACE_WORDS, re.IGNORECASE)
# An accession in apposition ("an Armadillo-type fold, IPR016024, across 111 to 442") heads nothing of its own.
CUE_WORDS = {"residues", "residue", "positions", "position", "at", "from", "over", "across", "spanning", "spans",
             "covering", "around", "about", "approximately", "roughly", "to", "the", "a", "an", "and", "is", "are",
             "ipr"}
# A colon right after the keyword's phrase opens the keyword's own items ("two tandem helix-turn-helix modules rather
# than one: a first SANT/Myb domain over roughly 8 to 57"); their domain words name the keyword's parts.
RE_OWN_COLON = re.compile(r"^[^\d:;.]{0,60}?:", re.IGNORECASE)
OWN_COLON_WORDS = 4
RE_CONTRAST = re.compile(r"\b(?:while|whereas|but)\b", re.IGNORECASE)
RE_KEYWORD_TAIL = re.compile(r"^[A-Za-z'-]+")
RE_LEAD_BREAK = re.compile(r"[,;:]|\b(?:and|plus|with|then)\b", re.IGNORECASE)
RE_CUE_COMMA = re.compile(r"^\W*(?:[\w/-]+\s*)?,\s*(?:residues?|positions?|at)\s*$", re.IGNORECASE)
# InterPro list items carry their entry type after the name ("Signal peptide, N-terminal delta (domain) [1-18]").
RE_ENTRY_TYPE = re.compile(r"\((?:domain|family|homologous_superfamily|repeat|conserved_site|active_site|"
                           r"binding_site|ptm|site)\)", re.IGNORECASE)
# The round-1 boundaries: a list behind one of them, or behind a word that is not list glue, is attached loosely.
RE_LOOSE = re.compile(r"\b(?:while|whereas|but)\b|,\s*(?:and\s+)?(?:a|an|the)\s", re.IGNORECASE)
# Backward, a hard boundary always separates a list from a later keyword; a soft one (an appositive or a relative
# pronoun) only when the keyword heads its own coordinates ("an ATP-binding loop at 412 to 420, the catalytic lysine
# at 442"), not when it labels the list before it ("histidine 303, histidine 349 and aspartate 404, a catalytic triad").
RE_BACK_HARD = re.compile(r"\band\s+(?:then\s+)?(?:a|an|the)\b|,\s*(?:and|but|while|whereas)\b|[;:]|\bwhile\b|"
                          r"\bwhereas\b", re.IGNORECASE)
RE_BACK_SOFT = re.compile(r",\s*(?:then\s+|with\s+)?(?:a|an|the)\s|,\s*(?:which|where)\b", re.IGNORECASE)
SOFT_HEAD_WORDS = 3
# A keyword heading coordinates of its own takes a list before it only through a copula or labelling verb ("the first
# eighteen residues ... read as a classic cleavable signal peptide: a short basic n-region at 1 to 2"), not by
# adjacency ("the ATP-contacting residues at 569, ..., 971 with magnesium ligands at 569, 571").
LABEL_VERBS = (r"is|are|was|were|read|reads|form|forms|formed|forming|mark|marks|marked|define|defines|defined|"
               r"constitute|constitutes|line|lines|lined|build|builds|built|carry|carries|carrying|comprise|comprises|"
               r"make|makes|serve|serves|act|acts")
RE_LABEL_VERB = re.compile(r"\b(?:%s)\b" % LABEL_VERBS, re.IGNORECASE)
# The list a new_list_head drop would take from a later keyword keeps it when its own head names the keyword's type
# ("substrate-contact residues at 56, 103, ..., 340, calcium ligands at 102", "a single helix at 170 to 194, to a type
# I single-pass topology"); the head is read back over the list to the previous break. A list item is stripped one at a
# time, since a repeated group backtracks badly on long lists.
RE_HEAD_BREAK = re.compile(r"[,;:(\[]|\b(?:and|plus|with|then|while|whereas|but)\b", re.IGNORECASE)
RE_LIST_ITEM_END = re.compile(r"\b\d{1,4}(?:\s*(?:-|–|to|through)\s*\d{1,4})?\s*(?:,|\band\b|\bor\b)?\s*$",
                              re.IGNORECASE)
# Helices that do not cross the membrane ("a second pore module with intramembrane helix 225 to 237").
RE_NOT_MEMBRANE_HELIX = re.compile(r"\bintramembrane\b|\bamphipathic\b|\bre-?entrant\b|\bpore helix\b", re.IGNORECASE)
# A coordinate that is the subject of a predicate naming the asking keyword claims both types ("a conserved
# active-site acid/base (captured by the NTYK motif at residues 81, 104, 134, 138) protonates the carbonyl oxygen and
# coordinates the nucleotide-sugar", "IPR033130 (..., active site, residues 312-320), which encodes the canonical HExxH
# zinc-binding motif", "histidine 18 is singled out as the acceptor-substrate pocket").
# Only identity and role verbs: "the disulfide from 116 to 199 holds the second extracellular loop" and "a single
# transmembrane helix at 397 to 419 makes this a type I membrane protein" do not name the coordinate as the keyword.
PREDICATE_VERBS = (r"is|are|was|were|read|reads|form|forms|mark|marks|define|defines|constitute|constitutes|line|lines|"
                   r"build|builds|carry|carries|comprise|comprises|serve|serves|act|acts|singled\s+out|named|"
                   r"protonates?|deprotonates?|encodes?")
RE_OWN_PREDICATE = re.compile(
    r"^(?:(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)\d{1,4}\b(?:\s*(?:-|–|—|to|through)\s*\d{1,4}\b)?)*"
    r"[\s)\]]*(?:,\s*which\s+)?(?:%s)\b" % PREDICATE_VERBS, re.IGNORECASE)
# The keyword must sit in the predicate's own clause ("... at 257 to 259 is the localization clue that the catalytic
# core does not by itself give").
RE_PREDICATE_RELATIVE = re.compile(r"\b(?:that|which|where|whose|who|when)\b", re.IGNORECASE)
RE_PREDICATE_BREAK = re.compile(r"[,;:.()\[\]]")
RE_OBJECT_END = re.compile(r"[,;:.()]|\b(?:that|which|while|whereas|but)\b", re.IGNORECASE)
RE_OBJECT_JOIN = re.compile(r"\band\b", re.IGNORECASE)
OBJECT_REACH = 80
RE_SEQUENCE_WORD = re.compile(r"\b(?:followed|preceded|then|next|after|before|downstream|upstream|beside|adjacent|"
                              r"flanking|flanked)\b", re.IGNORECASE)
RE_LANDMARK = re.compile(r"\b(?:(?:residues?|positions?)\s+(?:away\s+)?from|past|beyond|beside|next to|adjacent to|"
                         r"after|before|following|preceding|upstream of|downstream of|flanking|behind|outside|over|"
                         r"above|below|beneath|under)\s+"
                         r"(?:the\s+|its\s+|a\s+|an\s+)?(?:[\w-]+\s+){0,2}$", re.IGNORECASE)
# A landmark word in a new clause governs that clause's noun, not the coordinate ("a further cysteine ligand at 812,
# 815, 840 and 1071, so I read two ThDP-fold regions flanking an iron-sulfur relay").
RE_LANDMARK_CLAUSE = re.compile(r",\s*(?:so|thus|hence|therefore)\b|\bI (?:read|take|see|place)\b|\bwhich\b|\bwhere\b",
                                re.IGNORECASE)
# After ", which" a keyword deep in the clause labels the list only as the predicate of a labelling verb ("which form
# the ATP-binding loop"); one in a prepositional phrase does not ("which I read as family-level echoes across the
# transmembrane and carboxyl modules").
RE_RELATIVE_PREP = re.compile(r"\b(?:across|throughout|between|among)\s+(?:the\s+|its\s+|a\s+|an\s+)?"
                              r"(?:[\w-]+\s+)?$", re.IGNORECASE)
# A relative clause set off inside a claim ("around 304 to 313, where the pocket residues 304, 307 and 308 sit, is the
# oxygen-binding groove") does not separate the claim from its head.
RE_RELATIVE = re.compile(r",\s*(?:where|which|that|whose|in which)\b(?:[^,]|,\s*\d)*,", re.IGNORECASE)
# Where a keyword's own phrase ends; a coordinate before it is the keyword's own. A chemical locant ("the sn-2 ester
# bond") is no coordinate.
RE_OWN_END = re.compile(r"[,;.]\s|\b(?:and|while|whereas|but|which|that)\b", re.IGNORECASE)
RE_OWN_COORD = re.compile(r"(?<![A-Za-z0-9])(?<![A-Za-z]-)\d{1,4}\b|\b(?:%s)[\s-]?\d{1,4}\b"
                          % "|".join([c.title() for c in rc.AA3] + [c.upper() for c in rc.AA3]))
# A later head takes a loosely attached list only through a short predicate ("two asparagines at 38 and 192 carry
# N-linked glycans", "residues 222 and 223 as further magnesium ligands"), not through a place ("within the ...").
HEAD_REACH = 40
HOLD_WORDS = 1
OWNER_REACH = 300
# Two keywords joined into one head ("the scattered active-site and nucleotide-binding residues at 48-51") both own its
# list.
RE_JOINT = re.compile(r"[\s-]*(?:and/or|and|or|/)\s*", re.IGNORECASE)
RE_COPULA_COMMA = re.compile(r"^\s*,\s*(?=(?:is|are|was|were|remain|remains)\b)", re.IGNORECASE)
RE_LOCATION = re.compile(r"\b(?:within|in|inside|outside|on|near|nearby|of|across|along|between|flanking|beside|"
                         r"into|from|toward|towards|at|via|through|to|around|surrounding|adjacent|next|close|behind|"
                         r"beneath|under|over|above|below|against)\b", re.IGNORECASE)
# "to" before a head that is a verb marks a purpose, not a place ("the histidine at 39 with aspartate 63 to activate
# water").
RE_INFINITIVE = re.compile(r"\bto\s*$", re.IGNORECASE)
RE_VERB_HEAD = re.compile(r"(?:activat|accept|donat|deprotonat|protonat|stabiliz|polariz|attack|cleav|hydrolys|"
                          r"hydrolyz|bind|coordinat|contact|chelat)", re.IGNORECASE)
RE_SIGNAL_PARTS = re.compile(r"\b[nhc]-region|polar cap|signal peptide|signal sequence|cleavable signal",
                             re.IGNORECASE)
# Named elements that are no feature of any scored type take the coordinates they head: the myosin motor's relay,
# SH1 helix, converter and lever arm, a sterile alpha motif, the integrin-binding RGD triplet.
RIVAL_NOUNS = (r"disorder\w*|low[- ]complexity|compositional\w*|\bbias\w*|\blinker\b|\bhinge\b|interface|\bchannel\b|"
               r"\binteraction (?:sites?|surfaces?|regions?|patch(?:es)?|motifs?|residues?)\b|"
               r"activation loop|activation segment|propeptide|mature chain|mature protein|\bstub\b|"
               r"\brelay (?:loop|helix)\b|\bSH1 helix\b|\bconverter\b|\blever arm\b|sterile alpha motif|\bRGD\b|"
               r"arginine-glycine-aspartate|cell[- ]attachment")
# The sides of the membrane take coordinates from every type but the topological domain, which they name (round 5).
RIVAL_SIDES = r"cytoplasmic side|cytosolic side|far side|lumenal side|luminal side|extracellular side|non-cytoplasmic"
RE_RIVAL = re.compile(RIVAL_NOUNS + "|" + RIVAL_SIDES, re.IGNORECASE)
RE_RIVAL_TOPOLOGY = re.compile(RIVAL_NOUNS, re.IGNORECASE)
# A coordinate a helix leaves on one side of the membrane is no helix ("leaving only 1401 to 1418 in the cytosol",
# "followed by only five residues, 684 to 688, on the cytosolic side").
RE_SIDE_AFTER = re.compile(
    r"\s*,?\s*(?:(?:lies|lie|sits|sit|is|are|remains?|returns?|faces?|facing|exposed|left|lying|sitting|resides?)\s+)?"
    r"(?:(?:in|on|into|to|toward|towards)\s+)?the\s+(?:cytosol|cytoplasm|lumen|periplasm|matrix|extracellular space|"
    r"(?:cytoplasmic|cytosolic|extracellular|lumenal|luminal|periplasmic|exoplasmic|intracellular|outer|inner|far|"
    r"non-cytoplasmic)\s+(?:side|face|surface|leaflet))\b", re.IGNORECASE)
# Residues the positive-inside rule names flank a helix ("lysines at 22 and 23, are the positive-inside signature of an
# uncleaved signal anchor").
RE_POSITIVE_INSIDE = re.compile(r"positive[- ]inside", re.IGNORECASE)
RE_PART_LABEL = re.compile(
    r"\b[nhc][- ]regions?\b|hydrophobic core|polar cap|(?:basic|polar|charged|cleavage) (?:amino[- ]terminal |"
    r"N-terminal |carboxyl[- ]terminal )?(?:region|segment|stretch|tail)", re.IGNORECASE)
PART_REACH = 40
PART_TYPES = ("signal_peptide", "transit_peptide")
NTERM_TYPES = ("signal_peptide", "transit_peptide")
# Shortest claim from residue 1 read as the whole peptide rather than its n-region ("an n-region at 1 to 7").
WHOLE_MIN = 12
NTERM_START = 3
UNION_TYPES = ("coiled_coil",)
RE_SIDE_IN = re.compile(r"cytoplasm|cytosol", re.IGNORECASE)
RE_SIDE_OUT = re.compile(r"extracellular|lumenal|luminal|periplasm|vesicular", re.IGNORECASE)
# Only for telling whether a sentence names both sides; "outside" makes a side mixed and so falls back to either side.
RE_SIDE_WORD = re.compile(r"non[-‐‑]?cyto(?:plasm|sol)\w*|cytoplasm\w*|cytosol\w*|extracellular|lumenal|luminal|"
                          r"periplasm\w*|vesicular|\boutside\b|exoplasmic", re.IGNORECASE)
RE_SIDE_OUT_WORD = re.compile(r"non|extracellular|lumenal|luminal|periplasm|vesicular|outside|exoplasmic",
                              re.IGNORECASE)
# A coordinate whose own list item names a side takes that side, not the keyword's ("a cytosolic loop at 69 to 95
# returns inward, a long stretch at 117 to 180 faces out again").
RE_ITEM_BREAK = re.compile(r"[,;:()\[\]]|\b(?:and|with|while|whereas|but|then)\b", re.IGNORECASE)
RE_ITEM_SIDE = re.compile(r"non[-‐‑]?cyto(?:plasm|sol)\w*|cytoplasm\w*|cytosol\w*|extracellular|lumenal|luminal|"
                          r"\blumen\b|periplasm\w*|exoplasmic|\boutward\b|\binward\b|\bfaces?\s+(?:out|in)\b|"
                          r"\boutside\s+(?:of\s+)?the\s+cell\b", re.IGNORECASE)
RE_ITEM_OUT = re.compile(r"non|extracellular|lumen|luminal|periplasm|exoplasmic|outward|\bout\b|outside", re.IGNORECASE)
# Final round: a side predicate whose subject is a new terminus, tail or "both" does not reach back over a list another
# head or predicate holds ("seven crossings, at 2 to 22, ..., 248 to 265, with the amino terminus on the periplasmic
# side"; "loops at 95 to 103 ... reach the far side, so both termini sit on the cytosolic face"), unless the list's own
# words name its side ("78-101, 163-198 and 261-273 outside, so the amino terminus is extracellular").
RE_SIDE_SUBJECT = re.compile(r"\b(?:with|so|while|whereas)\s+(?:the\s+|its\s+)?(?:both\s+)?(?:(?:amino|carboxyl|N|C)"
                             r"[- ]?(?:terminal\s+)?)?(?:termin(?:us|i)|ends?|tails?)\b|\bso\s+both\b", re.IGNORECASE)
RE_LIST_SIDE = re.compile(r"\b(?:inside|outside|inward|outward|cytosol\w*|cytoplasm\w*|extracellular|lumen\w*|"
                          r"periplasm\w*|exoplasmic)\b|\bfaces?\s+(?:in|out)\b", re.IGNORECASE)
# A coordinate placed in a helix is no topological domain ("Two common variants, at 136 in the third helix and 223 in
# the third cytosolic loop"; "a glutamate at 63 in the cytosolic loop between the second and third helices and another
# at 92 at the far end of the third").
ORDINALS = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|final"
RE_HELIX_PLACE = re.compile(
    r"\s*(?:in|within|inside|at|on|near)\s+(?:the\s+|its\s+)?(?:(?:(?:far|near|cytosolic|cytoplasmic|extracellular|"
    r"outer|inner|amino|carboxyl|N|C)[- ](?:terminal\s+)?)?ends?\s+of\s+(?:the\s+)?)?(?:(?:%s)\s+)?"
    r"(?:(?:transmembrane|hydrophobic|membrane)[- ])?(?:hel(?:ix|ices)|crossings?|TM\d*)\b|"
    r"\s*(?:in|at|on|near)\s+the\s+(?:(?:far|near)\s+)?ends?\s+of\s+the\s+(?:%s)\b(?=\s*[,.;)])" % (ORDINALS, ORDINALS),
    re.IGNORECASE)
# A small range before a helix noun counts helices ("typically forming a 10-12 transmembrane-helix bundle", "roughly
# 13-14 transmembrane helices"); a residue range of a helix is longer ("the 49-66 helix").
RE_COUNT_NOUN = re.compile(r"[\s-]*(?:(?:transmembrane|TM|membrane[- ]spanning)[\s-]*(?:hel(?:ix|ices)|helical|"
                           r"segments|spans|passes|domains?)|helices|passes)\b", re.IGNORECASE)
COUNT_SPREAD = 6
# Another molecule's numbering: a tRNA, rRNA or nucleotide position ("the D/T loops that house position 26", "at
# position 37 of tRNA").
RE_OTHER_NUMBERING_LEAD = re.compile(
    r"\b(?:tRNAs?|rRNAs?|anticodon|wobble|D/T\s+loops?|[DT][- ]loops?|nucleotides?|nucleosides?|uridines?|cytidines?|"
    r"guanosines?|adenosines?|guanines?|adenines?|cytosines?|uracils?|queuosine)\b[^.;:]{0,50}\bpositions?\s+$",
    re.IGNORECASE)
RE_OTHER_NUMBERING_TRAIL = re.compile(
    r"(?:\s*(?:,|and|or|to|-|–)\s*\d{1,4})*\s+(?:of|in|on|within)\s+(?:the\s+|a\s+|its\s+|their\s+)?(?:[\w-]+\s+)?"
    r"(?:tRNAs?|rRNAs?|anticodon|histones?)\b", re.IGNORECASE)
NUMBERING_REACH = 80
# A position governed by one of these states where a feature ends, not a residue inside it ("an N-terminal signal
# peptide (preceding residue 10)"). For a signal or transit peptide it is read as the peptide's end: before N ends at
# N - 1, after or until N at N. For a region type it is the region's edge ("After residue 172 the chain becomes
# coiled": the coil starts at 173), scored against the same edge of the nearest true feature; for a point type the
# excluded side carries no position and is dropped.
BOUNDARY_TARGET = r"\s+(?:the\s+)?(?:(?:residues?|positions?|amino acids?)\s+|(?:%s)[\s-]?)?$" % AA_WORDS
RE_BOUNDARY_BEFORE = re.compile(r"\b(?:preceding|precedes?|before|upstream of)" + BOUNDARY_TARGET, re.IGNORECASE)
RE_BOUNDARY_AFTER = re.compile(r"\b(?:after|following|follows?|beyond|past|downstream of)" + BOUNDARY_TARGET,
                               re.IGNORECASE)
RE_BOUNDARY_UNTIL = re.compile(r"\b(?:until|up to|ending at|ends? at)" + BOUNDARY_TARGET, re.IGNORECASE)
BOUNDARY_REACH = 40
EDGE_TYPES = ("transmembrane", "coiled_coil", "topological_domain", "dna_binding", "zinc_finger")
# The boundary word opens the clause that describes the region ("After residue 172 the chain becomes coiled", "the coil
# starts after 172"); after a noun it limits the noun ("every functional element after residue 27 is lumenal").
RE_EDGE_LEAD = re.compile(r"(?:^|[.;:,(]|\b(?:and|then|but|while|so|begins?|starts?|ends?|stops?|runs?|continues?|"
                          r"extends?|resumes?))\s*$", re.IGNORECASE)
POINT_TYPES = ("modified_residue", "glycosylation", "lipidation")
# A disulfide range is a cysteine pair when a bond word heads it ("Four disulfide bonds, 50 to 65, 67 to 77"); when a
# domain-level noun or an entry's citation heads it, it is a region said to hold bonds, which places no bond
# ("IPR003582 (ShKT domain, residues 276-320), a compact disulfide-stabilized module", "SRCR domains at 70 to 170, 273
# to 373", "from 369 to 424, sits an EGF-like domain").
RE_PAIR_HEAD = re.compile(r"\b(?:disulfides?|disulphides?|bonds?|bridges?|pairs?|paired|pairing|links?|linked|"
                          r"cystines?|S-S|between)\b", re.IGNORECASE)
# A bond word that modifies a region noun names the region ("IPR054717 (Secretory peroxidase, disulphide-bonds motif,
# residues 64-75)", "a disulfide-rich domain"), not a pair.
RE_BOND_REGION = re.compile(r"\b(?:disulfide|disulphide|cystine)s?(?:[- ](?:bonds?|bonded|bridges?|rich|stabili[sz]ed|"
                            r"linked|knotted))?[- ](?:[\w-]+\s+)?(?:motifs?|domains?|modules?|regions?|folds?|"
                            r"repeats?)\b", re.IGNORECASE)
# The region noun has to head the list itself, joined to it by cues alone ("SRCR domains at 70 to 170", "(ShKT domain,
# residues 276-320)"); "33 to 37 in the first repeat, 61 to 74" and "written out for the first and fourth repeats as 30
# to 42" are pair lists.
RE_REGION_DIRECT = re.compile(
    r"\b(?:domains?|superfamil(?:y|ies)|famil(?:y|ies)|folds?|modules?|repeats?|motifs?|regions?|matches|match|"
    r"entr(?:y|ies))\b[\s,(\[~]*(?:(?:at|from|over|across|spanning|spans?|covering|covers|residues?|positions?|"
    r"roughly|about|around|approximately|between|sits? over|runs?)\b[\s,(\[~]*)*$", re.IGNORECASE)
RE_REGION_AFTER = re.compile(r"\W{0,3}(?:sits|folds as|forms|is|are|makes up|constitutes)\s+(?:an?|the)\s+"
                             r"(?:[\w-]+\s+){0,3}(?:domains?|modules?|repeats?|folds?)\b", re.IGNORECASE)
# A range said to hold bonds is a region ("an LRRNT cap at 38 to 72, matched again at 45 to 76, carries two disulfide
# bonds", "the extracellular loop from 126 to 180 is the largest extracellular element and carries two disulfide
# pairs").
RE_REGION_HOLDS = re.compile(
    r"\)?,?\s+(?:(?:is|are)\s+(?:the\s+)?(?:[\w-]+\s+){0,4}?(?:and|that)\s+)?(?:holds?|carries|carry|contains?|"
    r"harbou?rs?|bears?|(?:is|are) (?:locked|stabili[sz]ed|cross-?linked|knotted|braced|held|closed|tied) by)\s+"
    r"(?:[\w-]+\s+){0,3}(?:disulfide|disulphide|cystine)", re.IGNORECASE)
REGION_REACH = 160

# --- InterPro ---
# Final round: a degenerate accession run ("IPR003650, IPR003651, IPR003652, ...", up to 692 in one generation) names no
# entries, as a counting run cites no residues: LOOP_RUN or more mentions in a row, each a different accession at most
# ACCESSION_STEP numbers from the last and starting within ACCESSION_GAP characters of it. A run of eight would also
# take real consecutive entries ("RRM 1", "RRM 2", "RRM 3" at IPR034172 to IPR034174; nine SMCHD1 Ig-like entries).
ACCESSION_STEP = 3
ACCESSION_GAP = 160

# --- truth ---
SIGNALP_E = "interpro_match:signalp_e"
SIGNALP_G = ("interpro_match:signalp_g+", "interpro_match:signalp_g-")
EUKARYOTE_MODE = "e"
PROKARYOTE_MODE = "g"
PROKARYOTE_DOMAINS = ("Bacteria", "Archaea")
RE_PASS_MEMBRANE = re.compile(r"(?:single|multi)-pass", re.IGNORECASE)
RE_METAL_EXTRA = re.compile(r"\d?Fe-\d?S|metal cation|K\(\+\)|Na\(\+\)|molybd", re.IGNORECASE)
# Round 5: the site labels the claim side reads as active sites too ("Active site cavity", "active site pocket",
# "catalytic network", "catalytic core", "stabilizes intermediate"); lids, flaps, loops and clefts stay out.
RE_SITE_ACTIVE = re.compile(
    r"^ACT_SITE:|^(?:putative |probable )?(?:active sites?(?: cavity| pocket)?|catalytic (?:residues?|sites?|"
    r"site residues|triad|dyad|tetrad|acid|network|core))\s*$|^nucleophile\b|^stabili[sz]es intermediate\s*$",
    re.IGNORECASE)
REVIEWED = "Reviewed"
RE_SITE_BINDING = re.compile(
    r"^(?:BINDING|NP_BIND|METAL):|^REGION: [^.]*binding|"
    r"\b(?:ATP|ADP|AMP|GTP|GDP|NTP|dNTP|nucleotide|nucleotidyl|NAD|NADP|NAD\(P\)|NADP\(H\)|NAD\+|FAD|FMN|heme|haem|"
    r"Ca\d?\+?|calcium|Zn\d?\+?|zinc|Mg\d?\+?|magnesium|Mn\d?\+?|manganese|metal|Fe|iron|Fe-S|Cu|copper|cation|"
    r"substrate|ligand|cofactor|pyridoxal|PLP|S-adenosylmethionine|SAM|CoA|coenzyme|molybdopterin|sugar|"
    r"carbohydrate|mannose|NAG|lipid|phospholipid|phosphoinositide|PtdIns|phosphate|polyP|TPP|TDP|GSH|glutamate|"
    r"product|siderophore|hyaluronan|starch|glycerol|shikimate|ketoglutarate|oxalacetate|nucleoside|purine)"
    r"[\w()+/-]*[- ](?:binding|ligand)|\b(?:metal|Mg\d?\+?|Di-iron) ligands?\b|binuclear metal", re.IGNORECASE)
RE_SITE_NOT_LIGAND = re.compile(r"\b(?:%s)\b|nucleic|surface" % NOT_LIGAND, re.IGNORECASE)
# PROSITE and Pfam binding-site matches whose bound object is a protein: the IQ motif's EF-hand (calmodulin) binding
# site, Aurora-A, Ankyrin-G and the beta-tubulin autoregulation site. The object is the word before "binding", since
# "Protein kinase, ATP binding site" binds ATP.
RE_RECORD_NOT_LIGAND = re.compile(r"\b(?:%s|autoregulation)[- ]binding" % NOT_LIGAND, re.IGNORECASE)
BINDING_RECORD = "interpro_match:binding_site"
# Heme counts as a metal site because patterns.RE_METAL_DESC counts UniProt heme BINDING records as metal.
RE_SITE_METAL = re.compile(
    r"^METAL:|^BINDING: (?:Ca|Mg|Zn|Fe|Mn|Cu|Co|Ni|K|Na)\b|^BINDING: (?:Fe|Cu|divalent metal) cation|"
    r"^BINDING: (?:heme|haem)|\b(?:heme|haem)\b[- ](?:\w+ )?binding|"
    r"\b(?:Ca|Zn|Mg|Mn|Fe|Cu|Co|Ni)\d?\+?[- ](?:ion )?(?:binding|ligand)|"
    r"\b(?:calcium|zinc|magnesium|manganese|iron|copper|metal|cation)[- ](?:ion )?(?:binding|ligand)|"
    r"\[\d?Fe-\d?S\]|Fe-S cluster|iron-sulfur|Di-iron|binuclear metal", re.IGNORECASE)
RE_MIXED_METAL = re.compile(r"(?:GTP|ATP|substrate|AMP|NTP)[/-](?:Mg|Mn)", re.IGNORECASE)
SITE_TYPES = ("active_site", "binding_site", "metal_binding")
PROCESSED_CHAIN_TYPES = ("uniprot:CHAIN", "uniprot:PEPTIDE")
# Final round, truth:
# - A UniProt SIGNAL ending before residue SIGNAL_MIN_END is the remnant of a signal the fragment's start cuts off
#   ("SIGNAL 1-1", "1-3"); the shortest real one in the test set ends at 12.
# - A UniProt BINDING record whose ligand is a macromolecule ("RNA", "ssDNA", "a protein", "a peptide") is no ligand
#   site, by the NOT_LIGAND test the claims and the InterProScan records already pass.
# - A tool helix on the C-terminal propeptide of a curated GPI-anchored protein is its GPI signal, removed in the mature
#   form, not a transmembrane segment.
# - A TMHMM helix on a Phobius signal peptide that Phobius does not call a helix is the signal's h-region, as on a
#   UniProt or SignalP signal; but a protein whose location is a single-pass membrane protein other than type I, left
#   with no helix by these two rules, keeps the helix they removed as its uncleaved amino-terminal signal anchor.
SIGNAL_MIN_END = 10
RE_ANCHOR_TOPOLOGY = re.compile(r"single-pass (?:type (?:II|III) )?membrane protein", re.IGNORECASE)
RE_UNIPROT_NOT_LIGAND = re.compile(r"^(?:an?\s+)?(?:glyco)?(?:%s)$" % NOT_LIGAND, re.IGNORECASE)
RE_GPI = re.compile(r"GPI-anchor", re.IGNORECASE)
PHOBIUS_SIGNAL = "SIGNAL_PEPTIDE"
PHOBIUS_HELIX = "TRANSMEMBRANE"
# Final round, presence truth completeness: protein-level evidence that a protein has the
# type where its residue-level records are silent. It joins the presence truth only, never the span truth, and none of
# it is in either prompt.
# - Active site: UniProt curates the cofactor or substrate of many enzymes but no ACT_SITE (60% of active-site false
#   positives read were true assertions on such enzymes). An EC number in the name makes the protein an enzyme, except
#   the RING-type E3 transferase, which acts through the E2's cysteine; with no EC number, a CATALYTIC ACTIVITY reaction
#   does when it is a chemical change, not a transport event ("K(+)(in) = K(+)(out)" on channels and carriers).
# - Transmembrane: a UniProt location naming a single- or multi-pass membrane protein.
# - Coiled coil: an InterPro entry or UniProt domain or region named for a coiled coil, a SNARE motif, a bZIP or a
#   leucine zipper; the CHCH domain ("coiled coil-helix-coiled coil-helix") is named for helices, not a coiled coil.
RE_EC = re.compile(r"\(EC ([\d]+\.[\d-]+\.[\d-]+\.[\dn-]+)\)")
RING_EC = "2.3.2.27"
CATALYTIC_ACTIVITY = "CATALYTIC ACTIVITY"
RE_TRANSPORT_REACTION = re.compile(r"\((?:in|out)\)")
# UniProt mostly writes the typed form ("Single-pass type I membrane protein"), which the rule meant to include.
RE_PASS_LOCATION = re.compile(r"(?:single|multi)-pass (?:type (?:I|II|III|IV) )?membrane protein", re.IGNORECASE)
RE_COIL_NAME = re.compile(r"coiled[- ]coil(?![- ]helix)|leucine zipper|\bSNARE\b|\bbZIP\b", re.IGNORECASE)
COIL_NAME_FEATURES = ("DOMAIN", "REGION")
ENTRY_LISTS = ("interpro_formatted", "interpro_formatted_iscan")


def load_reference(path: Path = REFERENCE) -> dict:
    """
    Read the bundled grounding_reference.json: taxon domains, viral host domains and retired InterPro entries.

    Args:
        path: The JSON file.

    Returns:
        The parsed document.

    Raises:
        FileNotFoundError: If the file is missing; it ships with the repository beside this module.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; restore evals/grounding/grounding_reference.json from git")
    return json.loads(path.read_text())


def signalp_mode(protein_id: str, taxid: float | None, reference: dict) -> str | None:
    """
    Choose the SignalP mode that fits the organism: eukaryotic for eukaryotes and their viruses, Gram for prokaryotes.

    Args:
        protein_id: UniProt accession.
        taxid: NCBI taxon id of the organism.
        reference: The reference document.

    Returns:
        EUKARYOTE_MODE, PROKARYOTE_MODE, or None when the domain is unknown (every mode is then kept).
    """
    domain = reference["taxon_domain"].get(str(int(taxid))) if taxid == taxid and taxid else None
    if domain == "Viruses":
        domain = reference["virus_host_domain"].get(protein_id)
    if domain == "Eukaryota":
        return EUKARYOTE_MODE
    if domain in PROKARYOTE_DOMAINS:
        return PROKARYOTE_MODE
    return None


def overlapping(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    """
    Whether a span overlaps any of a list of spans.

    Args:
        span: (start, end).
        others: Spans to test against.

    Returns:
        True if any shares a residue with it.
    """
    return any(min(span[1], b) >= max(span[0], a) for a, b in others)


def side_of(text: str) -> str:
    """
    Read which side of the membrane a topological description names.

    Args:
        text: A TOPO_DOM description or a claim's keyword.

    Returns:
        "in" (cytoplasmic), "out" (extracellular, lumenal, periplasmic) or "" when neither.
    """
    if RE_SIDE_IN.search(text) and not RE_SIDE_OUT.search(text):
        return "in"
    if RE_SIDE_OUT.search(text) and not RE_SIDE_IN.search(text):
        return "out"
    return ""


def presence_evidence(row: dict) -> dict[str, bool]:
    """
    Protein-level evidence that a protein has a presence type its residue-level records may not show.

    Args:
        row: Test-set row (protein_names, comments, subcellular_location, features and the two InterPro entry lists).

    Returns:
        {"active_site": an enzyme by EC number or chemical reaction, "transmembrane": a single- or multi-pass membrane
        protein, "coiled_coil": an entry, domain or region named for a coiled coil}.
    """
    ecs = RE_EC.findall(row.get("protein_names") or "")
    reactions = [c["text"] for c in row.get("comments") or [] if c.get("topic") == CATALYTIC_ACTIVITY]
    if ecs:
        enzyme = any(not ec.startswith(RING_EC) for ec in ecs)
    else:
        enzyme = any(not RE_TRANSPORT_REACTION.search((text or "").split(";")[0]) for text in reactions)
    names = [line for key in ENTRY_LISTS for line in (row.get(key) or "").splitlines()]
    names += [f.get("desc") or "" for f in row.get("features") or []
              if f["src"] == "uniprot" and f["type"] in COIL_NAME_FEATURES]
    return {"active_site": enzyme,
            "transmembrane": bool(RE_PASS_LOCATION.search(row.get("subcellular_location") or "")),
            "coiled_coil": any(RE_COIL_NAME.search(name) for name in names)}


def build_truth(row: dict, parents: dict[str, str], reference: dict) -> dict:
    """
    Build one protein's truth: feature spans per type (with the audit's truth fixes), InterPro accessions closed upward,
    member signatures and processed-chain starts.

    Args:
        row: Test-set row (protein_id, sequence, length, features, interpro_ids, taxid, subcellular_location,
            reviewed, protein_names, comments, interpro_formatted, interpro_formatted_iscan).
        parents: InterPro accession to its parent, from every protein's features.
        reference: The reference document.

    Returns:
        {"length", "sequence", "types", "sites", "notes", "evidence", "interpro", "interpro_closed", "members",
        "chain_starts"}; types and sites map a feature type to [(start, end, label)], notes to the truth fixes that
        changed it, evidence to the protein-level presence evidence (presence_evidence).
    """
    keyed = collections.defaultdict(list)
    members = set()
    for feat in row["features"] or []:
        if feat.get("member_acc"):
            members.add(feat["member_acc"])
        start, end = feat.get("start"), feat.get("end")
        if not (start and end and end >= start):
            continue
        keyed[f"{feat['src']}:{feat['type']}"].append((int(start), int(end), feat.get("desc") or "",
                                                        feat.get("member_acc") or ""))
    plain = {key: [(s, e) for s, e, _, _ in values] for key, values in keyed.items()}
    types, sites, notes = {}, {}, collections.defaultdict(list)
    for name, (keys, _) in rc.FEATURE_TYPES.items():
        types[name] = [(s, e, "") for key in keys for s, e in plain.get(key, [])]
    records = keyed.get(BINDING_RECORD, [])
    ligand = [(s, e, "") for s, e, desc, _ in records if not RE_RECORD_NOT_LIGAND.search(desc)]
    if len(ligand) < len(records):
        notes["binding_site"].append("non_ligand_record")
    curated = keyed.get("uniprot:BINDING", [])
    curated_ligand = [(s, e, "") for s, e, desc, _ in curated
                      if not RE_UNIPROT_NOT_LIGAND.match(desc.split(" ChEBI")[0].strip())]
    if len(curated_ligand) < len(curated):
        notes["binding_site"].append("non_ligand_uniprot")
    types["binding_site"] = curated_ligand + ligand

    mode = signalp_mode(row["protein_id"], row.get("taxid"), reference)
    uniprot_signal = [span for span in plain.get("uniprot:SIGNAL", []) if span[1] >= SIGNAL_MIN_END]
    if len(uniprot_signal) < len(plain.get("uniprot:SIGNAL", [])):
        notes["signal_peptide"].append("signal_remnant")
    uniprot_tm = list(plain.get("uniprot:TRANSMEM", []))
    reviewed = row.get("reviewed") == REVIEWED
    signalp = []
    if mode in (EUKARYOTE_MODE, None):
        signalp += plain.get(SIGNALP_E, [])
    elif plain.get(SIGNALP_E):
        notes["signal_peptide"].append("signalp_mode")
    if mode in (PROKARYOTE_MODE, None):
        signalp += [s for key in SIGNALP_G for s in plain.get(key, [])]
    elif any(plain.get(key) for key in SIGNALP_G):
        notes["signal_peptide"].append("signalp_mode")
    # A curated (Swiss-Prot) helix with no curated signal over it is an uncleaved signal anchor that SignalP reads as a
    # cleavable peptide.
    signal = list(uniprot_signal)
    for span in signalp:
        if reviewed and overlapping(span, uniprot_tm) and not overlapping(span, uniprot_signal):
            notes["signal_peptide"].append("signalp_on_curated_anchor")
            continue
        signal.append(span)
    types["signal_peptide"] = [(s, e, "") for s, e in signal]

    phobius = keyed.get("interpro_match:phobius", [])
    phobius_tm = [(s, e) for s, e, _, member in phobius if member == PHOBIUS_HELIX]
    # As for SignalP, a Phobius signal over a curated helix with no curated signal is an uncleaved anchor.
    phobius_signal = [(s, e) for s, e, _, member in phobius if member == PHOBIUS_SIGNAL and not (
        reviewed and overlapping((s, e), uniprot_tm) and not overlapping((s, e), uniprot_signal))]
    gpi_sites = [s for s, _, desc, _ in keyed.get("uniprot:LIPID", []) if RE_GPI.search(desc)]
    gpi_propeptides = [(s, e) for s, e in plain.get("uniprot:PROPEP", [])
                       if reviewed and any(s > site for site in gpi_sites)]
    helices = list(uniprot_tm)
    on_tool_signal = []
    for span in plain.get("interpro_match:tmhmm", []):
        if overlapping(span, signal) and not overlapping(span, phobius_tm):
            notes["transmembrane"].append("tmhmm_on_signal_peptide")
            on_tool_signal.append(span)
            continue
        if overlapping(span, phobius_signal) and not overlapping(span, phobius_tm):
            notes["transmembrane"].append("tmhmm_on_phobius_signal")
            on_tool_signal.append(span)
            continue
        # A tool helix on a curated signal peptide is its h-region, whatever Phobius says.
        if reviewed and overlapping(span, uniprot_signal) and not overlapping(span, uniprot_tm):
            notes["transmembrane"].append("tmhmm_on_curated_signal")
            continue
        if overlapping(span, gpi_propeptides) and not overlapping(span, uniprot_tm):
            notes["transmembrane"].append("helix_on_gpi_propeptide")
            continue
        helices.append(span)
    if RE_PASS_MEMBRANE.search(row.get("subcellular_location") or ""):
        for span in phobius_tm:
            if overlapping(span, gpi_propeptides) and not overlapping(span, uniprot_tm):
                notes["transmembrane"].append("helix_on_gpi_propeptide")
                continue
            if not overlapping(span, helices) and not overlapping(span, signal):
                helices.append(span)
                notes["transmembrane"].append("phobius_on_membrane_protein")
    if not helices and on_tool_signal and RE_ANCHOR_TOPOLOGY.search(row.get("subcellular_location") or ""):
        helices.extend(on_tool_signal)
        notes["transmembrane"].append("signal_anchor_topology")
    types["transmembrane"] = [(s, e, "") for s, e in helices]

    metal = []
    for s, e, desc, _ in keyed.get("uniprot:BINDING", []):
        if rc.RE_METAL_DESC.search(desc):
            metal.append((s, e, ""))
        elif RE_METAL_EXTRA.search(desc):
            metal.append((s, e, ""))
            notes["metal_binding"].append("metal_description")
    types["metal_binding"] = metal
    types["topological_domain"] = [(s, e, side_of(desc)) for s, e, desc, _ in keyed.get("uniprot:TOPO_DOM", [])]

    site_records = keyed.get("interpro_site:site", [])
    sites["active_site"] = [(s, e, "") for s, e, desc, _ in site_records if RE_SITE_ACTIVE.search(desc)]
    sites["binding_site"] = [(s, e, "") for s, e, desc, _ in site_records
                             if RE_SITE_BINDING.search(desc) and not RE_SITE_NOT_LIGAND.search(desc)]
    sites["metal_binding"] = [(s, e, "") for s, e, desc, _ in site_records
                              if RE_SITE_METAL.search(desc) and not RE_MIXED_METAL.search(desc)]

    raw = row["interpro_ids"]
    interpro = set(IPR_RE.findall(" ".join(raw) if isinstance(raw, list) else str(raw or "")))
    closed = set(interpro)
    for acc in interpro:
        seen = set()
        while acc in parents and acc not in seen:
            seen.add(acc)
            acc = parents[acc]
            closed.add(acc)
    sequence = row.get("sequence") or ""
    chain_starts = sorted({s for key in PROCESSED_CHAIN_TYPES for s, _ in plain.get(key, []) if s > 1})
    return {"length": row["length"], "sequence": sequence, "types": types, "sites": sites, "notes": dict(notes),
            "evidence": presence_evidence(row), "interpro": interpro, "interpro_closed": closed, "members": members,
            "chain_starts": chain_starts}


def split_files(test_set: Path) -> list[Path]:
    """
    The parquet files that hold the test split.

    Args:
        test_set: A parquet file, or a directory of corpus shards (the split's TEST_SHARDS are taken).

    Returns:
        The files, sorted.

    Raises:
        FileNotFoundError: If the path is missing or the directory holds no test shard.
    """
    test_set = Path(test_set)
    shards = [test_set] if test_set.is_file() else sorted(test_set.glob(TEST_SHARDS))
    if not shards:
        raise FileNotFoundError(f"no {TEST_SHARDS} under {test_set}; pass --test-set with a local parquet of the "
                                "benchmark split (README, Evaluate: scripts/data/fetch_corpus.py)")
    return shards


def load_truth(reference: dict | None = None, test_set: Path = TEST_SET) -> dict[str, dict]:
    """
    Build every test protein's truth from the test-set parquet.

    Args:
        reference: The reference document; read from REFERENCE when None.
        test_set: The v2 test split, a parquet file or a directory of shards (split_files).

    Returns:
        protein_id to its truth (build_truth).

    Raises:
        FileNotFoundError: If the test split is not found.
    """
    reference = reference or load_reference()
    columns = ["protein_id", "sequence", "length", "features", "interpro_ids", "taxid", "subcellular_location",
               "reviewed", "protein_names", "comments", *ENTRY_LISTS]
    rows = []
    for shard in split_files(test_set):
        # wanglab/bioreason-pro-v2 has no `interpro_formatted_iscan`; its entries are in `interpro_formatted`.
        present = set(pq.read_schema(shard).names)
        wanted = [c for c in columns if c in present or c not in ENTRY_LISTS]
        rows += pq.read_table(shard, columns=wanted).to_pylist()
    parents = {}
    for row in rows:
        for feat in row["features"] or []:
            if feat.get("acc") and feat.get("parent_acc"):
                parents[feat["acc"]] = feat["parent_acc"]
    return {row["protein_id"]: build_truth(row, parents, reference) for row in rows}


def load_records(path: Path, pattern: str = RESULT_PATTERN) -> list[dict]:
    """
    Read one model's eval records: an eval.py results directory, or a JSON-lines export of one (.jsonl or .jsonl.gz).

    Args:
        path: The results directory or the export.
        pattern: Which result files of a directory to read (RESULT_PATTERN, one per protein).

    Returns:
        The records, in file-name order for a directory and line order for an export.

    Raises:
        FileNotFoundError: If the path is missing or the directory holds no matching file.
        ValueError: If a protein appears twice, which would score it twice.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no {pattern} under {path}")
        records = [json.loads(file.read_text()) for file in files]
    elif path.is_file():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        raise FileNotFoundError(path)
    counts = collections.Counter(record["protein_id"] for record in records)
    repeated = sorted(pid for pid, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(f"{len(repeated)} proteins appear more than once in {path}, e.g. {repeated[:3]}")
    return records


def type_truth(truth: dict, name: str, sites: bool = True) -> list[tuple[int, int, str]]:
    """
    One feature type's true spans for a protein.

    Args:
        truth: The protein's truth.
        name: Feature type.
        sites: Whether InterProScan site records join the active-site, binding-site and metal-binding truth.

    Returns:
        (start, end, label) per true feature.
    """
    spans = list(truth["types"].get(name, []))
    if sites:
        spans += truth["sites"].get(name, [])
    return spans


def presence_truth(truth: dict, name: str, sites: bool = True, evidence: bool = True) -> bool:
    """
    Whether a protein has a presence type: a true feature of it, or protein-level evidence (presence_evidence).

    Args:
        truth: The protein's truth.
        name: Feature type.
        sites: Whether InterProScan site records join the active- and binding-site truth.
        evidence: Whether the protein-level evidence counts; False gives the residue-level records alone.

    Returns:
        True if the protein has the type.
    """
    return bool(type_truth(truth, name, sites)) or (evidence and truth.get("evidence", {}).get(name, False))


def usable(record: dict, truth: dict | None) -> bool:
    """
    Whether a record is scored for residues, presence and spans.

    Args:
        record: Exported eval record.
        truth: The protein's truth, or None if the protein is not in the test set.

    Returns:
        True if it has a sequence, a generation and a length under the truncation cap, and a truth.
    """
    seq = record.get("protein_sequence") or ""
    text = record.get("generated_response") or ""
    length = record.get("sequence_length") or len(seq)
    return bool(seq and text and length and length < rc.TRUNCATION_CAP and truth is not None)


def sentence(text: str, start: int, end: int) -> tuple[str, str]:
    """
    The part of a claim's sentence before and after it, capped at LEAD and TRAIL characters.

    Args:
        text: The generation.
        start: Claim start.
        end: Claim end.

    Returns:
        Text before the claim and text after it, within the sentence.
    """
    lead = text[max(0, start - LEAD):start]
    cut = list(RE_SENT_START.finditer(lead))
    if cut:
        lead = lead[cut[-1].end():]
    trail = text[end:end + TRAIL]
    stop = RE_SENT_START.search(trail)
    if stop:
        trail = trail[:stop.start()]
    return lead, trail


def paragraph(text: str, start: int, end: int) -> str:
    """
    The line of the generation a claim sits on; the generations write one paragraph or list item per line.

    Args:
        text: The generation.
        start: Claim start.
        end: Claim end.

    Returns:
        The text from the previous line break to the next one.
    """
    line_end = text.find("\n", end)
    return text[text.rfind("\n", 0, start) + 1:line_end if line_end >= 0 else len(text)]


def located_claims(text: str) -> list[tuple[str, int, int, int, str, str]]:
    """
    Every high-tier residue-identity claim with where it sits: the three-letter and full-name forms
    (rc.RE_AA3, rc.RE_FULL, lists by rc.RE_ENUM), then the "<residue> at N" forms they miss.

    Args:
        text: The generation.

    Returns:
        (amino acid, position, start, end, matched text, form) per claim; form is "scored" or "at_form".
    """
    out = []
    for match in rc.RE_AA3.finditer(text):
        out.append((rc.AA3[match.group(1).lower()], int(match.group(2)), match.start(), match.end(), match.group(0),
                    "scored"))
    for match in rc.RE_FULL.finditer(text):
        aa = rc.AA_FULL[match.group(1).lower()]
        out.append((aa, int(match.group(3)), match.start(), match.end(), match.group(0), "scored"))
        cursor = match.end()
        while match.group(2):
            tail = rc.RE_ENUM.match(text, cursor)
            if not tail:
                break
            out.append((aa, int(tail.group(1)), tail.start(1), tail.end(1), "(enum) " + tail.group(0), "scored"))
            cursor = tail.end()
    for match in RE_MISSED.finditer(text):
        # "aspartate, glutamate and asparagine at 241, 244 and 260" pairs respectively, and "arginine to lysine at
        # 219" names the mutant: neither reads as one residue at one position.
        if RE_MULTI_AA.search(text[max(0, match.start() - 40):match.start()]):
            continue
        word = match.group(1).lower()
        aa = rc.AA3.get(word) or rc.AA_FULL.get(word)
        out.append((aa, int(match.group(3)), match.start(), match.end(), match.group(0), "at_form"))
        cursor = match.end()
        while match.group(2):
            tail = rc.RE_ENUM.match(text, cursor)
            if not tail:
                break
            out.append((aa, int(tail.group(1)), tail.start(1), tail.end(1), "(enum) " + tail.group(0), "at_form"))
            cursor = tail.end()
    return out


def loop_flags(located: list[tuple]) -> list[bool]:
    """
    Mark claims inside counting runs (counting_runs): LOOP_RUN or more claims of one residue, each within LOOP_GAP
    characters of the last and within LOOP_STEP positions of it or one constant step on.

    Args:
        located: Claims as located_claims returns them.

    Returns:
        One flag per claim, in the input order.
    """
    order = sorted(range(len(located)), key=lambda i: located[i][2])
    flags = [False] * len(located)
    items = [(located[i][0], located[i][1], located[i][2], located[i][3]) for i in order]
    for run in counting_runs(items):
        for k in run:
            flags[order[k]] = True
    return flags


def counting_runs(items: list[tuple[str, int, int, int]]) -> list[list[int]]:
    """
    The degenerate counting runs in a sequence of numbered items in text order: LOOP_RUN or more items of one label,
    each within LOOP_GAP characters of the last, that either stay within LOOP_STEP positions of the last ("Gly13, Gly14,
    Gly15, ...") or advance by one constant step ("Serines 100, 104, 108, ..., 344").

    Args:
        items: (label, position, start, end) per item, in text order; the label is the residue for residue claims and
            "" for span coordinates.

    Returns:
        Each run as a list of indices into items.
    """
    runs = []
    for constant in (False, True):
        run, step = [0] if items else [], None
        for k in range(1, len(items)):
            a, b = items[k - 1], items[k]
            delta = b[1] - a[1]
            near = a[0] == b[0] and b[2] - a[3] <= LOOP_GAP
            if near and not constant and abs(delta) <= LOOP_STEP:
                run.append(k)
                continue
            if near and constant and delta > 0 and step in (None, delta):
                run.append(k)
                step = delta
                continue
            if len(run) >= LOOP_RUN:
                runs.append(run)
            run, step = ([k - 1, k], delta) if near and constant and delta > 0 else ([k], None)
        if len(run) >= LOOP_RUN:
            runs.append(run)
    return runs


def in_partner_list(text: str, start: int) -> bool:
    """
    Whether a claim sits in a list item under an interaction-partners heading.

    Args:
        text: The generation.
        start: Claim start.

    Returns:
        True if the claim's line is a "- " item and the heading above it names interaction partners.
    """
    line_start = text.rfind("\n", 0, start) + 1
    if not text[line_start:start].lstrip().startswith("-"):
        return False
    block = text[max(0, line_start - 3000):line_start]
    headers = list(RE_LIST_HEADER.finditer(block))
    return bool(headers) and bool(RE_PARTNER_HEADER.search(headers[-1].group(0)))


def claim_category(text: str, claim: tuple, loop: bool) -> str:
    """
    Classify one residue claim; the first rule that fires wins.

    Args:
        text: The generation.
        claim: A located claim.
        loop: Whether it is inside a counting run.

    Returns:
        loop, name, range, chemical, ubiquitin, histone, ctd, zinc, substrate (another molecule's residue: a named
        substrate or partner), homolog or self; only self and homolog are claims about this protein's residue at that
        position.
    """
    aa, pos, start, end, match, _ = claim
    if loop:
        return "loop"
    lead, trail = sentence(text, start, end)
    token = match.replace("(enum) ", "")
    letters = re.match(r"[A-Za-z]+", token)
    if letters and len(letters.group(0)) == 3 and letters.group(0).isupper():
        return "name"
    # "- Protein kinase acting on serine 53" is a partner list item naming this protein's own residue.
    listed = in_partner_list(text, start) and not re.search(r"\b(?:on|at)\s+$", lead)
    if listed or RE_LIKE.match(trail) or RE_NAME_LEAD.search(lead):
        return "name"
    if not match.startswith("(enum)") and (RE_ALIAS_LEAD.search(lead) or RE_ALIAS_TRAIL.match(trail)
                                           or RE_CLASS_TRAIL.match(trail) or token in introduced_names(text)):
        return "name"
    if not match.startswith("(enum)") and RE_WINDOW_TRAIL.match(trail):
        return "range"
    if RE_CHEM.match(trail) or (aa == "H" and pos == 1 and RE_TRIAD_LEAD.search(lead) and RE_TRIAD_TRAIL.match(trail)):
        return "chemical"
    if aa in "KM" and pos in UB_POSITIONS and (RE_UB_AFTER.match(trail) or RE_UB_NEAR.match(trail)
                                               or (aa == "K" and RE_UB_LEAD.search(lead))):
        return "ubiquitin"
    if aa == "G" and pos in UB_GLYCINES and (RE_UB_GLY_LEAD.search(lead) or RE_UB_AFTER.match(trail)):
        return "ubiquitin"
    if RE_HISTONE_LEAD.search(lead[-60:]) or RE_HISTONE_TRAIL.match(trail):
        return "histone"
    # The heptad is often named a sentence earlier ("Dephosphorylating the CTD reshapes the transcription cycle.
    # Removing Ser-5 phosphate ..., while Ser-2 dephosphorylation modulates elongation"), so the whole paragraph counts.
    if aa in "STY" and pos <= 7 and RE_CTD.search(paragraph(text, start, end)):
        return "ctd"
    if aa in "CH" and pos <= 6 and (RE_ZINC_ADJ_LEAD.search(lead) or RE_ZINC_ADJ_TRAIL.match(trail)
                                    or RE_ZINC_WORD.match(trail)):
        return "zinc"
    quoted = lead.endswith(("'", "‘")) and trail.startswith(("'", "’"))
    if quoted and "autophosph" not in (lead + trail).lower():
        return "substrate"
    if RE_SUBSTRATE_LEAD.search(lead) or RE_SUBSTRATE_ACT.search(lead) or RE_SUBSTRATE_TRAIL.match(trail):
        return "substrate"
    partner = RE_PARTNER_TRAIL.match(trail)
    if partner and partner.group(1) not in PARTNER_NOT and not RE_PARTNER_NOT.match(partner.group(1)):
        return "substrate"
    if RE_PARTNER_SUBUNIT.match(trail):
        return "substrate"
    if RE_HOMOLOG.search(lead + token + trail):
        return "homolog"
    return "self"


@functools.lru_cache(maxsize=NAME_CACHE)
def introduced_names(text: str) -> frozenset[str]:
    """
    The residue tokens a generation introduces as protein names ("a protein called Lys1", "the peroxisomal protein
    Lys1"), which name that protein wherever they recur.

    Args:
        text: The generation.

    Returns:
        The tokens, as written ("Lys1").
    """
    return frozenset(m.group(0) for m in RE_COMPACT_TOKEN.finditer(text)
                     if RE_NAME_INTRO.search(text[max(0, m.start() - NAME_INTRO_REACH):m.start()]))


def mature_numbering(sequence: str, aa: str, pos: int, chain_starts: list[int]) -> bool:
    """
    Whether a wrong claim at position 1 is right in mature-chain numbering ("Thr1" of a proteasome beta subunit).

    Args:
        sequence: The precursor sequence.
        aa: Claimed amino acid.
        pos: Claimed position.
        chain_starts: Starts (beyond 1) of the protein's UniProt CHAIN and PEPTIDE records.

    Returns:
        True if the claim is at position 1, the precursor does not start with that residue, and a processed chain
        does.
    """
    if pos != 1 or not sequence or sequence[0] == aa:
        return False
    return any(start <= len(sequence) and sequence[start - 1] == aa for start in chain_starts)


def residue_claims(record: dict, truth: dict | None) -> list[dict]:
    """
    Every high-tier residue-identity claim of one generation, with its category and verdict.

    Args:
        record: Exported eval record (protein_sequence, generated_response, sequence_length, input_prompt).
        truth: The protein's truth.

    Returns:
        One dict per claim: aa, pos, start, end, match, form, category, counted (scored as a claim about this
        protein), correct (None when not counted), rule ("mature_numbering" when that made it right), in_prompt and
        context. Empty when the record is not usable.
    """
    if not usable(record, truth):
        return []
    seq = record["protein_sequence"]
    text = record["generated_response"]
    prompt_coords = set(RE_PROMPT_COORD.findall(record.get("input_prompt") or ""))
    located = located_claims(text)
    flags = loop_flags(located) if located else []
    mixed = mixed_list_offsets(text)
    out = []
    for claim, loop in zip(located, flags):
        aa, pos, start, end, match, form = claim
        lead, trail = sentence(text, start, end)
        row = {"aa": aa, "pos": pos, "start": start, "end": end, "match": match, "form": form,
               "in_prompt": str(pos) in prompt_coords, "context": lead + text[start:end] + trail, "rule": ""}
        if not 1 <= pos <= len(seq):
            row.update(category="out_of_range", counted=False, correct=None)
            out.append(row)
            continue
        category = claim_category(text, claim, loop)
        if category in COUNTED_CATEGORIES and start in mixed:
            category = "mixed_list"
        counted = category in COUNTED_CATEGORIES
        correct = seq[pos - 1] == aa
        if counted and not correct and mature_numbering(seq, aa, pos, truth["chain_starts"]):
            correct = True
            row["rule"] = "mature_numbering"
        row.update(category=category, counted=counted, correct=correct if counted else None)
        out.append(row)
    return out


def mixed_list_offsets(text: str) -> set[int]:
    """
    Where the claims of a residue list headed by two plural residue words start ("the zinc site at cysteines and
    histidines 33, 36, 50 and 52"), which the original enumeration would score as all histidines.

    Args:
        text: The generation.

    Returns:
        Start offsets of the list's head claim and of each enumerated number, as located_claims records them.
    """
    out = set()
    for match in rc.RE_FULL.finditer(text):
        if not match.group(2) or not RE_MIXED_HEAD.search(text[max(0, match.start() - LEAD):match.start()]):
            continue
        out.add(match.start())
        cursor = match.end()
        while tail := rc.RE_ENUM.match(text, cursor):
            out.add(tail.start(1))
            cursor = tail.end()
    return out


def answer_layout(text: str) -> dict[str, int]:
    """
    Locate the end of the reasoning and the start of the answer's interaction-partners section.

    Args:
        text: The generation.

    Returns:
        {"think_end": offset of "</think>" or -1, "partner_start": offset of the partners heading or -1}.
    """
    think_end = text.find("</think>")
    partner = RE_PARTNER_HEADER.search(text, think_end) if think_end >= 0 else None
    return {"think_end": think_end, "partner_start": partner.start() if partner else -1}


def contrast_in_scope(lead: str, last: re.Match) -> bool:
    """
    Whether a keyword sits inside the scope of "rather than" or "instead of" that ends its lead.

    Args:
        lead: Text before the keyword, as the negation check reads it.
        last: The last clause break in it.

    Returns:
        True if the break is "rather" + "than" or "instead" + "of" and no new clause starts before the keyword.
    """
    word = last.group(0).lower()
    after = lead[last.end():]
    head = after.lstrip().lower()
    if not ((word == "rather" and head.startswith("than")) or (word == "instead" and head.startswith("of"))):
        return False
    return not RE_SCOPE_END.search(after)


def go_term_name(text: str, hit: re.Match, layout: dict[str, int]) -> bool:
    """
    Whether a keyword is part of a GO term's name rather than a statement about the protein.

    Args:
        text: The generation.
        hit: The keyword match.
        layout: answer_layout of the text.

    Returns:
        True on an answer list line carrying a GO id, or right before or after a GO id in the same term name.
    """
    start, end = hit.span()
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:line_end if line_end >= 0 else len(text)]
    if 0 <= layout["think_end"] < start and line.lstrip().startswith("-") and RE_GO_ID.search(line):
        return True
    return go_id_after(text, end) or bool(RE_GO_BEFORE.search(text[max(0, start - 80):start]))


def go_id_after(text: str, end: int) -> bool:
    """
    Whether RE_GO_AFTER matches at end, answered without it when no GO id can follow the term-name characters there.

    Args:
        text: The generation.
        end: Where the keyword ends.

    Returns:
        RE_GO_AFTER.match(text, end) as a bool.
    """
    stop = RE_GO_NAME_RUN.match(text, end).end()
    if text[max(end, stop - 2):stop + 1].upper() != "GO:" and text[stop:stop + 4].upper() != "(GO:":
        return False
    return bool(RE_GO_AFTER.match(text, end))


def dismissed_prediction(text: str, hit: re.Match, name: str) -> bool:
    """
    Whether a keyword reports a tool's call that the same sentence then rejects ("One predictor calls 5 to 23 a
    transmembrane helix, but that is the hydrophobic core of the signal").

    Args:
        text: The generation.
        hit: The keyword match.
        name: Feature type.

    Returns:
        True if a tool cue governs the keyword in its clause and a "but" later in the sentence leads to a dismissal.
    """
    start, end = hit.span()
    stops = [m.end() for m in RE_SENTENCE_STOP.finditer(text, max(0, start - SENTENCE_REACH), start)]
    before = text[stops[-1] if stops else max(0, start - SENTENCE_REACH):start]
    stop = RE_SENTENCE_STOP.search(text, end, end + SENTENCE_REACH)
    after = text[end:stop.start() if stop else min(len(text), end + SENTENCE_REACH)]
    cues = list(RE_TOOL_CUE.finditer(before))
    governed = bool(cues) and not rc.RE_CLAUSE_BREAK.search(before[cues[-1].end():])
    if governed and (RE_MISREAD.search(before[cues[-1].start():]) or RE_UNPERSUADED.search(after)):
        return True
    if not (governed or RE_TOOL_CUE.fullmatch(hit.group(0)) or RE_TOOL_AFTER.match(after)):
        return False
    if RE_UNPERSUADED.search(after) or later_denial(text, hit, name):
        return True
    if name == "transmembrane" and (RE_IDENTIFIED.search(after) or RE_SIGNAL_CAUGHT.search(before)):
        return True
    but = RE_BUT.search(after)
    if not but:
        return False
    rest = after[but.end():]
    return bool(RE_DISMISSAL.search(rest) or (name == "transmembrane" and RE_DISMISSAL_TM.search(rest)))


def later_denial(text: str, hit: re.Match, name: str) -> bool:
    """
    Whether the paragraph goes on to deny the type a tool's call named ("the predictors split between a cleavable signal
    and a transmembrane anchor ..., a lipid-anchored protein ... rather than a transmembrane one").

    Args:
        text: The generation.
        hit: The keyword match the tool's call governs.
        name: Feature type.

    Returns:
        True if the next keyword of the type in the same paragraph, within DENIAL_REACH and other than a signal anchor,
        is negated or set aside by "rather than", and not by a limited denial ("no second transmembrane helix"), by "not
        merely", or by a denial that a new statement ends before the keyword (", so", "and for").
    """
    line_end = text.find("\n", hit.end())
    stop = min(line_end if line_end >= 0 else len(text), hit.end() + DENIAL_REACH)
    for later in PATTERNS[name].finditer(text, hit.end(), stop):
        if re.fullmatch(SIGNAL_ANCHOR, later.group(0), re.IGNORECASE):
            continue
        lead = text[max(0, later.start() - rc.NEGATION_WINDOW):later.start()]
        breaks = list(rc.RE_CLAUSE_BREAK.finditer(lead))
        last = breaks[-1] if breaks else None
        clause = lead[last.end():] if last else lead
        if last and contrast_in_scope(lead, last):
            return True
        cues = [m for m in rc.RE_NEGATION.finditer(clause)
                if not RE_UNINTERRUPTED.match(clause, m.start()) and not RE_NOT_MERELY.match(clause, m.start())]
        if not cues or RE_LIMITED_DENIAL.search(clause[cues[-1].start():]) \
                or RE_DENIAL_RELEASE.search(clause, cues[-1].end()):
            return False
        return negated(text, later, clause)
    return False


def negated(text: str, hit: re.Match, clause: str) -> bool:
    """
    Whether a negation in the keyword's clause reaches it: the last negation cue governs unless a new main clause with a
    verb of its own starts after it (", and the decisive chemistry is", "with no hydrophobic core signature, fits") and
    the keyword's own predicate is not negated after it.

    Args:
        text: The generation.
        hit: The keyword match.
        clause: The keyword's clause before it, as hit_reason cuts it.

    Returns:
        True if the keyword is negated.
    """
    cues = [m for m in rc.RE_NEGATION.finditer(clause) if not RE_UNINTERRUPTED.match(clause, m.start())]
    cues += list(RE_EXTRA_NEGATION.finditer(clause))
    if not cues:
        return False
    last = max(cues, key=lambda m: m.end())
    stop = RE_CLAUSE_STOP.search(text, hit.end(), hit.end() + SENTENCE_REACH)
    verb_after = RE_CLAUSE_VERB.search(text, hit.end(), stop.start() if stop else hit.end() + SENTENCE_REACH)
    boundaries = [m for m in RE_NEW_CLAUSE.finditer(clause, last.end())
                  if RE_CLAUSE_VERB.search(clause, m.end()) or verb_after]
    if RE_WITH_NEGATION.search(clause, 0, last.end()):
        boundaries += list(RE_COMMA_VERB.finditer(clause, last.end()))
    if not boundaries:
        return True
    return bool(RE_TRAILING_NEGATION.match(text, hit.end()))


def sentence_lead(text: str, start: int, reach: int) -> str:
    """
    The part of a keyword's sentence before it, capped at a reach.

    Args:
        text: The generation.
        start: Keyword start.
        reach: Most characters to take.

    Returns:
        The text from the sentence's start (or the reach) to the keyword.
    """
    stops = [m.end() for m in RE_SENTENCE_STOP.finditer(text, max(0, start - reach), start)]
    return text[stops[-1] if stops else max(0, start - reach):start]


def other_molecule(text: str, hit: re.Match, name: str) -> bool:
    """
    Whether a keyword names another molecule's feature rather than the protein's: the noun it modifies is a partner, a
    class in the plural or (for a membrane word) a membrane machine's part ("transmembrane cargo", "type II and type I
    transmembrane serine/threonine kinase receptors", "the membrane-embedded Vo sector", "the transmembrane segments of
    hydrophobic precursor proteins"), a partner or a plural owns it ("the client's transmembrane helices"), it is part
    of a partner's name ("Transmembrane protein 231"), the protein binds or docks onto the molecule it describes
    ("docked onto the transmembrane permease"), or it is said of a partner ("The one contact I have to work with is a
    small integral membrane protein of the endoplasmic reticulum, SMIM12", "a transmembrane protein is exactly the kind
    of partner").

    Args:
        text: The generation.
        hit: The keyword match.
        name: Feature type.

    Returns:
        True if so, unless the keyword names a family the protein belongs to ("one of the multi-pass membrane proteins",
        "like other transmembrane receptors") or the protein is said to be or form the noun.
    """
    lead = sentence_lead(text, hit.start(), OTHER_REACH)
    after = text[hit.end():hit.end() + OTHER_REACH]
    if RE_FAMILY_LEAD.search(lead) or RE_SELF_PREDICATE_LEAD.search(lead):
        return False
    if RE_OTHER_POSSESSIVE.search(lead) or RE_PARTNER_SUBJECT.search(lead) or RE_OTHER_SUBJECT.search(lead):
        return True
    if RE_VERB_HIT.search(hit.group(0)):
        return False
    modifier = bool(RE_MODIFIER_HIT.search(hit.group(0)))
    if (RE_OTHER_HEAD_LIST if modifier else RE_OTHER_HEAD).match(after) or RE_OTHER_OF.match(after):
        return True
    if RE_NAMED_PARTNER.match(after) or RE_OTHER_PREDICATE.match(after):
        return True
    if RE_PARTNER_ACT.search(lead) and RE_MOLECULE_HEAD.match(after):
        return True
    return name == "transmembrane" and modifier and bool(RE_OTHER_MACHINE.match(after))


def peripheral_anchor(text: str, hit: re.Match) -> bool:
    """
    Whether a "membrane anchor" is a lipid, amphipathic or lipid-binding-domain anchor rather than a transmembrane
    segment ("an amino-terminal C2 membrane anchor", "a fourteen-carbon acyl chain ... is a membrane anchor"), or one
    the sentence calls peripheral or not crossing the bilayer.

    Args:
        text: The generation.
        hit: The keyword match.

    Returns:
        True if so.
    """
    if not RE_ANCHOR_HIT.fullmatch(hit.group(0)):
        return False
    lead = sentence_lead(text, hit.start(), ANCHOR_LEAD)
    stop = RE_SENTENCE_STOP.search(text, hit.end(), hit.end() + SENTENCE_REACH)
    after = text[hit.end():stop.start() if stop else min(len(text), hit.end() + SENTENCE_REACH)]
    return bool(RE_ANCHOR_KIND.search(lead) or RE_ANCHOR_DENIAL.search(after))


def rejected_alternative(clause: str) -> bool:
    """
    Whether a keyword is the reading the model sets aside: the alternative after "than" ("better read as a buried
    hydrophobic helix than as a true transmembrane anchor"), what it rejects ("the one reading I reject is the
    transmembrane assignment") or what something is mistaken for ("long enough to be mistaken for a transmembrane
    helix").

    Args:
        clause: The keyword's clause before it.

    Returns:
        True if so.
    """
    return bool(RE_THAN_LEAD.search(clause) or RE_REJECT_LEAD.search(clause))


def partner_pocket(text: str, hit: re.Match) -> bool:
    """
    Whether a nucleotide or substrate pocket, cleft or domain belongs to a partner the protein acts on: a client
    GTPase's nucleotide pocket an exchange or activating factor opens ("binds Ras-GDP, pries open the nucleotide
    pocket"), or a pocket the protein inserts into or binds when a GTPase is named or a partner owns it ("it inserts
    into the nucleotide-binding pocket" of a Rab, "binds the Hsp70 nucleotide-binding domain", "inserts into the
    protease substrate-binding cleft", "invades the nucleotide pocket of EF-Tu").

    Args:
        text: The generation.
        hit: The keyword match.

    Returns:
        True if so.
    """
    before = sentence_lead(text, hit.start(), PARTNER_REACH)
    if RE_NUCLEOTIDE_POCKET.search(hit.group(0)) and RE_GTPASE_PARTNER.search(before):
        return True
    pocket = RE_POCKET_NOUN.match(text, hit.start(), hit.end() + POCKET_VIEW)
    if not pocket:
        return False
    verb = RE_PARTNER_VERB.search(before)
    return bool(verb and (RE_GTPASE_PARTNER.search(before) or RE_POCKET_OWNER.search(verb.group(0))
                          or RE_POCKET_OF.match(text, pocket.end())))


def receiver_phosphate(text: str, hit: re.Match) -> bool:
    """
    Whether an aspartylphosphate is a receiver domain's phosphorylated aspartate, a modification, rather than a covalent
    catalytic intermediate ("residue 68 is the phosphoacceptor, a 4-aspartylphosphate").

    Args:
        text: The generation.
        hit: The keyword match.

    Returns:
        True if the hit names an aspartylphosphate that no "intermediate" follows and its paragraph names a receiver,
        response regulator, phosphoacceptor or phosphorelay.
    """
    if not RE_ASPARTYL.search(hit.group(0)) or "intermediate" in hit.group(0).lower():
        return False
    if RE_INTERMEDIATE_AFTER.match(text, hit.end()):
        return False
    return bool(RE_RECEIVER.search(paragraph(text, hit.start(), hit.end())))


def hit_reason(text: str, hit: re.Match, name: str, layout: dict[str, int]) -> str:
    """
    Decide whether one keyword hit asserts the feature of this protein, and if not, why.

    Args:
        text: The generation.
        hit: The keyword match.
        name: Feature type.
        layout: answer_layout of the text.

    Returns:
        "" if it asserts the feature, else the reason: rather_than, negated, nowhere, hypothetical, non_prefix,
        other_entity, partner_section, function_phrase, cut_word, resemblance, go_term_name, non_ligand_binding,
        protein_name, partner_pocket, rejected_alternative, other_molecule, peripheral_anchor, receiver_phosphate or
        dismissed_prediction.
    """
    start = hit.start()
    lead = text[max(0, start - rc.NEGATION_WINDOW):start]
    breaks = list(rc.RE_CLAUSE_BREAK.finditer(lead))
    last = breaks[-1] if breaks else None
    clause = lead[last.end():] if last else lead
    if last and contrast_in_scope(lead, last):
        return "rather_than"
    if negated(text, hit, clause):
        return "negated"
    if RE_NOWHERE.search(clause):
        return "nowhere"
    if hypothetical(text, hit, clause):
        return "hypothetical"
    # "a non-cytoplasmic loop at 167 to 185" names the other side of the membrane rather than denying a domain.
    if RE_NON_PREFIX.search(lead) and name != "topological_domain":
        return "non_prefix"
    if name in rc.SELF_ATTRIBUTION_TYPES and rc.RE_OTHER_ENTITY.search(clause):
        return "other_entity"
    if 0 <= layout["partner_start"] < start:
        return "partner_section"
    if RE_FUNCTION_PHRASE.match(text, start) or (RE_PASS_HIT.fullmatch(hit.group(0)) and RE_PASS_IDIOM.search(lead)) \
            or (RE_ANCHOR_HIT.fullmatch(hit.group(0)) and RE_PROTEIN_MEMBRANE_LEAD.search(lead)):
        return "function_phrase"
    if RE_CUT_RESIDUE.fullmatch(hit.group(0)) and text[hit.end():hit.end() + 1].isalpha():
        word = hit.group(0)[-3:] + RE_LETTERS.match(text, hit.end()).group(0)
        if not RE_RESIDUE_WORD.match(word):
            return "cut_word"
    # Domain names keep their "-like" ("a homeodomain-like fold across 8 to 61").
    if name in TOOL_TYPES and RE_LIKE_SUFFIX.match(text, hit.end()):
        return "resemblance"
    if go_term_name(text, hit, layout):
        return "go_term_name"
    # "calcium-binding site" names its own object; a bare one takes it from the word before ("EF-hand binding site") or
    # from the DNA it lies on ("binding sites in enhancers").
    bare = RE_BARE_BINDING.match(hit.group(0))
    if name == "binding_site" and ((bare and (RE_NON_LIGAND_BEFORE.search(lead[-40:])
                                              or RE_DNA_SITES_AFTER.match(text, hit.end())
                                              or RE_DNA_SITES_LEAD.search(lead)))
                                   or RE_RRM_NAME.match(text, start)):
        return "non_ligand_binding"
    named = RE_PROTEIN_NAME.search(text, max(0, start - PROTEIN_NAME_LEAD), hit.end() + PROTEIN_NAME_LEAD)
    if name == "binding_site" and named and named.start() <= start < named.end():
        return "protein_name"
    if name == "binding_site" and partner_pocket(text, hit):
        return "partner_pocket"
    if rejected_alternative(clause) or (name == "transmembrane" and RE_SIGNAL_READ.search(clause)):
        return "rejected_alternative"
    if other_molecule(text, hit, name):
        return "other_molecule"
    if name == "transmembrane" and peripheral_anchor(text, hit):
        return "peripheral_anchor"
    if name == "active_site" and receiver_phosphate(text, hit):
        return "receiver_phosphate"
    if dismissed_prediction(text, hit, name):
        return "dismissed_prediction"
    return ""


def hypothetical(text: str, hit: re.Match, clause: str) -> bool:
    """
    Whether a keyword names what the model would need to see rather than what it sees: inside an "if", "unless" or
    "whether" clause, after "demonstrated", before "would change my mind", "would settle" or "would be odd", after
    "too ... to be", or in a sentence about what would change the model's mind ("What would change my mind is evidence
    of ... a transmembrane anchor").

    Args:
        text: The generation.
        hit: The keyword match.
        clause: The keyword's clause before it, as hit_reason cuts it.

    Returns:
        True if the keyword is hypothetical.
    """
    condition = list(RE_IF.finditer(clause))
    if condition and not RE_MAIN_CLAUSE.search(clause[condition[-1].end():]):
        return True
    if RE_DEMONSTRATED.search(clause) or RE_TOO_TO_BE.search(clause) or RE_WOULD_LEAD.search(clause):
        return True
    if RE_MIND_LEAD.search(sentence_lead(text, hit.start(), SENTENCE_REACH)):
        return True
    return bool(RE_WOULD_AFTER.match(text, hit.end()))


def confusion(said: bool, true: bool) -> str:
    """
    Name a presence decision's cell of the confusion matrix.

    Args:
        said: Whether the text asserts the feature.
        true: Whether the protein has it.

    Returns:
        TP, FP, FN or TN.
    """
    return {(True, True): "TP", (True, False): "FP", (False, True): "FN", (False, False): "TN"}[(said, true)]


def presence(record: dict, truth: dict | None, sites: bool = True, evidence: bool = True) -> dict[str, dict]:
    """
    Decide, for the five tool-annotated feature types, whether the generation asserts the feature (the regex reader;
    the LLM reader is the reader of record) and whether the protein has it.

    Args:
        record: Exported eval record.
        truth: The protein's truth.
        sites: Whether InterProScan site records join the active- and binding-site truth.
        evidence: Whether protein-level evidence joins the truth (presence_truth).

    Returns:
        Feature type to {"said", "true", "verdict", "hits"}, hits being {"text", "start", "reason"} per keyword match
        (reason "" when it counts). Empty when the record is not usable.
    """
    if not usable(record, truth):
        return {}
    text = record["generated_response"]
    layout = answer_layout(text)
    out = {}
    for name in TOOL_TYPES:
        hits = [{"text": m.group(0), "start": m.start(), "reason": hit_reason(text, m, name, layout)}
                for m in PATTERNS[name].finditer(text)]
        said = any(not h["reason"] for h in hits)
        true = presence_truth(truth, name, sites, evidence)
        out[name] = {"said": said, "true": true, "verdict": confusion(said, true), "hits": hits}
    return out


def count_value(token: str) -> int:
    """
    Read a count written in digits or words.

    Args:
        token: "23", "twenty-three", "twenty three".

    Returns:
        The number.
    """
    if token.isdigit():
        return int(token)
    return sum(NUMBER_WORDS[part] for part in re.split(r"[- ]", token.lower()))


def window_bounds(text: str, hit: re.Match) -> tuple[int, int]:
    """
    Bound a keyword's clause window at sentence ends and line breaks.

    Args:
        text: The generation.
        hit: The keyword match.

    Returns:
        Start and end offsets.
    """
    start = max(0, hit.start() - rc.KEYWORD_WINDOW_BACK)
    before = text[start:hit.start()]
    cuts = list(rc.RE_SENTENCE_END.finditer(before)) + list(re.finditer(r"\n", before))
    if cuts:
        start += max(c.end() for c in cuts)
    after = text[hit.end():hit.end() + rc.KEYWORD_WINDOW]
    stops = [m.start() for m in (rc.RE_SENTENCE_END.search(after), re.search(r"\n", after)) if m]
    return start, hit.end() + (min(stops) if stops else len(after))


def cut_number(text: str, w0: int, w1: int, lo: int, hi: int) -> bool:
    """
    Whether a coordinate at text[lo:hi] is a number the window edge cut in two ("aspartate 3" from "aspartate 396").

    Args:
        text: The generation.
        w0: Window start.
        w1: Window end.
        lo: Coordinate start.
        hi: Coordinate end.

    Returns:
        True if the coordinate touches an edge and the text continues with a digit there.
    """
    return (lo == w0 and w0 > 0 and text[w0 - 1].isdigit()) or (hi == w1 and w1 < len(text) and text[w1].isdigit())


def glue(segment: str, name: str = "") -> bool:
    """
    Whether a stretch between two coordinates is list glue rather than a new noun phrase.

    Args:
        segment: The text between them.
        name: Feature type whose own nouns count as glue too, together with generic nouns and quantity words; ""
            for the list words alone.

    Returns:
        True if every word is glue.
    """
    words = {w.lower() for w in RE_WORD.findall(segment)} - LIST_WORDS
    return not words if not name else not words - GENERIC_WORDS - TYPE_WORDS.get(name, set())


def tight(gap: str) -> bool:
    """
    Whether a list hangs directly on its keyword: the keyword's own phrase runs to the first coordinate without a
    break ("an active-site set at 23, 24"), and only list words join the coordinates after it.

    Args:
        gap: The text between the keyword and the coordinate.

    Returns:
        True if attached directly, a comma before a cue word and a colon opening the keyword's own items included
        ("active site, residues 153-165", "crosses the membrane six times: 61 to 89"); a loose attachment can lose the
        list to a nearer head ("Six disulfide bonds stitch the domain together, and two asparagines at 38 and 192 carry
        N-linked glycans").
    """
    digits = list(RE_COORD.finditer(gap))
    lead = gap[:digits[0].start()] if digits else gap
    lead = "" if RE_CUE_COMMA.match(lead) else lead
    colon = own_colon(lead)
    if colon >= 0 and not {w.lower() for w in RE_WORD.findall(lead[colon:])} - CUE_WORDS:
        lead = lead[:colon]
    if RE_LOOSE.search(gap) or RE_LEAD_BREAK.search(lead):
        return False
    # A colon after a list opens a new statement about it ("at 8, 9, 39, 91, 121 and 123: Asp8, Asp39, Asn91 and Ser123
    # are ... a polar cluster that can bind a divalent metal").
    joins = [gap[a.end():b.start() if b else len(gap)] for a, b in zip(digits, digits[1:] + [None])]
    return all(":" not in join and glue(join) for join in joins)


def typed_types(segment: str) -> set[str]:
    """
    The feature types a phrase names, by a keyword or by a typed word; "non_ligand" for DNA, RNA or protein binding.

    Args:
        segment: The phrase.

    Returns:
        Feature type names.
    """
    found = {"non_ligand"} if RE_NOT_LIGAND_TYPED.search(segment) else set()
    # "a scattered DNA-contact surface" names DNA, not a ligand contact.
    rest = RE_NOT_LIGAND_TYPED.sub(" ", segment)
    found |= {other for other, pattern in PATTERNS.items() if pattern.search(rest)}
    found |= {other for other, pattern in TYPED_WORDS.items() if pattern.search(rest)}
    return found


def residue_letters(segment: str) -> set[str]:
    """
    The amino acids a phrase names ("two asparagines", "cysteines", "Ser").

    Args:
        segment: The phrase.

    Returns:
        One-letter codes.
    """
    out = set()
    for word in RE_WORD.findall(segment):
        lower = word.lower()
        stem = lower[:-1] if lower.endswith("s") and lower[:-1] in rc.AA_FULL else lower
        if stem in rc.AA_FULL:
            out.add(rc.AA_FULL[stem])
        elif lower in rc.AA3 and (word.istitle() or word.isupper()):
            out.add(rc.AA3[lower])
    return out


def rival_head(gap: str, name: str, colon: int = -1) -> bool:
    """
    Whether the phrase that last heads coordinates in a forward gap heads a list of its own: it names another feature
    type and not this one ("a catalytic set at 111" after a magnesium site), it is a domain-level phrase or a named
    element right before coordinates ("and the kinase domain at", "and a cell attachment site at"), it names a residue
    the feature cannot sit on ("and two asparagines at 38 and 192" after a disulfide), or, after a metal site, it names
    another ligand right before coordinates ("prenylated FMN is contacted at 168 to 170"). Parts, generic nouns and
    quantity words head nothing of their own ("a hydrophobic h-region at 4 to 15", "plus a scattered set across the
    C-terminal lobe at"), and a later phrase naming the keyword's own type takes the list back ("an ion transport
    domain, from 408 to 667, and inside it I read six crossings: S1 at 408 to 431, S2 at 450 to 475").

    A place named on the way is not a head ("the largest extracellular segment between the fifth and sixth crossings,
    around 272 to 311"), and a domain-level noun heads only the coordinates right after it ("the disulfide density
    across the disintegrin and cysteine-rich domains, with the annotated pair at 589 to 600" keeps the pair), not
    those it is joined to by "and" ("191 to 210 in the first domain and 175 to 210").

    Args:
        gap: The text from the keyword's end to the coordinate.
        name: The keyword's feature type.
        colon: Offset in the gap of a colon opening the keyword's own items, whose domain words are its parts (a
            family still heads its own range, and a named element does after the first item: "the carboxyl-proximal
            heme pocket: a conserved P450 site at 435 to 444" is the pocket); -1 when there is none.

    Returns:
        True if such a phrase lies between them.
    """
    allowed = SEGMENT_ALLOWED.get(name, {name})
    first = True
    rival = False
    items_seen = False
    bounds = [0] + [x for m in RE_COORD.finditer(gap) for x in (m.start(), m.end())] + [len(gap)]
    for piece_start, piece_end in zip(bounds[0::2], bounds[1::2]):
        parts = RE_SEGMENT_SPLIT.split(gap[piece_start:piece_end])
        segments, separators = parts[0::2], [""] + parts[1::2]
        # The phrase that heads the next coordinate is the last one with a word beyond range cues ("C-terminal domain,
        # residues 227-247").
        heading = max((i for i, s in enumerate(segments) if {w.lower() for w in RE_WORD.findall(s)} - CUE_WORDS),
                      default=-1)
        after_place = False
        offset = piece_start
        for index, (segment, separator) in enumerate(zip(segments, separators)):
            offset += len(separator)
            listed = 0 <= colon < offset
            first_item = listed and not items_seen and bool(RE_WORD.search(segment))
            items_seen = items_seen or first_item
            offset += len(segment)
            if name in RESIDUE_TYPES and residue_letters(segment) - set(RESIDUE_TYPES[name]):
                return True
            place = bool(RE_PLACE_START.match(segment)) or (separator.lower() == "and" and after_place)
            after_place = place or bool(RE_PLACE.search(segment))
            if first:
                first = False
                continue
            types = typed_types(segment)
            if name == "metal_binding" and "binding_site" in types and not RE_OTHER_LIGAND.search(segment):
                types = (types - {"binding_site"}) | {"metal_binding"}
            own = bool(types & allowed) or (name == "dna_binding" and "non_ligand" in types)
            if own:
                rival = False
                continue
            other = types - allowed - ({"non_ligand"} if name == "dna_binding" else set())
            # A heading phrase joined to the coordinate by "and" modifies the coordinate before it ("191 to 210 in the
            # first domain and 175 to 210").
            direct = not any(sep.lower() in JOINERS for sep in separators[index + 1:])
            heads = index == heading and direct
            element = RE_ELEMENT.search(segment) and not first_item and not RE_POSSESSIVE_ELEMENT.search(segment)
            level = RE_FAMILY.search(segment) or element or (not listed and RE_DOMAIN_LEVEL.search(segment))
            if (other and not place) or (heads and level):
                rival = True
            elif heads and name == "metal_binding" and RE_OTHER_LIGAND.search(segment):
                rival = True
            elif heads and name == "transmembrane" and RE_TOPOLOGY_PART.search(segment):
                rival = True
    return rival


def has_own_coordinates(text: str, end: int, w1: int) -> bool:
    """
    Whether a keyword heads coordinates of its own in its phrase ("the catalytic lysine at 442").

    Args:
        text: The generation.
        end: The keyword's end.
        w1: Window end.

    Returns:
        True if a coordinate follows before the phrase ends.
    """
    tail = text[end:w1]
    stop = RE_OWN_END.search(tail)
    return bool(RE_OWN_COORD.search(tail[:stop.start()] if stop else tail))


def rival_nouns(name: str) -> re.Pattern:
    """
    The rival nouns that take coordinates from a type's keyword: the sides of the membrane rival every type but the
    topological domain, which they name.

    Args:
        name: The keyword's feature type.

    Returns:
        RE_RIVAL_TOPOLOGY for a topological domain, else RE_RIVAL.
    """
    return RE_RIVAL_TOPOLOGY if name == "topological_domain" else RE_RIVAL


def names_head(segment: str) -> bool:
    """
    Whether a phrase names something that heads coordinates of its own: a feature keyword or typed word of any type,
    a rival noun, a non-ligand binding phrase, a domain-level noun or a named element.

    Args:
        segment: The phrase.

    Returns:
        True if it does.
    """
    sources = list(PATTERNS.values()) + list(TYPED_WORDS.values()) + [RE_RIVAL, RE_NOT_LIGAND_TYPED, RE_DOMAIN_LEVEL,
                                                                        RE_ELEMENT]
    return any(pattern.search(segment) for pattern in sources)


def cites(text: str, accession_end: int, lo: int) -> bool:
    """
    Whether a coordinate is on an InterPro entry's own coordinate list: the citation's first coordinate after the
    accession (digits in the entry's name do not count), or one its statement runs on to before any other head.

    Args:
        text: The generation.
        accession_end: End of the accession.
        lo: Coordinate start.

    Returns:
        True if the entry's citation holds the coordinate, in the same sentence: every coordinate between them is
        joined to the next by enumeration, or by words naming no head and starting no new noun phrase, so "IPR036640
        across 252 to 585, with helices at 274 to 292" holds 252-585 only, while "IPR001609 marks the myosin head
        repeatedly from 71-758 down to short internal spans such as 107-126" and "IPR016024, across 111 to 442, leaving
        33 to 110" hold every range; a residue named after an uncited entry ("under IPR017972, with cysteine 451") is
        not on its list.
    """
    between = text[accession_end:lo]
    if RE_ACCESSION.search(between) or RE_CITATION_BREAK.search(between):
        return False
    first = RE_CITATION_START.search(between)
    if not first and RE_RESIDUE_END.search(between):
        return False
    # The reach bounds the entry's name and verb; its coordinate list may run on.
    if len(between[:first.start()] if first else between) > CITATION_REACH:
        return False
    if not first:
        return not RE_CLAUSE_START.search(between)
    digits = list(RE_COORD.finditer(between, first.start()))
    if not digits:
        return True
    joins = [between[a.end():b.start()] for a, b in zip(digits, digits[1:])] + [between[digits[-1].end():]]
    return all(RE_ENUM_GLUE.match(join) or not (names_head(join) or RE_BACK_HARD.search(join) or RE_LOOSE.search(join))
               for join in joins)


def own_accession(text: str, hit_end: int, acc: re.Match) -> bool:
    """
    Whether an accession after a keyword is the keyword's own entry, cited in apposition ("a putative metal-binding site
    (IPR024722, residues 234-245)") or listed after it.

    Args:
        text: The generation.
        hit_end: The keyword's end.
        acc: The accession match.

    Returns:
        True if apposed_accession holds for it, or if it is joined by list glue alone to the keyword's own
        cited accession ("two TM helix signatures IPR013110 (..., residues 706-774) and IPR057588 (..., residues
        883-906)").
    """
    if apposed_accession(text, hit_end, acc):
        return True
    previous = list(RE_ACCESSION.finditer(text, hit_end, acc.start()))
    if not previous or not apposed_accession(text, hit_end, previous[0]):
        return False
    for first, second in zip(previous, previous[1:] + [acc]):
        tail = RE_CITATION_TAIL.match(text[first.end():second.start()])
        if not RE_JOINER.match(text[first.end() + (tail.end() if tail else 0):second.start()]):
            return False
    return True


def apposed_accession(text: str, hit_end: int, acc: re.Match) -> bool:
    """
    Whether an accession right after a keyword is cited in apposition to it ("a putative metal-binding site
    (IPR024722, residues 234-245)").

    Args:
        text: The generation.
        hit_end: The keyword's end.
        acc: The accession match.

    Returns:
        True if only a few words of the keyword's noun phrase, none a place, lie between them, and an accession after a
        comma is cited rather than the subject of a verb.
    """
    lead = text[hit_end:acc.start()]
    if not RE_OWN_ACCESSION_LEAD.match(lead) or RE_LEAD_PLACE.search(lead):
        return False
    return not lead.rstrip().endswith(",") or bool(RE_CITED_AFTER.match(text, acc.end()))


def apposed(text: str, start: int, end: int) -> bool:
    """
    Whether the text from the end of an entry's citation to a later keyword sets the keyword in apposition to the
    entry ("IPR002487 (Transcription factor, K-box), a coiled-coil module").

    Args:
        text: The generation.
        start: Where the citation's accession or coordinate ends.
        end: The keyword's start.

    Returns:
        True if the keyword is the entry's own description.
    """
    return bool(RE_APPOSITION.match(RE_CITATION_TAIL.sub("", text[start:end], count=1)))


def entry_citation(text: str, hit: re.Match, lo: int) -> bool:
    """
    Whether a keyword sits in an InterPro entry's name and the coordinate is on that entry's citation ("IPR001440
    Coiled-coil vimentin, conserved site, residues 1568-1584").

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.

    Returns:
        True if so; the entry's coordinates are then the keyword's own whatever its name says after the keyword (a
        family entry's are not, family_entry).
    """
    line_start = text.rfind("\n", 0, hit.start()) + 1
    accessions = list(RE_ACCESSION.finditer(text, max(line_start, hit.start() - ENTRY_REACH), hit.start()))
    if not accessions or not RE_ENTRY_NAME_LEAD.match(text[accessions[-1].end():hit.start()]):
        return False
    return cites(text, accessions[-1].end(), lo)


def family_entry(text: str, hit: re.Match, lo: int) -> bool:
    """
    Whether a keyword sits in the name of a family entry whose citation holds the coordinate ("IPR050463 (Distal-less
    homeobox transcription factors family) spanning residues 12-286"): the range is the lineage's extent, not a
    placement of the keyword's feature.

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.

    Returns:
        True if entry_citation holds and the entry's name, read to its end (a bracket, its coordinates or its type
        label), ends with "family", or its type label is "(family)".
    """
    if not entry_citation(text, hit, lo):
        return False
    line_start = text.rfind("\n", 0, hit.start()) + 1
    accession = list(RE_ACCESSION.finditer(text, max(line_start, hit.start() - ENTRY_REACH), hit.start()))[-1]
    rest = text[hit.end():lo]
    end = RE_ENTRY_END.search(rest)
    name = text[accession.end():hit.end()] + (rest[:end.start()] if end else rest)
    return bool(RE_FAMILY_END.search(name) or (end and RE_FAMILY_LABEL.match(rest, end.start())))


def broader_family(text: str, hit: re.Match, lo: int) -> bool:
    """
    Whether the keyword modifies a family or superfamily name the model calls broader and that name heads the
    coordinate ("the broader RING/FYVE/PHD zinc-finger superfamily at 197 to 269").

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.

    Returns:
        True if "broader" (or wider, generic, fold-level) leads the keyword's phrase and only a few modifiers, the
        family noun and cues lie between the keyword and the coordinate.
    """
    return bool(RE_BROAD_LEAD.search(text, max(0, hit.start() - BROAD_REACH), hit.start())
                and RE_FAMILY_PHRASE.match(text, hit.end(), lo))


def dismissed_family(text: str, hit: re.Match) -> bool:
    """
    Whether the entry whose name holds the keyword is a family or superfamily the generation sets aside where it cites
    it, by a dismissal ("IPR027417 (P-loop containing ... homologous superfamily, residues 1-25), but ... this signal
    reflects a generic P-loop-like motif rather than an NTPase function") or by calling it broader ("IPR013083 marks the
    broader RING/FYVE/PHD zinc-finger superfamily at 197 to 269"); this holds wherever the entry is listed again.

    Args:
        text: The generation.
        hit: The keyword match, inside the entry's name.

    Returns:
        True if some citation of the entry names a family before its first coordinate and calls it broader there, or
        its sentence goes on to a "but" followed by a dismissal.
    """
    line_start = text.rfind("\n", 0, hit.start()) + 1
    accessions = list(RE_ACCESSION.finditer(text, max(line_start, hit.start() - ENTRY_REACH), hit.start()))
    if not accessions:
        return False
    for cited in re.finditer(accessions[-1].group(0), text):
        stop = RE_SENTENCE_STOP.search(text, cited.end(), cited.end() + SENTENCE_REACH)
        after = text[cited.end():stop.start() if stop else min(len(text), cited.end() + SENTENCE_REACH)]
        # The entry's own statement ends where another entry is cited, and a family placed or named for another entry
        # is not this one's ("(IPR017441) and catalytic (IPR008271) motifs inside the broader kinase-like superfamily
        # (IPR011009)").
        other = RE_ACCESSION.search(after)
        first = RE_CITATION_START.search(after, 0, other.start() if other else len(after))
        family = RE_FAMILY.search(after, 0, first.start() if first else (other.start() if other else len(after)))
        if not family or RE_LEAD_PLACE.search(after, 0, family.start()) \
                or (other and RE_NAMES_NEXT.match(after[family.end():other.start()])):
            continue
        after = after[:other.start()] if other else after
        but = RE_BUT.search(after)
        if RE_BROAD_LEAD.search(after, max(0, family.start() - BROAD_REACH), family.start()) \
                or (but and RE_DISMISSAL.search(after, but.end())):
            return True
    return False


def own_colon(gap: str) -> int:
    """
    Where a colon right after the keyword's phrase opens the keyword's own items.

    Args:
        gap: The text from the keyword's end to the coordinate.

    Returns:
        The colon's offset in the gap, or -1 when there is none, the keyword's phrase runs past OWN_COLON_WORDS words
        before it, or the items cite an InterPro entry (its range is the entry's, "the catalytic machinery: IPR000668,
        the papain C-terminal peptidase domain, spans 80 to 329").
    """
    colon = RE_OWN_COLON.match(gap)
    if not colon or len(RE_WORD.findall(colon.group(0))) > OWN_COLON_WORDS or "IPR" in gap:
        return -1
    return colon.end() - 1


def accession_blocks(text: str, hit: re.Match, lo: int, hi: int) -> bool:
    """
    Whether an InterPro accession means the coordinate belongs to that entry rather than to the keyword.
    Forward, an accession between them is passed when it is the keyword's own in apposition ("a putative metal-binding
    site (IPR024722, residues 234-245)"), when it trails a coordinate ("a copper-binding motif at 711 to 724
    (IPR049947) with copper ligands at 551"), or when the coordinate is not on its list ("IPR036640 across 252 to 585,
    with helices at 274 to 292"). Backward, the entry that owns the coordinate (an accession between them, or one whose
    citation holds the coordinate, searched back to the line's start) takes it unless the keyword stands in apposition
    to that entry ("residues 85-178 form IPR002487 (Transcription factor, K-box), a coiled-coil module"); "The
    N-terminus (residues 1-24) carries IPR040920 (...), immediately followed by a trio of signal peptide annotations"
    is the repeat's.

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.
        hi: Coordinate end.

    Returns:
        True if an accession takes the coordinate.
    """
    if lo >= hit.end():
        for acc in RE_ACCESSION.finditer(text, hit.end(), lo):
            if own_accession(text, hit.end(), acc):
                continue
            if RE_TRAILING_ACCESSION.search(text[hit.end():acc.start()]):
                continue
            if cites(text, acc.end(), lo):
                return True
        return False
    accessions = list(RE_ACCESSION.finditer(text, hi, hit.start()))
    if any(RE_COORD.search(text[acc.end():hit.start()]) for acc in accessions):
        return True
    if accessions:
        last = accessions[-1].end()
        if not (RE_ENTRY_NAME_LEAD.match(text[last:hit.start()]) or apposed(text, last, hit.start())):
            return True
    reach = max(0, lo - OWNER_REACH)
    stops = [m.end() for m in RE_CITATION_BREAK.finditer(text, reach, lo)]
    start = stops[-1] if stops else reach
    owners = list(RE_ACCESSION.finditer(text, start, lo))
    if not owners:
        return False
    owner = owners[-1]
    if RE_TRAILING_ACCESSION.search(text[start:owner.start()]) or not cites(text, owner.end(), lo):
        return False
    return not apposed(text, hi, hit.start())


def forward_reason(text: str, hit: re.Match, lo: int, hi: int, name: str, w0: int, w1: int) -> tuple[str, bool]:
    """
    Whether a coordinate after a keyword belongs to it: the nearest head wins.

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.
        hi: Coordinate end.
        name: The keyword's feature type.
        w0: Window start.
        w1: Window end.

    Returns:
        ("" if it does, else accession_between, broader_family, family_entry, dismissed_family, rival_noun,
        new_phrase, nearer_keyword, rival_head, new_clause or owned_by_later_head) and whether the attachment is tight.
    """
    gap = RE_ACCESSION.sub("IPR", text[joint_end(text, hit.end(), lo, name):lo])
    # The rest of a keyword the pattern ends inside of ("cytoplasmic loop|s at 1-112") is the keyword's own.
    if hit.group(0)[-1:].isalpha():
        gap = RE_KEYWORD_TAIL.sub("", gap, count=1)
    if accession_blocks(text, hit, lo, hi):
        return "accession_between", False
    if broader_family(text, hit, lo):
        return "broader_family", False
    if family_entry(text, hit, lo):
        return "family_entry", False
    if entry_citation(text, hit, lo):
        return ("dismissed_family", False) if dismissed_family(text, hit) else ("", True)
    if rival_nouns(name).search(gap):
        return "rival_noun", False
    if RE_CONTRAST.search(gap):
        return "new_phrase", False
    # "short extracellular loops, larger cytoplasmic loops at 1-112": the nearer keyword of the type takes the list.
    if PATTERNS[name].search(gap):
        return "nearer_keyword", False
    head_gap = RE_SHORT_PAREN.sub("", RE_ENTRY_TYPE.sub(" ", gap))
    if rival_head(blank_own_bracket(head_gap), name, own_colon(head_gap)) and not own_item(text, hi, w1, name):
        return "rival_head", False
    if name in VERB_HEAD_TYPES and RE_VERB_HIT.search(hit.group(0)) and new_verb_clause(text, hit, hi, lo, name):
        return "new_clause", False
    if tight(gap):
        return "", True
    # The keyword's own colon list stays its own past a later predicate ("the active-site constellation: histidines at
    # 12 and 14, ..., histidine 170 and aspartate 218 are assigned to metal binding").
    if own_colon(head_gap) < 0 and owned_after(text, hi, w1, name):
        return "owned_by_later_head", False
    return "", False


def new_verb_clause(text: str, hit: re.Match, hi: int, lo: int, name: str) -> bool:
    """
    Whether a coordinate after a metal or ligand verb ("coordinate a single zinc ion") sits in a new clause that does
    not make it the verb's: past a clause break, its own clause gives it another role or (for a metal) another ligand,
    or it names neither the type nor the verb's site again and is an item added by "with" or, for a metal, a clause
    with a verb of its own or a verbless continuation naming another object.

    Args:
        text: The generation.
        hit: The verb keyword match.
        hi: Coordinate end.
        lo: Coordinate start.
        name: The keyword's feature type (metal_binding or binding_site).

    Returns:
        True if the coordinate is not the verb's.
    """
    breaks = list(RE_VERB_BREAK.finditer(text, hit.end(), lo))
    if not breaks:
        return False
    start = breaks[-1].end()
    # The clause runs past the keyword's window to its own end or to the next item with coordinates of its own ("the G4
    # NKAD at 622 to 625 and G5 SAK at 658 to 660 specify guanine, and the scattered ..."; "the G2 threonine at 541 ...
    # sit in switch I and switch II, the G4 NKAD at 622"), not to an appositive ("and lysine 15, the invariant
    # beta-strand lysine, is listed among the ATP contacts").
    reach = min(len(text), hi + SENTENCE_REACH)
    end = RE_VERB_CLAUSE_END.search(text, hi, reach)
    stop = end.start() if end else reach
    for item in RE_VERB_BREAK.finditer(text, hi, stop):
        item_end = RE_ITEM_STOP.search(text, item.end(), stop)
        if RE_COORD.search(text, item.end(), item_end.start() if item_end else stop):
            stop = item.start()
            break
    clause = text[start:stop]
    if RE_OTHER_ROLE.search(clause):
        return True
    if name == "metal_binding" and (RE_OTHER_LIGAND.search(clause) or RE_OTHER_BOUND.search(clause)):
        return True
    named = [PATTERNS[name], TYPED_WORDS[name], RE_VERB_OWN[name], RE_SAME_REFERENCE]
    if name == "binding_site":
        named.append(TYPED_WORDS["metal_binding"])
    if any(pattern.search(clause) for pattern in named):
        return False
    if RE_WITH_BREAK.match(breaks[-1].group(0)):
        return True
    # A ligand's binding elements are often enumerated without restating the ligand ("binds GTP, the G2 threonine at 62
    # sits in Switch I, the G3 DCPG motif at 81-84 opens Switch II"); a metal's ligands are named as such.
    if name != "metal_binding":
        return False
    words = {w.lower() for w in RE_WORD.findall(text[start:lo])}
    return bool(RE_CLAUSE_PREDICATE.search(clause) or words - LIST_WORDS - GENERIC_WORDS - GAPPING_WORDS)


def own_item(text: str, hi: int, w1: int, name: str) -> bool:
    """
    Whether the words after a coordinate, up to its list item's end, name the keyword's type and no other, so a rival
    head before it does not take it ("a conserved P450 signature at 466 to 475, and at 473 a cysteine named as the axial
    ligand to the heme iron").

    Args:
        text: The generation.
        hi: Coordinate end.
        w1: Window end.
        name: The keyword's feature type.

    Returns:
        True if so.
    """
    after = text[hi:w1]
    stop = RE_ITEM_BREAK.search(after)
    item = after[:stop.start() if stop else len(after)]
    if RE_COORD.search(item):
        return False
    types = typed_types(item)
    allowed = SEGMENT_ALLOWED.get(name, {name})
    return bool(types & allowed) and not types - allowed - COMPATIBLE.get(name, set())


def blank_own_bracket(gap: str) -> str:
    """
    Blank the bracketed entry name of an accession cited right after the keyword, so its words ("conserved site",
    "zinc-binding domain") do not read as a rival head; a family name is kept.

    Args:
        gap: The forward gap, accessions written "IPR".

    Returns:
        The gap with the name replaced by "X", or unchanged.
    """
    own = RE_OWN_BRACKET.match(gap)
    if not own or RE_FAMILY.search(own.group(2)):
        return gap
    return gap[:own.start(2)] + "X" + gap[own.end(2):]


def backward_reason(text: str, hit: re.Match, lo: int, hi: int, name: str, w0: int, w1: int) -> tuple[str, bool]:
    """
    Whether a coordinate before a keyword belongs to it.

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.
        hi: Coordinate end.
        name: The keyword's feature type.
        w0: Window start.
        w1: Window end.

    Returns:
        ("" if it does, else accession_between, rival_noun, separate_list, new_phrase, landmark, new_list_head,
        relative_clause, owned_by_rival_noun or owned_by_other_type) and whether the attachment is tight.
    """
    gap = RE_RELATIVE.sub(" ", RE_ACCESSION.sub("IPR", text[hi:hit.start()]))
    if accession_blocks(text, hit, lo, hi):
        return "accession_between", False
    if rival_nouns(name).search(gap):
        return "rival_noun", False
    digits = list(RE_COORD.finditer(gap))
    # Every stretch up to the last number must be list glue, not only the first: "histidines at 322, 326 and 332 are
    # flagged as zinc ligands and glutamate 323 as the catalytic residue" keeps the histidines off the catalytic one.
    pieces = [gap[:digits[0].start()]] + [gap[a.end():b.start()] for a, b in zip(digits, digits[1:])] if digits else []
    if not all(glue(piece, name) for piece in pieces):
        return "separate_list", False
    tail = gap[digits[-1].end():] if digits else gap
    if RE_BACK_HARD.search(tail):
        return "new_phrase", False
    # A keyword named as a landmark outside the coordinate does not own it ("the cysteine at 269, sitting in the
    # catalytic domain just past the SAM-binding loop"); "within the ATP-binding loop" would.
    landmark = RE_LANDMARK.search(tail)
    if landmark and not RE_LANDMARK_CLAUSE.search(tail[:landmark.start()]):
        return "landmark", False
    # A keyword with coordinates of its own heads its own list ("..., with a coiled-coil prediction at 384 to 411"), so
    # it takes the list before it only through a labelling verb, not by adjacency ("38 to 57, 64 to 81 and 101 to 125,
    # with the last also matching a helical transmembrane span from 102 to 124").
    softs = list(RE_BACK_SOFT.finditer(tail))
    deep = len(RE_WORD.findall(tail[softs[-1].end():])) if softs else 0
    own = has_own_coordinates(text, hit.end(), w1)
    headed = False
    if own and ((softs and deep <= SOFT_HEAD_WORDS) or not RE_LABEL_VERB.search(tail)):
        if not own_type_head(text, lo, name):
            return "new_list_head", False
        headed = True
    if softs and deep > SOFT_HEAD_WORDS and softs[-1].group(0).strip(", ").lower() in ("which", "where") \
            and RE_RELATIVE_PREP.search(tail):
        return "relative_clause", False
    close = not headed and not softs and "," not in tail and len(tail) <= HEAD_REACH
    # A pair is no predicate of a range ("the extracellular loop from 126 to 180 ... carries two disulfide pairs"), and
    # a keyword heading its own list relates two placed things ("a cytosolic loop from 432 to 669 carrying a coiled-coil
    # at 473 to 493"). Round 5: the coordinate's own predicate also beats a rival noun before it ("residues 1 to 39 are
    # disordered and non-cytoplasmic, 40 to 60 form a single hydrophobic transmembrane helix").
    predicated = name != "disulfide" and not own and own_predicate(text, hi, hit, name)
    if owned_by_rival(text, w0, lo, name) and not predicated:
        return "owned_by_rival_noun", False
    if owned_before(text, w0, lo, hi, name) and not predicated:
        return "owned_by_other_type", False
    return "", close


def list_head(text: str, lo: int) -> str:
    """
    The phrase heading the list a coordinate sits in: back over the list to its first coordinate, then back to the
    previous break, within the sentence.

    Args:
        text: The generation.
        lo: Coordinate start.

    Returns:
        The heading phrase ("Around them are substrate-contact residues at "), which no earlier coordinate is part of
        ("a hydrophobic h-region over 10 to 21 that a transmembrane predictor also catches as a helix from 7 to 29" is
        headed by the predictor's helix).
    """
    start = max(0, lo - OWNER_REACH)
    stops = [m.end() for m in RE_CITATION_BREAK.finditer(text, start, lo)]
    before = text[stops[-1] if stops else start:lo]
    while (item := RE_LIST_ITEM_END.search(before)) and item.start() < len(before):
        before = before[:item.start()]
    cuts = [m.end() for m in RE_HEAD_BREAK.finditer(before)] + [m.end() for m in RE_COORD.finditer(before)]
    return before[max(cuts, default=0):]


def own_type_head(text: str, lo: int, name: str) -> bool:
    """
    Whether the phrase heading a coordinate's list names the keyword's own type and no other ("a carbohydrate-binding
    set at 28, 97, ...", "a single helix at 170 to 194"), so a later keyword of that type may keep the list it would
    otherwise leave to that head.

    Args:
        text: The generation.
        lo: Coordinate start.
        name: The keyword's feature type.

    Returns:
        True if it does; a non-ligand head ("a peptide-binding pocket at 299, 309, 313") or a helix that does not cross
        the membrane ("intramembrane helix 225 to 237") does not.
    """
    head = list_head(text, lo)
    types = typed_types(head)
    allowed = SEGMENT_ALLOWED.get(name, {name})
    own = allowed - ({"transmembrane"} if name != "transmembrane" else set())
    if not types & own or types - allowed - ({"non_ligand"} if name == "dna_binding" else set()):
        return False
    return not (name == "transmembrane" and RE_NOT_MEMBRANE_HELIX.search(head))


def own_predicate(text: str, hi: int, hit: re.Match, name: str) -> bool:
    """
    Whether the text from a coordinate to a later keyword is the coordinate's own predicate, so that the keyword claims
    it even though another type's head holds it ("... (captured by the NTYK motif at residues 81, 104, 134, 138)
    protonates the carbonyl oxygen and coordinates the nucleotide-sugar"). An object joined to another type's
    ("IPR... (..., residues 650-673) and IPR008271 (..., active site, residues 686-698) define the nucleotide-binding
    pocket and catalytic base") is read respectively.

    Args:
        text: The generation.
        hi: Coordinate end.
        hit: The keyword match.
        name: The keyword's feature type.

    Returns:
        True if a predicate verb opens it (past the rest of the list, a closing bracket and ", which") and no break,
        other coordinate, new subject, place or sequence word follows, nor does an object of another type joined to the
        keyword's.
    """
    after = text[hit.end():hit.end() + OBJECT_REACH]
    stop = RE_OBJECT_END.search(after)
    joined = RE_OBJECT_JOIN.search(after[:stop.start() if stop else len(after)])
    allowed = SEGMENT_ALLOWED.get(name, {name}) | COMPATIBLE.get(name, set())
    if joined and typed_types(after[joined.end():stop.start() if stop else len(after)]) - allowed:
        return False
    between = text[hi:hit.start()]
    lead = RE_OWN_PREDICATE.match(between)
    if not lead:
        return False
    rest = between[lead.end():]
    return not (RE_PREDICATE_BREAK.search(rest) or RE_COORD.search(rest) or RE_BACK_HARD.search(rest)
                or RE_LOCATION.search(rest) or RE_PREDICATE_RELATIVE.search(rest) or RE_SEQUENCE_WORD.search(between))


def joint_end(text: str, end: int, lo: int, name: str) -> int:
    """
    Where a keyword's head ends when another type's keyword is joined to it ("active-site and nucleotide-binding
    residues at 48-51"): the joined keyword is part of the head, not a rival between it and the coordinate.

    Args:
        text: The generation.
        end: The keyword's end.
        lo: Coordinate start.
        name: The keyword's feature type.

    Returns:
        The end of the joined keyword, or the keyword's own end when none is joined.
    """
    joint = RE_JOINT.match(text, end, lo)
    if not joint or joint.end() == end:
        return end
    for other, pattern in PATTERNS.items():
        m = pattern.match(text, joint.end(), lo) if other != name else None
        if m:
            return m.end()
    return end


def owned_before(text: str, w0: int, lo: int, hi: int, name: str) -> bool:
    """
    Whether a coordinate already belongs to a keyword or typed word of another type that precedes it ("an ATP-binding
    loop at 412 to 420, the catalytic lysine at 442": 412-420 is the loop's, not the lysine's; "a single helix crosses
    at 719 to 739, and 740 to 797 returns to the cytosol, which makes this a type I membrane protein": 719-739 is the
    helix's).

    Args:
        text: The generation.
        w0: Window start.
        lo: Coordinate start.
        hi: Coordinate end.
        name: The feature type asking.

    Returns:
        True if the nearest preceding head in the window is another type's and holds the coordinate tightly (a typed
        word also across a short verb); a loose hold yields to the nearer head ("Six disulfide bonds stitch the domain
        together, and two asparagines at 38 and 192 carry N-linked glycans"), and a helix does not hold what it leaves
        on one side of the membrane ("a single helix crosses at 719 to 739, and 740 to 797 returns to the cytosol").
    """
    # The owner may head a list longer than the keyword's window, so it is sought back to the sentence's start.
    reach = max(0, lo - OWNER_REACH)
    stops = [m.end() for m in RE_CITATION_BREAK.finditer(text, reach, lo)]
    w0 = min(w0, stops[-1] if stops else reach)
    before = text[w0:lo]
    last = None
    for other, pattern in PATTERNS.items():
        # Searched up to the coordinate's end, since keywords like "crossings at " look ahead to its digits.
        for m in pattern.finditer(text, w0, hi):
            if m.end() <= lo and (last is None or m.end() > last[1].end()):
                last = (other, m)
    # A bare helix is any helix ("The I-helix region around 304 to 313"), so transmembrane words own nothing here.
    excluded = SEGMENT_ALLOWED.get(name, {name}) | COMPATIBLE.get(name, set()) | {name, "transmembrane"}
    typed = [(m, other) for other, pattern in TYPED_WORDS.items() if other not in excluded
             for m in pattern.finditer(before)]
    if typed:
        word, _ = max(typed, key=lambda item: item[0].end())
        gap = before[word.end():]
        if (last is None or word.end() > last[1].end()) and holds_directly(gap):
            return True
    if last is None or last[0] == name or last[0] in COMPATIBLE.get(name, ()):
        return False
    if PATTERNS[name].fullmatch(last[1].group(0)):
        return False
    if last[0] == "transmembrane" and RE_SIDE_AFTER.match(text, hi):
        return False
    lead = text[max(w0, last[1].start() - HEAD_REACH):last[1].start()]
    joined = [m for m in PATTERNS[name].finditer(lead) if RE_JOINT.fullmatch(lead[m.end():])]
    if joined:
        return False
    gap = text[last[1].end():lo]
    return tight(gap) and not RE_ACCESSION.search(gap) and not rival_nouns(name).search(gap)


def holds_directly(gap: str) -> bool:
    """
    Whether a typed word heads the coordinate after it directly: no break, no accession or rival noun, and at most
    HOLD_WORDS words beyond cues and list words between them ("the proton-accepting aspartate at 1251").

    Args:
        gap: The text from the typed word's end to the coordinate.

    Returns:
        True if it does.
    """
    if RE_LEAD_BREAK.search(gap) or RE_ACCESSION.search(gap) or RE_RIVAL.search(gap) or RE_COORD.search(gap):
        return False
    words = {w.lower() for w in RE_WORD.findall(gap)} - CUE_WORDS - LIST_WORDS
    return len(words) <= HOLD_WORDS


def owned_after(text: str, hi: int, w1: int, name: str) -> bool:
    """
    Whether a loosely attached coordinate belongs to a later head of another type that it is the subject of ("and two
    asparagines at 38 and 192 carry N-linked glycans"; "a scattered set of positions at 85, ..., 224 forms an
    inhibitor-binding surface"; "residues 222 and 223 as further magnesium ligands"; "uses the histidine at 39 with
    aspartate 63 to activate water").

    Args:
        text: The generation.
        hi: Coordinate end.
        w1: Window end.
        name: The feature type asking.

    Returns:
        True if the first head after the coordinate names another type (or a rival noun), the coordinate's list runs
        into it through list glue and a short predicate, not a place, and the head has no coordinates of its own.
    """
    after = text[hi:w1]
    allowed = SEGMENT_ALLOWED.get(name, {name})
    typed = [(m, other) for source in (PATTERNS, TYPED_WORDS) for other, pattern in source.items()
             if other not in allowed for m in pattern.finditer(after)]
    # A metal site's list may run into a binding phrase naming no other ligand ("... form one binding site").
    heads = [m for m, other in typed if not (name in ("metal_binding", "zinc_finger") and other == "binding_site"
                                             and not RE_OTHER_LIGAND.search(m.group(0)))]
    heads += list(rival_nouns(name).finditer(after))
    if name != "dna_binding":
        heads += list(RE_NOT_LIGAND_TYPED.finditer(after))
    if not heads:
        return False
    head = min(heads, key=lambda m: m.start())
    path = after[:head.start()]
    if PATTERNS[name].search(path) or (name in TYPED_WORDS and TYPED_WORDS[name].search(path)):
        return False
    digits = list(RE_COORD.finditer(path))
    pieces = ([path[:digits[0].start()]] + [path[a.end():b.start()] for a, b in zip(digits, digits[1:])]
              if digits else [])
    # The coordinate's own list has to run on by enumeration or a joined subject ("the asparagine at 489 with the
    # aspartate at 502 chelate the magnesium ions"); "375, a histidine at 24 named as the acceptor substrate-binding
    # pocket" starts a new item.
    if not all(RE_ENUM_GLUE.match(piece) or ("," not in piece and glue(piece)) for piece in pieces):
        return False
    tail = path[digits[-1].end():] if digits else path
    # A comma ends the list ("a polar c-region at 16 to 24, followed by the mature chain") unless a copula follows it
    # ("the last twenty residues, 742 to 763, are disordered").
    comma = "," in RE_COPULA_COMMA.sub("", tail, count=1)
    place = RE_INFINITIVE.sub("", tail) if RE_VERB_HEAD.match(after, head.start()) else tail
    if len(tail) > HEAD_REACH or comma or RE_BACK_HARD.search(tail) or RE_LOCATION.search(place):
        return False
    return not has_own_coordinates(text, hi + head.end(), w1)


def bind_reason(text: str, hit: re.Match, lo: int, hi: int, name: str, w0: int, w1: int) -> tuple[str, bool]:
    """
    Decide whether a coordinate at text[lo:hi] belongs to a keyword hit, and if not, why.

    Args:
        text: The generation.
        hit: The keyword match.
        lo: Coordinate start.
        hi: Coordinate end.
        name: The keyword's feature type.
        w0: Window start.
        w1: Window end.

    Returns:
        ("" if it binds, else cut_number, other_molecule, helix_place, topological_remainder, flanking_residue,
        predicate_forward, other_subject, other_type_between, signal_part or a forward_reason / backward_reason) and
        whether the attachment is tight.
    """
    if cut_number(text, w0, w1, lo, hi):
        return "cut_number", False
    if other_numbering(text, lo, hi):
        return "other_molecule", False
    if name == "topological_domain" and RE_HELIX_PLACE.match(text, hi):
        return "helix_place", False
    if name == "transmembrane" and RE_SIDE_AFTER.match(text, hi):
        return "topological_remainder", False
    if name == "transmembrane" and RE_POSITIVE_INSIDE.search(text, min(hi, hit.end()), max(lo, hit.start())):
        return "flanking_residue", False
    forward = lo >= hit.end()
    # A side predicate ("54 to 455 lie on the non-cytoplasmic side") states the side of the range before it.
    side_predicate = name == "topological_domain" and RE_SIDE_PREDICATE.fullmatch(hit.group(0))
    if forward and side_predicate:
        return "predicate_forward", False
    if not forward and side_predicate:
        switch = RE_SIDE_SUBJECT.search(text, hi, hit.start())
        if switch and not RE_LIST_SIDE.search(text, hi, switch.start()):
            return "other_subject", False
    gap = text[joint_end(text, hit.end(), lo, name):lo] if forward else text[hi:hit.start()]
    rivals = (p for other, p in PATTERNS.items() if other != name and other not in COMPATIBLE.get(name, ()))
    # A list item headed by the keyword's own type stays the keyword's past another type's keyword ("two membrane
    # anchors: a short cytosolic amino end at 1 to 6, a first helix at 7 to 27").
    if any(pattern.search(gap) for pattern in rivals) and not (forward and own_type_head(text, lo, name)):
        return "other_type_between", False
    if name == "transmembrane" and RE_SIGNAL_PARTS.search(gap):
        return "signal_part", False
    if forward:
        return forward_reason(text, hit, lo, hi, name, w0, w1)
    return backward_reason(text, hit, lo, hi, name, w0, w1)


def owned_by_rival(text: str, w0: int, lo: int, name: str) -> bool:
    """
    Whether a coordinate read backward from a keyword is the list of a rival noun before it ("an activation loop
    spanning 304 to 324 that faces the substrate pocket": the span is the loop's).

    Args:
        text: The generation.
        w0: Window start.
        lo: Coordinate start.
        name: The feature type asking.

    Returns:
        True if the nearest rival noun before the coordinate reaches it through list words alone, with no clause
        boundary and no keyword of this type in between.
    """
    before = text[w0:lo]
    rivals = list(rival_nouns(name).finditer(before))
    if not rivals:
        return False
    tail = before[rivals[-1].end():]
    return (glue(tail) and not RE_BACK_HARD.search(tail) and not RE_BACK_SOFT.search(tail)
            and not PATTERNS[name].search(tail))


def part_label(text: str, w0: int, lo: int) -> bool:
    """
    Whether a coordinate is labelled as an n-, h- or c-region, a part of a signal or transit peptide.

    Args:
        text: The generation.
        w0: Window start.
        lo: Coordinate start.

    Returns:
        True if the label sits between the previous number and the coordinate, within PART_REACH characters.
    """
    lead = text[max(w0, lo - PART_REACH):lo]
    digits = list(re.finditer(r"\d", lead))
    if digits:
        lead = lead[digits[-1].end():]
    return bool(RE_PART_LABEL.search(lead))


def loose_candidates(text: str, hit: re.Match) -> list[tuple[int, int, int, str]]:
    """
    The coordinates the original extractor read around a keyword (rc.clause_window, rc.RE_SPAN, rc.RE_POSITION),
    located, for attributing a claim the new extractor drops.

    Args:
        text: The generation.
        hit: The keyword match.

    Returns:
        (start, end, offset in the text, source) per candidate; source is range, borrowed_start, position or residue.
    """
    before = text[max(0, hit.start() - rc.KEYWORD_WINDOW_BACK):hit.start()]
    breaks = list(rc.RE_SENTENCE_END.finditer(before))
    w0 = hit.start() - len(before) + (breaks[-1].end() if breaks else 0)
    window = rc.clause_window(text, hit.start(), hit.end())
    borrowed = {(int(a), int(b)) for a, b in rc.RE_ACCESSION_SPAN.findall(window)}
    out = []
    ranges = []
    for m in rc.RE_SPAN.finditer(window):
        a, b = int(m.group(1)), int(m.group(2))
        if b >= a and (a, b) not in borrowed:
            ranges.append((a, b))
            out.append((a, b, w0 + m.start(), "range"))
    for m in rc.RE_POSITION.finditer(window):
        inside_word = m.start() > 0 and window[m.start() - 1].isalpha()
        for v in re.finditer(r"\d{1,4}", m.group(1)):
            value = int(v.group(0))
            if any(lo <= value <= hi for lo, hi in ranges):
                continue
            if any(value == a for a, _ in borrowed):
                source = "borrowed_start"
            else:
                source = "position_in_word" if inside_word else "position"
            out.append((value, value, w0 + m.start(1) + v.start(), source))
    for m in list(rc.RE_AA3.finditer(window)) + list(rc.RE_FULL.finditer(window)):
        number = m.group(2) if m.re is rc.RE_AA3 else m.group(3)
        values = [(int(number), w0 + m.end() - len(number))]
        if m.re is rc.RE_FULL and m.group(2):
            cursor = m.end()
            while tail := rc.RE_ENUM.match(window, cursor):
                values.append((int(tail.group(1)), w0 + tail.start(1)))
                cursor = tail.end()
        out.extend((v, v, lo, "residue") for v, lo in values if not any(a <= v <= b for a, b in ranges))
    return out


def unread_reason(w0: int, w1: int, offset: int, source: str) -> str:
    """
    Why the new window does not even read a coordinate the old window read.

    Args:
        w0: New window start.
        w1: New window end.
        offset: The coordinate's offset in the text.
        source: Its loose_candidates source.

    Returns:
        line_break_window, accession_range_start, position_word_boundary, or not_read.
    """
    if offset < w0 or offset >= w1:
        return "line_break_window"
    if source == "borrowed_start":
        return "accession_range_start"
    if source == "position_in_word":
        return "position_word_boundary"
    return "not_read"


def add_candidate(found: list[dict], start: int, end: int, kind: str, reason: str, extra: dict) -> None:
    """
    Record one candidate coordinate of a keyword window, kept or rejected.

    Args:
        found: The list to append to.
        start: Claimed start.
        end: Claimed end.
        kind: range, first_n, last_n, pair, position, residue, boundary (an N-terminal peptide's end) or edge_start /
            edge_end (a region's edge stated by a boundary word).
        reason: "" when kept, else why not.
        extra: keyword, hit_start, side, part, window and the coordinate's offsets lo and hi.
    """
    found.append({"start": start, "end": end, "kind": kind, "reason": reason, **extra})


def window_sides(window: str) -> set[str]:
    """
    The sides of the membrane a window names.

    Args:
        window: The keyword window.

    Returns:
        A subset of {"in", "out"}.
    """
    return {"out" if RE_SIDE_OUT_WORD.match(m.group(0)) else "in" for m in RE_SIDE_WORD.finditer(window)}


def other_numbering(text: str, lo: int, hi: int) -> bool:
    """
    Whether a coordinate numbers another molecule: a tRNA, rRNA or nucleotide position ("the D/T loops that house
    position 26", "N1-methylguanine at position 9 of tRNA").

    Args:
        text: The generation.
        lo: Coordinate start.
        hi: Coordinate end.

    Returns:
        True if a nucleic-acid word leads to "position" right before it, or "of/in the tRNA" follows it.
    """
    return bool(RE_OTHER_NUMBERING_LEAD.search(text, max(0, lo - NUMBERING_REACH), lo)
                or RE_OTHER_NUMBERING_TRAIL.match(text, hi))


def item_side(text: str, w0: int, w1: int, lo: int, hi: int) -> str:
    """
    The side of the membrane a coordinate's own list item names ("a long stretch at 117 to 180 faces out again").

    Args:
        text: The generation.
        w0: Window start.
        w1: Window end.
        lo: Coordinate start.
        hi: Coordinate end.

    Returns:
        "in", "out", or "" when the item names neither or both.
    """
    before, after = text[w0:lo], text[hi:w1]
    cuts = [m.end() for m in RE_ITEM_BREAK.finditer(before)] + [m.end() for m in RE_COORD.finditer(before)]
    stops = [m.start() for m in RE_ITEM_BREAK.finditer(after)] + [m.start() for m in RE_COORD.finditer(after)]
    item = before[max(cuts, default=0):] + " " + after[:min(stops, default=len(after))]
    sides = {"out" if RE_ITEM_OUT.search(m.group(0)) else "in" for m in RE_ITEM_SIDE.finditer(item)}
    return sides.pop() if len(sides) == 1 else ""


def occurrence(base: dict, side: str, mixed: bool, close: bool, part: bool, lo: int, hi: int,
               own_side: str = "") -> dict:
    """
    The per-candidate record of a keyword occurrence.

    Args:
        base: keyword, hit_start and window.
        side: The keyword's side of the membrane, or "".
        mixed: Whether the window names both sides.
        close: Whether the coordinate hangs directly on the keyword.
        part: Whether the coordinate is labelled as a part of a signal or transit peptide.
        lo: Coordinate start.
        hi: Coordinate end.
        own_side: The side the coordinate's own list item names, which wins over the keyword's; "" for none.

    Returns:
        The record add_candidate extends.
    """
    return {**base, "side": own_side or (side if close or not mixed else ""), "part": part, "lo": lo, "hi": hi}


def boundary_word(lead: str) -> str:
    """
    Read which boundary word governs a position ("preceding residue 10", "cleaved after Ala-22", "up to residue 30").

    Args:
        lead: The text right before the position's number.

    Returns:
        "before", "after", "until", or "" when none does.
    """
    if RE_BOUNDARY_BEFORE.search(lead):
        return "before"
    if RE_BOUNDARY_AFTER.search(lead):
        return "after"
    if RE_BOUNDARY_UNTIL.search(lead):
        return "until"
    return ""


def edge_clause(text: str, w0: int, lo: int) -> bool:
    """
    Whether the boundary word before a position opens a clause, so the position is a region's edge rather than a limit
    on some other noun.

    Args:
        text: The generation.
        w0: Window start.
        lo: The position's offset.

    Returns:
        True if only the start of the clause, a conjunction or a verb of extent comes before the boundary word.
    """
    lead = text[max(w0, lo - BOUNDARY_REACH):lo]
    word = max((m for regex in (RE_BOUNDARY_BEFORE, RE_BOUNDARY_AFTER) for m in [regex.search(lead)] if m),
               key=lambda m: m.start())
    return bool(RE_EDGE_LEAD.search(text[w0:max(w0, lo - BOUNDARY_REACH) + word.start()]))


def swallowed(text: str, value: int, lo: int, bound: list[tuple[int, int, int, int]], name: str) -> bool:
    """
    Whether a single position restates part of a range the keyword holds. For a site type, it lies inside a range
    earlier in the same list item ("at 120 to 130, with Gly125"; "the ATP-binding region at 10 to 30, including
    glycine 12, glycine 14 and lysine 18"), and a single in a list of its own is its own claim ("SAM-contacting
    positions at 37, 41, 48, 75, 96, 124 and 125, with a tighter site across 74 to 80, 96 to 97 and 123 to 125"). For a
    disulfide it is a cysteine of a pair the keyword lists anywhere. For a region type any held range containing it
    says more than the position does ("a hydrophobic core at 8 to 15 ..., with arginines at 2, 6 and 14").

    Args:
        text: The generation.
        value: The position.
        lo: Its offset in the text.
        bound: The keyword's held ranges as (start, end, lo, hi).
        name: Feature type.

    Returns:
        True if the single adds no claim of its own.
    """
    for start, end, r_lo, r_hi in bound:
        if name == "disulfide":
            if value in (start, end):
                return True
            continue
        if not start <= value <= end:
            continue
        if name not in SITE_TYPES + POINT_TYPES:
            return True
        if r_hi > lo:
            continue
        between = text[r_hi:lo]
        inside = all(start <= int(m.group(0)) <= end for m in RE_COORD.finditer(between))
        if inside and not rc.RE_SENTENCE_END.search(between) and not names_head(between):
            return True
    return False


def window_candidates(text: str, hit: re.Match, name: str, length: int) -> list[dict]:
    """
    Every coordinate in one keyword's window, with whether it binds to the keyword.

    Args:
        text: The generation.
        hit: The keyword match (already judged to assert the feature).
        name: Feature type.
        length: Chain length, for "the last N residues".

    Returns:
        Candidates as add_candidate records them.
    """
    w0, w1 = window_bounds(text, hit)
    window = text[w0:w1]
    found = []
    side = side_of(hit.group(0))
    prefix = text[max(0, hit.start() - 6):hit.start()]
    if name == "topological_domain" and side == "in" and (RE_NON_PREFIX.search(prefix)
                                                          or RE_NON_INSIDE.search(hit.group(0))):
        side = "out"
    # When the sentence names both sides, only a list hanging directly on its keyword takes that keyword's side
    # ("... 377 to 401, so the subtype signature sits in the extracellular amino terminus, the extracellular loops and
    # the cytosolic tail" names no side for 377-401); the others are scored against either side.
    mixed = name == "topological_domain" and len(window_sides(window)) > 1
    base = {"keyword": hit.group(0), "hit_start": hit.start(), "window": window}
    borrowed = {(int(a), int(b)) for a, b in rc.RE_ACCESSION_SPAN.findall(window)}
    range_text = [(w0 + m.start(), w0 + m.end(), (int(m.group(1)), int(m.group(2))))
                  for m in rc.RE_SPAN.finditer(window)]
    for m in rc.RE_SPAN.finditer(window):
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            continue
        lo, hi = w0 + m.start(), w0 + m.end()
        part = name in PART_TYPES and part_label(text, w0, lo)
        reason, close = bind_reason(text, hit, lo, hi, name, w0, w1)
        if not reason and b - a <= COUNT_SPREAD and RE_COUNT_NOUN.match(text, hi):
            reason = "count"
        if not reason and name == "disulfide" and disulfide_region(text, w0, lo, hi):
            reason = "disulfide_region"
        add_candidate(found, a, b, "range", reason, occurrence(base, side, mixed, close, part, lo, hi,
                                                               sided(text, w0, w1, lo, hi, name)))
    for regex, kind in ((RE_FIRST_N, "first_n"), (RE_LAST_N, "last_n")):
        for m in regex.finditer(window):
            n = count_value(m.group(1))
            lo, hi = w0 + m.start(), w0 + m.end()
            if not 0 < n < length:
                continue
            span = (1, n) if kind == "first_n" else (length - n + 1, length)
            reason, close = bind_reason(text, hit, lo, hi, name, w0, w1)
            add_candidate(found, *span, kind, reason, occurrence(base, side, mixed, close, False, lo, hi,
                                                                 sided(text, w0, w1, lo, hi, name)))
    if name == "disulfide":
        for m in pair_matches(window):
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            if a == b:
                continue
            lo, hi = w0 + m.start(1), w0 + m.end(2)
            reason, close = bind_reason(text, hit, lo, hi, name, w0, w1)
            add_candidate(found, a, b, "pair", reason, occurrence(base, side, mixed, close, False, lo, hi))
    pairs = {(c["start"], c["end"]) for c in found if c["kind"] == "pair"}
    # A single position is skipped when its digits are a range's own ("at 23" of "at 23 to 45") or it lies inside a
    # range this keyword holds; a domain range that belongs to another entry no longer swallows the residues named
    # near it, and "glutamate 451" next to "contacts at 451-452" is its own claim.
    cap = rc.MAX_SPAN.get(name, 0)
    bound = [(c["start"], c["end"], c["lo"], c["hi"]) for c in found
             if c["kind"] in ("range", "pair") and not c["reason"] and (not cap or c["end"] - c["start"] + 1 <= cap)]
    singles = []
    for m in RE_POSITION.finditer(window):
        for v in re.finditer(r"\d{1,4}", m.group(1)):
            singles.append((int(v.group(0)), w0 + m.start(1) + v.start(), w0 + m.start(1) + v.end(), "position", ""))
    for m in list(rc.RE_AA3.finditer(window)) + list(rc.RE_FULL.finditer(window)):
        if m.re is rc.RE_AA3:
            aa, number = rc.AA3[m.group(1).lower()], m.group(2)
        else:
            aa, number = rc.AA_FULL[m.group(1).lower()], m.group(3)
        located = [(aa, int(number), w0 + m.start(), w0 + m.end(), m.group(0), "scored")]
        if m.re is rc.RE_FULL and m.group(2):
            cursor = m.end()
            while tail := rc.RE_ENUM.match(window, cursor):
                located.append((aa, int(tail.group(1)), w0 + tail.start(1), w0 + tail.end(1),
                                "(enum) " + tail.group(0), "scored"))
                cursor = tail.end()
        for claim in located:
            # "L-aspartate 4-semialdehyde" or "histone H3 lysine 4" is not a residue of this protein.
            category = claim_category(text, claim, False)
            singles.append((claim[1], claim[3] - len(str(claim[1])), claim[3], "residue",
                            "" if category in COUNTED_CATEGORIES else "not_a_residue:" + category))
    for value, lo, hi, kind, why in singles:
        part = name in PART_TYPES and part_label(text, w0, lo)
        own = next((span for r0, r1, span in range_text if r0 <= lo < r1), None)
        word = boundary_word(text[max(w0, lo - BOUNDARY_REACH):lo])
        start = stop = value
        close = False
        if word and name in NTERM_TYPES:
            start, stop, kind = 1, value - 1 if word == "before" else value, "boundary"
        elif word in ("before", "after") and name in EDGE_TYPES and edge_clause(text, w0, lo):
            start = stop = value - 1 if word == "before" else value + 1
            kind = "edge_end" if word == "before" else "edge_start"
        if why:
            reason = why
        elif any(value in pair for pair in pairs):
            reason = "disulfide_pair_endpoint"
        elif own is not None:
            reason = "accession_range_start" if own in borrowed else "range_endpoint"
        elif word in ("before", "after") and kind not in ("edge_start", "edge_end") and name not in NTERM_TYPES:
            reason = "boundary_word"
        elif kind not in ("boundary", "edge_start", "edge_end") and swallowed(text, value, lo, bound, name):
            reason = "inside_bound_range"
        else:
            reason, close = bind_reason(text, hit, lo, hi, name, w0, w1)
        if 0 < start <= stop:
            add_candidate(found, start, stop, kind, reason, occurrence(base, side, mixed, close, part, lo, hi,
                                                                       sided(text, w0, w1, lo, hi, name)))
    return found


def pair_matches(window: str) -> list[re.Match]:
    """
    Every cysteine pair a window states, including the further pairs a pair verb lists.

    Args:
        window: The keyword window.

    Returns:
        Matches whose groups 1 and 2 are the two cysteines.
    """
    out = [m for regex in RE_PAIRS for m in regex.finditer(window)]
    for m in RE_PAIR_VERB.finditer(window):
        cursor = m.end()
        while more := RE_PAIR_NEXT.match(window, cursor):
            out.append(more)
            cursor = more.end()
    return out


def sided(text: str, w0: int, w1: int, lo: int, hi: int, name: str) -> str:
    """
    The side a topological claim's own list item names; item_side for topological domains, "" for other types.

    Args:
        text: The generation.
        w0: Window start.
        w1: Window end.
        lo: Coordinate start.
        hi: Coordinate end.
        name: Feature type.

    Returns:
        "in", "out" or "".
    """
    return item_side(text, w0, w1, lo, hi) if name == "topological_domain" else ""


def disulfide_region(text: str, w0: int, lo: int, hi: int) -> bool:
    """
    Whether a disulfide range names a region said to hold bonds rather than a cysteine pair: the phrase heading its
    list (looked for before the list's first coordinate) is a domain-level noun joined to it by cues alone, or an
    InterPro entry's citation holds it, or a domain-level noun or the holding of bonds is predicated of it right after
    ("from 369 to 424, sits an EGF-like domain", "the extracellular loop from 112 to 280 carries two disulfide
    bonds"). A heading phrase naming a bond, or placing the list ("in the first repeat"), makes a pair; a bond word
    that only modifies the region noun ("disulphide-bonds motif, residues 64-75") does not.

    Args:
        text: The generation.
        w0: Window start.
        lo: Coordinate start.
        hi: Coordinate end.

    Returns:
        True for a region, False for a pair (also when nothing heads it).
    """
    if RE_REGION_AFTER.match(text, hi) or RE_REGION_HOLDS.match(text, hi):
        return True
    first = lo
    for m in reversed(list(RE_COORD.finditer(text, max(w0, lo - REGION_REACH), lo))):
        if not RE_ENUM_GLUE.match(text[m.end():first]):
            break
        first = m.start()
    stops = [m.end() for m in RE_CITATION_BREAK.finditer(text, max(0, first - REGION_REACH), first)]
    start = stops[-1] if stops else max(0, first - REGION_REACH)
    owners = list(RE_ACCESSION.finditer(text, start, first))
    cited = RE_BOND_REGION.sub(" ", text[owners[-1].end():first]) if owners else ""
    if owners and cites(text, owners[-1].end(), first) and not RE_PAIR_HEAD.search(cited):
        return True
    parts = RE_SEGMENT_SPLIT.split(text[max(w0, first - REGION_REACH):first])
    segments = [seg for seg in parts[0::2] if {w.lower() for w in RE_WORD.findall(seg)} - CUE_WORDS]
    if not segments:
        return False
    head = RE_COORD.split(segments[-1])[-1]
    tail = text[max(w0, first - REGION_REACH):first]
    return (bool(RE_REGION_DIRECT.search(tail)) and not RE_PLACE_START.match(head)
            and not RE_PAIR_HEAD.search(RE_BOND_REGION.sub(" ", head)))


def claimed_spans(text: str, name: str, length: int, layout: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """
    Extract one feature type's claimed spans, and the candidates that were not bound, with why.

    Args:
        text: The generation.
        name: Feature type.
        length: Chain length.
        layout: answer_layout of the text.

    Returns:
        Kept claims (deduplicated by span, first occurrence kept) and rejected candidates; each a dict with start,
        end, kind, reason, keyword, hit_start, side, part and window.
    """
    cap = rc.MAX_SPAN.get(name, 0)
    kept, rejected, seen = [], [], set()
    looped = run_offsets(text)
    starts = {m.start() for m in PATTERNS[name].finditer(text)}
    for hit in rc.FEATURE_TYPES[name][1].finditer(text):
        if hit.start() not in starts:
            for a, b, _, _ in loose_candidates(text, hit):
                add_candidate(rejected, a, b, "loose", "hit:pattern_change",
                              {"keyword": hit.group(0), "hit_start": hit.start(), "side": "", "part": False,
                               "window": rc.clause_window(text, hit.start(), hit.end())})
    for hit in PATTERNS[name].finditer(text):
        reason = hit_reason(text, hit, name, layout)
        loose = loose_candidates(text, hit)
        old_window = {"keyword": hit.group(0), "hit_start": hit.start(), "side": "", "part": False,
                      "window": rc.clause_window(text, hit.start(), hit.end())}
        if reason:
            for a, b, _, _ in loose:
                add_candidate(rejected, a, b, "loose", "hit:" + reason, old_window)
            continue
        candidates = window_candidates(text, hit, name, length)
        read = {(c["start"], c["end"]) for c in candidates}
        w0, w1 = window_bounds(text, hit)
        for a, b, offset, source in loose:
            if (a, b) not in read:
                add_candidate(rejected, a, b, "loose", unread_reason(w0, w1, offset, source), old_window)
        for cand in candidates:
            span = (cand["start"], cand["end"])
            if not cand["reason"] and any(offset in looped for offset in range(cand["lo"], cand["hi"])):
                cand["reason"] = "loop"
            if not cand["reason"] and cap and span[1] - span[0] + 1 > cap:
                cand["reason"] = "too_long"
            if not cand["reason"] and span[0] == span[1] == length and length_phrase(text, cand["lo"], cand["hi"]):
                cand["reason"] = "chain_length"
            if cand["reason"]:
                rejected.append(cand)
            elif span not in seen:
                seen.add(span)
                kept.append(cand)
    return kept, rejected


def run_offsets(text: str) -> set[int]:
    """
    Where the numbers of degenerate counting runs start (counting_runs over every coordinate of the text, with no
    residue label), so a span claim inside one is no placement, as a residue claim inside one is no citation.

    Args:
        text: The generation.

    Returns:
        Start offsets of the numbers inside runs.
    """
    items = [(RUN_LABEL, int(m.group(0)), m.start(), m.end()) for m in RE_COORD.finditer(text)]
    return {items[k][2] for run in counting_runs(items) for k in run}


def length_phrase(text: str, lo: int, hi: int) -> bool:
    """
    Whether a position at the chain's last residue is stated as the chain's length or end ("the chain is 856 residues",
    "ends at 211", "the carboxyl terminus at 856") rather than as a site in a list.

    Args:
        text: The generation.
        lo: Coordinate start.
        hi: Coordinate end.

    Returns:
        True if a length unit follows it or a length or end phrase leads to it.
    """
    return bool(RE_LENGTH_AFTER.match(text, hi) or RE_LENGTH_BEFORE.search(text[max(0, lo - LENGTH_REACH):lo]))


def region_distance(claim: tuple[int, int], truth: tuple[int, int]) -> int:
    """
    Measure how far a claimed span lies from one true span (the base rule).

    A single-residue claim or annotation, or a claim lying wholly inside the true feature, is 0 when one lies within the
    other and else the gap, so a feature described by its parts ("an n-region over 1 to 7, an h-region over 8 to 18")
    places each part right; two spans otherwise score the larger of the start and end errors, so a claim running past
    the feature is off by its overhang.

    Args:
        claim: Claimed (start, end).
        truth: True (start, end).

    Returns:
        Distance in residues.
    """
    (c0, c1), (t0, t1) = claim, truth
    if c0 == c1 or t0 == t1 or (t0 <= c0 and c1 <= t1):
        return max(t0 - c1, c0 - t1, 0)
    return max(abs(c0 - t0), abs(c1 - t1))


def pair_distance(claim: tuple[int, int], bond: tuple[int, int]) -> int:
    """
    Score a disulfide claim against one true bond by its cysteines.

    Args:
        claim: Claimed (start, end); start == end for a single cysteine.
        bond: True (cys1, cys2).

    Returns:
        Distance in residues.
    """
    if claim[0] == claim[1]:
        return min(abs(claim[0] - bond[0]), abs(claim[0] - bond[1]))
    return max(abs(claim[0] - bond[0]), abs(claim[1] - bond[1]))


def overhang_distance(claim: tuple[int, int], truth: tuple[int, int]) -> int:
    """
    Score a labelled part of a feature by how far it runs outside the true feature.

    Args:
        claim: Claimed (start, end).
        truth: True (start, end).

    Returns:
        The overhang if the two overlap, else the gap between them.
    """
    (c0, c1), (t0, t1) = claim, truth
    if min(c1, t1) >= max(c0, t0):
        return max(t0 - c0, c1 - t1, 0)
    return max(t0 - c1, c0 - t1)


def claim_distance(claim: dict, truth: list[tuple[int, int, str]], name: str) -> tuple[int, int, str]:
    """
    Measure a claim's distance to the nearest true feature of its type, by the type's rule.

    Args:
        claim: A kept claim of claimed_spans.
        truth: The type's true (start, end, label) spans; not empty.
        name: Feature type.

    Returns:
        The distance, the distance by region_distance alone, and the rule that set it ("region" for that rule).
    """
    span = (claim["start"], claim["end"])
    plain = [(s, e) for s, e, _ in truth]
    base = min(region_distance(span, t) for t in plain)
    if name == "disulfide":
        return min(pair_distance(span, t) for t in plain), base, "disulfide_pair"
    if claim.get("part"):
        return min(overhang_distance(span, t) for t in plain), base, "part_overhang"
    starts = [t for t in plain if t[0] <= NTERM_START]
    if name in NTERM_TYPES and claim.get("kind") == "boundary" and starts:
        return min(abs(span[1] - t[1]) for t in starts), base, "boundary_end"
    if claim.get("kind") == "edge_start":
        return min(abs(span[0] - t[0]) for t in plain), base, "edge_start"
    if claim.get("kind") == "edge_end":
        return min(abs(span[1] - t[1]) for t in plain), base, "edge_end"
    if name in NTERM_TYPES and span[0] == 1 and span[1] >= WHOLE_MIN and starts:
        return min(abs(span[1] - t[1]) for t in starts), base, "whole_peptide_end"
    if name in UNION_TYPES:
        hit = [t for t in plain if min(span[1], t[1]) >= max(span[0], t[0])]
        if len(hit) >= 2:
            merged = region_distance(span, (min(t[0] for t in hit), max(t[1] for t in hit)))
            return min(base, merged), base, "coiled_coil_union"
    if name == "topological_domain" and claim.get("side"):
        same = [(s, e) for s, e, label in truth if label == claim["side"]]
        if same:
            return min(region_distance(span, t) for t in same), base, "topology_side"
    return base, base, "region"


def spans(record: dict, truth: dict | None, sites: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Every claimed feature span of one generation on the types the protein carries, with its distance to the nearest
    true feature of the type; plus the candidates the extractor did not bind.

    Args:
        record: Exported eval record.
        truth: The protein's truth.
        sites: Whether InterProScan site records join the active-site, binding-site and metal-binding truth.

    Returns:
        Scored claims (type, start, end, distance, base_distance, rule, within, kind, keyword, part, side, window) and
        rejected candidates (type, start, end, reason, keyword, window), the latter on every type.
    """
    if not usable(record, truth):
        return [], []
    text = record["generated_response"]
    length = record.get("sequence_length") or len(record["protein_sequence"])
    layout = answer_layout(text)
    scored, rejected = [], []
    for name in rc.FEATURE_TYPES:
        kept, dropped = claimed_spans(text, name, length, layout)
        rejected.extend({"type": name, **cand} for cand in dropped)
        true = type_truth(truth, name, sites)
        if not true:
            continue
        for claim in kept:
            dist, base, rule = claim_distance(claim, true, name)
            scored.append({"type": name, **claim, "distance": dist, "base_distance": base, "rule": rule,
                           "within": dist <= WITHIN})
    return scored, rejected


def interpro(record: dict, truth: dict | None, reference: dict, skip_runs: bool = True) -> list[dict]:
    """
    Every InterPro accession a generation names, with its verdict against the protein's InterPro entries.

    Args:
        record: Exported eval record.
        truth: The protein's truth (None scores every accession false).
        reference: The reference document (retired entries).
        skip_runs: Whether mentions inside degenerate accession runs are left out (the final round's rule); False
            scores them like any other mention, for comparison.

    Returns:
        One dict per distinct accession named outside degenerate accession runs (accession_run_offsets): acc, verdict
        (matched, parent, retired or false), correct, in_prompt and context (around its first such mention).
    """
    text = record.get("generated_response") or ""
    prompt = record.get("input_prompt") or ""
    shown = set(IPR_RE.findall(prompt[prompt.rfind("<|im_start|>user"):]))
    retired = reference["retired_interpro"]
    raw = truth["interpro"] if truth else set()
    closed = truth["interpro_closed"] if truth else set()
    members = truth["members"] if truth else set()
    runs = accession_run_offsets(text) if skip_runs else set()
    out, seen = [], set()
    for m in IPR_RE.finditer(text):
        acc = m.group(0)
        if acc in seen or m.start() in runs:
            continue
        seen.add(acc)
        if acc in raw:
            verdict = "matched"
        elif acc in closed:
            verdict = "parent"
        elif set(retired.get(acc) or ()) & members:
            verdict = "retired"
        else:
            verdict = "false"
        out.append({"acc": acc, "verdict": verdict, "correct": verdict != "false", "in_prompt": acc in shown,
                    "context": text[max(0, m.start() - CONTEXT):m.end() + CONTEXT]})
    return out


def accession_run_offsets(text: str) -> set[int]:
    """
    Where the mentions of degenerate accession runs start: LOOP_RUN or more InterPro mentions in a row, each a different
    accession at most ACCESSION_STEP numbers from the last and starting within ACCESSION_GAP characters of it
    ("IPR050098, IPR050097, IPR050096, ...").

    Args:
        text: The generation.

    Returns:
        Start offsets of the mentions inside runs.
    """
    mentions = [(int(m.group(0)[3:]), m.start()) for m in IPR_RE.finditer(text)]
    out, run = set(), mentions[:1]
    for previous, current in zip(mentions, mentions[1:]):
        step = abs(current[0] - previous[0])
        if 0 < step <= ACCESSION_STEP and current[1] - previous[1] <= ACCESSION_GAP:
            run.append(current)
            continue
        if len(run) >= LOOP_RUN:
            out |= {start for _, start in run}
        run = [current]
    if len(run) >= LOOP_RUN:
        out |= {start for _, start in run}
    return out


def residue_accuracy(hits: int, claims: int) -> dict:
    """
    Report cited residue accuracy with its claim count and whether the count clears the reporting floor.

    Args:
        hits: Counted claims that are correct.
        claims: Counted claims.

    Returns:
        {"accuracy", "claims", "reported"}; accuracy is None without claims.
    """
    return {"accuracy": hits / claims if claims else None, "claims": claims, "reported": claims >= MIN_RESIDUE_CLAIMS}


# =====================================================================================================================
# The LLM presence reader (`read`)
# =====================================================================================================================

# cache/ is gitignored, so the default cache stays out of git.
DEFAULT_CACHE = REPO / "cache" / "presence_llm"
TYPES = ("transmembrane", "signal_peptide", "coiled_coil", "active_site", "binding_site")

BASE_URL_VAR = "JUDGE_BASE_URL"
API_KEY_VAR = "JUDGE_API_KEY"
# Part of PROMPT_MD5, so a cache read by one model is never served as another's.
MODEL = os.environ.get("JUDGE_MODEL", "")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# On gold_dev the effort field barely moves this route (median reasoning 4.4k-4.6k tokens at low, high and max) and
# agreement differs by less than run-to-run noise; max matches the final judge's setting.
DEFAULT_EFFORT = "max"
MAX_TOKENS = 32000
# Replies take a median 22-39 s and at most 257 s of 1,900 measured; some connections are never answered at all, so a
# stalled call is cut and retried rather than left to hold a worker.
TIMEOUT = 300
RETRIES = 6
RATE_LIMIT_RETRIES = 200
BACKOFF = 10
RATE_LIMIT_WAIT = (20, 45)
QUOTE_WORDS = 40
PROGRESS_EVERY = 25
# The generations use Nemotron's own reasoning delimiters, which its tokenizer reads as the reader's reasoning even
# inside user text (llm_judge.py); the reader sees them under a neutral tag.
THINK_TAGS = (("<think>", "<reasoning>"), ("</think>", "</reasoning>"))

PROMPT_VERSION = "p3"
PROMPT_TEMPLATE = """You are an expert protein biologist auditing a model-written analysis of one protein. Read the TEXT \
at the end (the model's reasoning, inside <reasoning> tags, followed by its final answer) and decide, for each of five \
sequence-feature types, whether the TEXT ASSERTS that THIS protein, the protein the text is about, has that feature.

You judge what the text claims, not whether the claim is right. Do not use your own knowledge of the protein: a claim \
the text makes counts even if you think it is wrong, and a feature the protein surely has does not count unless the \
text says so.

WHAT COUNTS AS AN ASSERTION
- A statement anywhere in the reasoning or the answer that this protein has the feature: "a transmembrane helix at 23 \
to 45", "the first 20 residues are a cleavable signal peptide", "a predicted coiled coil at 184 to 204", "the catalytic \
serine 195", "zinc is coordinated by Cys38 and Cys41".
- A hedged claim the text still affirms: "likely a signal peptide", "probably anchored by one transmembrane helix", \
"a putative ATP-binding site".
- A tool or database call that the text relays as a property of this protein without rejecting it.
- An inference from family membership that the text applies to this protein ("like other microsomal P450s, it is \
anchored by an N-terminal transmembrane helix").
- A precursor feature: "synthesised with a signal peptide that is later removed" asserts a signal peptide.

WHAT DOES NOT COUNT
- A feature of another molecule: a partner, client, cargo, substrate, receptor, a subunit of a complex other than \
this protein, a homolog, or a family in general when the text does not apply it to this protein ("its partners are \
polytopic membrane proteins", "the E2's catalytic cysteine", "the clients' transmembrane segments", "the RNA forms \
the catalytic pocket").
- A hypothetical, conditional or counterfactual: "if it had a signal peptide", "a membrane enzyme would need", "I \
would change my mind if a transmembrane helix were found".
- A reading the text raises and then rejects or explains away: "a predictor calls 534 to 552 transmembrane, but I read \
it as a hydrophobic regulatory segment", "the signal-peptide call is best read as a false positive", "the stretch is \
a signal rather than a membrane anchor".
- A negation or absence: "no transmembrane helix", "lacks a catalytic triad", "no cleavable signal".
- A name used as a name: GO terms and activity labels ("transmembrane transporter activity", "zinc ion binding" or \
"lipid binding" given as a function), and InterPro entry, domain, fold or protein names ("guanine nucleotide-binding \
protein G", "the CHCH, coiled coil-helix-coiled coil-helix, domain", "the Zn(2)Cys(6) DNA-binding domain", "a RING \
finger", "a P-loop NTPase fold"), unless the text separately states the feature for this protein ("the six cysteines \
of the Zn(2)Cys(6) domain coordinate two zinc ions").
- A word that only shares letters with the feature: "signaling domain", "secretory signaling", "coiled bodies".
If the text asserts a feature and later explicitly retracts it, the retraction wins. If it affirms the feature \
anywhere as its own reading of this protein and never retracts it, the feature is asserted.

THE FIVE FEATURE TYPES
1. transmembrane: a segment of this protein that spans the lipid bilayer. Counts: a transmembrane helix, segment or \
domain; "single-pass", "multi-pass", "type I or type II membrane protein"; "crosses the membrane at 95 to 112"; an \
integral membrane protein held by membrane-spanning segments; an uncleaved signal anchor (a signal anchor is a \
transmembrane helix); membrane-spanning beta-barrel strands. A membrane domain the text describes with crossings \
counts even when it also has re-entrant loops ("a re-entrant segment between crossings"). Does not count: peripheral \
membrane association; lipid anchors (GPI, myristoyl, palmitoyl, prenyl); amphipathic, in-plane or re-entrant helices \
that do not cross the bilayer when nothing else does; membrane-binding domains (C2, PH, BAR, FYVE); \
"membrane-associated"; transport of a molecule across a membrane.
2. signal_peptide: a cleavable amino-terminal signal peptide that sends the chain into the secretory pathway or \
across the bacterial membrane (a secretory, Sec or Tat signal peptide, a lipoprotein signal peptide, a class III \
prepilin signal peptide), including the signal peptide of a precursor. Does not count: an uncleaved signal anchor \
(that is transmembrane); a mitochondrial or chloroplast transit peptide or presequence; nuclear localization, nuclear \
export, peroxisomal or ER-retention signals; a C-terminal GPI-attachment signal; signaling domains.
3. coiled_coil: a coiled-coil segment, region or domain of this protein: "a coiled coil at 812 to 839", heptad \
repeats that form a coiled coil, a leucine zipper, a SNARE coiled-coil motif, the coiled coil of a tripartite motif. \
These texts also name a coiled-coil prediction as "a coiled segment", "a coiled stretch" or "a coil at 79 to 99"; \
that counts, even when the text goes on to give the segment a role such as a hinge, linker or dimerization surface. \
Does not count: a coiled-coil protein that is a partner; "random coil"; a region described only as disordered or as \
a loop.
4. active_site: the catalytic site of this protein's own enzyme activity: an active site; catalytic residues (a \
catalytic nucleophile, general acid or base, proton donor or acceptor, charge relay); a catalytic triad or dyad; the \
catalytic centre, core, pocket, cleft or machinery of this protein ("a single catalytic core", "the ferrous iron \
centre where oxygen is activated"); an active-site signature the text accepts as this protein's. Does not count: \
naming the reaction or the enzyme class without a site or residue ("catalyzes hydrolysis", "is a kinase"); a \
catalytic domain, module or half with no site, residue or centre named; the active site of a partner, substrate or \
complex that this protein binds, docks into, regulates or completes; a denial ("lacks the catalytic residues", \
"catalytically dead"); catalytic-site signatures or residues the text names only to conclude that this protein's \
domain is degenerate, dead or not an enzyme.
5. binding_site: a site of this protein that binds a ligand other than DNA, RNA, a protein or a peptide: a metal \
ion (zinc, iron, calcium, magnesium, copper, manganese), a cofactor or prosthetic group (heme, FAD, FMN, NAD(P), PLP, \
SAM, coenzyme A, biotin, an iron-sulfur cluster), a nucleotide (ATP, GTP; a P-loop or Walker A phosphate-binding \
loop), a small molecule (a substrate, lipid, amino acid, phosphate, or an ion or solute a transporter binds), or a \
carbohydrate (a sugar, an oligosaccharide, or a polysaccharide such as chitin, starch, peptidoglycan, heparin or \
another glycosaminoglycan), stated as residues, a pocket, cleft, cavity, groove, loop or motif of this protein, or as \
this protein binding or coordinating the ligand. Also counts: a substrate-binding pocket, cleft or residue set of an \
enzyme, even when it is part of the active site (the two types are decided separately, and one pocket can assert \
both); a metal-coordinating active site ("a metal-coordinating active-site geometry"); a site the text calls a \
ligand-binding, cofactor-binding or substrate-binding site, unless the ligand it names is DNA, RNA, a protein or a \
peptide. Does not count: binding of DNA, RNA, other nucleic acids, proteins or peptides (including phosphopeptides, \
targeting signals, histone tails and ubiquitin); a partner's ligand site; a bare activity label with no site of this \
protein ("its function is lipid binding"); adhering to membranes, lipid bilayers or cell surfaces in general ("it \
binds lipids and coats host membranes"), as opposed to a named lipid or headgroup held by a motif, pocket or residues \
of this protein.

OUTPUT
Read the whole text, the reasoning and every part of the final answer, before deciding. Return one JSON object with \
one entry per type. In each entry, "quote" holds the decisive words from the text, copied verbatim (at most \
{quote_words} words), or "" when the text never mentions the feature; "asserted" is true or false. Decide every type \
on its own.

TEXT
<<<
{generation}
>>>"""

ENTRY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["quote", "asserted"],
                "properties": {"quote": {"type": "string"}, "asserted": {"type": "boolean"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(TYPES),
          "properties": {name: ENTRY_SCHEMA for name in TYPES}}
PROMPT_MD5 = hashlib.md5((PROMPT_TEMPLATE + json.dumps(SCHEMA, sort_keys=True) + MODEL).encode()).hexdigest()[:8]
RE_FENCED = re.compile(r"```(?:json)?\s*(.+?)```", re.S)

PRINT_LOCK = threading.Lock()


class Retryable(RuntimeError):
    """A transport error, timeout, 5xx, truncated reply or unreadable verdict: worth another attempt."""


class RateLimited(Retryable):
    """A 429 from the shared endpoint: wait and try again without spending an attempt."""


def api_settings(base_url: str | None = None) -> tuple[str, str]:
    """
    The endpoint base URL and key, read from the environment as evals/judge/llm_judge.py reads them.

    Args:
        base_url: An explicit base URL; else JUDGE_BASE_URL.

    Returns:
        (base URL, key).

    Raises:
        RuntimeError: If the model, the base URL or the key is not set.
    """
    key = os.environ.get(API_KEY_VAR)
    base = base_url or os.environ.get(BASE_URL_VAR)
    if not MODEL:
        raise RuntimeError("JUDGE_MODEL not set: the reader model's name at the endpoint")
    if not base:
        raise RuntimeError("%s not set (or pass --base-url)" % BASE_URL_VAR)
    if not key:
        raise RuntimeError("%s not set (keep it in the environment, never in a job spec)" % API_KEY_VAR)
    return base.rstrip("/"), key


def build_prompt(generation: str) -> str:
    """
    The reader's prompt for one generation.

    Args:
        generation: The generated response, reasoning and answer.

    Returns:
        The prompt, with the generation's reasoning tags renamed.
    """
    text = generation
    for tag, neutral in THINK_TAGS:
        text = text.replace(tag, neutral)
    return PROMPT_TEMPLATE.format(quote_words=QUOTE_WORDS, generation=text)


def text_md5(text: str) -> str:
    """
    A short fingerprint of a generation, stored with its reading so a changed generation is never served stale.

    Args:
        text: The generation.

    Returns:
        The first 12 hex digits of its md5.
    """
    return hashlib.md5(text.encode()).hexdigest()[:12]


def cache_file(cache_dir: Path, name: str, protein_id: str, effort: str) -> Path:
    """
    Where one reading is cached.

    Args:
        cache_dir: Cache root.
        name: The run's name, such as "rl-features"; readings of different runs never share a folder.
        protein_id: UniProt accession.
        effort: Reasoning effort.

    Returns:
        <cache_dir>/<prompt version>/<effort>/<name>/<protein_id>.json.
    """
    return Path(cache_dir) / PROMPT_VERSION / effort / name / f"{protein_id}.json"


def extract_json(text: str) -> dict | None:
    """
    Pull the verdict object out of a completion: bare JSON, a fenced block, or the outermost braces.

    Args:
        text: The completion's content.

    Returns:
        The parsed object, or None when nothing parses.
    """
    text = (text or "").strip()
    candidates = [text]
    fenced = RE_FENCED.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"):text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def validate(verdict: dict | None) -> list[str]:
    """
    Check a verdict against the schema.

    Args:
        verdict: The parsed reply.

    Returns:
        Problems; empty when the verdict is usable.
    """
    if not isinstance(verdict, dict):
        return ["unparseable"]
    problems = []
    for name in TYPES:
        entry = verdict.get(name)
        if not isinstance(entry, dict):
            problems.append(f"{name}: missing")
        elif not isinstance(entry.get("asserted"), bool):
            problems.append(f"{name}: asserted is not a boolean")
        elif not isinstance(entry.get("quote"), str):
            problems.append(f"{name}: quote is not a string")
    return problems


def post_chat(prompt: str, effort: str, base: str, key: str, timeout: int = TIMEOUT) -> dict:
    """
    One chat completion with strict JSON output at temperature 0.

    Args:
        prompt: The reader's prompt.
        effort: Reasoning effort, sent as reasoning_effort.
        base: API base URL.
        key: API key; never logged.
        timeout: Seconds before the request is abandoned.

    Returns:
        {"content", "thinking", "finish_reason", "usage"}.

    Raises:
        RateLimited: On HTTP 429.
        Retryable: On transport errors, timeouts, 408/409/5xx, or a reply cut at max_tokens.
        RuntimeError: On any other HTTP error, which is a bug in the request.
    """
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
               "temperature": 0, "reasoning_effort": effort,
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "presence", "strict": True, "schema": SCHEMA}}}
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as err:
        message = f"HTTP {err.code}: {err.read().decode(errors='replace')[:300]}"
        if err.code == 429:
            raise RateLimited(message) from err
        if err.code in (408, 409) or err.code >= 500:
            raise Retryable(message) from err
        raise RuntimeError(message) from err
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as err:
        raise Retryable(f"{type(err).__name__}: {str(err)[:200]}") from err
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if choice.get("finish_reason") == "length":
        raise Retryable("reply cut at max_tokens")
    usage = body.get("usage") or {}
    return {"content": message.get("content") or "", "thinking": message.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}}


def load_cached(path: Path, generation: str) -> dict | None:
    """
    A cached reading, if it is usable for this prompt and this generation.

    Args:
        path: The cache file.
        generation: The generation it must have read.

    Returns:
        The record, or None when absent, invalid or read from another generation.

    Raises:
        RuntimeError: If the record was written under the same prompt version but by another model (JUDGE_MODEL) or a
            different prompt, which means the prompt was edited without bumping PROMPT_VERSION.
    """
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("prompt_md5") != PROMPT_MD5:
        raise RuntimeError(f"{path} was read with prompt {record.get('prompt_md5')} by {record.get('model')!r}, not "
                           f"{PROMPT_MD5} by {MODEL!r}; set JUDGE_MODEL to the reader's model, or bump PROMPT_VERSION "
                           f"after editing the prompt")
    if record.get("generation_md5") != text_md5(generation) or record.get("problems"):
        return None
    return record


def read_one(name: str, protein_id: str, generation: str, effort: str, settings: tuple[str, str], cache_dir: Path,
             keep_thinking: bool = True) -> dict:
    """
    Read one generation, from the cache when it holds a usable reading, else from the endpoint with retries.

    Args:
        name: The run's name (cache_file).
        protein_id: UniProt accession.
        generation: The generated response.
        effort: Reasoning effort.
        settings: (base URL, key).
        cache_dir: Cache root.
        keep_thinking: Whether to store the reader's reasoning text with the record.

    Returns:
        The record: cell (the run's name), protein_id, prompt and generation fingerprints, verdict, raw reply, usage,
        timing, problems and "cached" (True when served from the cache).

    Raises:
        Retryable: When every attempt failed in transport.
        RuntimeError: On a non-retryable HTTP error.
    """
    path = cache_file(cache_dir, name, protein_id, effort)
    cached = load_cached(path, generation)
    if cached is not None:
        return {**cached, "cached": True}
    prompt = build_prompt(generation)
    start, attempts, waits, problems, reply = time.time(), 0, 0, ["not called"], {}
    while attempts < RETRIES:
        try:
            reply = post_chat(prompt, effort, *settings)
        except RateLimited:
            waits += 1
            if waits > RATE_LIMIT_RETRIES:
                raise
            time.sleep(random.uniform(*RATE_LIMIT_WAIT))
            continue
        except Retryable:
            attempts += 1
            if attempts >= RETRIES:
                raise
            time.sleep(BACKOFF * attempts + random.uniform(0, BACKOFF))
            continue
        attempts += 1
        verdict = extract_json(reply["content"])
        problems = validate(verdict)
        if not problems:
            break
    record = {"cell": name, "protein_id": protein_id, "prompt_version": PROMPT_VERSION, "prompt_md5": PROMPT_MD5,
              "model": MODEL, "effort": effort, "generation_md5": text_md5(generation),
              "verdict": extract_json(reply.get("content", "")) if not problems else None, "problems": problems,
              "raw": reply.get("content", ""), "finish_reason": reply.get("finish_reason"),
              "thinking": reply.get("thinking", "") if keep_thinking else "",
              "thinking_chars": len(reply.get("thinking", "")), "usage": reply.get("usage", {}),
              "attempts": attempts, "rate_limit_waits": waits, "elapsed_s": round(time.time() - start, 1),
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(record, ensure_ascii=False))
    os.replace(partial, path)
    return {**record, "cached": False}


def said(record: dict | None) -> dict[str, bool] | None:
    """
    The reader's presence decisions.

    Args:
        record: A reading from read_one.

    Returns:
        Feature type to whether the text asserts it, or None when the reading is unusable.
    """
    if not record or record.get("problems") or not record.get("verdict"):
        return None
    return {name: bool(record["verdict"][name]["asserted"]) for name in TYPES}


def cached_said(name: str, generations: dict[str, str], effort: str, cache_dir: Path) -> dict[str, dict[str, bool]]:
    """
    The cached presence decisions for one run, without calling the endpoint; for `score`.

    Args:
        name: The run's name (cache_file).
        generations: protein_id to the generation each reading must have read.
        effort: Reasoning effort of the readings.
        cache_dir: Cache root.

    Returns:
        protein_id to feature type to whether the text asserts it, for every usable cached reading.

    Raises:
        RuntimeError: If a cached reading was written by a different prompt under this prompt version.
    """
    out = {}
    for pid, text in generations.items():
        flags = said(load_cached(cache_file(cache_dir, name, pid, effort), text))
        if flags is not None:
            out[pid] = flags
    return out


def load_generations(path: Path, pattern: str) -> dict[str, str]:
    """
    One run's generations, from an eval.py results directory or a JSON-lines export (load_records).

    Args:
        path: The results directory or export.
        pattern: Which result files of a directory to read.

    Returns:
        protein_id to generated response (empty generations dropped).
    """
    records = load_records(path, pattern)
    return {r["protein_id"]: r["generated_response"] for r in records if r.get("generated_response")}


def run_name(path: Path) -> str:
    """
    The default name of a run: its results directory's name, or its export's name without the .jsonl(.gz) suffix.

    Args:
        path: The results directory or export.

    Returns:
        The name.
    """
    name = Path(path).resolve().name
    for suffix in (".gz", ".jsonl"):
        name = name.removesuffix(suffix)
    return name


def read_progress(done: int, total: int, fresh: int, failed: int, start: float) -> None:
    """
    Print one progress line to stderr.

    Args:
        done: Jobs finished.
        total: Jobs in the run.
        fresh: Jobs that called the endpoint.
        failed: Jobs that ended in an error.
        start: Run start time.
    """
    elapsed = time.time() - start
    rate = fresh / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate / 60 if rate > 0 else float("nan")
    with PRINT_LOCK:
        print(f"  {done}/{total} done, {fresh} called, {failed} failed, {elapsed / 60:.1f} min, "
              f"{rate * 60:.1f} calls/min, eta {eta:.1f} min", file=sys.stderr, flush=True)


def run(jobs: list[tuple[str, str, str]], effort: str, workers: int, cache_dir: Path, base_url: str | None = None,
        keep_thinking: bool = True) -> dict:
    """
    Read many generations with a worker pool.

    Args:
        jobs: (run name, protein_id, generation) per reading.
        effort: Reasoning effort.
        workers: Concurrent calls.
        cache_dir: Cache root.
        base_url: An explicit endpoint base URL (api_settings).
        keep_thinking: Whether to store the reader's reasoning text.

    Returns:
        (run name, protein_id) to its record, or to {"error": message} when every attempt failed.

    Raises:
        RuntimeError: If no key is set.
    """
    settings = api_settings(base_url)
    results, fresh, failed, start = {}, 0, 0, time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_one, name, pid, text, effort, settings, cache_dir, keep_thinking): (name, pid)
                   for name, pid, text in jobs}
        for done, future in enumerate(cf.as_completed(futures), 1):
            key = futures[future]
            try:
                record = future.result()
                fresh += not record["cached"]
            except (Retryable, RuntimeError) as err:
                record = {"error": str(err)[:300]}
                failed += 1
                with PRINT_LOCK:
                    print(f"  {key[0]} {key[1]}: {record['error']}", file=sys.stderr, flush=True)
            results[key] = record
            if done % PROGRESS_EVERY == 0 or done == len(futures):
                read_progress(done, len(futures), fresh, failed, start)
    return results


def read_main(args: argparse.Namespace) -> int:
    """
    `read`: read one run's generations and print its assertion rates and throughput.

    Args:
        args: The parsed options.

    Returns:
        0 when every generation has a usable reading, 2 otherwise (rerun to fill the gap from the cache).
    """
    name = args.name or run_name(args.results)
    generations = load_generations(args.results, args.pattern)
    pids = sorted(generations)[:args.limit] if args.limit else sorted(generations)
    jobs = [(name, pid, generations[pid]) for pid in pids]
    if args.print_prompt:
        print(build_prompt(jobs[0][2]))
        return 0
    print(f"prompt {PROMPT_VERSION} ({PROMPT_MD5}), model {MODEL}, effort {args.effort}, {len(jobs)} generations of "
          f"{name}, {args.workers} workers, cache {args.cache_dir}", file=sys.stderr)
    start = time.time()
    results = run(jobs, args.effort, args.workers, args.cache_dir, args.base_url, not args.no_thinking)
    wall = time.time() - start
    flags = [said(record) for record in results.values()]
    usable_flags = [f for f in flags if f]
    rates = " ".join(f"{kind} {sum(f[kind] for f in usable_flags) / max(1, len(usable_flags)):.3f}" for kind in TYPES)
    print(f"{name}: {len(usable_flags)}/{len(flags)} usable; asserted {rates}")
    called = [r for r in results.values() if "error" not in r and not r["cached"]]
    if called:
        seconds = sorted(r["elapsed_s"] for r in called)
        print(f"called {len(called)} in {wall / 60:.1f} min ({len(called) / wall * 60:.1f}/min); latency median "
              f"{seconds[len(seconds) // 2]:.0f} s, p90 {seconds[int(0.9 * (len(seconds) - 1))]:.0f} s; reasoning "
              f"tokens median {sorted(r['usage'].get('reasoning_tokens') or 0 for r in called)[len(called) // 2]}")
    return 0 if len(usable_flags) == len(flags) else 2


# =====================================================================================================================
# Scoring a run (`score`)
# =====================================================================================================================

WORKERS = 16
CHUNK = 8
DRAWS = 1000
SEED = 23
TABLE_VERDICTS = {"said": ("TP", "FP"), "true": ("TP", "FN"), "tp": ("TP",)}
# Table column prefix and metric prefix of each presence reader.
READERS = {"exist": "presence", "regex_exist": "regex_presence"}
TABLE_COLUMNS = ["protein_id", "rip_n", "rip_hit", "ipr_said", "ipr_correct", "ipr_said_noloop", "ipr_correct_noloop",
                 "presence_read"]
TABLE_COLUMNS += [f"{prefix}_{name}_{key}" for prefix in READERS for name in TOOL_TYPES
                  for key in TABLE_VERDICTS]
SPAN_COLUMNS = ["protein_id", "type", "start", "end", "distance", "within5", "distance_nosite", "within5_nosite"]
SPAN_VIEWS = {"span": "distance", "span_nosite": "distance_nosite"}
LABELS = {"transmembrane": "transmembrane", "signal_peptide": "signal peptide", "coiled_coil": "coiled coil",
          "active_site": "active site", "binding_site": "binding site"}
SHORT = {"transmembrane": "TM", "signal_peptide": "SP", "coiled_coil": "CC", "active_site": "AS", "binding_site": "BS"}

# Installed in the parent before the pool forks, so workers share them without pickling.
STATE = {}


def install(truth: dict, reference: dict, llm: dict) -> None:
    """
    Install the truth, the reference document and the LLM readings the workers read.

    Args:
        truth: load_truth().
        reference: load_reference().
        llm: protein_id to the LLM reader's presence decisions (cached_said).
    """
    STATE.update(truth=truth, reference=reference, llm=llm)


def presence_columns(verdicts: dict[str, str], prefix: str) -> dict[str, int | str]:
    """
    One reader's presence decisions as said, true and tp columns per type.

    Args:
        verdicts: Feature type to TP, FP, FN or TN; a type the reader did not decide is absent.
        prefix: Column prefix ("exist" or "regex_exist").

    Returns:
        Column to 0 or 1, or "" for every column of an undecided type.
    """
    out = {}
    for name in TOOL_TYPES:
        verdict = verdicts.get(name)
        for key, counted in TABLE_VERDICTS.items():
            out[f"{prefix}_{name}_{key}"] = "" if verdict is None else int(verdict in counted)
    return out


def blank(value: int | None) -> int | str:
    """
    A table cell for a value that may be missing.

    Args:
        value: The value, or None.

    Returns:
        The value, or "" for None.
    """
    return "" if value is None else value


def flag(distance: int | None) -> str:
    """
    A table cell for whether a distance is within WITHIN.

    Args:
        distance: Distance to the nearest true feature, or None without truth of the type.

    Returns:
        "1", "0", or "" without truth.
    """
    return "" if distance is None else str(int(distance <= WITHIN))


def span_rows(record: dict, truth: dict) -> list[list]:
    """
    Every span claim of one usable record, scored with and without the InterProScan site records in the truth.

    Args:
        record: The eval record.
        truth: The protein's truth.

    Returns:
        One SPAN_COLUMNS row per claim scored under either truth: the claims scored with the site records first, in
        the extractor's order, then those scored only without them.
    """
    scored, _ = spans(record, truth)
    nosite, _ = spans(record, truth, sites=False)
    full = {(c["type"], c["start"], c["end"]): c["distance"] for c in scored}
    plain = {(c["type"], c["start"], c["end"]): c["distance"] for c in nosite}
    rows = []
    for key in list(full) + [key for key in plain if key not in full]:
        distance, distance_nosite = full.get(key), plain.get(key)
        rows.append([record["protein_id"], *key, blank(distance), flag(distance), blank(distance_nosite),
                     flag(distance_nosite)])
    return rows


def score(record: dict) -> tuple[dict, list[list]]:
    """
    Score one record on the four metrics.

    Args:
        record: The eval record.

    Returns:
        Its per-protein table row (TABLE_COLUMNS) and its span rows (SPAN_COLUMNS).
    """
    pid = record["protein_id"]
    truth = STATE["truth"].get(pid)
    counted = [c for c in residue_claims(record, truth) if c["counted"]]
    named = interpro(record, truth, STATE["reference"])
    named_runs = interpro(record, truth, STATE["reference"], skip_runs=False)
    row = {"protein_id": pid, "rip_n": len(counted), "rip_hit": sum(int(c["correct"]) for c in counted),
           "ipr_said": len(named), "ipr_correct": sum(int(i["correct"]) for i in named),
           "ipr_said_noloop": len(named_runs), "ipr_correct_noloop": sum(int(i["correct"]) for i in named_runs),
           "presence_read": 0}
    llm, regex, spans = {}, {}, []
    if usable(record, truth):
        found = presence(record, truth)
        flags = STATE["llm"].get(pid)
        row["presence_read"] = int(flags is not None)
        for name in TOOL_TYPES:
            regex[name] = found[name]["verdict"]
            if flags is not None:
                llm[name] = confusion(flags[name], found[name]["true"])
        spans = span_rows(record, truth)
    row.update(presence_columns(llm, "exist"))
    row.update(presence_columns(regex, "regex_exist"))
    return row, spans


def write_tables(out: Path, name: str, rows: list[dict], spans: list[list]) -> None:
    """
    Write the per-protein and per-span tables.

    Args:
        out: Output directory.
        name: The run's name.
        rows: Per-protein rows.
        spans: Span rows.
    """
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / f"{name}.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(out / f"{name}.spans.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SPAN_COLUMNS)
        writer.writerows(spans)


def table_arrays(rows: list[dict], spans: list[list]) -> dict[str, np.ndarray]:
    """
    The per-protein counts the metrics pool, one value per protein, blanks as 0.

    Args:
        rows: Per-protein rows.
        spans: Span rows.

    Returns:
        Column name to values; plus <view>_n and <view>_within (SPAN_VIEWS) counting each protein's scored span claims
        and those within WITHIN.
    """
    out = {name: np.array([float(row[name] or 0) for row in rows]) for name in TABLE_COLUMNS[1:]}
    index = {row["protein_id"]: i for i, row in enumerate(rows)}
    for view, column in SPAN_VIEWS.items():
        position = SPAN_COLUMNS.index(column)
        scored = [(index[span[0]], span[position]) for span in spans if span[position] != ""]
        proteins = np.array([p for p, _ in scored], dtype=int)
        within = np.array([d <= WITHIN for _, d in scored], dtype=float)
        out[f"{view}_n"] = np.bincount(proteins, minlength=len(rows)).astype(float)
        out[f"{view}_within"] = np.bincount(proteins, weights=within, minlength=len(rows))
    return out


def metrics(table: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute every metric for each set of protein weights (one row per bootstrap draw; ones for the point estimate).

    Args:
        table: table_arrays().
        weights: Draws x proteins, how often each protein is drawn.

    Returns:
        Metric name to one value per draw.
    """
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["residue_claims"] = weights @ table["rip_n"]
        out["residue_accuracy"] = (weights @ table["rip_hit"]) / out["residue_claims"]
        out["interpro_said"] = weights @ table["ipr_said"]
        out["interpro_precision"] = (weights @ table["ipr_correct"]) / out["interpro_said"]
        out["interpro_said_runs_kept"] = weights @ table["ipr_said_noloop"]
        out["interpro_precision_runs_kept"] = (weights @ table["ipr_correct_noloop"]) / out["interpro_said_runs_kept"]
        for prefix, label in READERS.items():
            scores = {"precision": [], "recall": [], "f1": []}
            for name in TOOL_TYPES:
                said = weights @ table[f"{prefix}_{name}_said"]
                true = weights @ table[f"{prefix}_{name}_true"]
                tp = weights @ table[f"{prefix}_{name}_tp"]
                values = {"precision": tp / said, "recall": tp / true, "f1": 2 * tp / (said + true)}
                for stat, value in values.items():
                    out[f"{label}_{stat}_{name}"] = value
                    if name in F1_TYPES:
                        scores[stat].append(value)
            for stat, values in scores.items():
                out[f"{label}_{stat}_tool"] = np.nanmean(values, axis=0)
        for view in SPAN_VIEWS:
            out[f"{view}_claims"] = weights @ table[f"{view}_n"]
            out[f"{view}_within_{WITHIN}"] = (weights @ table[f"{view}_within"]) / out[f"{view}_claims"]
    return out


def resample(size: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw bootstrap weights: how often each protein appears in each resample.

    Args:
        size: Number of proteins.
        draws: Number of resamples.
        rng: Random generator.

    Returns:
        draws x size counts.
    """
    picks = rng.integers(0, size, size=(draws, size))
    return np.stack([np.bincount(row, minlength=size) for row in picks]).astype(float)


def summarise(table: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, dict]:
    """
    Point estimates with 95% percentile intervals over proteins.

    Args:
        table: table_arrays().
        draws: Bootstrap resamples; 0 gives no interval.
        seed: Random seed.

    Returns:
        Metric to {"value", "low", "high"} (low and high None without draws).
    """
    size = len(table["rip_n"])
    point = metrics(table, np.ones((1, size)))
    spread = metrics(table, resample(size, draws, np.random.default_rng(seed))) if draws else {}
    out = {}
    for name, value in point.items():
        low = high = None
        if draws and not np.all(np.isnan(spread[name])):
            low, high = (float(v) for v in np.nanpercentile(spread[name], [2.5, 97.5]))
        out[name] = {"value": float(value[0]), "low": low, "high": high}
    return out


def interval(entry: dict) -> str:
    """
    A value with its interval, for printing.

    Args:
        entry: {"value", "low", "high"}.

    Returns:
        "0.957 [0.951, 0.963]", or the value alone without an interval.
    """
    if entry["low"] is None:
        return f"{entry['value']:.3f}"
    return f"{entry['value']:.3f} [{entry['low']:.3f}, {entry['high']:.3f}]"


def report(name: str, summary: dict) -> None:
    """
    Print the four metrics and presence per type.

    Args:
        name: The run's name.
        summary: The summary document (main).
    """
    values = summary["metrics"]
    counts = summary["counts"]
    types = ", ".join(SHORT[t] for t in F1_TYPES)
    print(f"\n{name}: {counts['records']:,} records; {counts['usable']:,} usable for residue, presence and span "
          f"(sequence, generation and truth, under {rc.TRUNCATION_CAP:,} residues); {counts['llm_read']:,} "
          f"with an LLM presence reading\n")
    residue_note = "" if summary["residue_reported"] else (
        f"  under {MIN_RESIDUE_CLAIMS} claims: too few to report")
    lines = [
        ("cited residue accuracy", interval(values["residue_accuracy"]),
         f"{values['residue_claims']['value']:,.0f} claims{residue_note}"),
        ("InterPro precision", interval(values["interpro_precision"]),
         f"{values['interpro_said']['value']:,.0f} accessions"),
        (f"presence F1 ({types}), LLM reader", interval(values["presence_f1_tool"]), ""),
        (f"presence F1 ({types}), regex reader", interval(values["regex_presence_f1_tool"]), ""),
        (f"span within {WITHIN} residues", interval(values[f"span_within_{WITHIN}"]),
         f"{values['span_claims']['value']:,.0f} spans"),
        (f"span within {WITHIN} residues, no site records",
         interval(values[f"span_nosite_within_{WITHIN}"]),
         f"{values['span_nosite_claims']['value']:,.0f} spans"),
    ]
    for label, value, note in lines:
        print(f"  {label:46} {value:24} {note}")
    print("\n  presence per type, precision / recall / F1, LLM reader | regex reader")
    for kind in TOOL_TYPES:
        parts = []
        for label in READERS.values():
            parts.append(" / ".join(f"{values[f'{label}_{stat}_{kind}']['value']:.3f}"
                                    for stat in ("precision", "recall", "f1")))
        note = "  (recall only)" if kind in RECALL_ONLY_TYPES else ""
        print(f"    {LABELS[kind]:16} {parts[0]} | {parts[1]}{note}")


def score_main(args: argparse.Namespace) -> int:
    """
    `score`: score one run, write its tables and summary, and print the metrics.

    Args:
        args: The parsed options.

    Returns:
        0, or 2 when some usable record has no LLM presence reading (its presence columns stay blank).
    """
    name = args.name or run_name(args.results)
    out = args.out or args.results.resolve().parent / f"{name}-grounding"
    records = load_records(args.results, args.pattern)
    reference = load_reference()
    truth = load_truth(reference, args.test_set)
    generations = {r["protein_id"]: r["generated_response"] for r in records if r.get("generated_response")}
    llm = cached_said(name, generations, args.effort, args.cache_dir)
    install(truth, reference, llm)
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            results = pool.map(score, records, chunksize=CHUNK)
    else:
        results = [score(record) for record in records]

    rows = [row for row, _ in results]
    spans = [span for _, claims in results for span in claims]
    write_tables(out, name, rows, spans)
    n_usable = sum(usable(r, truth.get(r["protein_id"])) for r in records)
    read = sum(row["presence_read"] for row in rows)
    values = summarise(table_arrays(rows, spans), args.bootstrap, args.seed)
    summary = {
        "name": name, "results": str(args.results.resolve()), "test_set": str(args.test_set),
        "counts": {"records": len(records), "test_proteins": sum(r["protein_id"] in truth for r in records),
                   "usable": n_usable, "llm_read": read},
        "presence_llm": {"prompt_version": PROMPT_VERSION, "prompt_md5": PROMPT_MD5,
                         "model": MODEL, "effort": args.effort, "cache_dir": str(args.cache_dir)},
        "f1_types": F1_TYPES, "recall_only_types": list(RECALL_ONLY_TYPES),
        "min_residue_claims": MIN_RESIDUE_CLAIMS,
        "residue_reported": values["residue_claims"]["value"] >= MIN_RESIDUE_CLAIMS,
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "interval": "95% percentile, over proteins"},
        "metrics": values,
    }
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    report(name, summary)
    missing = summary["counts"]["records"] - summary["counts"]["test_proteins"]
    if missing:
        print(f"\nWARNING: {missing} records are not v2 test proteins; they score only on InterPro, every accession "
              f"false", file=sys.stderr)
    if read < n_usable:
        print(f"\nWARNING: {n_usable - read} usable records have no LLM presence reading in {args.cache_dir} under "
              f"'{name}'; the LLM presence values cover the {read} read. Run `read` first.", file=sys.stderr)
    print(f"\ntables and summary -> {out}")
    return 0 if read == n_usable else 2


def parse_args() -> argparse.Namespace:
    """
    Command-line options: `read` (LLM presence readings) and `score`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    read = commands.add_parser("read", help="LLM presence readings of a run, cached and resumable")
    read.add_argument("results", type=Path, help="eval.py results directory, or a .jsonl(.gz) export of one")
    read.add_argument("--name", help="the run's name in the cache (default: the results directory's name)")
    read.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    read.add_argument("--pattern", default=RESULT_PATTERN, help="result files of a directory to read")
    read.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORTS)
    read.add_argument("--base-url", default=None, help="endpoint base URL (default JUDGE_BASE_URL)")
    read.add_argument("--workers", type=int, default=64, help="concurrent calls; 192 is comfortable")
    read.add_argument("--limit", type=int, default=0, help="at most this many generations")
    read.add_argument("--no-thinking", action="store_true", help="do not store the reader's reasoning")
    read.add_argument("--print-prompt", action="store_true", help="print the first prompt and exit")

    score = commands.add_parser("score", help="score a run with the cached readings; write tables and summary")
    score.add_argument("results", type=Path, help="eval.py results directory, or a .jsonl(.gz) export of one")
    score.add_argument("--name", help="the run's name, for the LLM cache and the output files (default: the "
                                      "results directory's name)")
    score.add_argument("--out", type=Path, default=None, help="output directory (default: <results>-grounding)")
    score.add_argument("--test-set", type=Path, default=TEST_SET,
                       help="the v2 test split: a parquet file or a directory of shards")
    score.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                       help="the LLM presence reader's cache (read --cache-dir)")
    score.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORTS)
    score.add_argument("--pattern", default=RESULT_PATTERN, help="result files of a directory to read")
    score.add_argument("--workers", type=int, default=WORKERS)
    score.add_argument("--bootstrap", type=int, default=DRAWS, help="bootstrap resamples over proteins (0: none)")
    score.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    """
    Run the chosen command.

    Returns:
        Its exit status.
    """
    args = parse_args()
    return read_main(args) if args.command == "read" else score_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
