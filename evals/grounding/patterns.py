# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The claim patterns the audited grounding scorer (`grounding.py`) reads residue-level claims with.

Every grounding metric is a regex over free prose, so these patterns are the measurement
instrument: the feature vocabulary (FEATURE_TYPES), residue names, coordinate and span forms,
negation and attribution cues, and the window a coordinate may be bound to. `grounding.py` builds
its rules on top of them; `tests/test_grounding.py` pins the resulting decisions.
"""
import re


#: max_length_protein; at or above this the chain the model was shown was truncated.
TRUNCATION_CAP = 2000

#: Characters AFTER and BEFORE a feature keyword that a coordinate may be bound to.
#: Both directions, because these generations state the span either way round -- "a
#: transmembrane helix at 23 to 45" and "residues 43 to 63 are flagged as coiled" are
#: equally common, and a forward-only window sees only the first.
KEYWORD_WINDOW = 170
KEYWORD_WINDOW_BACK = 110

#: A sentence boundary truncates the window on both sides. Without it a 170-character
#: reach routinely crosses into the next clause and attributes its coordinates to this
#: feature -- the neighbouring-clause theft that makes SITE precision unreliable.
RE_SENTENCE_END = re.compile(r"[.;]\s")

#: A span stated as a range. `through`/`until` and the em dash are included because the
#: model uses all of them; `[1-113]` from an InterPro citation matches too, which is why
#: RE_ACCESSION_SPAN exists to take those back out.
RE_SPAN = re.compile(r"\b(\d{1,4})\s*(?:-|\u2013|\u2014|to|through|until)\s*(\d{1,4})\b",
                     re.IGNORECASE)

#: A range that belongs to an InterPro citation rather than to the feature being
#: described: "IPR036188 [34-341]". Attributing those to a nearby keyword was inflating
#: span recall for whatever feature happened to be mentioned next to a domain call.
RE_ACCESSION_SPAN = re.compile(r"IPR\d{6}[^0-9]{0,12}(\d{1,4})\s*[-\u2013]\s*(\d{1,4})")

#: Longest span each feature is allowed to claim, the 99th percentile of the annotated
#: lengths on the test split. A claim longer than anything UniProt annotates for that
#: feature is a domain or fold span swept in from the same sentence, not the feature --
#: an "active site" of 245 residues was the commonest SITE false positive. Capping at p99
#: rather than at the observed maximum forfeits at most 1% of true spans and removes the
#: whole class. The point features really are point features: MOD_RES, CARBOHYD, LIPID
#: and metal BINDING are length 1 at p99 AND at the maximum.
MAX_SPAN = {
    "transmembrane": 30, "signal_peptide": 41, "disulfide": 248, "zinc_finger": 87,
    "active_site": 20, "binding_site": 25, "coiled_coil": 255, "modified_residue": 3,
    "glycosylation": 3, "lipidation": 3, "transit_peptide": 90, "dna_binding": 298,
    "topological_domain": 802, "metal_binding": 3,
}

#: Explicitly marked single positions, including comma lists: "residues 46, 263 and 459".
#: Binding and active sites are stated this way far more often than as ranges, and the old
#: fallback -- the first two bare integers in the window -- picked up GO and InterPro
#: digits instead.
RE_POSITION = re.compile(
    r"(?:residues?|positions?|at|is|are)\s+(\d{1,4}(?:\s*,\s*\d{1,4})*"
    r"(?:\s*,?\s*and\s+\d{1,4})?)", re.IGNORECASE)

#: How far back a negation cue is searched for. Scoped to the CLAUSE rather than to a
#: fixed window: at 48 characters "I find no long apolar run that could be a signal anchor
#: or transmembrane helix" scored as an ASSERTION of a transmembrane helix, because the
#: cue sat 52 characters back. The cap is a backstop for prose with no punctuation.
NEGATION_WINDOW = 260

#: What ends the clause a negation can reach across. Sentence punctuation, and the
#: contrastive conjunctions -- without those, "there is no signal peptide, but a
#: transmembrane helix spans 23-45" would have its TM claim negated by the signal clause.
RE_CLAUSE_BREAK = re.compile(
    r"[.;:]|\b(?:but|yet|however|whereas|while|although|though|despite|"
    r"the only|instead|rather|so I|therefore)\b", re.IGNORECASE)

#: Markers that the thing just named belongs to a SUBSTRATE or a binding partner rather
#: than to this protein: "giving a phosphoserine product", "a phosphotyrosine-recognition
#: module". Applied per clause, exactly like negation, and only to the features in
#: SELF_ATTRIBUTION_TYPES -- `active_site` legitimately says "catalytic" about itself.
RE_OTHER_ENTITY = re.compile(
    r"\b(?:giving|gives|yields?|produces?|producing|product|converts?|converting|"
    r"transfers?|transferring|catalys\w+|catalyz\w+|recognis\w+|recogniz\w+|"
    r"acts? on|reader)\b", re.IGNORECASE)

#: Features whose vocabulary is shared between "this protein carries it" and "this protein
#: does it to something else". Modification chemistry is the whole class.
SELF_ATTRIBUTION_TYPES = ("modified_residue", "glycosylation", "lipidation")

RE_NEGATION = re.compile(
    r"\b(?:no|not|non(?!-)|none|never|without|lacks?|lacking|absent|absence|devoid|"
    r"neither|nor|unlikely|rather than|instead of|would be|could be|"
    r"cannot|can ?not|unable|fails? to|is ?n.t|are ?n.t|does ?n.t|do ?n.t|"
    r"nothing|no part|free of|rules? out|argues against|too \w+ to)\b",
    re.IGNORECASE)

AA3 = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C", "gln": "Q", "glu": "E",
    "gly": "G", "his": "H", "ile": "I", "leu": "L", "lys": "K", "met": "M", "phe": "F",
    "pro": "P", "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}

AA_FULL = {
    "alanine": "A", "arginine": "R", "asparagine": "N", "aspartate": "D",
    "aspartic acid": "D", "cysteine": "C", "glutamine": "Q", "glutamate": "E",
    "glutamic acid": "E", "glycine": "G", "histidine": "H", "isoleucine": "I",
    "leucine": "L", "lysine": "K", "methionine": "M", "phenylalanine": "F",
    "proline": "P", "serine": "S", "threonine": "T", "tryptophan": "W",
    "tyrosine": "Y", "valine": "V",
}

#: "Asp146", "Asp-146", "Asp 146". Case-sensitive on purpose: "his" and "met" are ordinary
#: English words, so a lowercase three-letter code before a number is not a residue claim.
RE_AA3 = re.compile(r"\b(%s)[\s-]?(\d{1,4})\b"
                    % "|".join([c.title() for c in AA3] + [c.upper() for c in AA3]))

#: "aspartate 146", "cysteines 34". Same trust as the three-letter form.
RE_FULL = re.compile(r"\b(%s)(s?)[\s-]?(\d{1,4})\b" % "|".join(sorted(AA_FULL, key=len,
                                                                     reverse=True)),
                     re.IGNORECASE)

#: The tail of "cysteines 34, 56 and 78" -- a plural name distributes over a number list,
#: and taking only the first number would undercount every disulfide and metal site.
RE_ENUM = re.compile(r"\s*(?:,|and|&|/)\s*(\d{1,4})\b")

#: Metals named in a UniProt BINDING record's description. Metal coordination is a
#: distinct claim the model makes constantly, and UniProt folded the old METAL feature
#: into BINDING in 2022, so it is recovered here by description rather than by type.
RE_METAL_DESC = re.compile(
    r"\b(zinc|Zn|calcium|Ca|magnesium|Mg|iron|Fe|copper|Cu|manganese|Mn|nickel|Ni|"
    r"cobalt|Co|cadmium|molybdenum|heme|haem|Fe\(?2\)?-S|iron-sulfur)\b", re.IGNORECASE)

#: Feature type -> (annotation keys it is stored under, regex naming it in prose).
#: The annotation keys are "<src>:<type>" as they appear in the protein table.
#:
#: Each pattern is a compromise measured rather than guessed, against a deliberately
#: over-broad lexicon whose disagreements were read. Three defects that found, all fixed here:
#:
#:   "a type I SINGLE-PASS membrane protein"   -- TM recall was understated
#:   "29 to 51 form a helical SIGNAL-ANCHOR"   -- only the unhyphenated form matched
#:   "COORDINATED lipid turnover"              -- the organisational sense of the word
#:                                                counted as a binding-site claim
FEATURE_TYPES = {
    "transmembrane": (
        ("uniprot:TRANSMEM", "interpro_match:tmhmm"),
        re.compile(r"transmembrane|\bTMD?s?\b|\bTMHs?\b|\b7TM\b|"
                   r"membrane[- ](?:spanning|embedded)|integral membrane|"
                   r"single[- ]pass|multi[- ]?pass|bitopic|polytopic|"
                   r"(?:spans?|spanning|traverses?|crosses) the (?:lipid )?(?:bi)?"
                   r"(?:layer|membrane)|membrane anchor|"
                   # A hydrophobic stretch is not necessarily transmembrane: "no
                   # transmembrane helix ... while the only strongly hydrophobic segment
                   # is the very amino terminus" scored as an assertion. The forms that
                   # do mean it ("crosses the membrane") are already matched above.
                   r"hydrophobic (?:helix|helices|segment|stretch|span) "
                   r"(?:that |which )?(?:crosses|spans|traverses|inserts)", re.IGNORECASE)),
    "signal_peptide": (
        ("uniprot:SIGNAL", "interpro_match:signalp_e", "interpro_match:signalp_g+",
         "interpro_match:signalp_g-"),
        re.compile(r"signal[- ]peptide|signal sequence|signal[- ]anchor|SignalP|"
                   r"secreto?ry signal|secretion signal|cleavable signal|"
                   r"leader (?:peptide|sequence)|N[- ]terminal signal", re.IGNORECASE)),
    "disulfide": (
        ("uniprot:DISULFID",),
        re.compile(r"disulfide|disulphide|cystine|\bS[-\u2013]S\b|"
                   r"(?:cysteine|cys) bridge", re.IGNORECASE)),
    "zinc_finger": (
        ("uniprot:ZN_FING",),
        # ZN_FING is the classic zinc finger. RING, PHD, LIM, FYVE, B-box and zinc
        # ribbons are zinc-binding modules UniProt annotates under other feature types,
        # so including them only costs precision: measured over all 5,939, dropping them
        # moves F1 0.383 -> 0.431 at a recall cost of 0.015.
        re.compile(r"zinc[- ]finger|\bZnFs?\b|\bC2H2\b|\bC3H1\b|\bCCHC\b|\bCCHH\b",
                   re.IGNORECASE)),
    "active_site": (
        ("uniprot:ACT_SITE", "interpro_match:active_site"),
        re.compile(r"active[- ]site|catalytic (?:residues?|sites?|centre|center|triad|"
                   r"dyad|tetrad|machinery|core|pocket|base|acid|nucleophile|"
                   r"Asp|Glu|His|Ser|Cys|Lys|Tyr|Thr|Arg|"
                   r"aspartate|glutamate|histidine|serine|cysteine|lysine|tyrosine|"
                   r"threonine|arginine)|nucleophilic (?:cysteine|serine|attack)|"
                   r"general (?:acid|base)|proton donor|reactive cysteine|"
                   r"catalytically essential", re.IGNORECASE)),
    "binding_site": (
        ("uniprot:BINDING", "interpro_match:binding_site"),
        # A binding NOUN is required. Bare "X binding" matched the answer's GO-term list
        # ("heme binding GO:0020037"), and "coordinat(es|ing|ion)" is dropped entirely
        # unless it names what is coordinated.
        # Requiring a noun after "X-binding" cost 0.16 of recall; dropping the
        # requirement but excluding a following GO id keeps the recall without
        # re-admitting the answer's own GO list ("heme binding GO:0020037"), which is a
        # term name rather than a residue claim. Measured over all 5,939: F1 0.548 ->
        # 0.577, and the unguarded version scores 0.581 by counting the GO list.
        re.compile(r"binding (?:sites?|residues?|pockets?|clefts?|grooves?|loop|motif)|"
                   r"(?:substrate|ligand|nucleotide|cofactor|metal|ATP|GTP|NAD|NADP|"
                   r"FAD|FMN|SAM|heme|haem)[- ]binding\b(?!\s*GO:)|"
                   r"\bP[- ]loop\b|Walker [AB]|"
                   r"contacts? (?:the )?(?:substrate|ligand|cofactor|nucleotide|ATP|GTP|"
                   r"NAD|FAD|heme)|"
                   r"coordinat(?:es|ing|ed) (?:a |the |two |three |four )?"
                   r"(?:zinc|Zn|calcium|Ca|magnesium|Mg|iron|Fe|copper|Cu|manganese|Mn|"
                   r"metal|ion|heme|haem|substrate|nucleotide)\b", re.IGNORECASE)),
    "coiled_coil": (
        ("uniprot:COILED", "interpro_match:coiled_coil", "interpro_match:coils"),
        re.compile(r"coiled[- ]?coil|\bcoiled\b|heptad|leucine zipper|"
                   r"(?:alpha|\u03b1)[- ]helical coil", re.IGNORECASE)),

    # --- added after the first pass; none has an InterProScan record type, so none can
    # --- be relayed as a coordinate line (descriptions are a separate question).
    "modified_residue": (
        ("uniprot:MOD_RES",),
        re.compile(r"(?:phosphorylation|acetylation|methylation|hydroxylation|"
                   r"sumoylation|ubiquitination) (?:sites?|residues?|of Ser|of Thr|of Tyr)|"
                   r"phospho[- ]?(?:serine|threonine|tyrosine|acceptor)s?\b"
                   r"(?!\s*[- ]?(?:phosphatase|aminotransferase|synthase|synthetase|"
                   r"kinase|dehydrogenase|transferase|lyase|decarboxylase|reductase|"
                   r"recognition|binding|reader|product|residue product))|"
                   r"(?:phosphorylat|acetylat|methylat|hydroxylat)(?:ion|ed) at\b|"
                   r"modified residues?|post[- ]translational modification|\bPTMs?\b",
                   re.IGNORECASE)),
    "glycosylation": (
        ("uniprot:CARBOHYD",),
        re.compile(r"glycosylat(?:ion|ed)|glycosylation (?:sites?|sequons?)|"
                   r"[NO][- ]linked glycan|[NO][- ]glycosylation|\bsequons?\b|"
                   r"glycoprotein", re.IGNORECASE)),
    "lipidation": (
        ("uniprot:LIPID",),
        re.compile(r"(?:myristoylat|palmitoylat|prenylat|farnesylat|geranylgeranylat|"
                   r"lipidat|S[- ]acylat)(?:ion|ed)\s+"
                   r"(?:at|on|sites?|residues?|cysteines?|glycine|anchor)|"
                   r"(?:myristoylation|palmitoylation|prenylation|farnesylation) motif|"
                   r"GPI[- ]anchor|lipid anchor|\bCAAX\b", re.IGNORECASE)),
    "transit_peptide": (
        ("uniprot:TRANSIT",),
        re.compile(r"transit peptide|mitochondrial (?:targeting|import) "
                   r"(?:sequence|peptide|signal)|\bMTS\b|chloroplast transit|"
                   r"presequence|matrix[- ]targeting", re.IGNORECASE)),
    "dna_binding": (
        ("uniprot:DNA_BIND",),
        # UniProt DNA_BIND marks a structural DNA-binding REGION (HTH, homeobox, bZIP),
        # not the functional statement "binds DNA". Matching the functional form against
        # a structural annotation gave precision 0.21 at recall 0.99 -- the pattern was
        # answering a different question from the ground truth.
        re.compile(r"DNA[- ]binding (?:domain|motif|region)|\bDBD\b|"
                   r"helix[- ]turn[- ]helix|\bHTH\b|homeodomain|homeobox|\bbZIP\b|"
                   r"basic helix[- ]loop[- ]helix|\bbHLH\b|winged[- ]helix|"
                   r"recognition helix", re.IGNORECASE)),
    "topological_domain": (
        ("uniprot:TOPO_DOM",),
        # TOPO_DOM is a topological domain OF A MEMBRANE PROTEIN. "at the cytoplasmic
        # face of the plasma membrane" is a subcellular-localisation idiom and bare
        # "topology" is used for localisation reasoning, so both fired on soluble
        # proteins -- which is where precision 0.25 at recall 0.95 came from.
        re.compile(r"cytoplasmic (?:domain|loop|tail|segment)|"
                   r"extracellular (?:domain|loop|segment)|"
                   r"(?:lumenal|luminal|periplasmic) (?:domain|loop|segment|tail)|"
                   # `non-cytoplasmic` is SignalP/Phobius vocabulary for "outside" and
                   # fires on secreted proteins, which carry no TOPO_DOM. Measured over
                   # all 5,939: dropping it moves F1 0.294 -> 0.467, precision 0.19 -> 0.36.

                   r"(?:type I{1,3}|N[- ]?in|N[- ]?out|C[- ]?in|C[- ]?out) "
                   r"(?:membrane|orientation|topology)", re.IGNORECASE)),
    "metal_binding": (
        ("uniprot:BINDING_METAL",),
        re.compile(r"metal[- ]binding\b(?!\s*GO:)|metal[- ]coordinat|"
                   r"(?:zinc|calcium|magnesium|iron|copper|manganese|heme|haem)"
                   r"[- ]binding\b(?!\s*GO:)|"
                   r"iron[-\u2013]sulfur|\bFe[-\u2013]S\b|"
                   r"coordinat(?:es|ing|ed) (?:a |the |two |three |four )?"
                   r"(?:zinc|Zn|calcium|Ca|magnesium|Mg|iron|Fe|copper|Cu|manganese|Mn|"
                   r"metal|heme|haem)\b", re.IGNORECASE)),
}


def clause_window(text: str, start: int, end: int) -> str:
    """The text around a keyword, truncated at the nearest sentence boundary each way.

    Args:
        text: the generated response.
        start: keyword match start.
        end: keyword match end.

    Returns:
        The window a coordinate may be bound to.
    """
    before = text[max(0, start - KEYWORD_WINDOW_BACK):start]
    breaks = list(RE_SENTENCE_END.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]

    after = text[end:end + KEYWORD_WINDOW]
    stop = RE_SENTENCE_END.search(after)
    if stop:
        after = after[:stop.start()]
    return before + text[start:end] + after

