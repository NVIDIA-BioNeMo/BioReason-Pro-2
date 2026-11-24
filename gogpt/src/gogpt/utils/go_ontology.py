# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""GO term names and the text rendering of GO-GPT predictions."""

ASPECT_LABELS = {
    "MF": "Molecular Function (MF)",
    "BP": "Biological Process (BP)",
    "CC": "Cellular Component (CC)",
}


def load_go_names(obo_file):
    """{GO id: name} for every non-obsolete term in an OBO file, alt_ids included.

    Matches goatools' GODag (what BioReason-Pro 2 uses): obsolete terms are left out, an
    alt_id resolves to its primary term's name, and "_" in names becomes a space.
    """
    names = {}
    term = None

    def flush(t):
        if t and t.get("id") and not t.get("obsolete"):
            name = t.get("name", "").replace("_", " ")
            for go_id in [t["id"]] + t.get("alt_ids", []):
                names[go_id] = name

    with open(obo_file) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("["):
                flush(term)
                term = {} if line == "[Term]" else None
            elif term is not None and ": " in line:
                key, value = line.split(": ", 1)
                if key == "id":
                    term["id"] = value
                elif key == "name":
                    term["name"] = value
                elif key == "alt_id":
                    term.setdefault("alt_ids", []).append(value)
                elif key == "is_obsolete" and value == "true":
                    term["obsolete"] = True
    flush(term)
    return names


def format_predictions(predictions, go_names):
    """Render {"MF": [...], "BP": [...], "CC": [...]} as the text block BioReason-Pro 2
    reads as "Initial GO term speculations".

    One line per aspect, always in MF, BP, CC order: the aspect label, ": ", then the
    terms as "GO:xxxxxxx (name)" joined by ", " in GO-GPT's generation order. An aspect
    without predictions keeps its label with nothing after ": ". Unknown ids are named
    "Unknown".
    """
    lines = []
    for aspect, label in ASPECT_LABELS.items():
        terms = ", ".join(f"{g} ({go_names.get(g, 'Unknown')})" for g in predictions.get(aspect, []))
        lines.append(f"{label}: {terms}")
    return "\n".join(lines)
