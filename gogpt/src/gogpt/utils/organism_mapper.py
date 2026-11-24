# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import re
from collections import Counter
from pathlib import Path

UNKNOWN = "<UNKNOWN>"


class OrganismMapper:
    """Maps an organism key (an NCBI taxonomy id as a string, e.g. "9606") to the index
    of its organism embedding. Index 0 is <UNKNOWN>: organisms outside the top
    `top_n_organisms` of the training data, and organisms that are not given.

    `names` optionally maps each key to a display name (e.g. "Homo sapiens (Human)") so
    callers can pass a name instead of a taxonomy id; see resolve().
    """

    def __init__(self, organisms, top_n_organisms=None, names=None):
        valid = [o for o in organisms if o is not None]
        counts = Counter(valid)
        if top_n_organisms is not None:
            kept = [o for o, _ in counts.most_common(top_n_organisms)]
        else:
            kept = list(counts)
        self.organism_to_idx = {UNKNOWN: 0}
        for idx, organism in enumerate(sorted(kept), 1):
            self.organism_to_idx[organism] = idx
        self.idx_to_organism = {i: o for o, i in self.organism_to_idx.items()}
        self.top_n_organisms = top_n_organisms
        self.total_original_organisms = len(counts)
        self.vocab_size = len(self.organism_to_idx)
        self.names = {k: v for k, v in (names or {}).items() if k in self.organism_to_idx}

    def get_vocab_size(self):
        return self.vocab_size

    def map_organism(self, organism):
        """Index for an exact organism key; unknown keys and None map to 0."""
        if organism is None:
            return 0
        return self.organism_to_idx.get(organism, 0)

    def resolve(self, organism):
        """Organism key for a taxonomy id (int or str) or a display name; None if unknown.

        Names match either the full display name ("Homo sapiens (Human)") or the
        scientific name alone ("Homo sapiens"), case-insensitively.
        """
        if organism is None:
            return None
        if isinstance(organism, (int,)) and not isinstance(organism, bool):
            organism = str(organism)
        organism = str(organism).strip()
        if organism in self.organism_to_idx and organism != UNKNOWN:
            return organism
        lookup = self._name_lookup()
        return lookup.get(organism.lower())

    def _name_lookup(self):
        if not hasattr(self, "_lookup"):
            lookup = {}
            for key, name in self.names.items():
                lookup[name.lower()] = key
                scientific = re.sub(r"\s*\(.*$", "", name).strip().lower()
                lookup.setdefault(scientific, key)
            self._lookup = lookup
        return self._lookup

    # ------------------------------------------------------------------ serialization

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({
                "organism_to_idx": self.organism_to_idx,
                "names": self.names,
                "top_n_organisms": self.top_n_organisms,
            }, f, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(Path(path)) as f:
            d = json.load(f)
        mapper = cls.__new__(cls)
        mapper.organism_to_idx = dict(d["organism_to_idx"])
        if mapper.organism_to_idx.get(UNKNOWN) != 0:
            raise ValueError(f"{path}: {UNKNOWN} must map to index 0")
        mapper.idx_to_organism = {i: o for o, i in mapper.organism_to_idx.items()}
        if sorted(mapper.idx_to_organism) != list(range(len(mapper.organism_to_idx))):
            raise ValueError(f"{path}: organism indices are not contiguous from 0")
        mapper.vocab_size = len(mapper.organism_to_idx)
        mapper.names = dict(d.get("names", {}))
        mapper.top_n_organisms = d.get("top_n_organisms")
        mapper.total_original_organisms = None
        return mapper
