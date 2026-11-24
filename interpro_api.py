#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""InterProScan results for new proteins, as the feature records the model was trained on.

    python interpro_api.py --sequence MSSQQYQQQRRK...                         # online
    python interpro_api.py --fasta proteins.fasta --output features.json --email you@your.org
    python interpro_api.py --fasta proteins.fasta --output features.json --interproscan_dir <dir>

The prompt's residue-feature block was trained on InterPro's member-database matches and
residue sites (feature records with `src` `interpro_match` and `interpro_site`). This module
turns InterProScan's JSON into those records, field for field, so that
`scripts/data/build_question.py` renders them through the same renderer, cap and order as the
training prompts.

Two sources:

- online: InterPro's precomputed InterProScan matches, looked up by the sequence's MD5. This
  covers every sequence in UniParc, needs no account and returns at once. A sequence that is
  not there is submitted to EBI's InterProScan service, which needs a real email address
  (`--email`) and is rate-limited by EBI's fair-use policy.
- local: an InterProScan 5 installation (`interproscan.sh -f json`), for batches and for
  sequences EBI does not have.

Only the member databases the model saw in training are kept. Current InterProScan releases
predict transmembrane helices and signal peptides with TMbed and SignalP 6, which replaced
the TMHMM and SignalP 4 the corpus was built with, so their records are dropped; Phobius,
which the corpus also carries, still reports both.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

MATCHES_API_URL = "https://www.ebi.ac.uk/interpro/matches/api/matches"
IPRSCAN_API_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

REQUEST_TIMEOUT = 60
#: EBI's queue time varies widely (minutes for one protein), so the wait is generous.
POLL_TIMEOUT = 1800
POLL_INTERVAL = 10
#: EBI's fair-use policy allows 30 concurrent jobs per user.
ONLINE_WORKERS = 25

#: InterProScan library name, upper-cased with punctuation removed -> the corpus's `member_db`.
#: The precomputed service spells libraries for display (`CATH-Gene3D`), InterProScan's own
#: JSON as enum names (`CATH_GENE3D`); both normalise to the same key.
MEMBER_DB = {
    "CATHFUNFAM": "cathfunfam", "FUNFAM": "cathfunfam",
    "CATHGENE3D": "cathgene3d", "GENE3D": "cathgene3d",
    "CDD": "cdd",
    "COILS": "coils",
    "ELM": "elm",
    "HAMAP": "hamap",
    "MOBIDBLITE": "mobidblt", "MOBIDB": "mobidblt",
    "NCBIFAM": "ncbifam",
    "PANTHER": "panther",
    "PFAM": "pfam",
    "PHOBIUS": "phobius",
    "PIRSF": "pirsf",
    "PIRSR": "pirsr",
    "PRINTS": "prints",
    "PROSITEPATTERNS": "prosite",
    "PROSITEPROFILES": "profile",
    "SFLD": "sfld",
    "SIGNALPEUK": "signalp_e",
    "SIGNALPGRAMPOSITIVE": "signalp_g+",
    "SIGNALPGRAMNEGATIVE": "signalp_g-",
    "SMART": "smart",
    "SUPERFAMILY": "ssf",
    "TMHMM": "tmhmm",
}

#: Predictors without InterPro entries: the corpus types them by database and, for most,
#: carries no description, so the renderer labels them by `member_acc`.
PREDICTOR_DBS = frozenset(("coils", "mobidblt", "phobius", "signalp_e", "signalp_g+",
                           "signalp_g-", "tmhmm", "elm"))
#: Of the predictors, the one whose corpus records carry the signature name as `desc`.
NAMED_PREDICTOR_DBS = frozenset(("mobidblt",))
#: CATH-FunFam matches are typed `match` and carry no description in the corpus.
FUNFAM_TYPE = "match"
#: `source` of the precomputed service's InterPro-N predictions.
INTERPRO_N_SOURCE = "InterPro-N"
#: Databases whose sites, not matches, enter the block: a PIRSR match is only a carrier.
SITE_ONLY_DBS = frozenset(("pirsr",))
#: PIRSR site descriptions in the corpus are InterPro 109's text, which ended most of them
#: (79% on a 150-protein sample) in a full stop and wrote a bare `DISULFID` as `DISULFID:`.
#: Later releases dropped both; restoring them is right for most sites and off by one
#: character for the rest.
PERIOD_SITE_DBS = frozenset(("pirsr",))

FEATURE_KEYS = ("src", "type", "start", "end", "desc", "acc", "member_db", "member_acc", "residue")


def _library_key(name: str) -> str:
    """Normalise an InterProScan library name.

    Args:
        name: library as InterProScan or the precomputed service spells it.

    Returns:
        Upper-case alphanumerics only.
    """
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def _record(**fields) -> Dict:
    """A feature record with every key the corpus schema uses for InterPro.

    Args:
        **fields: the values to set.

    Returns:
        The record, unset keys None.
    """
    return {key: fields.get(key) for key in FEATURE_KEYS}


def _match_type(signature: Dict, member_db: str) -> Optional[str]:
    """The corpus `type` of one match.

    Args:
        signature: InterProScan signature object.
        member_db: its corpus database name.

    Returns:
        The entry type when the signature is integrated, else the signature's own type, with
        the predictor and CATH-FunFam conventions of the corpus.
    """
    if member_db in PREDICTOR_DBS:
        return member_db
    if member_db == "cathfunfam":
        return FUNFAM_TYPE
    entry = signature.get("entry") or {}
    kind = entry.get("type") or signature.get("type")
    return kind.lower() if kind else None


def _match_desc(signature: Dict, member_db: str) -> Optional[str]:
    """The corpus `desc` of one match.

    Args:
        signature: InterProScan signature object.
        member_db: its corpus database name.

    Returns:
        The InterPro entry's full name when integrated, else the signature's description;
        the signature name for MobiDB-lite and nothing for the other predictors and FunFam.
    """
    if member_db in NAMED_PREDICTOR_DBS:
        return signature.get("name")
    if member_db in PREDICTOR_DBS or member_db == "cathfunfam":
        return None
    entry = signature.get("entry") or {}
    return entry.get("description") or entry.get("name") or signature.get("description")


def _site_desc(description: Optional[str], member_db: str) -> Optional[str]:
    """The corpus `desc` of one residue site.

    Args:
        description: InterProScan's site description.
        member_db: the site's database.

    Returns:
        The description, in the corpus's PIRSR punctuation.
    """
    if not description or member_db not in PERIOD_SITE_DBS:
        return description
    if ":" not in description:
        return description + ":"
    return description if description.endswith(".") else description + "."


def to_features(matches: List[Dict]) -> List[Dict]:
    """InterProScan matches -> `interpro_match` and `interpro_site` feature records.

    One record per location fragment, so a discontinuous match stays separate fragments as in
    the corpus, and one per site location.

    Args:
        matches: the `matches` list of one protein in InterProScan JSON.

    Returns:
        The records, in input order; `build_question` sorts and caps them.
    """
    records = []
    for match in matches or []:
        # The precomputed service also returns InterPro-N's predicted matches, which the corpus
        # does not carry. They are told apart by `source`, not by a null signature type:
        # InterProScan's own JSON leaves the signature type null on every match.
        if match.get("source") == INTERPRO_N_SOURCE:
            continue
        signature = match.get("signature") or {}
        library = (signature.get("signatureLibraryRelease") or {}).get("library")
        member_db = MEMBER_DB.get(_library_key(library))
        if member_db is None:
            continue
        entry = signature.get("entry") or {}
        accession = signature.get("accession")
        for location in match.get("locations") or []:
            if member_db not in SITE_ONLY_DBS:
                fragments = location.get("location-fragments") or [location]
                for fragment in fragments:
                    records.append(_record(
                        src="interpro_match", type=_match_type(signature, member_db),
                        start=int(fragment["start"]), end=int(fragment["end"]),
                        desc=_match_desc(signature, member_db), acc=entry.get("accession"),
                        member_db=member_db, member_acc=accession))
            for site in location.get("sites") or []:
                for spot in site.get("siteLocations") or []:
                    records.append(_record(
                        src="interpro_site", type="site",
                        start=int(spot["start"]), end=int(spot["end"]),
                        desc=_site_desc(site.get("description"), member_db),
                        member_db=member_db, member_acc=accession, residue=spot.get("residue")))
    return records


def sequence_md5(sequence: str) -> str:
    """The MD5 InterPro keys sequences by.

    Args:
        sequence: amino-acid sequence.

    Returns:
        Upper-case hex digest.
    """
    return hashlib.md5(sequence.upper().encode()).hexdigest().upper()


def lookup_precomputed(sequence: str) -> Optional[List[Dict]]:
    """InterPro's precomputed InterProScan matches for a sequence.

    Args:
        sequence: amino-acid sequence.

    Returns:
        The matches, or None when InterPro has not scanned this sequence.

    Raises:
        requests.HTTPError: on any other HTTP failure.
    """
    resp = requests.get(f"{MATCHES_API_URL}/{sequence_md5(sequence)}", timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("matches") or []


def run_ebi_job(sequence: str, email: str) -> List[Dict]:
    """Run InterProScan on EBI's service and wait for it.

    Args:
        sequence: amino-acid sequence.
        email: a real address; EBI blocks placeholder ones.

    Returns:
        The protein's matches.

    Raises:
        RuntimeError: when EBI refuses the job or it fails.
        TimeoutError: when it does not finish within POLL_TIMEOUT.
    """
    resp = requests.post(f"{IPRSCAN_API_URL}/run", timeout=REQUEST_TIMEOUT,
                         data={"email": email, "sequence": sequence, "stype": "p",
                               "goterms": "false", "pathways": "false"})
    if not resp.ok:
        raise RuntimeError(f"EBI refused the InterProScan job ({resp.status_code}): {resp.text[:300]}")
    job = resp.text.strip()

    deadline = time.time() + POLL_TIMEOUT
    status = "UNKNOWN"
    while time.time() < deadline:
        status = requests.get(f"{IPRSCAN_API_URL}/status/{job}", timeout=REQUEST_TIMEOUT).text.strip()
        if status == "FINISHED":
            break
        if status in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError(f"InterProScan job {job} ended {status}")
        time.sleep(POLL_INTERVAL)
    else:
        raise TimeoutError(f"InterProScan job {job} still {status} after {POLL_TIMEOUT}s")

    result = requests.get(f"{IPRSCAN_API_URL}/result/{job}/json", timeout=REQUEST_TIMEOUT)
    result.raise_for_status()
    proteins = result.json().get("results") or []
    if not proteins:
        return []
    return proteins[0].get("matches") or []


def scan_online(sequence: str, email: Optional[str] = None) -> List[Dict]:
    """Feature records for one sequence: precomputed matches, else an EBI job.

    Args:
        sequence: amino-acid sequence.
        email: needed only for a sequence InterPro has not scanned.

    Returns:
        The feature records.

    Raises:
        RuntimeError: when the sequence needs an EBI job and no email was given.
    """
    matches = lookup_precomputed(sequence)
    if matches is None:
        if not email:
            raise RuntimeError("not in InterPro's precomputed matches; pass --email to run "
                               "InterProScan on EBI, or use a local installation")
        matches = run_ebi_job(sequence, email)
    return to_features(matches)


def scan_local(sequences: Dict[str, str], interproscan_dir: str, cpu: int = 8) -> Dict[str, List[Dict]]:
    """Feature records for many sequences from a local InterProScan 5.

    Args:
        sequences: protein id -> sequence.
        interproscan_dir: the installation (contains `interproscan.sh`).
        cpu: worker threads for InterProScan.

    Returns:
        protein id -> feature records; a protein with no match maps to [].

    Raises:
        subprocess.CalledProcessError: when InterProScan fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fasta = os.path.join(tmp, "input.fasta")
        output = os.path.join(tmp, "output.json")
        with open(fasta, "w") as handle:
            for pid, seq in sequences.items():
                handle.write(f">{pid}\n{seq}\n")
        subprocess.run([os.path.join(interproscan_dir, "interproscan.sh"), "-i", fasta,
                        "-f", "json", "-o", output, "-cpu", str(cpu)], check=True)
        with open(output) as handle:
            results = json.load(handle).get("results") or []

    by_id = {pid: [] for pid in sequences}
    for protein in results:
        for xref in protein.get("xref") or []:
            if xref.get("id") in by_id:
                by_id[xref["id"]] = to_features(protein.get("matches"))
    return by_id


def scan_online_one(item: tuple) -> tuple:
    """One `scan_online` call for a thread pool, with its error captured.

    Args:
        item: (protein id, sequence, email).

    Returns:
        (protein id, records or None, error message or None).
    """
    pid, sequence, email = item
    try:
        return pid, scan_online(sequence, email), None
    except (requests.RequestException, RuntimeError, TimeoutError, ValueError) as exc:
        return pid, None, str(exc)


def scan_many(sequences: Dict[str, str], email: Optional[str] = None,
              interproscan_dir: Optional[str] = None, cpu: int = 8,
              workers: int = ONLINE_WORKERS) -> Dict[str, List[Dict]]:
    """Feature records for many sequences, locally or online.

    Args:
        sequences: protein id -> sequence.
        email: for EBI jobs, online only.
        interproscan_dir: a local InterProScan; online when None.
        cpu: InterProScan threads, local only.
        workers: concurrent online requests.

    Returns:
        protein id -> feature records. A protein that failed online is left out, with a
        warning, so the caller can fall back to the no-feature prompt for it.
    """
    if interproscan_dir:
        return scan_local(sequences, interproscan_dir, cpu)
    out = {}
    items = [(pid, seq, email) for pid, seq in sequences.items()]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        for pid, records, error in pool.map(scan_online_one, items):
            if error:
                print(f"  InterPro failed for {pid}: {error}", file=sys.stderr)
            else:
                out[pid] = records
    return out


def read_fasta(path: str) -> Dict[str, str]:
    """Read a FASTA file.

    Args:
        path: FASTA file.

    Returns:
        id (the first word of the header) -> sequence.
    """
    sequences, pid = {}, None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                pid = line[1:].split()[0]
                sequences[pid] = ""
            elif pid is not None:
                sequences[pid] += line
    return sequences


def main() -> int:
    """CLI: print one sequence's records, or write a FASTA file's as JSON.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", help="one amino-acid sequence")
    parser.add_argument("--fasta", help="FASTA file of proteins")
    parser.add_argument("--output", help="JSON file, protein id -> feature records (with --fasta)")
    parser.add_argument("--email", default=None, help="your email, for EBI InterProScan jobs")
    parser.add_argument("--interproscan_dir", default=None, help="a local InterProScan 5 installation")
    parser.add_argument("--cpu", type=int, default=8, help="InterProScan threads (local)")
    args = parser.parse_args()

    if bool(args.sequence) == bool(args.fasta):
        parser.error("give exactly one of --sequence and --fasta")
    if args.fasta and not args.output:
        parser.error("--output is required with --fasta")

    sequences = read_fasta(args.fasta) if args.fasta else {"query": args.sequence.strip()}
    features = scan_many(sequences, args.email, args.interproscan_dir, args.cpu)
    if args.fasta:
        with open(args.output, "w") as handle:
            json.dump(features, handle, indent=1)
        print(f"{len(features)}/{len(sequences)} proteins written to {args.output}")
    else:
        print(json.dumps(features.get("query"), indent=1))
    return 0 if len(features) == len(sequences) else 1


if __name__ == "__main__":
    raise SystemExit(main())
