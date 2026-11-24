#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Download the assets BioReason-Pro 2 training and evaluation need but that are not
part of the git repository:

  * protein structures   -> wanglab/bioreason-pro-structures
  * GO term embeddings   -> wanglab/bioreason-pro-go-embeddings
  * IA weights           -> data/IA_CAFA6.tsv, the CAFA 6 information accretion the
                            benchmark is scored with, from the CAFA 6 Kaggle competition
                            (needs Kaggle credentials and the competition's rules accepted;
                            checked by md5 against the file every published number used)

Every shard is extracted by default, so the local mirror is complete. Pass
--referenced-only to keep just the files the corpus (wanglab/bioreason-pro-v2) names,
which is a fraction of the disk.

The embeddings and structures land as plain directories that you point the scripts at, the
IA weights at the path `bioreason2/paths.py` resolves:

    python scripts/data/download_assets.py --dest assets

    # -> assets/structures      (STRUCTURE_DIR)
    # -> assets/go_embeddings   (GO_EMBEDDINGS_PATH)
    # -> data/IA_CAFA6.tsv      (BRP_IA_FILE)

The download is resumable: re-running skips shards that are already extracted. The IA weights
come last and a failure there does not stop the rest, since only scoring needs them.

Structures are optional. Prediction and evaluation run sequence-only without them
(USE_STRUCTURE=False for eval.sh); training refuses a structures directory that does not
cover the table unless BRP_ALLOW_PARTIAL_STRUCTURES=1, since the released checkpoints were
trained with them and skipping changes the model. Use --verify to confirm that what you
downloaded actually covers the corpus (see the silent-fallback note below).
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bioreason2 import paths as _paths

STRUCTURES_REPO = _paths.HF_STRUCTURES
GO_EMBEDDINGS_REPO = _paths.HF_GO_EMBEDDINGS

# The corpus's (config, split)s of training and the benchmark. Both the coverage check and
# --referenced-only read structure_path from these.
CORPUS_REPO = _paths.HF_RAW_DATASET
CORPUS_SPLITS = (("proteins", "train"), ("proteins_extra", "train"), ("proteins_extra", "ground"),
                 ("proteins_extra", "test"))
# Read alone: some columns are all-null in the last train shards, so a full load fails to cast them.
STRUCTURE_COLUMN = "structure_path"

# CAFA publishes its IA weights without a license, so they are fetched from the source.
IA_COMPETITION = "cafa-6-protein-function-prediction"
IA_KAGGLE_FILE = "IA.tsv"
IA_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/competitions/data/download/{competition}/{file}"
#: The weights every published number was scored with; anything else changes every weighted metric.
IA_MD5 = "368d7288f0d99dfe6aee594a7bc9d9a6"
IA_TIMEOUT_S = 120
KAGGLE_CONFIG = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_shard(tar_path: str, dest_dir: str, keep: set = None) -> int:
    """Extract one shard into dest_dir as a flat directory of .cif files.

    Members inside the shards are gzipped (`AF-XXXX-F1-model_v4.cif.gz`) but the
    `structure_path` column of the datasets refers to the *decompressed* name
    (`AF-XXXX-F1-model_v4.cif`). We therefore gunzip on the way out. Getting this
    wrong is silent: collate.py checks os.path.exists() and falls back to empty
    coordinates without warning, so every structure would be ignored.

    `keep`, when given, restricts extraction to the filenames the corpus actually
    references (~263k of the ~370k shipped). Default is None: extract everything, so
    the local mirror is a complete copy of the published structures.
    """
    written = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name.endswith(".gz"):
                name = name[: -len(".gz")]
            if keep is not None and name not in keep:
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            data = fileobj.read()
            if os.path.basename(member.name).endswith(".gz"):
                data = gzip.decompress(data)
            with open(os.path.join(dest_dir, name), "wb") as out:
                out.write(data)
            written += 1
    return written


def _referenced_structure_names() -> set:
    """Filenames the corpus names in structure_path, across CORPUS_SPLITS."""
    from datasets import load_dataset

    names = set()
    for config, split in CORPUS_SPLITS:
        ds = load_dataset(CORPUS_REPO, config, split=split, columns=[STRUCTURE_COLUMN])
        names.update(p for p in ds[STRUCTURE_COLUMN] if p)
    return names


def _shard_marker(dest_dir: str, shard: str) -> str:
    return os.path.join(dest_dir, ".shards", shard.replace("/", "__") + ".done")


def download_structures(dest_dir: str, num_workers: int, keep_archives: bool,
                        referenced_only: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, ".shards"), exist_ok=True)

    keep = None
    if referenced_only:
        print(f"structures: resolving which files {CORPUS_REPO} references …")
        try:
            keep = _referenced_structure_names()
        except Exception as exc:
            raise SystemExit(f"--referenced-only could not read {CORPUS_REPO} ({exc}); "
                             "rerun when it is reachable, or drop the flag to extract everything") from exc
        print(f"  {len(keep)} referenced (shards hold ~370k; "
              f"filtering saves ~40 GB of disk)")

    api = HfApi()
    available = sorted(
        f for f in api.list_repo_files(STRUCTURES_REPO, repo_type="dataset")
        if f.endswith(".tar.gz")
    )
    # Every shard: the v2 corpus names structures across all of them.
    shards = available
    if not available:
        raise SystemExit(
            f"{STRUCTURES_REPO} exposes no .tar.gz shards. Downloading nothing and "
            "reporting success is how a run ends up silently sequence-only.")
    todo = [s for s in shards if not os.path.exists(_shard_marker(dest_dir, s))]
    print(f"structures: {len(shards)} of {len(available)} shards, {len(todo)} to fetch")
    if not todo:
        print("  already complete")
        return

    archive_dir = os.path.join(dest_dir, ".archives") if keep_archives else tempfile.mkdtemp()
    os.makedirs(archive_dir, exist_ok=True)

    def one(shard: str) -> tuple:
        # local_dir puts a real file here rather than a symlink into a blob cache,
        # so deleting it actually reclaims the space. With cache_dir the blob
        # survives and the full ~34 GB of shards accumulates.
        kwargs = dict(repo_type="dataset")
        if keep_archives:
            kwargs["cache_dir"] = archive_dir
        else:
            kwargs["local_dir"] = archive_dir
        path = hf_hub_download(STRUCTURES_REPO, shard, **kwargs)
        n = _extract_shard(path, dest_dir, keep)
        if not keep_archives:
            try:
                os.remove(path)
            except OSError:
                pass
        open(_shard_marker(dest_dir, shard), "w").close()
        return shard, n

    total = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(one, s): s for s in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            shard, n = fut.result()
            total += n
    print(f"  extracted {total} structure files into {dest_dir}")

    if not keep_archives and archive_dir.startswith(tempfile.gettempdir()):
        shutil.rmtree(archive_dir, ignore_errors=True)


def download_go_embeddings(dest_dir: str) -> None:
    parent = os.path.dirname(os.path.abspath(dest_dir)) or "."
    os.makedirs(parent, exist_ok=True)

    if os.path.isdir(dest_dir) and len(os.listdir(dest_dir)) > 40000:
        print(f"go embeddings: already present in {dest_dir} "
              f"({len(os.listdir(dest_dir))} files)")
        return

    print("go embeddings: downloading …")
    with tempfile.TemporaryDirectory() as tmp:
        path = hf_hub_download(
            GO_EMBEDDINGS_REPO, "go_embeddings.tar.gz", repo_type="dataset", cache_dir=tmp
        )
        print("  extracting …")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(parent)
    n = len(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else 0
    print(f"  {n} embedding files in {dest_dir}")


def _md5(data: bytes) -> str:
    """Hex md5 of `data`."""
    return hashlib.md5(data).hexdigest()


def _kaggle_credentials() -> tuple:
    """(username, key) from KAGGLE_USERNAME / KAGGLE_KEY, else ~/.kaggle/kaggle.json.

    Raises:
        RuntimeError: when neither gives credentials.
    """
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    if os.path.isfile(KAGGLE_CONFIG):
        with open(KAGGLE_CONFIG) as handle:
            config = json.load(handle)
        if config.get("username") and config.get("key"):
            return config["username"], config["key"]
    raise RuntimeError("IA weights: no Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY, or "
                       f"save kaggle.json to {KAGGLE_CONFIG}, and accept the rules of the "
                       f"{IA_COMPETITION} competition; or pass --skip-ia and place the file yourself.")


def download_ia(dest_file: str) -> None:
    """Fetch CAFA 6's IA weights into `dest_file`, refusing any file but the published one.

    Raises:
        RuntimeError: no credentials, a failed download, or a file whose md5 is not IA_MD5.
    """
    if os.path.isfile(dest_file):
        with open(dest_file, "rb") as handle:
            if _md5(handle.read()) == IA_MD5:
                print(f"ia weights: already present in {dest_file}")
                return
    print(f"ia weights: downloading {IA_KAGGLE_FILE} from the {IA_COMPETITION} competition …")
    url = IA_DOWNLOAD_URL.format(competition=IA_COMPETITION, file=IA_KAGGLE_FILE)
    response = requests.get(url, auth=_kaggle_credentials(), timeout=IA_TIMEOUT_S)
    if response.status_code != 200:
        raise RuntimeError(f"IA weights: HTTP {response.status_code} from Kaggle; check the credentials "
                           f"and that the {IA_COMPETITION} rules are accepted")
    data = response.content
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            data = archive.read(IA_KAGGLE_FILE)
    if _md5(data) != IA_MD5:
        raise RuntimeError(f"IA weights: md5 {_md5(data)} is not the published {IA_MD5}; refusing to "
                           "write weights that would change every weighted number")
    os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
    with open(dest_file, "wb") as handle:
        handle.write(data)
    print(f"  {dest_file} (md5 ok)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(structures_dir: str, go_dir: str, ia_file: str = _paths.IA_CAFA6,
           check_ia: bool = True) -> int:
    """Check that the downloaded assets actually cover the corpus.

    Args:
        structures_dir: the structures mirror.
        go_dir: the GO embeddings.
        ia_file: the CAFA 6 IA weights.
        check_ia: False to skip the IA check (--skip-ia).

    Returns:
        The number of problems found.
    """
    problems = 0

    if not check_ia:
        print(f"[skip] ia weights: {ia_file} not checked (--skip-ia)")
    elif os.path.isfile(ia_file):
        with open(ia_file, "rb") as handle:
            ok = _md5(handle.read()) == IA_MD5
        print(f"[{'ok ' if ok else 'FAIL'}] ia weights: {ia_file} "
              f"{'is' if ok else 'is NOT'} the published CAFA 6 file")
        problems += 0 if ok else 1
    else:
        print(f"[FAIL] ia weights: {ia_file} does not exist (scoring and GO-GPT validation need it)")
        problems += 1

    if os.path.isdir(go_dir):
        n = len([f for f in os.listdir(go_dir) if f.endswith(".safetensors")])
        ok = n >= 43000
        print(f"[{'ok ' if ok else 'FAIL'}] go embeddings: {n} .safetensors "
              f"(expected ~43,248) in {go_dir}")
        problems += 0 if ok else 1
    else:
        print(f"[FAIL] go embeddings: {go_dir} does not exist")
        problems += 1

    if not os.path.isdir(structures_dir):
        print(f"[skip] structures: {structures_dir} does not exist "
              f"(prediction and evaluation run sequence-only)")
        return problems

    try:
        from datasets import load_dataset
    except ImportError:
        print("[skip] structures coverage: `datasets` not installed")
        return problems

    for config, split in CORPUS_SPLITS:
        try:
            ds = load_dataset(CORPUS_REPO, config, split=split, columns=[STRUCTURE_COLUMN])
        except Exception as exc:  # offline: the local files were still checked above
            print(f"[skip] structures coverage: {CORPUS_REPO} is not reachable ({type(exc).__name__})")
            return problems
        want = [p for p in ds[STRUCTURE_COLUMN] if p]
        # load.py joins STRUCTURE_DIR with the BARE FILENAME, so check that way.
        have = sum(1 for p in want
                   if os.path.exists(os.path.join(structures_dir, os.path.basename(p))))
        pct = 100.0 * have / len(want) if want else 0.0
        ok = pct >= 99.0
        print(f"[{'ok ' if ok else 'FAIL'}] structures for {CORPUS_REPO}:{config}/{split}: "
              f"{have}/{len(want)} resolve ({pct:.1f}%)")
        if not ok:
            print("        NOTE: unresolved structures are silently ignored at "
                  "training time (empty coordinates), so this must be ~100%.")
            problems += 1

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download BioReason-Pro 2's GO embeddings, structures and CAFA 6 IA weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dest", default=_paths.ASSETS_DIR,
                    help="Parent directory; creates <dest>/structures and <dest>/go_embeddings")
    ap.add_argument("--structures-dir", default=None, help="Override structures destination")
    ap.add_argument("--go-embeddings-dir", default=None, help="Override GO embeddings destination")
    ap.add_argument("--skip-structures", action="store_true",
                    help="No structures: prediction and evaluation run sequence-only; training needs "
                         "BRP_ALLOW_PARTIAL_STRUCTURES=1")
    ap.add_argument("--skip-go-embeddings", action="store_true")
    ap.add_argument("--ia-file", default=_paths.IA_CAFA6,
                    help="Where the CAFA 6 IA weights go (BRP_IA_FILE moves the default)")
    ap.add_argument("--skip-ia", action="store_true",
                    help="Do not fetch the IA weights (they need Kaggle credentials)")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Parallel shard downloads")
    ap.add_argument("--keep-archives", action="store_true",
                    help="Keep the downloaded .tar.gz shards instead of deleting them")
    ap.add_argument("--referenced-only", action="store_true",
                    help="Extract only the ~263k structures the corpus's training and benchmark "
                         "proteins reference (~108 GB) instead of all ~370k in the shards (~150 GB, "
                         "the default)")
    ap.add_argument("--verify", action="store_true",
                    help="Only verify existing downloads against the corpus (with --skip-ia, "
                         "without the IA weights; a missing structures directory is skipped)")
    args = ap.parse_args()

    structures_dir = args.structures_dir or os.path.join(args.dest, "structures")
    go_dir = args.go_embeddings_dir or os.path.join(args.dest, "go_embeddings")

    if args.verify:
        return 1 if verify(structures_dir, go_dir, args.ia_file, not args.skip_ia) else 0

    if not args.skip_go_embeddings:
        download_go_embeddings(go_dir)
    if not args.skip_structures:
        download_structures(structures_dir, args.num_workers, args.keep_archives,
                            args.referenced_only)
    # Last, and not fatal: only scoring and GO-GPT validation need the IA weights, and they need
    # Kaggle credentials that inference does not.
    if not args.skip_ia:
        try:
            download_ia(args.ia_file)
        except (RuntimeError, requests.RequestException) as exc:
            print(f"\n{exc}")

    print("\nVerifying …")
    problems = verify(structures_dir, go_dir, args.ia_file, not args.skip_ia)
    if problems:
        print(f"\n{problems} problem(s) found.")
        return 1

    print("\nAll good. Point the training script at:")
    print(f"  STRUCTURE_DIR={os.path.abspath(structures_dir)}")
    print(f"  GO_EMBEDDINGS_PATH={os.path.abspath(go_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
