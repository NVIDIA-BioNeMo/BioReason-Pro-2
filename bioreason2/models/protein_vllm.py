# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import glob
from typing import Optional, List, Union

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

from bioreason2.models import model_family as _mf
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from vllm import LLM, SamplingParams

from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.pl.chat_template_pl import get_chat_template
from bioreason2.models.protein_encoder import create_protein_encoder
from bioreason2.models import projection_norm
from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
from bioreason2.models.special_tokens import get_all_special_tokens, get_token



def _env_true(name: str) -> bool:
    """Read a boolean switch from the environment.

    Args:
        name: Environment variable to read.

    Returns:
        True when set to 1/true/yes, case-insensitively.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _configure_vllm_env() -> None:
    """Environment fixes vLLM needs before its first import.

    `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM's PyTorch top-k/top-p sampler instead of
    flashinfer's, the sampler every published evaluation ran with, so sampled outputs
    come from the same implementation. It also keeps vLLM from importing flashinfer for
    sampling. Costs nothing here: the flashinfer sampler is a throughput optimisation,
    not a correctness requirement.

    Set with setdefault so an explicit value from the launcher always wins.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


_configure_vllm_env()


def _disable_mrope(cfg):
    """Strip M-RoPE from a Qwen3.5 config so vLLM falls back to 1D RoPE.

    Required because prompt_embeds and prompt_token_ids are mutually exclusive in
    vLLM, and M-RoPE position computation needs the token ids. Safe for text-only
    input: with no image/video sections all three M-RoPE dimensions get identical
    positions, making it equivalent to standard 1D RoPE.
    """
    text_cfg = getattr(cfg, "text_config", cfg)
    rope = getattr(text_cfg, "rope_parameters", None)
    if rope and isinstance(rope, dict):
        rope.pop("mrope_section", None)
        rope.pop("mrope_interleaved", None)
        return cfg
    # getattr means a config without rope_parameters silently keeps M-RoPE; say so.
    print(f"⚠️ _disable_mrope: no rope_parameters on {type(text_cfg).__name__}, M-RoPE NOT stripped")
    return cfg


class ProteinLLMModel(nn.Module):
    """
    A combined model that processes both protein sequences and text inputs.

    The model uses a protein encoder (ESM3) to extract features from protein sequences
    and a text model (LLM) to process text inputs and generate responses. The protein features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        protein_model_name: str = "esm3_sm_open_v1",
        cache_dir: Optional[str] = None,
        max_length_protein: int = 2048,
        max_length_text: int = 4096,
        text_model_finetune: bool = False,
        protein_embedding_layer: int = -1,
        go_model_finetune: bool = False,
        attn_implementation: str = "flash_attention_2",
        go_obo_path: Optional[str] = None,
        precomputed_embeddings_path: Optional[str] = None,
        go_hidden_dim: int = 512,
        go_num_gat_layers: int = 3,
        go_num_heads: int = 8,
        go_num_reduced_embeddings: int = 200,
        go_embedding_dim: int = 2560,
        protein_projection_layernorm: bool = False,
        protein_projection_norm: str = "",
        protein_projection_norm_target: float = 0.0,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 32768,
        max_num_seqs: int = 256,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize the ProteinLLMModel for vLLM inference.

        Args:
            ckpt_dir: Directory or identifier of the pre-trained text model.
            protein_model_name: Name of the protein model to be used (ESM3).
            cache_dir: Directory to cache the models.
            max_length_protein: Maximum length of protein sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 4096.
            text_model_finetune: Whether to finetune the text model. Defaults to False.
            protein_embedding_layer: ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models.
            go_model_finetune: Whether to finetune the GO graph encoder. Defaults to False.
            attn_implementation: Attention implementation to use. Defaults to "flash_attention_2".
            go_obo_path: Path to GO ontology OBO file. If None, GO encoder will be disabled.
            precomputed_embeddings_path: Directory with GO embeddings .safetensors files.
            go_hidden_dim: Hidden dimension for GO GAT layers. Defaults to 512.
            go_num_gat_layers: Number of GAT layers in GO encoder. Defaults to 3.
            go_num_heads: Number of attention heads in GO GAT. Defaults to 8.
            go_num_reduced_embeddings: Number of reduced GO embeddings. Defaults to 200.
            go_embedding_dim: GO embedding dimension. Defaults to 2560.
            protein_projection_layernorm: Append a LayerNorm to the protein projection. Defaults to False.
            protein_projection_norm: Normalisation variant for the projection tail, one of
                `bioreason2.models.projection_norm.VARIANTS`. Empty derives it from
                `protein_projection_layernorm`. Defaults to "".
            protein_projection_norm_target: Initial output RMS for the variants that take
                one; 0 uses the variant's default. Defaults to 0.0.
            gpu_memory_utilization: GPU memory utilization for vLLM. Defaults to 0.4.
            max_model_len: Maximum length of the model. Defaults to 32768.
            tensor_parallel_size: GPUs to shard the weights over. 1 fits models up
                to ~9B on an 80 GB card; a 27B is ~55 GB of bf16 weights and needs
                >=2, since gpu_memory_utilization is vLLM's share only and ESM3,
                the GO encoder and the prompt embeddings allocate outside it.
            max_num_seqs: Maximum number of sequences to process concurrently in vLLM. Defaults to 256.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.protein_embedding_layer = protein_embedding_layer
        self.go_model_finetune = go_model_finetune
        self.max_length_protein = max_length_protein
        self.max_length_text = max_length_text
        self.max_model_len = max_model_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load the text model and tokenizer
        self.model_family = _mf.detect_family(ckpt_dir)
        _llm_kwargs = dict(
            model=ckpt_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prompt_embeds=True,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            max_num_seqs=max_num_seqs,
            dtype=self.dtype,
            tensor_parallel_size=tensor_parallel_size,
            # Qwen3.5 is multimodal by default. Two overrides are required:
            #  * language_model_only skips loading the vision encoder we never feed.
            #  * M-RoPE must be stripped. Qwen3.5 advertises mrope_section in
            #    rope_parameters, and M-RoPE derives positions from prompt_token_ids
            #    -- which vLLM treats as mutually exclusive with prompt_embeds. Since
            #    this model MUST use prompt_embeds (it substitutes protein/GO vectors),
            #    M-RoPE has to go. For text-only input M-RoPE is functionally identical
            #    to 1D RoPE: all three sections receive the same positions.
            language_model_only=True,
            hf_overrides=_disable_mrope,
        )
        print("vLLM: Qwen3.5 -- language_model_only=True, M-RoPE disabled")

        self.text_model = LLM(**_llm_kwargs)

        self.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.text_config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.text_tokenizer.chat_template = get_chat_template(ckpt_dir)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add special tokens from centralized module
        all_special_tokens = get_all_special_tokens()
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})

        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("protein_pad"))
        self.go_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("go_graph_pad"))

        # # Resize text model embeddings to accommodate new tokens
        # self.text_model.resize_token_embeddings(len(self.text_tokenizer))

        # Load the protein encoder (ESM3) -- unless a precomputed
        # embedding cache is supplied, in which case ESM3 is never called and
        # loading it would only cost ~9 s and ~2.6 GiB of GPU that vLLM can use.
        self.protein_embed_cache = os.environ.get("PROTEIN_EMBED_CACHE", "") or None
        self.protein_embed_cache_write = os.environ.get("PROTEIN_EMBED_CACHE_WRITE", "") or None
        self._preloaded = None
        self._cache_write_warned = False

        if self.protein_embed_cache and self.protein_embed_cache_write:
            raise ValueError(
                "PROTEIN_EMBED_CACHE and PROTEIN_EMBED_CACHE_WRITE are mutually exclusive: "
                "reading a cache means ESM3 is never run, so there is nothing to write."
            )

        if self.protein_embed_cache_write:
            os.makedirs(self.protein_embed_cache_write, exist_ok=True)
            print(f"🗃️  Writing protein embeddings to {self.protein_embed_cache_write}")

        if self.protein_embed_cache:
            if not os.path.isdir(self.protein_embed_cache):
                raise FileNotFoundError(
                    f"PROTEIN_EMBED_CACHE={self.protein_embed_cache} is not a directory"
                )
            print(f"🗃️  Protein embedding cache: {self.protein_embed_cache} (ESM3 NOT loaded)")
            self.protein_encoder = None
            self.protein_model = None
            self._cache_hits = 0
            self._cache_misses = 0
            self._preloaded = self._preload_cache() if _env_true("PROTEIN_EMBED_CACHE_PRELOAD") else None
        else:
            self.protein_encoder = create_protein_encoder(protein_model_name, embedding_layer=protein_embedding_layer)
            self.protein_model = self.protein_encoder.model
            self.protein_model = self.protein_model.to(self.device, dtype=self.dtype)

        # Get embedding dimensions
        # Qwen3.5 nests the language dims under text_config; the top-level
        # hidden_size there is the vision tower's width.
        self.text_hidden_size = _mf.text_hidden_size(self.text_config)
        # 1536 for esm3_sm_open_v1; taken from the projection when ESM3 is absent
        # so the cached path does not need the encoder just to read a constant.
        self.protein_hidden_size = (
            self.protein_encoder.embedding_dim if self.protein_encoder is not None
            else self.protein_projection[0].in_features
            if getattr(self, "protein_projection", None) is not None else 1536
        )

        # Initialize GO graph encoder if paths are provided
        self.go_encoder = None
        self.go_embeddings_cache = {}  # Cache for GO embeddings when encoder is frozen

        if go_obo_path is not None and precomputed_embeddings_path is not None:
            self.go_encoder = create_go_graph_encoder_pipeline(
                go_obo_path=go_obo_path,
                precomputed_embeddings_path=precomputed_embeddings_path,
                hidden_dim=go_hidden_dim,
                num_gat_layers=go_num_gat_layers,
                num_heads=go_num_heads,
                num_reduced_embeddings=go_num_reduced_embeddings,
                embedding_dim=go_embedding_dim,
                embeddings_load_to=str(self.device),
            )

        # Always create projection layer — weights loaded from go_projection.pt later.
        # Needed even without encoder when using cached go_embedding.pt.
        self.go_projection = nn.Sequential(
            nn.Linear(go_embedding_dim, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        ).to(device=self.device, dtype=self.dtype)

        # Create projection layer to map protein embeddings to text model's embedding space
        self.protein_projection_norm = projection_norm.resolve(
            protein_projection_norm, protein_projection_layernorm)
        self.protein_projection_layernorm = protein_projection_layernorm
        self.protein_projection = projection_norm.build_protein_projection(
            self.protein_hidden_size, self.text_hidden_size,
            self.protein_projection_norm, protein_projection_norm_target,
        ).to(device=self.device, dtype=self.dtype)

        # load custom components
        self.load_custom_components(ckpt_dir)

        # Initialize all models in eval mode with frozen parameters by default
        # Training setup will be handled by train_protein_llm.py
        self._setup_default_eval_mode()

        # Create processor for handling inputs
        self.processor = PLProcessor(tokenizer=self.text_tokenizer)

        # Initialize embedding layer on CPU first for safer multi-GPU loading
        _vocab = _mf.text_vocab_size(self.text_config)
        self._embedding_layer = nn.Embedding(_vocab, self.text_hidden_size)
        print(
            f"🧬 Initialized local embedding layer with shape: ({_vocab}, {self.text_hidden_size})"
        )

        # Load the input-embedding matrix out of the checkpoint.
        #
        # The key name is NOT stable across families. Qwen3 writes
        # `model.embed_tokens.weight`; a converted Qwen3.5 checkpoint writes
        # `language_model.model.embed_tokens.weight` (the vLLM-compatible layout the
        # converter produces). Try every known spelling, and scan the shard as a last
        # resort rather than assuming one.
        #
        # This block used to catch its own failure, print a warning, and CONTINUE WITH
        # RANDOM EMBEDDINGS. That produced a complete 8,630-protein evaluation of pure
        # noise -- degenerate repetition, zero GO terms -- with the only clue buried in
        # a per-chunk log. Never degrade silently here: a wrong embedding matrix makes
        # every downstream number meaningless, so it must be fatal.
        _CANDIDATES = (
            "model.embed_tokens.weight",                  # Qwen3 / plain causal LMs
            "language_model.model.embed_tokens.weight",   # converted Qwen3.5
            "model.language_model.embed_tokens.weight",   # HF-native Qwen3.5
            "embed_tokens.weight",
        )
        shard_files = [os.path.join(ckpt_dir, "model.safetensors")]
        shard_files = [f for f in shard_files if os.path.exists(f)]
        shard_files += sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors found in {ckpt_dir}")

        embed_weights = None
        found_key = None
        seen_keys = []
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                seen_keys = keys if not seen_keys else seen_keys
                for cand in _CANDIDATES:
                    if cand in keys:
                        embed_weights = f.get_tensor(cand)
                        found_key = f"{cand}  ({os.path.basename(shard_file)})"
                        break
                if embed_weights is None:
                    # Last resort: any key that looks like the input embedding.
                    for k in keys:
                        if k.endswith("embed_tokens.weight"):
                            embed_weights = f.get_tensor(k)
                            found_key = f"{k}  ({os.path.basename(shard_file)})"
                            break
            if embed_weights is not None:
                break

        if embed_weights is None:
            sample = [k for k in seen_keys if "embed" in k][:5] or seen_keys[:5]
            raise RuntimeError(
                f"Could not find the input-embedding tensor in {ckpt_dir}. "
                f"Tried {_CANDIDATES} and any *embed_tokens.weight. "
                f"Keys resembling embeddings: {sample}. Refusing to run with random "
                f"embeddings -- every generation would be noise."
            )

        if tuple(embed_weights.shape) != tuple(self._embedding_layer.weight.shape):
            raise RuntimeError(
                f"Embedding shape mismatch: checkpoint {tuple(embed_weights.shape)} vs "
                f"model {tuple(self._embedding_layer.weight.shape)}."
            )
        self._embedding_layer.weight.data = embed_weights
        print(f"✅ Loaded embedding weights from {found_key}")

        # Move the embedding layer to the correct device and dtype
        self._embedding_layer = self._embedding_layer.to(self.device, dtype=self.dtype)
        print(f"🧬 Moved embedding layer to device: {self.device}")

    def load_custom_components(
        self,
        llm_dir: str,
    ) -> None:

        # ===== Protein projection =====
        projection_path = os.path.join(llm_dir, "protein_projection.pt")
        if os.path.exists(projection_path):
            state = torch.load(projection_path, map_location=self.device)
            self.protein_projection.load_state_dict(state, strict=True)
            self.protein_projection = self.protein_projection.to(device=self.device, dtype=self.dtype)
            print(f"✅ Loaded protein projection from {projection_path} (converted to bfloat16)")
        else:
            raise FileNotFoundError(f"Protein projection not found at {projection_path}")

        # ESM3 is not read from the checkpoint: it stays frozen and comes from the `esm` package.

        # ===== GO encoder + projection =====
        if self.go_encoder is not None:
            # Load GO encoder weights
            go_encoder_path = os.path.join(llm_dir, "go_encoder.pt")
            if os.path.exists(go_encoder_path):
                go_state = torch.load(go_encoder_path, map_location=self.device)
                self.go_encoder.load_state_dict(go_state, strict=True)
                self.go_encoder = self.go_encoder.to(device=self.device, dtype=self.dtype)
                print(f"✅ Loaded GO encoder from {go_encoder_path} (converted to bfloat16)")
            else:
                # A warning here let eval score randomly-initialised GO weights as a result.
                raise FileNotFoundError(f"GO encoder weights not found at {go_encoder_path}")

        if self.go_projection is not None:
            # Load GO projection weights
            go_proj_path = os.path.join(llm_dir, "go_projection.pt")
            if os.path.exists(go_proj_path):
                state = torch.load(go_proj_path, map_location=self.device)
                self.go_projection.load_state_dict(state, strict=True)
                self.go_projection = self.go_projection.to(device=self.device, dtype=self.dtype)
                print(f"✅ Loaded GO projection from {go_proj_path} (converted to bfloat16)")
            else:
                raise FileNotFoundError(f"GO projection weights not found at {go_proj_path}")

            # Load pre-computed GO embeddings if available (skips GO encoder forward pass)
            go_embedding_path = os.path.join(llm_dir, "go_embedding.pt")
            if os.path.exists(go_embedding_path):
                cached_embedding = torch.load(go_embedding_path, map_location=self.device)
                cached_embedding = cached_embedding.to(device=self.device, dtype=self.dtype)
                self.go_embeddings_cache["all"] = cached_embedding
                print(f"✅ Loaded pre-computed GO embedding from {go_embedding_path} (will skip GO encoder)")
            else:
                print(f"ℹ️ No pre-computed GO embedding found at {go_embedding_path}, will use GO encoder")

    def _setup_default_eval_mode(self):
        """
        Set all model components to eval mode with frozen parameters by default.
        Training setup will be handled by train_protein_llm.py.
        Note: Text model parameter freezing is skipped since vLLM handles it.
        """
        # vLLM text model: Skip parameter freezing (vLLM handles this)
        # The vLLM LLM class manages its own parameters
        
        # Protein encoder: frozen, never trained
        if self.protein_encoder is not None:
            self.protein_encoder.freeze()
        
        # Protein projection: eval mode, frozen
        self.protein_projection.eval()
        for param in self.protein_projection.parameters():
            param.requires_grad = False
            
        # GO encoder: eval mode, frozen
        if self.go_encoder is not None:
            self.go_encoder.eval()
            for param in self.go_encoder.parameters():
                param.requires_grad = False
        
        # GO projection: eval mode, frozen
        if self.go_projection is not None:
            self.go_projection.eval()
            for param in self.go_projection.parameters():
                param.requires_grad = False

    def process_protein_embeddings(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
        protein_ids: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """
        Process protein sequences and structures to obtain embeddings using the protein encoder.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch
            structure_coords: Optional tensor containing the structure coordinates

        Returns:
            List of tensor embeddings for each batch item
        """
        if self.protein_embed_cache:
            batch_protein_embeddings = self._embeddings_from_cache(
                protein_ids=protein_ids,
                protein_sequences=protein_sequences,
                batch_idx_map=batch_idx_map,
                batch_size=batch_size,
            )
        else:
            batch_protein_embeddings = self.protein_encoder.encode_sequences(
                protein_sequences=protein_sequences,
                batch_idx_map=batch_idx_map,
                batch_size=batch_size,
                structure_coords=structure_coords,
            )
            if self.protein_embed_cache_write:
                try:
                    self._write_cache(protein_ids, protein_sequences, batch_idx_map,
                                      batch_protein_embeddings)
                except Exception as exc:
                    if not self._cache_write_warned:
                        self._cache_write_warned = True
                        print(f"⚠️  protein embedding cache write failed: {exc}")

        # Project all embeddings to text embedding space
        for i in range(batch_size):
            if batch_protein_embeddings[i].numel() > 0:  # Check if tensor is not empty
                batch_protein_embeddings[i] = batch_protein_embeddings[i].to(
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )
                batch_protein_embeddings[i] = self.protein_projection(batch_protein_embeddings[i])
            else:
                # Ensure empty tensors have correct dimensions
                batch_protein_embeddings[i] = torch.zeros(
                    (0, self.text_hidden_size),
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )

        return batch_protein_embeddings

    def _write_cache(
        self,
        protein_ids: Optional[List[str]],
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_embeddings: List[torch.Tensor],
    ) -> None:
        """Persist what ESM3 just returned, one safetensors file per protein_id.

        Args:
            protein_ids: One id per sequence, in the same order as protein_sequences.
            protein_sequences: The sequences that were encoded.
            batch_idx_map: Batch item each sequence belongs to.
            batch_embeddings: Per-batch-item embeddings returned by the encoder.

        Returns:
            None. Files already present are left alone, so this is resumable.
        """
        protein_ids = protein_ids or []

        if len(protein_ids) != len(protein_sequences):
            return

        counts = {}
        for item in batch_idx_map:
            counts[item] = counts.get(item, 0) + 1

        for idx, (pid, seq) in enumerate(zip(protein_ids, protein_sequences)):
            item = batch_idx_map[idx]

            # batch_embeddings[item] concatenates every protein mapped to that item,
            # so only a single-protein item can be split back apart by id.
            if counts[item] != 1 or not pid:
                continue

            emb = batch_embeddings[item]
            if emb.numel() == 0 or emb.shape[0] != len(seq) + 2:
                continue

            path = os.path.join(self.protein_embed_cache_write, f"{pid}.safetensors")
            if os.path.exists(path):
                continue

            # Store the encoder's own dtype. Downcasting to bf16 saves half the disk
            # and loses byte-identity: the opening tokens still match, then greedy
            # decoding amplifies the rounding and the traces diverge.
            tmp = f"{path}.{os.getpid()}.tmp"
            save_file({"emb": emb.detach().cpu().contiguous()}, tmp)
            os.replace(tmp, path)

    def _preload_cache(self) -> dict:
        """Read every cached embedding into RAM, so a batch costs no NFS round trips.

        Returns:
            Mapping of protein_id to its [L+2, 1536] bf16 embedding.
        """
        cached = {}

        for name in os.listdir(self.protein_embed_cache):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(self.protein_embed_cache, name)
            cached[name[: -len(".safetensors")]] = load_file(path)["emb"]

        megabytes = sum(v.numel() * v.element_size() for v in cached.values()) / 1e6
        dtype = next(iter(cached.values())).dtype if cached else "empty"
        print(f"🗃️  Preloaded {len(cached):,} protein embeddings into RAM "
              f"({megabytes:,.0f} MB, {dtype})")

        return cached

    def _embeddings_from_cache(
        self,
        protein_ids: Optional[List[str]],
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """Per-residue ESM3 layer-37 embeddings read from cache instead of computed.

        The cache stores exactly what the encoder returns: [L+2, 1536] bf16, BOS/EOS
        included. A miss or a length mismatch is FATAL rather than a silent fallback --
        the model hard-fails when protein token count differs from embedding count, and
        a quietly wrong embedding is the failure mode that already produced one full
        evaluation of noise.

        Args:
            protein_ids: One id per sequence, in the same order as protein_sequences.
            protein_sequences: The sequences whose embeddings are wanted.
            batch_idx_map: Batch item each sequence belongs to.
            batch_size: Number of items in the batch.

        Returns:
            One tensor per batch item, concatenated over that item's proteins.
        """
        # Zero sequences is legitimate: the `sequence` ablation removes the protein
        # slot entirely, so there is nothing to look up. Only a LENGTH MISMATCH is
        # an error -- that means ids were lost somewhere in the plumbing.
        protein_ids = protein_ids or []
        if len(protein_ids) != len(protein_sequences):
            raise ValueError(
                f"protein_embed_cache needs one protein_id per sequence; got "
                f"{len(protein_ids)} ids for {len(protein_sequences)} sequences"
            )
        if not protein_sequences:
            return [torch.zeros((0, self.protein_hidden_size)) for _ in range(batch_size)]
        out = [[] for _ in range(batch_size)]
        for idx, (pid, seq) in enumerate(zip(protein_ids, protein_sequences)):
            emb = self._preloaded.get(pid) if self._preloaded is not None else None

            if emb is None:
                f = os.path.join(self.protein_embed_cache, f"{pid}.safetensors")
                if not os.path.exists(f):
                    self._cache_misses += 1
                    raise FileNotFoundError(
                        f"No cached embedding for {pid} at {f}. Build it with "
                        f"PROTEIN_EMBED_CACHE_WRITE over this split, or drop "
                        f"PROTEIN_EMBED_CACHE to use ESM3."
                    )
                emb = load_file(f)["emb"]

            if emb.shape[0] != len(seq) + 2:
                raise ValueError(
                    f"Cached embedding for {pid} has {emb.shape[0]} rows for a "
                    f"{len(seq)}-residue sequence (expected {len(seq) + 2} incl. BOS/EOS). "
                    f"The cache was built for a different truncation length."
                )
            self._cache_hits += 1
            out[batch_idx_map[idx]].append(emb)
        return [
            torch.cat(v, dim=0) if v else torch.zeros((0, self.protein_hidden_size))
            for v in out
        ]

    def process_go_aspects(
        self,
        go_aspects: Optional[List[str]] = None,
        batch_size: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """
        GO embeddings for a batch: the whole ontology, encoded once and shared by every example.

        Args:
            go_aspects: One entry per batch item; None means the batch has no GO tokens.
            batch_size: Number of items in the batch

        Returns:
            Optional list of tensors with GO embeddings, one per batch item, each of shape
            (200, text_hidden_size). Returns None if no GO encoder is available.
        """
        if go_aspects is None:
            return None
        if self.go_encoder is None and not self.go_embeddings_cache:
            return None

        batch_go_embeddings = []

        aspect_key = "all"

        # Check cache if not finetuning (cache contains encoder output)
        if not self.go_model_finetune and aspect_key in self.go_embeddings_cache:
            reduced_embeddings = self.go_embeddings_cache[aspect_key]
        else:
            reduced_embeddings = self.go_encoder()  # (200, 2560)
            # Cache encoder output if not finetuning
            if not self.go_model_finetune:
                self.go_embeddings_cache[aspect_key] = reduced_embeddings

        # Duplicate for all batch items
        for i in range(batch_size):
            batch_go_embeddings.append(reduced_embeddings)

        # Project all embeddings to text embedding space
        if self.go_projection is not None:
            for i in range(len(batch_go_embeddings)):
                batch_go_embeddings[i] = batch_go_embeddings[i].to(
                    device=self.go_projection[0].weight.device,
                    dtype=self.go_projection[0].weight.dtype,
                )
                batch_go_embeddings[i] = self.go_projection(batch_go_embeddings[i])  # (200, text_hidden_size)

        return batch_go_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            labels: Labels for supervised fine-tuning
            go_aspects: GO aspects for protein sequences
            **kwargs: Additional arguments

        Returns:
            Outputs from the text model
        """
        raise RuntimeError("HF forward is not supported with vLLM. Use .generate() with prompt_embeds.")

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        protein_ids: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on protein and text inputs.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            go_aspects: GO aspects for protein sequences
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        # Ensure input_ids is on the same device as the embedding layer
        input_ids = input_ids.to(self.device)

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self._embedding_layer(input_ids)

        # Process GO aspects if provided
        go_embeddings = None
        if go_aspects is not None:
            go_embeddings = self.process_go_aspects(go_aspects, batch_size)

        # Process protein sequences if provided
        if protein_sequences is not None and batch_idx_map is not None:
            batch_protein_embeds = self.process_protein_embeddings(
                protein_sequences,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords,
                protein_ids=protein_ids,
            )

            # Find positions where protein tokens should be replaced
            mask = input_ids == self.protein_token_id

            # Count tokens and embeddings for validation
            n_protein_tokens = mask.sum().item()
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Ensure protein embeddings have the same dtype as text embeddings
            protein_embeds_flat = protein_embeds_flat.to(
                dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device
            )

            # Replace protein tokens with actual protein embeddings
            text_inputs_embeds[mask] = protein_embeds_flat

        # Process GO embeddings if provided
        if go_embeddings is not None:
            # Find positions where GO tokens should be replaced
            go_mask = input_ids == self.go_token_id

            # Count tokens and embeddings for validation
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: " f"embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                # Ensure GO embeddings have the same dtype and device as text embeddings
                go_embeds_flat = go_embeds_flat.to(
                    dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device
                )

                # Replace GO tokens with actual GO embeddings
                text_inputs_embeds[go_mask] = go_embeds_flat

        # vLLM's penalty path needs prompt_token_ids, which prompt_embeds lacks; it then
        # allocates until the host dies (2,010 GB MaxRSS measured).
        _rep = generation_kwargs.get("repetition_penalty", 1.0)
        if _rep != 1.0:
            raise ValueError(f"repetition_penalty={_rep} is unusable with prompt_embeds; use 1.0")

        # Generation with embeddings using vLLM
        sampling_params = SamplingParams(
            temperature=generation_kwargs.get("temperature", 0),
            top_p=generation_kwargs.get("top_p", 0.95),
            max_tokens=generation_kwargs.get("max_new_tokens", 1000),
            stop=generation_kwargs.get("stop", ["<|im_end|>"]),
            repetition_penalty=generation_kwargs.get("repetition_penalty", 1.0),
        )

        # Build requests for vLLM generation.
        #
        # Strip padding first. The processor right-pads the batch and pad_token is
        # eos_token, so a shorter sequence ends in trailing end-of-turn embeddings.
        # vLLM gets prompt_embeds with no attention mask, so it reads those as
        # "generation already finished" and returns an empty string. With batch_size
        # > 1 and unequal protein lengths this silently blanks part of every batch.
        pad_token_id = self.text_tokenizer.pad_token_id
        requests = []
        for i in range(batch_size):
            embeds = text_inputs_embeds[i]
            if pad_token_id is not None:
                non_pad = (input_ids[i] != pad_token_id).nonzero()
                if non_pad.numel() > 0:
                    embeds = embeds[non_pad[0].item(): non_pad[-1].item() + 1]
            requests.append({"prompt_embeds": embeds})

        # Generate for the entire batch in one call
        vllm_outputs = self.text_model.generate(requests, sampling_params=sampling_params)

        # Extract the generated text from the output objects
        return [output.outputs[0].text for output in vllm_outputs]
