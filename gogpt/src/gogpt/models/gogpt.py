# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""GO-GPT decoder.

A protein is a prefix of per-residue embeddings (from a frozen protein language model);
GO terms for one aspect are generated after it, one token at a time. Every block keeps
two streams with separate weights:

* protein stream: bidirectional self-attention over protein positions only,
* GO stream: attends to every protein position plus the previous GO tokens (causal).

Because protein queries never see GO keys, the protein stream does not depend on the GO
tokens. The model therefore runs it once (`encode_protein_prefix`) and then advances the
GO stream over the cached per-layer protein keys/values, in training and in generation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def make_norm(config):
    if config.use_rmsnorm:
        return nn.RMSNorm(config.n_embd, eps=1e-5)
    return LayerNorm(config.n_embd, bias=config.bias)


class ProteinProjection(nn.Module):
    """2-layer MLP from protein embedding dim to the model dim."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = (config.protein_embedding_dim + config.n_embd) // 2
        self.projection = nn.Sequential(
            nn.Linear(config.protein_embedding_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.projection(x)


class PrefixCausalAttention(nn.Module):
    """Separate Q/K/V and output projections for the protein and GO streams."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.go_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.protein_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.go_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.protein_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Gated attention: output * sigmoid(W x), x being the attention input.
        self.use_gated_attention = config.use_gated_attention
        if self.use_gated_attention:
            self.go_gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.protein_gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # QK-norm. k_norm is shared by both streams: their keys meet in one softmax, so
        # stream-specific key scales would skew cross- against self-attention.
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            head_dim = config.n_embd // config.n_head
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-5)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-5)

        self.resid_dropout = nn.Dropout(config.dropout)

    def _heads(self, x, qkv):
        B, T, C = x.size()
        q, k, v = qkv(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def protein_forward(self, protein_states, protein_mask=None):
        """Protein self-attention. Returns (k, v, output); k/v are what GO tokens attend to."""
        B, T, C = protein_states.size()
        q, k, v = self._heads(protein_states, self.protein_qkv)
        attn_mask = None
        if protein_mask is not None:
            attn_mask = protein_mask.to(torch.bool).view(B, 1, 1, T)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.use_gated_attention:
            y = y * torch.sigmoid(self.protein_gate_proj(protein_states))
        return k, v, self.resid_dropout(self.protein_proj(y))

    def go_forward(self, go_states, protein_k, protein_v, protein_mask=None, go_mask=None):
        """GO attention over all protein positions plus causal attention over GO tokens."""
        B, T_go, C = go_states.size()
        T_protein = protein_k.size(2)
        q, k, v = self._heads(go_states, self.go_qkv)
        if go_mask is not None:
            # Padded GO positions contribute nothing as queries, keys or values.
            valid = go_mask.view(B, 1, T_go, 1).to(q.dtype)
            q, k, v = q * valid, k * valid, v * valid

        k = torch.cat([protein_k.to(k.dtype), k], dim=2)
        v = torch.cat([protein_v.to(v.dtype), v], dim=2)

        mask = torch.ones((B, 1, T_go, T_protein + T_go), dtype=torch.bool, device=q.device)
        if protein_mask is not None:
            mask[:, :, :, :T_protein] = protein_mask.to(torch.bool).view(B, 1, 1, T_protein)
        causal = torch.tril(torch.ones(T_go, T_go, dtype=torch.bool, device=q.device))
        mask[:, :, :, T_protein:] = causal.view(1, 1, T_go, T_go)
        if go_mask is not None:
            mask[:, :, :, T_protein:] &= go_mask.to(torch.bool).view(B, 1, 1, T_go)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        if go_mask is not None:
            y = y * go_mask.view(B, 1, T_go, 1).to(y.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T_go, C)
        if self.use_gated_attention:
            y = y * torch.sigmoid(self.go_gate_proj(go_states))
        if go_mask is not None:
            y = y * go_mask.unsqueeze(-1).to(y.dtype)
        return self.resid_dropout(self.go_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.fc(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1_protein = make_norm(config)
        self.ln_1_go = make_norm(config)
        self.attn = PrefixCausalAttention(config)
        self.ln_2_protein = make_norm(config)
        self.ln_2_go = make_norm(config)
        self.mlp_protein = MLP(config)
        self.mlp_go = MLP(config)

    def protein_forward(self, protein_states, protein_mask):
        k, v, attn = self.attn.protein_forward(self.ln_1_protein(protein_states), protein_mask)
        protein_states = protein_states + attn
        protein_states = protein_states + self.mlp_protein(self.ln_2_protein(protein_states))
        return protein_states, k, v

    def go_forward(self, go_states, protein_k, protein_v, protein_mask, go_mask):
        go_states = go_states + self.attn.go_forward(
            self.ln_1_go(go_states), protein_k, protein_v, protein_mask, go_mask)
        return go_states + self.mlp_go(self.ln_2_go(go_states))


class GOGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.protein_projection = ProteinProjection(config)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),  # GO positions only
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # ln_f_protein is never used: only the GO stream is decoded. It is kept so
            # released checkpoints load with strict=True.
            ln_f_protein=make_norm(config),
            ln_f_go=make_norm(config),
        ))
        self.organism_embedding = nn.Embedding(config.organism_vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # Depth-scaled init for the residual-branch output projections.
        for name, p in self.named_parameters():
            if name.endswith(("attn.go_proj.weight", "attn.protein_proj.weight", "fc.2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.end_token_ids = (config.mf_end_token_id, config.bp_end_token_id, config.cc_end_token_id)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
        return n

    # ------------------------------------------------------------------ protein stream

    def encode_protein_prefix(self, protein_embeddings, protein_mask=None):
        """Run the protein stream through every block. Returns per-layer (k, v)."""
        x = protein_embeddings.to(self.protein_projection.projection[0].weight.dtype)
        if protein_mask is not None:
            # The projection carries biases, so padded rows must be zero going in.
            x = x * protein_mask.to(x.dtype).unsqueeze(-1)
        protein_states = self.protein_projection(x)
        prefix_kv = []
        for block in self.transformer.h:
            protein_states, k, v = block.protein_forward(protein_states, protein_mask)
            prefix_kv.append((k, v))
        return prefix_kv

    # ----------------------------------------------------------------------- GO stream

    def _go_hidden(self, go_tokens, go_mask, organism_id, prefix_kv, protein_mask):
        b, t = go_tokens.size()
        total = prefix_kv[0][0].size(2) + t
        if total > self.config.block_size:
            raise ValueError(f"protein + GO length {total} exceeds block_size {self.config.block_size}")
        positions = torch.arange(t, device=go_tokens.device).unsqueeze(0).expand(b, -1)
        x = self.transformer.wte(go_tokens) + self.transformer.wpe(positions)
        if organism_id is not None:
            x = x + self.organism_embedding(organism_id).unsqueeze(1)
        x = self.transformer.drop(x)
        for block, (k, v) in zip(self.transformer.h, prefix_kv):
            x = block.go_forward(x, k, v, protein_mask, go_mask)
        return self.transformer.ln_f_go(x)

    def forward(self, go_tokens, protein_embeddings, protein_mask=None, go_mask=None,
                organism_id=None, targets=None, label_smoothing=0.0):
        """Teacher-forced forward pass.

        Returns (logits, loss). With targets, logits cover every GO position and the loss
        is cross-entropy ignoring <pad> (id 0); without, only the last position's logits.
        """
        prefix_kv = self.encode_protein_prefix(protein_embeddings, protein_mask)
        h = self._go_hidden(go_tokens, go_mask, organism_id, prefix_kv, protein_mask)
        if targets is None:
            return self.lm_head(h[:, [-1], :]), None
        logits = self.lm_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=self.config.pad_token_id,
                               label_smoothing=label_smoothing)
        return logits, loss

    # ---------------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, protein_embeddings, protein_mask, go_tokens, max_new_tokens,
                 temperature=0.0, top_k=None, organism_id=None):
        """Generate GO tokens after the given start tokens ([B, 1], one aspect START each).

        temperature 0 is greedy decoding. A sequence is finished once it emits any aspect
        END token; it is padded with <pad> after that. Returns [B, 1 + n_generated].
        """
        prefix_kv = self.encode_protein_prefix(protein_embeddings, protein_mask)
        pad = self.config.pad_token_id
        B = go_tokens.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=go_tokens.device)
        go_mask = torch.ones_like(go_tokens, dtype=torch.bool)
        end_ids = torch.tensor(self.end_token_ids, device=go_tokens.device)
        greedy = temperature is None or temperature < 1e-8

        for _ in range(max_new_tokens):
            logits = self.lm_head(
                self._go_hidden(go_tokens, go_mask, organism_id, prefix_kv, protein_mask)[:, -1, :])
            if finished.any():
                logits[finished] = float("-inf")
                logits[finished, pad] = 0.0
            if greedy:
                next_tokens = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = torch.where(finished.unsqueeze(1), logits, logits / temperature)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    filtered = logits.masked_fill(logits < kth, float("-inf"))
                    logits = torch.where(finished.unsqueeze(1), logits, filtered)
                next_tokens = torch.multinomial(F.softmax(logits.float(), dim=-1), num_samples=1)
            go_tokens = torch.cat((go_tokens, next_tokens), dim=1)
            go_mask = torch.cat((go_mask, (~finished).unsqueeze(1)), dim=1)
            finished = finished | torch.isin(next_tokens.squeeze(-1), end_ids)
            if finished.all():
                break
        return go_tokens

    @torch.no_grad()
    def generate_beam_search(self, protein_embeddings, protein_mask, go_tokens, max_new_tokens,
                             beam_size=5, length_penalty=0.6, organism_id=None):
        """Beam search; returns the best sequence per example, right-padded with <pad>.

        A finished beam is scored as sum(log p) / (length ** length_penalty), the start
        token excluded from the length.
        """
        B = go_tokens.size(0)
        device = go_tokens.device
        V = self.config.vocab_size
        pad = self.config.pad_token_id
        end_ids = torch.tensor(self.end_token_ids, device=device)

        prefix_kv = [(k.repeat_interleave(beam_size, 0), v.repeat_interleave(beam_size, 0))
                     for k, v in self.encode_protein_prefix(protein_embeddings, protein_mask)]
        protein_mask = protein_mask.repeat_interleave(beam_size, 0) if protein_mask is not None else None
        organism_id = organism_id.repeat_interleave(beam_size, 0) if organism_id is not None else None

        seqs = go_tokens.repeat_interleave(beam_size, 0)
        go_mask = torch.ones_like(seqs, dtype=torch.bool)
        scores = torch.full((B * beam_size,), float("-inf"), device=device)
        scores[0::beam_size] = 0.0  # one live beam per example at the start
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)
        best, best_scores = [None] * B, [float("-inf")] * B

        for step in range(max_new_tokens):
            logits = self.lm_head(
                self._go_hidden(seqs, go_mask, organism_id, prefix_kv, protein_mask)[:, -1, :])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_probs[finished] = float("-inf")
            log_probs[finished, pad] = 0.0

            top_scores, top_idx = torch.topk((scores.unsqueeze(1) + log_probs).view(B, beam_size * V),
                                              beam_size, dim=1)
            src = (torch.arange(B, device=device).unsqueeze(1) * beam_size + top_idx // V).view(-1)
            new_tokens = (top_idx % V).view(-1, 1)
            seqs = torch.cat([seqs[src], new_tokens], dim=1)
            scores = top_scores.view(-1)
            old_finished = finished[src]
            just_finished = torch.isin(new_tokens.squeeze(-1), end_ids) & ~old_finished
            finished = old_finished | just_finished
            go_mask = torch.cat([go_mask[src], (~finished).unsqueeze(1)], dim=1)

            length = seqs.size(1) - 1
            for idx in just_finished.nonzero().flatten().tolist():
                b = idx // beam_size
                s = scores[idx].item() / (length ** length_penalty)
                if s > best_scores[b]:
                    best_scores[b], best[b] = s, seqs[idx].clone()

            if finished.all():
                break
            if all(x is not None for x in best):
                # Stop once no live beam can beat the best finished one, even if every
                # remaining token had probability 1.
                max_len = seqs.size(1) + (max_new_tokens - step - 1)
                can_improve = False
                for b in range(B):
                    live = ~finished[b * beam_size:(b + 1) * beam_size]
                    if live.any():
                        top_live = scores[b * beam_size:(b + 1) * beam_size][live].max().item()
                        if top_live / (max_len ** length_penalty) > best_scores[b]:
                            can_improve = True
                            break
                if not can_improve:
                    break

        results = []
        for b in range(B):
            if best[b] is None:
                beam = scores[b * beam_size:(b + 1) * beam_size].argmax().item()
                best[b] = seqs[b * beam_size + beam]
            results.append(best[b])
        T = max(r.size(0) for r in results)
        return torch.stack([F.pad(r, (0, T - r.size(0)), value=pad) for r in results])
