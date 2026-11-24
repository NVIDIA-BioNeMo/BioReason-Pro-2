# SkyRL for BioReason-Pro 2

BioReason-Pro 2's reinforcement learning runs on [SkyRL](https://github.com/NovaSky-AI/SkyRL) at upstream
commit `50a570c0` plus [`bioreason.patch`](bioreason.patch). [`fetch.sh`](fetch.sh) clones upstream at
that commit, applies the patch and checks the resulting tree;
[`scripts/rl/build_env.sh`](../scripts/rl/build_env.sh) runs it for you. The released RL model was
trained with a version of the patch whose `go_function` env also offered a GO-F1 reward and scored
validation on it; its reward, the judge alone, is unchanged here.

```bash
bash skyrl/fetch.sh            # -> skyrl/src
```

What the patch adds, all in `skyrl-train` and `skyrl-gym`:

- **Modalities** (`skyrl_train/modalities/`, `dataset/modalities.py`,
  `tokenization/multimodal_processor.py`): a `modalities:` config block registers an encoder and a
  trainable projection per modality. Prompts carry placeholder tokens; the generator expands them to
  each sample's span, the training forward splices the projected features into `inputs_embeds`, and
  the projection's weights are synced to vLLM, where a model registered by a vLLM plugin receives the
  raw features as `multi_modal_data` (`SKYRL_MM_MODALITY`).
- **Qwen3.5 (hybrid linear attention) under FSDP2**: a guard against multi-sequence packing on hybrid
  layers, logits for response tokens only (`SKYRL_RESPONSE_LOGITS_ONLY`), LoRA merged into the base
  weights and pushed to vLLM over CUDA IPC, replicated projection weights with their gradients
  all-reduced, and a refusal to train a checkpoint whose decoder weights did not load.
- **`go_function` env** (`skyrl-gym`): the protein-function reward, an LLM judge, from
  `bioreason2.rl`.
- **Training fixes and options**: a KL loss that carries its gradient, `onpolicy_exact_ratio` and
  `skip_old_logprob_pass`, a constant Dr. GRPO divisor, `grpo_baseline_floor`, unscored rewards filled
  with the group mean plus a stop file when too many go unscored, a generation timeout with a restart
  from the last checkpoint, exact resume of a chained run, checkpoint completeness checks, and token
  balancing across data-parallel ranks (`SKYRL_BALANCE_DP_TOKENS`).

The run-time switches are `SKYRL_*` environment variables, forwarded to the Ray workers.

SkyRL is under the Apache License 2.0 ([`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt)); the
patch is a modification of it under the same license, and the files it changes are the ones it names.
