"""SmolVLA-450M behind the common interface (CLAUDE.md §5).

Why this file instruments the forward pass instead of using forward hooks
------------------------------------------------------------------------
`SmolVLMWithExpertModel.forward` does **not** call `LlamaDecoderLayer.forward`. It
inlines the decoder-layer computation itself, calling `layer.self_attn.o_proj`,
`layer.post_attention_layernorm` and `layer.mlp` as separate modules and keeping the
residual stream in a local variable (see lerobot/policies/smolvla/smolvlm_with_expert.py,
`forward`). Two consequences, both of which silently break naive patching:

1. A `register_forward_hook` on `...layers[i]` never fires -- the module is never called.
2. The residual stream is read *twice* per layer: once as the input to `input_layernorm`
   and once as the skip connection. Hooking the norm would patch only the first read, so
   the patch would be half-applied and the resulting number would be quietly wrong.

So we re-implement that outer loop here with explicit taps. The re-implementation is
pinned to lerobot 0.6.0 and is guarded by `tests/test_patching.py::test_instrumented_
forward_matches_stock`, which asserts the instrumented path reproduces the stock forward
exactly when no patches are active. If lerobot changes, that test fails loudly.

Determinism
-----------
SmolVLA is a flow-matching policy: `sample_actions` starts from Gaussian noise and
integrates for `config.num_steps` (10). `predict_action_chunk` accepts an explicit
`noise` tensor, so we always pass one derived from a fixed seed. Without this, two
identical calls differ and patching correctness is untestable.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .base import Batch, ForwardResult, PatchValue, SiteSpec, VLAModel, resolve_patch

DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
DEFAULT_NOISE_SEED = 0

_TOWERS: tuple[str, str] = ("vlm", "expert")


class _Taps:
    """Per-forward recording/substitution state."""

    def __init__(
        self,
        capture: set[str],
        patches: Mapping[str, PatchValue],
    ) -> None:
        self.capture = capture
        self.patches = dict(patches)
        self.cache: dict[str, list[Tensor]] = {}
        self.counts: dict[str, int] = {}
        self.fired: set[str] = set()

    def tap(self, name: str, value: Tensor) -> Tensor:
        """Record and/or replace the activation at `name`. Returns what flows onward."""
        idx = self.counts.get(name, 0)
        self.counts[name] = idx + 1
        self.fired.add(name)

        if name in self.patches:
            # clone: the forward pass mutates these tensors in place (`out_emb += ...`),
            # which would otherwise silently corrupt the caller's patch tensor -- and that
            # tensor is often a cached activation being reused across many patch runs.
            value = resolve_patch(self.patches[name], value, idx).clone()
        if name in self.capture:
            # clone for the same reason, in the other direction: keep the cache immune to
            # the in-place arithmetic that happens immediately after this returns.
            self.cache.setdefault(name, []).append(value.detach().clone())
        return value


class SmolVLA(VLAModel):
    """SmolVLA-450M (`lerobot/smolvla_base`) with activation capture and patching."""

    def __init__(
        self,
        policy: Any,
        noise_seed: int = DEFAULT_NOISE_SEED,
        norm_stats: dict[str, Tensor] | None = None,
    ) -> None:
        self.policy = policy
        self.noise_seed = noise_seed
        self.config = policy.config
        self._vwe = policy.model.vlm_with_expert
        self.num_layers: int = int(self._vwe.num_vlm_layers)
        self._taps: _Taps | None = None
        self.norm_stats = norm_stats or {}

    # ------------------------------------------------------------------ load

    @classmethod
    def load(
        cls,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        noise_seed: int = DEFAULT_NOISE_SEED,
        **kwargs: Any,
    ) -> SmolVLA:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(checkpoint, **kwargs)
        policy.to(device)
        policy.eval()
        for p in policy.parameters():  # we train nothing (CLAUDE.md §6)
            p.requires_grad_(False)
        return cls(
            policy,
            noise_seed=noise_seed,
            norm_stats=load_norm_stats(checkpoint, device=device),
        )

    # --------------------------------------------------------- normalization

    @property
    def has_norm_stats(self) -> bool:
        return bool(self.norm_stats)

    def normalize_state(self, state: Tensor) -> Tensor:
        """Map raw proprioception into the policy's training distribution."""
        if "state_mean" not in self.norm_stats:
            return state
        mean = self.norm_stats["state_mean"]
        std = self.norm_stats["state_std"]
        return (state.to(mean.device) - mean) / std.clamp_min(1e-6)

    def unnormalize_action(self, action: Tensor) -> Tensor:
        """Map the policy's output back into raw units.

        Required before comparing a predicted action against a demonstration action:
        otherwise the two live in different scales and any distance between them is
        meaningless -- which would silently invalidate the directional M0 readout.
        """
        if "action_mean" not in self.norm_stats:
            return action
        mean = self.norm_stats["action_mean"]
        std = self.norm_stats["action_std"]
        return action.to(mean.device) * std + mean

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    # ----------------------------------------------------------------- sites

    def sites(self) -> list[SiteSpec]:
        """The full layer x tower x component grid.

        `expert` sites fire once per denoising step; `vlm` sites fire once, during the
        prefix pass that fills the KV cache.
        """
        out: list[SiteSpec] = []
        for tower in _TOWERS:
            for layer in range(self.num_layers):
                for comp in ("resid_pre", "attn_out", "mlp_out", "resid_post"):
                    out.append(
                        SiteSpec(
                            name=f"{tower}.L{layer}.{comp}",
                            tower=tower,  # type: ignore[arg-type]
                            component=comp,  # type: ignore[arg-type]
                            layer=layer,
                        )
                    )
            out.append(
                SiteSpec(
                    name=f"{tower}.final_norm",
                    tower=tower,  # type: ignore[arg-type]
                    component="final_norm",
                    layer=None,
                )
            )
        return out

    # ------------------------------------------------------------ inference

    def make_noise(self, batch_size: int, seed: int | None = None) -> Tensor:
        """Deterministic flow-matching noise. Same seed => same action, always."""
        g = torch.Generator(device="cpu").manual_seed(self.noise_seed if seed is None else seed)
        shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
        return torch.randn(shape, generator=g, dtype=torch.float32).to(self.device)

    def _batch_size(self, inputs: Batch) -> int:
        from lerobot.utils.constants import OBS_STATE

        return int(inputs[OBS_STATE].shape[0])

    def predict_action(
        self,
        inputs: Batch,
        noise: Tensor | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.forward_with_cache(inputs, sites=(), noise=noise, seed=seed, **kwargs).action

    def forward_with_cache(
        self,
        inputs: Batch,
        sites: Sequence[str] | None = None,
        patches: Mapping[str, PatchValue] | None = None,
        noise: Tensor | None = None,
        seed: int | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> ForwardResult:
        """Forward pass with optional capture and optional substitution.

        `strict` (default True) raises if a requested capture site or a patch site never
        fired. That is almost always a typo or a wrong assumption about which tower runs
        in which pass, and silently returning an empty cache would hide it.
        """
        want = set(sites or ())
        patches = dict(patches or {})
        known = set(self.site_names())
        unknown = (want | set(patches)) - known
        if unknown:
            raise KeyError(f"unknown site(s): {sorted(unknown)}")

        if noise is None:
            noise = self.make_noise(self._batch_size(inputs), seed=seed)

        # `predict_action_chunk` pushes observations into persistent queues, so without a
        # reset the Nth call sees state left over from call N-1. Every call here is an
        # independent offline evaluation of a fixed state, so carrying that over would be
        # a bug -- and one that would make patched/unpatched runs incomparable.
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        taps = _Taps(capture=want, patches=patches)
        with self._instrumented(taps):
            action = self.policy.predict_action_chunk(dict(inputs), noise=noise)

        if strict:
            missing = (want | set(patches)) - taps.fired
            if missing:
                raise RuntimeError(
                    f"site(s) never fired during the forward pass: {sorted(missing)}. "
                    "vlm.* sites fire only in the prefix pass; expert.* sites fire once "
                    "per denoising step."
                )
        return ForwardResult(action=action, cache=taps.cache)

    def patch(
        self,
        inputs: Batch,
        patches: Mapping[str, PatchValue],
        sites: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ForwardResult:
        return self.forward_with_cache(inputs, sites=sites, patches=patches, **kwargs)

    # ------------------------------------------------- instrumented forward

    @contextlib.contextmanager
    def _instrumented(self, taps: _Taps | None) -> Iterator[None]:
        """Swap `vlm_with_expert.forward` for the tapped re-implementation."""
        if taps is None:
            yield
            return
        original = self._vwe.forward
        self._taps = taps
        self._vwe.forward = self._tapped_forward  # type: ignore[method-assign]
        try:
            yield
        finally:
            self._vwe.forward = original  # type: ignore[method-assign]
            self._taps = None

    def _tapped_forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ) -> tuple[list[Tensor | None], Any]:
        """Mirror of SmolVLMWithExpertModel.forward (lerobot 0.6.0) with taps.

        Structure is kept deliberately close to upstream so the two can be diffed by eye,
        including the in-place residual adds (see the comment at the add site -- changing
        them breaks mixed-precision dtype propagation). The only additions are the
        `taps.tap(...)` calls. `test_instrumented_forward_matches_stock` asserts bitwise
        equality with upstream when no patch is active.
        """
        vwe = self._vwe
        taps = self._taps
        assert taps is not None

        models = [vwe.get_vlm_model().text_model, vwe.lm_expert]
        model_layers = vwe.get_model_layers(models)

        assert inputs_embeds is not None
        batch_size = 1
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = vwe.num_vlm_layers
        head_dim = vwe.vlm.config.text_config.head_dim

        for layer_idx in range(num_layers):
            every_n = vwe.self_attn_every_n_layers
            use_self_attn = (
                fill_kv_cache
                or "cross" not in vwe.attention_mode
                or (every_n > 0 and layer_idx % every_n == 0)
            )

            # resid_pre, tapped before the layer consumes it
            inputs_embeds = [
                taps.tap(f"{_TOWERS[i]}.L{layer_idx}.resid_pre", h) if h is not None else None
                for i, h in enumerate(inputs_embeds)
            ]

            fwd = vwe.forward_attn_layer if use_self_attn else vwe.forward_cross_attn_layer
            att_outputs, past_key_values = fwd(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )

            outputs_embeds: list[Tensor | None] = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]

                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue

                end = start + hidden_states.shape[1]
                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]

                # Residual adds stay IN-PLACE, exactly as upstream. This is not stylistic:
                # the model holds mixed bfloat16/float32 parameters, and `a += b` keeps
                # a's dtype while `a = a + b` promotes to the wider one. Promoting here
                # feeds float32 into a bfloat16 Linear and the forward dies several layers
                # later, far from the cause. Cache/patch safety is handled by cloning in
                # `_Taps.tap`, not by avoiding in-place ops.
                out_emb = layer.self_attn.o_proj(att_out)
                out_emb = taps.tap(f"{_TOWERS[i]}.L{layer_idx}.attn_out", out_emb)

                out_emb += hidden_states
                after_first_residual = out_emb.clone()

                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)
                out_emb = taps.tap(f"{_TOWERS[i]}.L{layer_idx}.mlp_out", out_emb)

                out_emb += after_first_residual
                out_emb = taps.tap(f"{_TOWERS[i]}.L{layer_idx}.resid_post", out_emb)

                outputs_embeds.append(out_emb)
                start = end if len(att_outputs) == 1 else 0

            inputs_embeds = outputs_embeds

        final: list[Tensor | None] = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                final.append(None)
                continue
            out_emb = models[i].norm(hidden_states)
            final.append(taps.tap(f"{_TOWERS[i]}.final_norm", out_emb))
        return final, past_key_values


# ------------------------------------------------------------ normalization stats


_NORM_KEYS = {
    "state_mean": "normalize_inputs.buffer_observation_state.mean",
    "state_std": "normalize_inputs.buffer_observation_state.std",
    "action_mean": "unnormalize_outputs.buffer_action.mean",
    "action_std": "unnormalize_outputs.buffer_action.std",
}


def load_norm_stats(checkpoint: str, device: str = "cpu") -> dict[str, Tensor]:
    """Recover the dataset normalization buffers a checkpoint was trained with.

    LeRobot 0.6.0 moved normalization out of the policy and into a processor pipeline
    built from `dataset_stats`, so loading an older finetuned checkpoint logs

        Unexpected key(s) when loading model: normalize_inputs.buffer_observation_state...

    and **drops the statistics on the floor**. The policy then silently consumes raw
    proprioception and emits actions in normalized units. Both arms of a contrastive pair
    are affected equally, so the divergence readout survives -- but any comparison against
    a demonstration action does not, because the two would be in different scales.

    Returns an empty dict when a checkpoint carries no such buffers (e.g. smolvla_base),
    in which case normalization is a no-op and the caller is told via `has_norm_stats`.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError:  # pragma: no cover
        return {}

    try:
        path = hf_hub_download(checkpoint, "model.safetensors")
    except Exception:
        local = Path(checkpoint) / "model.safetensors"
        if not local.exists():
            return {}
        path = str(local)

    stats: dict[str, Tensor] = {}
    try:
        with safe_open(path, framework="pt") as f:
            available = set(f.keys())
            for short, full in _NORM_KEYS.items():
                if full in available:
                    stats[short] = f.get_tensor(full).to(device=device, dtype=torch.float32)
    except Exception:
        return {}

    # All-or-nothing: a partial set would normalize one side and not the other.
    return stats if len(stats) == len(_NORM_KEYS) else {}


# ---------------------------------------------------------------- input building


def pair_pad_length(policy: Any, instructions: Sequence[str]) -> int | None:
    """Common token length for a set of instructions that will be patched across.

    Cross-run patching needs equal prefix lengths. Checkpoints that pad language to
    a fixed `tokenizer_max_length` already have them; checkpoints trained with
    `pad_language_to='longest'` do not, and forcing them to the fixed length is NOT
    behaviour-neutral (measured: actions move by up to 0.02). Padding the *pair* to
    its own longest is -- the model saw exactly that under batched 'longest' padding
    in training. Returns None when the checkpoint's native mode already aligns.
    """
    from transformers import AutoTokenizer

    cfg = policy.config
    if cfg.pad_language_to == "max_length":
        return None
    tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    tasks = [t if t.endswith("\n") else t + "\n" for t in instructions]
    enc = tok(tasks, padding="longest", max_length=cfg.tokenizer_max_length,
              truncation=True, return_tensors="pt")
    return int(enc["input_ids"].shape[1])


def language_token_positions(policy: Any, prefix_len: int, n_lang_tokens: int) -> list[int]:
    """Positions of the language tokens inside a VLM prefix of `prefix_len` tokens.

    lerobot 0.6.0 `embed_prefix` builds the prefix as [image tokens][language tokens]
    [one state token], so the language block is the n_lang_tokens slice ending one
    position before the end. Only valid with `add_image_special_tokens=False` (true of
    every checkpoint this repo uses); with special tokens the image block grows by two
    tokens per image and this arithmetic would silently point at the wrong slice, so
    the assert is load-bearing.
    """
    assert not policy.config.add_image_special_tokens, (
        "prefix layout assumes no image special tokens; recompute for this checkpoint"
    )
    assert getattr(policy.config, "prefix_length", -1) == -1, (
        "prefix_length padding appends AFTER the state token, so 'state is last' "
        "would be false and this slice would silently shift; recompute for this checkpoint"
    )
    end = prefix_len - 1  # the state token is last
    start = end - n_lang_tokens
    if start < 0:
        raise ValueError(f"{n_lang_tokens} language tokens cannot fit in prefix of {prefix_len}")
    return list(range(start, end))


def make_batch(
    images: Tensor | Mapping[str, Tensor],
    state: Tensor,
    instruction: str | Sequence[str],
    policy: Any,
    device: str | torch.device = "cpu",
    pad_to_length: int | None = None,
) -> Batch:
    """Build a SmolVLA input batch.

    Mirrors lerobot's preprocessing pipeline for the language path: a newline is appended
    to the task string and it is tokenized with the VLM tokenizer, right-padded to
    `config.tokenizer_max_length` (see `make_smolvla_pre_post_processors`).

    Note this does NOT apply dataset normalization to state/action. For paired
    comparisons (patched vs unpatched on the same state) that is fine and cancels; for
    absolute action values against a real LIBERO policy it does not. Normalization is
    wired in where real trajectories are used, not here.
    """
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )
    from transformers import AutoTokenizer

    cfg = policy.config
    tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)

    tasks = [instruction] if isinstance(instruction, str) else list(instruction)
    tasks = [t if t.endswith("\n") else t + "\n" for t in tasks]
    enc = tok(
        tasks,
        padding="max_length" if pad_to_length else cfg.pad_language_to,
        padding_side="right",
        max_length=pad_to_length or cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    batch: Batch = {
        OBS_STATE: state.to(device),
        OBS_LANGUAGE_TOKENS: enc["input_ids"].to(device),
        # Must be bool. `embed_prefix` concatenates this with the boolean image and state
        # masks; an int64 mask silently promotes the whole concatenation to int64 and the
        # attention kernel then dies on `torch.where(mask, ...)` deep inside the model.
        OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].bool().to(device),
    }

    image_keys = list(cfg.image_features)
    if isinstance(images, Tensor):
        if not image_keys:
            raise ValueError("policy config declares no image features")
        batch[image_keys[0]] = images.to(device)
    else:
        for k, v in images.items():
            batch[k] = v.to(device)
    return batch
