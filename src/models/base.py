"""Common VLA interface (CLAUDE.md §5).

Everything in `src/interp/` and `src/eval/` talks to models through this module only.
SmolVLA-vs-OpenVLA differences must not leak past this boundary.

Core concepts
-------------
`SiteSpec`  A named, taggable tap point in the network (layer x tower x component).

`Occurrence`  A module/computation may run **many times** inside a single action
    prediction. SmolVLA is a flow-matching policy: the VLM prefix runs once, then the
    action expert runs once per denoising step (10 by default). So a cache entry is a
    *list* of tensors, one per invocation, not a single tensor. Code that assumes one
    tensor per site is silently wrong on this architecture, which is why `single()`
    raises rather than picking the first.

`PatchValue`  What to substitute at a site. Either one tensor (broadcast to every
    occurrence), a list (one per occurrence), or a callable `(old, index) -> new`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

Batch = dict[str, Any]
PatchValue = Tensor | Sequence[Tensor] | Callable[[Tensor, int], Tensor]

Tower = Literal["vlm", "expert"]
Component = Literal["resid_pre", "resid_post", "attn_out", "mlp_out", "final_norm"]


@dataclass(frozen=True, order=True)
class SiteSpec:
    """A tap point.

    `name` is the stable public id used by interp/eval, e.g. ``"expert.L7.mlp_out"``.
    `tower`/`layer`/`component` are the structured form, used to build the
    layer x component grid for the M2 localization map.
    """

    name: str
    tower: Tower
    component: Component
    layer: int | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass
class ForwardResult:
    """Output of a (possibly patched) forward pass."""

    action: Tensor
    """Predicted action chunk, shape (batch, chunk_size, action_dim)."""

    cache: dict[str, list[Tensor]] = field(default_factory=dict)
    """site name -> one tensor per invocation of that site."""

    def occurrences(self, site: str) -> list[Tensor]:
        if site not in self.cache:
            raise KeyError(
                f"site {site!r} was not cached; requested sites were {sorted(self.cache)}"
            )
        return self.cache[site]

    def single(self, site: str) -> Tensor:
        """The one tensor at `site`, erroring if the site actually fired several times.

        Deliberately strict: silently returning occurrence 0 would make a multi-call
        site (any expert site under flow matching) look like a single-call site.
        """
        occ = self.occurrences(site)
        if len(occ) != 1:
            raise ValueError(
                f"site {site!r} fired {len(occ)} times, not once. Use occurrences() and "
                f"say which invocation you mean (expert sites fire once per denoising step)."
            )
        return occ[0]


def resolve_patch(value: PatchValue, old: Tensor, index: int) -> Tensor:
    """Turn a `PatchValue` into the tensor to substitute at invocation `index`."""
    if callable(value) and not isinstance(value, Tensor):
        return value(old, index)
    if isinstance(value, Tensor):
        new = value
    else:
        seq = list(value)
        if index >= len(seq):
            raise IndexError(
                f"patch sequence has {len(seq)} entries but site fired at least {index + 1} times"
            )
        new = seq[index]
    if new.shape != old.shape:
        raise ValueError(f"patch shape {tuple(new.shape)} != activation shape {tuple(old.shape)}")
    return new.to(dtype=old.dtype, device=old.device)


class VLAModel(ABC):
    """Common interface. See CLAUDE.md §5."""

    @classmethod
    @abstractmethod
    def load(cls, **kwargs: Any) -> VLAModel:
        """Load frozen public weights. We train nothing."""

    @abstractmethod
    def sites(self) -> list[SiteSpec]:
        """Every tap point available for patching, in a stable order."""

    @abstractmethod
    def predict_action(self, inputs: Batch, **kwargs: Any) -> Tensor:
        """Predicted action chunk. Must be deterministic given the same inputs."""

    @abstractmethod
    def forward_with_cache(
        self,
        inputs: Batch,
        sites: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ForwardResult:
        """Run a forward pass, recording activations at `sites`."""

    @abstractmethod
    def patch(
        self,
        inputs: Batch,
        patches: Mapping[str, PatchValue],
        sites: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ForwardResult:
        """Run a forward pass with `patches` substituted in, optionally caching `sites`."""

    # ---------------------------------------------------------------- helpers

    def site_names(self) -> list[str]:
        return [s.name for s in self.sites()]

    def sites_where(
        self,
        tower: Tower | None = None,
        component: Component | None = None,
        layer: int | None = None,
    ) -> list[SiteSpec]:
        """Filter the site grid -- used to build M2 sweeps."""
        out = self.sites()
        if tower is not None:
            out = [s for s in out if s.tower == tower]
        if component is not None:
            out = [s for s in out if s.component == component]
        if layer is not None:
            out = [s for s in out if s.layer == layer]
        return out


def action_divergence(a: Tensor, b: Tensor, kind: str = "l2") -> Tensor:
    """Offline action divergence -- the primary patching readout (CLAUDE.md §8).

    Compares two predicted action chunks of shape (batch, chunk, dim) and returns one
    scalar per batch element.
    """
    if a.shape != b.shape:
        raise ValueError(f"action shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    flat_a = a32.reshape(a32.shape[0], -1)
    flat_b = b32.reshape(b32.shape[0], -1)
    if kind == "l2":
        return torch.linalg.vector_norm(flat_a - flat_b, dim=-1)
    if kind == "mse":
        return ((flat_a - flat_b) ** 2).mean(dim=-1)
    if kind == "cosine":
        return 1.0 - torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=-1)
    raise ValueError(f"unknown divergence kind {kind!r}; expected l2, mse or cosine")
