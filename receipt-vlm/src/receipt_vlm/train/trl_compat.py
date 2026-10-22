"""CPU fallback for TRL's fused log-prob kernel (used only when CUDA is unavailable).

TRL >= 1.x computes GRPO per-token log-probs with a chunked Triton kernel
(``trl.kernels.chunked_logprob._ChunkedLogProbFunction``), which needs a GPU driver. For CPU
smoke runs we swap in a plain-PyTorch function with the same signature and the same logit
transform (``z = softcap * tanh(logits * scale / softcap) / temperature``). On GPU nothing is
patched, so real runs use TRL's kernel unchanged.
"""

from __future__ import annotations

from typing import Any

from receipt_vlm.utils.logging import get_logger

log = get_logger(__name__)


class TorchLogProb:
    """Drop-in for ``_ChunkedLogProbFunction`` (only ``apply`` is used by TRL)."""

    @staticmethod
    def apply(
        hidden: Any,
        weight: Any,
        bias: Any,
        targets: Any,
        temperature: float,
        chunk_size: int,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
        outputs: tuple[str, ...] = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1"),
    ) -> tuple[Any, Any, Any, Any, Any]:
        import torch

        z = hidden.float() @ weight.float().t()
        if bias is not None:
            z = z + bias.float()
        z = z * logit_scale
        if final_logit_softcapping is not None:
            z = final_logit_softcapping * torch.tanh(z / final_logit_softcapping)
        z = z / temperature
        logp = torch.log_softmax(z, dim=-1)
        log_probs = logp.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1) if "entropy" in outputs else None
        log_sum_sq = torch.logsumexp(2 * logp, dim=-1) if "log_sum_sq_probs" in outputs else None
        mean_logits = z.mean(-1) if "mean_logits" in outputs else None
        is_top1 = (z.argmax(-1) == targets).float() if "is_top1" in outputs else None
        return log_probs, entropy, log_sum_sq, mean_logits, is_top1


def install_cpu_logprob_fallback() -> bool:
    """Patch TRL's fused-head helper to use :class:`TorchLogProb` when there is no GPU."""
    import torch

    if torch.cuda.is_available():
        return False
    import trl.trainer.utils as tu

    if not hasattr(tu, "_ChunkedLogProbFunction"):
        return False
    tu._ChunkedLogProbFunction = TorchLogProb  # type: ignore[attr-defined]
    log.info("no CUDA: using the PyTorch log-prob fallback instead of TRL's Triton kernel")
    return True
