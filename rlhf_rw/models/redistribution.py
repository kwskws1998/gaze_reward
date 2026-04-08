"""
redistribution.py — Asymmetric Gaussian TRT redistribution for GazeReward.

Based on:
  "Leveraging Psychophysical Visual Span for Gaze-Augmented Reward Modeling"

For each fixation at token j with TRT value TRT_j, redistributes TRT to
neighboring tokens i using an asymmetric Gaussian:

    k_{j→i} = ∫_{c_{i,1}}^{c_{i,2}} N(x | 0, σ_side) dx

where σ_side depends on whether token i is to the left or right of j, and
σ is optionally scaled by TRT_j (dynamic span, α > 0):

    σ_j^side = σ_0^side + α · TRT_j

Distances are measured in character units to account for variable token lengths.

Usage
-----
Standalone (fixed sigma, no gradients):
    from redistribution import redistribute_trt_static
    g = redistribute_trt_static(trt, tokens, sigma_left=1.72, sigma_right=2.59)

Differentiable (learnable σ_0 and α, used during training):
    from redistribution import redistribute_trt_differentiable
    g = redistribute_trt_differentiable(trt, tokens, log_sig_left, log_sig_right,
                                         log_alpha_left, log_alpha_right)

Integration into reward_model_base.py
--------------------------------------
In MyRewardBase.process_fixations(), after computing raw TRT but before the projector:

    if getattr(self, 'use_redistribution', False):
        fixations = redistribute_trt_differentiable(
            fixations,                  # (batch, seq, 1) for fmv=1
            self._token_strings,        # list[list[str]], set in compute_fixations_cached
            self.log_sig_left,
            self.log_sig_right,
            self.log_alpha_left,
            self.log_alpha_right,
        )

To enable learnable redistribution, call enable_learnable_redistribution()
on your model instance after load_fx_model_1().
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gaussian CDF helpers
# ---------------------------------------------------------------------------

def _gaussian_cdf(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Φ(x/σ) — standard normal CDF evaluated at x/σ."""
    return 0.5 * (1.0 + torch.erf(x / (sigma * math.sqrt(2.0) + 1e-8)))


def _integral_gaussian(lo: torch.Tensor, hi: torch.Tensor,
                        sigma: torch.Tensor) -> torch.Tensor:
    """∫_{lo}^{hi} N(x|0,σ) dx = Φ(hi/σ) − Φ(lo/σ)."""
    return _gaussian_cdf(hi, sigma) - _gaussian_cdf(lo, sigma)


# ---------------------------------------------------------------------------
# Character-level distance helpers
# ---------------------------------------------------------------------------

def _token_char_centers_and_widths(tokens: list[str]):
    """
    Given a list of token strings (already decoded, no special prefix handling),
    return:
        centers : list[float]  — character-unit center of each token
        half_widths : list[float] — half the character width of each token
    Tokens with zero length get width 1 (treated as single char).
    """
    centers, half_widths = [], []
    cursor = 0.0
    for tok in tokens:
        w = max(len(tok), 1)
        centers.append(cursor + w / 2.0)
        half_widths.append(w / 2.0)
        cursor += w
    return centers, half_widths


# ---------------------------------------------------------------------------
# Static redistribution (no gradients, used at eval when sigma is fixed)
# ---------------------------------------------------------------------------

def redistribute_trt_static(
    trt: torch.Tensor,
    tokens: list[str],
    sigma_left: float = 1.72256,
    sigma_right: float = 2.59374,
    alpha_left: float = 0.0,
    alpha_right: float = 0.0,
) -> torch.Tensor:
    """
    Redistribute TRT values using a fixed asymmetric Gaussian.

    Parameters
    ----------
    trt : Tensor shape (seq,) or (seq, 1)
        Per-token TRT predictions (already mapped to model tokenizer space).
    tokens : list[str]
        Decoded token strings for the same sequence (len == seq).
    sigma_left, sigma_right : float
        Base standard deviations for left/right side of the Gaussian.
    alpha_left, alpha_right : float
        Coefficient scaling σ by TRT (0 = static span).

    Returns
    -------
    Tensor shape same as trt — redistributed TRT.
    """
    squeeze = trt.dim() == 2
    if squeeze:
        trt = trt.squeeze(-1)           # (seq,)

    S = len(trt)
    assert len(tokens) == S, f"tokens length {len(tokens)} != trt length {S}"

    centers, half_widths = _token_char_centers_and_widths(tokens)
    device = trt.device
    trt_cpu = trt.detach().float().cpu()

    g = torch.zeros(S, dtype=torch.float32)

    for j in range(S):
        trt_j = trt_cpu[j].item()
        sig_l = sigma_left  + alpha_left  * trt_j
        sig_r = sigma_right + alpha_right * trt_j
        sig_l = max(sig_l, 1e-4)
        sig_r = max(sig_r, 1e-4)

        c_j = centers[j]
        for i in range(S):
            c_i = centers[i]
            hw_i = half_widths[i]
            lo = c_i - hw_i - c_j
            hi = c_i + hw_i - c_j
            # Use left or right sigma depending on side
            # For a token spanning both sides, split at 0
            if hi <= 0:
                # entirely to the left
                k = _compute_integral_scalar(lo, hi, sig_l)
            elif lo >= 0:
                # entirely to the right
                k = _compute_integral_scalar(lo, hi, sig_r)
            else:
                # spans the center — split at 0
                k = (_compute_integral_scalar(lo, 0.0, sig_l) +
                     _compute_integral_scalar(0.0, hi, sig_r))
            g[i] += trt_j * k

    g = g.to(device)
    if squeeze:
        g = g.unsqueeze(-1)
    return g


def _compute_integral_scalar(lo: float, hi: float, sigma: float) -> float:
    """∫_{lo}^{hi} N(x|0,sigma) dx — scalar version."""
    sqrt2 = math.sqrt(2.0)
    return 0.5 * (math.erf(hi / (sigma * sqrt2)) - math.erf(lo / (sigma * sqrt2)))


# ---------------------------------------------------------------------------
# Differentiable redistribution (learnable parameters, used during training)
# ---------------------------------------------------------------------------

def redistribute_trt_differentiable(
    trt: torch.Tensor,
    tokens: list[str],
    log_sig_left: nn.Parameter,
    log_sig_right: nn.Parameter,
    log_alpha_left: nn.Parameter,
    log_alpha_right: nn.Parameter,
) -> torch.Tensor:
    """
    Differentiable asymmetric Gaussian redistribution.

    Parameters
    ----------
    trt : Tensor shape (seq,) or (seq, 1)
    tokens : list[str]
    log_sig_left, log_sig_right : nn.Parameter (scalar)
        Log of base sigma (ensures sigma > 0 via exp).
    log_alpha_left, log_alpha_right : nn.Parameter (scalar)
        Log of alpha coefficient (alpha >= 0 via softplus or exp).

    Returns
    -------
    Tensor same shape as trt, with gradients w.r.t. parameters.
    """
    squeeze = trt.dim() == 2
    if squeeze:
        trt = trt.squeeze(-1)   # (seq,)

    S = len(trt)
    assert len(tokens) == S

    device = trt.device
    sig_l0 = torch.exp(log_sig_left).to(device)
    sig_r0 = torch.exp(log_sig_right).to(device)
    alpha_l = torch.nn.functional.softplus(log_alpha_left).to(device)
    alpha_r = torch.nn.functional.softplus(log_alpha_right).to(device)

    centers, half_widths = _token_char_centers_and_widths(tokens)
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)       # (S,)
    hw_t      = torch.tensor(half_widths, dtype=torch.float32, device=device)   # (S,)

    # trt: (S,), broadcast to (S_j, S_i)
    trt_j = trt.unsqueeze(1)                         # (S, 1)
    sig_l_j = (sig_l0 + alpha_l * trt_j).clamp(min=1e-4)   # (S, 1)
    sig_r_j = (sig_r0 + alpha_r * trt_j).clamp(min=1e-4)   # (S, 1)

    # distances from fixation j to boundaries of token i
    # lo[j,i] = centers[i] - hw[i] - centers[j]
    # hi[j,i] = centers[i] + hw[i] - centers[j]
    c_j = centers_t.unsqueeze(1)   # (S, 1)
    c_i = centers_t.unsqueeze(0)   # (1, S)
    hw_i = hw_t.unsqueeze(0)       # (1, S)

    lo = (c_i - hw_i) - c_j       # (S, S)
    hi = (c_i + hw_i) - c_j       # (S, S)

    # Split integral at 0 for asymmetric sigma
    # left portion: ∫_{lo}^{min(hi,0)} using sig_l_j
    # right portion: ∫_{max(lo,0)}^{hi} using sig_r_j
    hi_left  = hi.clamp(max=0.0)
    lo_left  = lo
    hi_right = hi
    lo_right = lo.clamp(min=0.0)

    k_left  = _integral_gaussian(lo_left,  hi_left,  sig_l_j)   # (S, S)
    k_right = _integral_gaussian(lo_right, hi_right, sig_r_j)   # (S, S)

    # Zero out the degenerate cases
    k_left  = k_left  * (hi_left  > lo_left).float()
    k_right = k_right * (hi_right > lo_right).float()

    k = k_left + k_right                          # (S_j, S_i)

    # g[i] = Σ_j TRT_j * k[j,i]
    g = (trt.unsqueeze(1) * k).sum(dim=0)         # (S,)

    if squeeze:
        g = g.unsqueeze(-1)
    return g


# ---------------------------------------------------------------------------
# Mixin to add to MyRewardBase
# ---------------------------------------------------------------------------

class LearnableRedistributionMixin:
    """
    Add this to MyRewardBase to enable learnable asymmetric Gaussian redistribution.

    Call enable_learnable_redistribution() after load_fx_model_1().
    The four learnable scalar parameters are registered as nn.Parameters so
    they are picked up by the optimizer and saved with the model state_dict.

    Integration
    -----------
    In process_fixations(), after the fixations tensor is computed but
    before the projector, add:

        if getattr(self, 'use_redistribution', False):
            # fixations: (batch, seq) for fmv=1 after squeeze(-1)... 
            # actually shape is (batch, seq, 1) before unsqueeze in process_fixations
            # so we apply per-sequence
            redistributed = []
            for b in range(fixations.shape[0]):
                trt_b = fixations[b]  # (seq, 1) or (seq,)
                toks_b = self._redistribution_tokens[b]
                redistributed.append(
                    redistribute_trt_differentiable(
                        trt_b, toks_b,
                        self.log_sig_left, self.log_sig_right,
                        self.log_alpha_left, self.log_alpha_right,
                    )
                )
            fixations = torch.stack(redistributed, dim=0)
    """

    def enable_learnable_redistribution(
        self,
        init_sigma_left: float = 4.0,
        init_sigma_right: float = 5.5,
        init_alpha: float = 0.0,
    ):
        """
        Register learnable redistribution parameters.

        Parameters are stored as log values so they stay positive via exp/softplus:
            σ_0 = exp(log_sig)
            α   = softplus(log_alpha)

        init_sigma_left/right : initial σ_0 values (Legge 2001 defaults: 4.0, 5.5)
        init_alpha            : initial α (0 = static span)
        """
        self.use_redistribution = True
        self.log_sig_left  = nn.Parameter(
            torch.tensor(math.log(init_sigma_left),  dtype=torch.float32)
        )
        self.log_sig_right = nn.Parameter(
            torch.tensor(math.log(init_sigma_right), dtype=torch.float32)
        )
        # softplus^{-1}(0) = log(e^0 - 1) ≈ -inf, use small positive instead
        _sp_inv = math.log(math.exp(max(init_alpha, 1e-4)) - 1.0 + 1e-8)
        self.log_alpha_left  = nn.Parameter(
            torch.tensor(_sp_inv, dtype=torch.float32)
        )
        self.log_alpha_right = nn.Parameter(
            torch.tensor(_sp_inv, dtype=torch.float32)
        )

    @property
    def sigma_left(self) -> float:
        return self.log_sig_left.exp().item()

    @property
    def sigma_right(self) -> float:
        return self.log_sig_right.exp().item()

    @property
    def alpha_left(self) -> float:
        return torch.nn.functional.softplus(self.log_alpha_left).item()

    @property
    def alpha_right(self) -> float:
        return torch.nn.functional.softplus(self.log_alpha_right).item()
