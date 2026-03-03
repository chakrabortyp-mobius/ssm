"""
model/ssm.py
============
SelectiveSSM — Mamba-inspired Selective State Space Model.
Pure NumPy with real backprop-through-time (BPTT).

Architecture (per timestep t)
------------------------------
    h_t   = tanh( W_in @ x_t  + b_in )          input projection
    Δ_t   = softplus( W_dt · h_t + b_dt )        selective gate  (scalar > 0)
    Ā_t   = exp( Δ_t * A )                        discretised transition  (d_state,)
    B̄_t   = (Ā_t - 1) / A * Δ_t                  discretised injection   (d_state,)
    z_t   = Ā_t * z_{t-1}  +  B̄_t * (W_B @ h_t)  state update
    x̂_t   = W_out @ tanh(W_C @ z_t) + b_out      reconstruction

What each learned parameter does
---------------------------------
    A_log  : log(-A), diagonal state transition
             → controls how fast each latent dim forgets the past
    W_in   : which input features project into the hidden representation
    W_dt   : which features drive the selective gate Δ
             (large Δ = model resets state = SHOCK)
    W_B    : how strongly each feature injects into each latent dim
    W_C    : how latent dims relate during emission
    W_out  : how to reconstruct original input from latent state

Training objective
------------------
    Unsupervised reconstruction: loss = mean( ||x_t - x̂_t||² )
    Gradients via BPTT — no labels required.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.adam import AdamParam
from utils.activations import (
    softplus, softplus_grad,
    tanh_safe, tanh_grad,
)
from utils.logger import get_logger

log = get_logger("SelectiveSSM")


class SelectiveSSM:
    """
    Selective State Space Model with BPTT training.

    Parameters
    ----------
    d_input  : input feature dimension (= encoder output width)
    d_state  : latent state dimension  (= shock embedding size)
    d_inner  : internal projection dim (default = max(2*d_input, 32))
    lr       : Adam learning rate
    seed     : random seed for weight initialisation
    """

    def __init__(
        self,
        d_input: int,
        d_state: int = 16,
        d_inner: Optional[int] = None,
        lr:      float = 1e-3,
        seed:    int = 42,
    ):
        self.d_input = d_input
        self.d_state = d_state
        self.d_inner = d_inner or max(d_input * 2, 32)
        self.lr = lr

        self._init_params(seed)

        # Ordered list of parameter names — used in backward loop
        self._param_names: List[str] = [
            "A_log", "W_in", "b_in", "W_dt", "b_dt",
            "W_B", "W_C", "W_out", "b_out",
        ]
        log.debug(
            f"SelectiveSSM initialised: "
            f"d_input={d_input}, d_state={d_state}, d_inner={self.d_inner}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Parameter initialisation
    # ────────────────────────────────────────────────────────────────────────

    def _init_params(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.1

        def P(shape) -> AdamParam:
            return AdamParam(rng.normal(0, scale, shape), lr=self.lr)

        # Diagonal state transition: A = -exp(A_log), always negative
        self.A_log = AdamParam(rng.uniform(-1.0, -0.1, self.d_state), lr=self.lr)

        # Input projection
        self.W_in  = P((self.d_inner, self.d_input))
        self.b_in  = AdamParam(np.zeros(self.d_inner), lr=self.lr)

        # Delta gate (selective memory)
        self.W_dt  = P((self.d_inner,))
        self.b_dt  = AdamParam(np.array([0.5]), lr=self.lr)   # positive init → gate open

        # State injection
        self.W_B   = P((self.d_state, self.d_inner))

        # State emission
        self.W_C   = P((self.d_state, self.d_state))

        # Reconstruction
        self.W_out = P((self.d_input, self.d_state))
        self.b_out = AdamParam(np.zeros(self.d_input), lr=self.lr)

    # ────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        X: np.ndarray,
        store_cache: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Dict]]]:
        """
        Run the SSM over sequence X.

        Parameters
        ----------
        X           : (T, d_input)  encoded feature matrix
        store_cache : True during training (needed for backward)
                      False during inference (saves memory)

        Returns
        -------
        Z      : (T, d_state)   latent state trajectory = shock embeddings
        deltas : (T,)           gate values  (large Δ → state reset → shock)
        recon  : (T, d_input)   reconstructed input
        cache  : list of per-timestep intermediates, or None
        """
        T  = X.shape[0]
        A  = -np.exp(self.A_log.data)   # (d_state,)  always negative

        Z      = np.zeros((T, self.d_state))
        deltas = np.zeros(T)
        recon  = np.zeros_like(X)
        cache  = [] if store_cache else None

        z = np.zeros(self.d_state)   # initial hidden state

        for t in range(T):
            x_t = X[t]

            # ── Input projection ─────────────────────────────────────────
            pre_h = self.W_in.data @ x_t + self.b_in.data   # (d_inner,)
            h_t   = tanh_safe(pre_h)                         # (d_inner,)

            # ── Selective gate Δ ─────────────────────────────────────────
            pre_dt = float(self.W_dt.data @ h_t) + float(self.b_dt.data[0])
            dt     = float(softplus(np.array([pre_dt]))[0])     # scalar > 0

            # ── Discretise (Zero-Order Hold) ─────────────────────────────
            A_bar = np.exp(dt * A)                           # (d_state,)
            B_bar = (A_bar - 1.0) / (A + 1e-8) * dt         # (d_state,)
            B_t   = self.W_B.data @ h_t                      # (d_state,)

            # ── State update ─────────────────────────────────────────────
            z_prev = z.copy()
            z      = A_bar * z_prev + B_bar * B_t            # (d_state,)

            # ── Reconstruction ───────────────────────────────────────────
            pre_C = self.W_C.data @ z                        # (d_state,)
            act_C = tanh_safe(pre_C)                         # (d_state,)
            x_hat = self.W_out.data @ act_C + self.b_out.data   # (d_input,)

            Z[t]      = z
            deltas[t] = dt
            recon[t]  = x_hat

            if store_cache:
                cache.append({
                    "x_t":    x_t,
                    "pre_h":  pre_h,
                    "h_t":    h_t,
                    "pre_dt": pre_dt,
                    "dt":     dt,
                    "A":      A,
                    "A_bar":  A_bar,
                    "B_bar":  B_bar,
                    "B_t":    B_t,
                    "z_prev": z_prev,
                    "z":      z.copy(),
                    "pre_C":  pre_C,
                    "act_C":  act_C,
                    "x_hat":  x_hat,
                })

        return Z, deltas, recon, cache

    # ────────────────────────────────────────────────────────────────────────
    # Backward pass (BPTT)
    # ────────────────────────────────────────────────────────────────────────

    def backward(
        self,
        X:     np.ndarray,
        recon: np.ndarray,
        cache: List[Dict],
    ) -> None:
        """
        Backprop-through-time: compute gradients and run Adam updates.

        Loss: MSE reconstruction  L = (1/T) * Σ ||x_t - x̂_t||²

        Gradient flow (reversed over time):
            dL/dx̂_t → W_out, b_out
                     → W_C (through tanh)
                     → dz_t
                     → A_log, W_B, W_dt, b_dt (through state update)
                     → W_in, b_in (through h_t)
                     → dz_{t-1}   (passed back to previous timestep)
        """
        T  = X.shape[0]
        g: Dict[str, np.ndarray] = {
            p: np.zeros_like(getattr(self, p).data)
            for p in self._param_names
        }

        dz_next = np.zeros(self.d_state)   # gradient from t+1 into z_t

        for t in reversed(range(T)):
            c = cache[t]

            # ── Loss gradient into x̂_t ───────────────────────────────────
            # dL/dx̂_t = (2/T) * (x̂_t - x_t)
            dx_hat = (recon[t] - X[t]) * (2.0 / T)               # (d_input,)

            # ── W_out, b_out ──────────────────────────────────────────────
            g["W_out"] += np.outer(dx_hat, c["act_C"])
            g["b_out"] += dx_hat

            # ── Through tanh(W_C @ z_t) ───────────────────────────────────
            d_act_C = self.W_out.data.T @ dx_hat                   # (d_state,)
            d_pre_C = d_act_C * tanh_grad(c["pre_C"])              # (d_state,)
            g["W_C"] += np.outer(d_pre_C, c["z"])

            # ── Gradient into z_t (from reconstruction + next timestep) ──
            dz = self.W_C.data.T @ d_pre_C + dz_next              # (d_state,)

            # ── State update: z_t = Ā * z_{t-1} + B̄ * B_t ───────────────
            dz_prev = c["A_bar"] * dz                              # into z_{t-1}
            dB_bar  = dz * c["B_t"]
            dB_t    = dz * c["B_bar"]
            dA_bar  = dz * c["z_prev"] + dB_bar * (1.0 / (c["A"] + 1e-8)) * c["dt"]

            # ── A_log gradient ────────────────────────────────────────────
            # A = -exp(A_log)  →  dA/dA_log = -exp(A_log) = A
            # A_bar = exp(dt*A)  →  dA_bar/dA = A_bar * dt
            dA      = dA_bar * c["A_bar"] * c["dt"]
            dA_log  = dA * (-np.exp(self.A_log.data))
            g["A_log"] += dA_log

            # ── dt gradient ───────────────────────────────────────────────
            # B_bar = (A_bar-1)/A * dt  →  dB_bar/d_dt = (A_bar-1)/A
            # A_bar = exp(dt*A)         →  dA_bar/d_dt = A_bar * A
            dB_bar_wrt_dt = (c["A_bar"] - 1.0) / (c["A"] + 1e-8)
            d_dt = (
                (dB_bar * dB_bar_wrt_dt).sum()
                + (dA_bar * c["A_bar"] * c["A"]).sum()
            )

            # ── W_dt, b_dt ────────────────────────────────────────────────
            d_pre_dt = d_dt * float(softplus_grad(np.array([c["pre_dt"]]))[0])
            g["W_dt"] += d_pre_dt * c["h_t"]
            g["b_dt"] += np.array([d_pre_dt])

            # ── W_B ───────────────────────────────────────────────────────
            g["W_B"] += np.outer(dB_t, c["h_t"])
            dh_from_B = self.W_B.data.T @ dB_t                    # (d_inner,)

            # ── W_in, b_in ────────────────────────────────────────────────
            dh = dh_from_B + self.W_dt.data * d_pre_dt            # (d_inner,)
            d_pre_h = dh * tanh_grad(c["pre_h"])                   # (d_inner,)
            g["W_in"] += np.outer(d_pre_h, c["x_t"])
            g["b_in"] += d_pre_h

            # Pass gradient to previous timestep
            dz_next = dz_prev

        # ── Clip and apply Adam updates ───────────────────────────────────
        for p in self._param_names:
            getattr(self, p).step(g[p])   # AdamParam.step() clips internally

    # ────────────────────────────────────────────────────────────────────────
    # Convenience wrappers
    # ────────────────────────────────────────────────────────────────────────

    def train_step(self, X: np.ndarray) -> float:
        """Forward + backward + Adam update. Returns scalar loss."""
        _, _, recon, cache = self.forward(X, store_cache=True)
        loss = float(np.mean((X - recon) ** 2))
        self.backward(X, recon, cache)
        return loss

    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inference-only forward pass (no cache stored).

        Returns
        -------
        Z      : (T, d_state)
        deltas : (T,)
        recon  : (T, d_input)
        """
        Z, deltas, recon, _ = self.forward(X, store_cache=False)
        return Z, deltas, recon

    def param_norms(self) -> Dict[str, float]:
        """Return L2 norm of each parameter — useful for debugging."""
        return {
            p: float(np.linalg.norm(getattr(self, p).data))
            for p in self._param_names
        }
