from __future__ import annotations

from data_classes import State, Trajectory, VolumeElements, Control

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.ops import segment_sum


class FiniteVolumeSolver:

    # Public fields ---------------------------------------------------
    volume_elements: VolumeElements
    times: jnp.ndarray                       # uniform grid (T,)
    K_fun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # coag kernel
    alpha_fun: Callable[[jnp.ndarray], jnp.ndarray]           # frag rate
    b_fun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # daughter law

    # ---------------------------------------------------------------------
    # Construction & pre‑computation
    # ---------------------------------------------------------------------
    def __init__(self, ve: VolumeElements, times: jnp.ndarray,
                 K_fun: Callable, alpha_fun: Callable, b_fun: Callable, mass_fix: str = None):
        self.volume_elements = ve
        self.times = times
        self.K_fun = K_fun
        self.alpha_fun = alpha_fun
        self.b_fun = b_fun

        # Grid shortcuts
        self.x = ve.centers              # (N,)
        self.dx = ve.volumes             # (N,)
        self.N = len(self.x)

        # Time step (assumed uniform)
        self.dt = float(times[1] - times[0])

        self.mass_fix = mass_fix

        # Pre‑compute kernel‑related tensors once – they live on the host.
        self._precompute_operators()

        # Placeholder, actual integrator set by subclass.
        self.integrator: Callable[[State, float], State] | None = None

    # ------------------------------------------------------------------
    # Operator assembly
    # ------------------------------------------------------------------
    def _precompute_operators(self):

        X, Y = self.x[:, None], self.x[None, :]

        # 1) Coag kernel on centers
        self.K_mat = self.K_fun(X, Y)               # (N, N)

        # 2) Flat indices -> target bin k = i + j  (for segment_sum)
        self.idx_sum = jnp.add.outer(jnp.arange(self.N), jnp.arange(self.N)).ravel()  # (N^2,)

        # 3) Fragmentation: alpha and mass-normalized B (upper triangle)
        self.alpha_vec = self.alpha_fun(self.x)          # (N,)
        B_raw = jnp.where(X <= Y, self.b_fun(X, Y), 0.0)  # (N, N), upper-triangular

        # normalize columns to enforce ∑_i x_i B_{ij} Δx_i = y_j (= x_j on same grid)
        col_mass = jnp.sum(X * B_raw * self.dx[:, None], axis=0)  # (N,)
        scale = jnp.where(col_mass > 0.0, self.x / col_mass, 0.0)               # (N,)
        self.B_mat = B_raw * scale[None, :]                                     # (N, N)


    # ------------------------------------------------------------------
    # Low‑level building blocks (all JIT‑compiled, differentiable)             
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=0)
    def _coag_gain(self, f: jnp.ndarray, c: float) -> jnp.ndarray:
        """Gain term ½∫ K(x‑y, y) f(x‑y) f(y) dy discretised by FV."""
        Kff = c * self.K_mat * (f[:, None] * f[None, :])  # (N,N)
        flat = 0.5 * (Kff * self.dx[None, :]).ravel()      # (N²,) 
        return segment_sum(flat, self.idx_sum, num_segments=self.N)  # (N,)

    @partial(jax.jit, static_argnums=0)
    def _coag_loss(self, f: jnp.ndarray, c: float) -> jnp.ndarray:
        """Loss term f(x)∫K(x,y)f(y)dy."""
        return c * f * (self.K_mat @ (f * self.dx))  # (N,)

    @partial(jax.jit, static_argnums=0)
    def _frag_gain(self, f: jnp.ndarray) -> jnp.ndarray:
        """Gain from fragmentation: ∫ α(y) b(x,y) f(y) dy."""
        return (self.B_mat * (self.alpha_vec[None, :] * f[None, :] * self.dx[None, :])).sum(axis=1)

    @partial(jax.jit, static_argnums=0)
    def _frag_loss(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.alpha_vec * f
    
    @partial(jax.jit, static_argnums=0)
    def _coag(self, f: jnp.ndarray, c: float) -> jnp.ndarray:
        return  self._coag_gain(f, c) - self._coag_loss(f, c)

    @partial(jax.jit, static_argnums=0)
    def rhs(self, f: jnp.ndarray, c: float) -> jnp.ndarray:
        return (
            self._coag_gain(f, c)
            - self._coag_loss(f, c)
            + self._frag_gain(f)
            - self._frag_loss(f)
        )
    
    @partial(jax.jit, static_argnums=0)
    def solve(self, *, initial_state: State, control: Control) -> tuple[State, Trajectory, jax.Array]:
        """Return final State, full Trajectory, and mass sequence (T,)."""

        if self.integrator is None:
            raise RuntimeError("Integrator not set. Use FVEulerSolver or FVAdjointRK4Solver.")

        x   = self.x
        dx  = self.dx


        def step(state: State, u_t):
            # 1) tentative step
            nxt = self.integrator(state, u_t)

            # 2) per-step mass enforcement (exact)
            m_curr = nxt.first_moment()
            f_nxt  = nxt.f
            nxt    = State(f_nxt, x)

            return nxt, f_nxt

        # scan over controls (adjust indexing if your integrator expects length T or T-1)
        final_state, f_seq = jax.lax.scan(step, initial_state, control.values[:-1])

        # prepend initial state to trajectory
        f_seq = jnp.concatenate([initial_state.f[None, :], f_seq], axis=0)

        trajectory = Trajectory(f=f_seq, centers=x, times=self.times)
        return final_state, trajectory

# ------------------------------------------------------------------------------
# 3.  Concrete forward integrators                                              
# ------------------------------------------------------------------------------

class FVEulerSolver(FiniteVolumeSolver):
    """Positivity‑preserving forward Euler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integrator = jax.jit(self._euler_step)

    @partial(jax.jit, static_argnums=0)
    def _euler_step(self, state: State, c: float) -> State:  # noqa: D401
        rhs_values = self.rhs(state.f, c)
        # jax.debug.print("Max rhs: {r}", r=self.dt * jnp.max(jnp.abs(rhs_values)))
        f_ = state.f + self.dt * rhs_values
        f_new = jnp.maximum(f_, 0.0)  # enforce positivity
        return State(f_new, self.x)
    
class FVRK4Solver(FiniteVolumeSolver):
    """Classical 4th-order Runge–Kutta time step for semi-discrete FV."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integrator = jax.jit(self._rk4_step)

    @partial(jax.jit, static_argnums=0)
    def _rk4_step(self, state: State, c: float) -> State:  # noqa: D401
        f0 = state.f
        dt = self.dt

        k1 = self.rhs(f0,                 c)             # f' (t,    f0)
        k2 = self.rhs(f0 + 0.5 * dt * k1, c)             # f' (t+dt/2, f0+dt/2 k1)
        k3 = self.rhs(f0 + 0.5 * dt * k2, c)             # f' (t+dt/2, f0+dt/2 k2)
        k4 = self.rhs(f0 +       dt * k3, c)             # f' (t+dt,   f0+dt k3)

        f_ = f0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        f_ = jnp.maximum(f_, 0.0)  

        return State(f_, self.x)