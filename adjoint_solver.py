from __future__ import annotations

from data_classes import State, Trajectory, VolumeElements, Control

import dataclasses
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import chex

from functools import partial
import jax
import jax.numpy as jnp

class AdjointFiniteVolumeSolver:
    """Backward-in-time adjoint associated with :class:`FiniteVolumeSolver`."""

    # Public (PyTree) fields ---------------------------------------------------
    volume_elements: "VolumeElements"
    times: jnp.ndarray
    K_fun: callable
    alpha_fun: callable
    b_fun: callable

    # ---------------------------------------------------------------------
    # Construction & pre-computation
    # ---------------------------------------------------------------------
    def __init__(self, ve: VolumeElements, times: jnp.ndarray,
                 K_fun, alpha_fun, b_fun):
        self.volume_elements = ve
        self.times = times
        self.K_fun = K_fun
        self.alpha_fun = alpha_fun
        self.b_fun = b_fun

        # Grid shortcuts
        self.x  = ve.centers
        self.dx = ve.volumes
        self.N  = self.x.shape[0]

        # Time step (assumed uniform)
        self.dt = float(times[1] - times[0])

        # Operators
        self._precompute_operators()

        # Placeholder; subclasses set a JIT'd backstep (phi_next, f_t, u_t) -> phi_prev
        self.integrator = None

    # ------------------------------------------------------------------
    # Operator assembly (mirrors forward)
    # ------------------------------------------------------------------
    def _precompute_operators(self) -> None:
        X, Y = self.x[:, None], self.x[None, :]

        self.K_mat = self.K_fun(X, Y)                 # (N,N)

        self.alpha_vec = self.alpha_fun(self.x)       # (N,)
        B_raw = jnp.where(X <= Y, self.b_fun(X, Y), 0.0)  # (N,N)

        col_mass = jnp.sum((X * B_raw) * self.dx[:, None], axis=0)   # (N,)
        scale    = jnp.where(col_mass > 0.0, self.x / col_mass, 0.0) # (N,)
        self.B_mat = B_raw * scale[None, :]                          # (N,N)
        self.idx_sum = jnp.add.outer(jnp.arange(self.N), jnp.arange(self.N)).ravel()

    # ------------------------------------------------------------------
    # Adjoint RHS (used by integrators)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def rhs(self, f: jnp.ndarray, phi: jnp.ndarray, c: float) -> jnp.ndarray:
        dy = self.dx
        term_a = self.K_mat.T @ (f * phi * dy)
        term_b = phi * (self.K_mat @ (f * dy))
        X = self.x[:, None]; Y = self.x[None, :]
        x_plus_y = X + Y
        phi_xy   = jnp.interp(x_plus_y, self.x, phi, left=0.0, right=0.0)
        K_sum    = self.K_fun(x_plus_y, Y)            # assume pure JAX function
        term_c   = jnp.sum(K_sum * (f[None, :] * phi_xy * dy[None, :]), axis=1)
        frag_loss = self.alpha_vec * phi
        frag_gain = - self.alpha_vec * (self.B_mat.T @ (phi * dy))
        return c * (term_a + term_b - term_c) + frag_loss + frag_gain

    # ------------------------------------------------------------------
    # Backward time marching (vectorised over *all* steps)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def integrate(self, f_traj: Trajectory, control: Control, phi_T: jnp.ndarray) -> Trajectory:
        """Return full costate trajectory φ with shape (T+1, N), aligned to f_traj."""
        if self.integrator is None:
            raise RuntimeError("Adjoint integrator not set. Use AdjointEulerSolver or AdjointRK4Solver.")

        def back_step(phi_next, inputs):
            f_t, u_t = inputs
            phi_prev = self.integrator(phi_next, f_t, u_t)  # <- uses subclass integrator
            return phi_prev, phi_prev

        # f_traj is (T+1,N) at t0..tT; controls are (T,) for intervals
        f_inner = f_traj.f[1:]             # t0..t_{T-1}
        control_values = control.values[:-1]
        _, phi_bw = jax.lax.scan(back_step, phi_T, (f_inner[::-1], control_values[::-1]))
        phi_fw = phi_bw[::-1]
        phi_traj = jnp.concatenate([phi_fw, phi_T[None, :]], axis=0)
        return Trajectory(phi_traj, f_traj.centers, self.times)

    

# ----------------------------------------------------------------------
# Concrete adjoint integrators (mirror forward Euler / RK4 structure)
# ----------------------------------------------------------------------

class AdjointFVEulerSolver(AdjointFiniteVolumeSolver):
    """Backward explicit Euler for the adjoint."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integrator = jax.jit(self._euler_backstep)

    @partial(jax.jit, static_argnums=0)
    def _euler_backstep(self, phi_next: jnp.ndarray,
                        f_t: jnp.ndarray, u_t: float) -> jnp.ndarray:
        # φ_{n} = φ_{n+1} - dt * φ̇(t_{n+1})
        phi_dot = self.rhs(f_t, phi_next, u_t)
        return phi_next - self.dt * phi_dot


class AdjointFVRK4Solver(AdjointFiniteVolumeSolver):
    """Backward classical RK4 for the adjoint."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integrator = jax.jit(self._rk4_backstep)

    @partial(jax.jit, static_argnums=0)
    def _rk4_backstep(self, phi_next: jnp.ndarray,
                      f_t: jnp.ndarray, u_t: float) -> jnp.ndarray:
        # Integrate dφ/dt = g(f_t, φ, u_t) from t_{n+1} -> t_n with step -dt
        dt = self.dt

        k1 = self.rhs(f_t, phi_next,                 u_t)
        k2 = self.rhs(f_t, phi_next - 0.5 * dt * k1, u_t)
        k3 = self.rhs(f_t, phi_next - 0.5 * dt * k2, u_t)
        k4 = self.rhs(f_t, phi_next - dt * k3,       u_t)

        # φ_n
        return phi_next - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)