from __future__ import annotations 

import dataclasses
from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from flax import struct
import chex

@struct.dataclass
class Control:
    """Scalar control u(t) sampled on a uniform grid."""

    values: jnp.ndarray  # shape (T,)
    times: jnp.ndarray   # shape (T,)

    def time_interpolation(self, t_new: jax.Array) -> jax.Array:
        """Interpolate control values to new time points."""
        v_new = jnp.interp(t_new, self.times, self.values, self.values[0], self.values[-1])
        return Control(values=v_new, times=t_new)

    # ---------------------------------------------------------------------
    # Small plotting helper (kept outside the computational graph)
    # ---------------------------------------------------------------------
    def plot(self, *, u_min: float, u_max: float, figsize: tuple[int, int] = (8, 4)) -> None:  
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.step(self.times, self.values, where="post", label="u(t)")
        plt.ylim(u_min * 1.1, u_max * 1.1)
        plt.xlabel("time t")
        plt.ylabel("control amplitude")
        plt.title("Scalar control")
        plt.grid(True)
        plt.legend()
        plt.show()


@struct.dataclass
class State:
    """Spatial density f(x) (cell averages) and corresponding cell centres."""

    f: jnp.ndarray        # shape (N,) or (T, N)
    centers: jnp.ndarray  # shape (N,) or (T, N)

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    def zeroth_moment(self) -> float:
        dx = self.centers[1] - self.centers[0]
        return jnp.sum(self.f * dx)

    def first_moment(self) -> float:
        dx = self.centers[1] - self.centers[0]
        return jnp.sum(self.f * self.centers * dx)

    # ------------------------------------------------------------------
    # Diagnostics (outside the JAX tracers)
    # ------------------------------------------------------------------
    def plot(self, *, y_max: float | None = None, figsize: tuple[int, int] = (8, 4)) -> None:  # noqa: D401
        import matplotlib.pyplot as plt

        dx = float(self.centers[1] - self.centers[0])
        plt.figure(figsize=figsize)
        plt.bar(self.centers, self.f, width=dx, align="center", edgecolor="k")
        plt.xlabel("particle size x")
        plt.ylabel("density f(x)")
        plt.ylim(0.0, y_max or self.f.max() * 1.05)
        plt.title("Discrete particle density")
        plt.grid(True, axis="y", alpha=0.3)
        plt.show()
    
@struct.dataclass
class Trajectory:
    f: jax.Array  # shape (T, N)
    centers: jax.Array # shape (N,)
    times: jax.Array # shape (T,)

    # def __post_init__(self):
    #     # --- rank checks ---
    #     # chex.assert_rank(self.f, 2)
    #     # chex.assert_rank(self.centers, 1)
    #     # chex.assert_rank(self.times, 1)

    #     # --- axis-length match ---
    #     T, N = self.f.shape
    #     chex.assert_shape(self.centers, (N,))
    #     chex.assert_shape(self.times, (T - 1,))

    def get_state(self, idx: int) -> State:
        if (0 > idx) or ((idx >= self.times.shape[0]) and (idx != -1)):
            raise ValueError("Index out of bounds.")
        return State(f=self.f[idx, :], centers=self.centers)
    
    def get_last_state(self) -> State:
        return State(f=self.f[-1, :], centers=self.centers)

    def drop_initial_state(self) -> Trajectory:
        f = self.f[1 : ,...]
        times = self.times[1 : ,...]
        return Trajectory(f=f, centers=self.centers, times=times)

    def drop_final_state(self) -> Trajectory:
        f = self.f[:-1 ,...]
        times = self.times[:-1 ,...]
        return Trajectory(f=f, centers=self.centers, times=times)

    def reverse(self) -> Trajectory:
        f_rev = jnp.flip(self.f, axis=0)
        return Trajectory(f=f_rev, centers=self.centers, times=self.times)


@struct.dataclass
class VolumeElements:
    """Uniform or non‑uniform grid description (edges, centres, volumes)."""

    edges: jnp.ndarray    # shape (N+1,)
    centers: jnp.ndarray  # shape (N,)
    volumes: jnp.ndarray  # shape (N,)

    # ------------------------------------------------------------------
    # Factory & simple getters                                           
    # ------------------------------------------------------------------
    @classmethod
    def from_edges(cls, edges: jnp.ndarray) -> "VolumeElements":
        centres = 0.5 * (edges[:-1] + edges[1:])
        vols = edges[1:] - edges[:-1]
        return cls(edges=edges, centers=centres, volumes=vols)

    def get_centers(self) -> jnp.ndarray:  # noqa: D401
        return self.centers

    def get_volumes(self) -> jnp.ndarray:  # noqa: D401
        return self.volumes

    def get_edges(self) -> jnp.ndarray:  # noqa: D401
        return self.edges