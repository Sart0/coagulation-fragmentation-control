from functools import partial
from typing import Callable, Tuple

import jax
from jax import lax
import jax.numpy as jnp

from data_classes import State, Control, VolumeElements, Trajectory
from solver import FiniteVolumeSolver
from adjoint_solver import AdjointFiniteVolumeSolver

# -----------------------------------------------------------------------------
# Optimiser
# -----------------------------------------------------------------------------

class Optimizer:
    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        init_state: State,
        forward_solver: FiniteVolumeSolver,
        adjoint_solver: AdjointFiniteVolumeSolver,
        terminal_cost: Callable[[jnp.ndarray], jax.Array],
        running_cost: Callable[[Control, jnp.ndarray], jax.Array],
        terminal_cost_grad: Callable[[jax.Array], jax.Array],
        *,
        u_min: float,
        u_max: float,
        w_control: float = 1.0,
        armijo_rho: float = 0.5,
        armijo_sigma: float = 1.0e-2,
        max_ls_iter: int = 5,
        verbose: bool = False,
    ):
        # Store configuration -------------------------------------------------
        self.state0 = init_state
        self.solver = forward_solver
        self.adj_solver = adjoint_solver
        self.terminal_cost = terminal_cost
        self.running_cost = running_cost
        self.terminal_cost_grad = terminal_cost_grad

        self.t = forward_solver.times

        # Control bounds
        self.u_min, self.u_max = u_min, u_max

        # Spatial bookkeeping
        self.x = forward_solver.x
        self.dx = forward_solver.dx

        # Cost parameters
        self.w_control = w_control

        # Line‑search 
        self.armijo_rho, self.armijo_sigma, self.max_ls = armijo_rho, armijo_sigma, max_ls_iter

        self.verbose = verbose

    # ------------------------------------------------------------------
    # Loss function 
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def total_cost(self, f_traj: Trajectory, control: Control) -> float:
        """Weighted sum of terminal and running costs."""
        return self.terminal_cost(f_traj.get_last_state()) + self.running_cost(f_traj, control)


    @partial(jax.jit, static_argnums=0)
    def _loss(self, u_vec: jnp.ndarray) -> float:
        control = Control(jnp.clip(u_vec, self.u_min, self.u_max), self.t)
        state_T, traj, _ = self.solver.solve(initial_state=self.state0, control=control)
        return self.total_cost(traj, control)

    # ------------------------------------------------------------------
    # Backtracking line‑search (Armijo)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _armijo(
        self,
        u: jnp.ndarray,
        descent_dir: jnp.ndarray,
        loss: float,
        alpha0: float,
    ) -> Tuple[float, int]:
        """Return a step length *α* that satisfies the Armijo condition and the
        number of back‑tracking iterations performed."""

        def cond_fun(state):
            a, ls_iter = state
            u_new = u + a * descent_dir
            u_new_projected = jnp.clip(u_new, self.u_min, self.u_max)
            loss_new = self._loss(u_new_projected)
            squared_residual = jnp.sum((u_new_projected - u) ** 2 * self.solver.dt)
            armijo_fail = loss_new > loss - self.armijo_sigma / a * squared_residual
            return jnp.logical_and(armijo_fail, ls_iter < self.max_ls)

        def body_fun(state):
            a, ls_iter = state
            return a * self.armijo_rho, ls_iter + 1

        alpha_final, ls_steps = lax.while_loop(cond_fun, body_fun, (alpha0, 0))
        return alpha_final, ls_steps
    

    @partial(jax.jit, static_argnums=0)
    def _armijo_dir(
        self,
        u: jnp.ndarray,
        descent_dir: jnp.ndarray,
        loss: float,
        grad: jnp.ndarray,         # gradient at u
        alpha0: float,
    ) -> Tuple[float, int]:
        """
        Armijo backtracking for a generic descent direction with projection.
        Uses: f(u_new) <= f(u) + c1 * <grad, u_new - u>, where u_new = Proj(u + a d).
        """
        c1 = self.armijo_sigma  # e.g., 1e-4
        dt = jnp.asarray(self.solver.dt, dtype=u.dtype)

        def wdot(a, b):
            return jnp.sum(a * b) * dt

        def cond_fun(state):
            a, k = state
            u_new = u + a * descent_dir
            u_new = jnp.clip(u_new, self.u_min, self.u_max)
            loss_new = self._loss(u_new)

            # effective projected step direction (depends on a)
            step = u_new - u
            rhs  = loss + c1 * wdot(grad, step)
            armijo_fail = loss_new > rhs
            return jnp.logical_and(armijo_fail, k < self.max_ls)

        def body_fun(state):
            a, k = state
            return (a * self.armijo_rho, k + 1)  # 0<rho<1

        alpha_final, ls_steps = lax.while_loop(cond_fun, body_fun, (alpha0, 0))
        return alpha_final, ls_steps


    # ------------------------------------------------------------------
    # Adjoint helpers
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=0)
    def _loss_and_grad_adj(self, u_vec: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Return the loss and its gradient w.r.t. *u* using a discrete adjoint."""
        control = Control(jnp.clip(u_vec, self.u_min, self.u_max), self.t)
        state_T, f_traj, _ = self.solver.solve(initial_state=self.state0, control=control)
        # Integrate the adjoint backwards in time
        phi_T = self.terminal_cost_grad(state_T)
        phi_traj = self.adj_solver.integrate(f_traj, control, phi_T)

        rhs = jax.vmap(self.solver._coag, in_axes=(0, None))(f_traj.f, 1.0)
        grad_u = self.w_control * (u_vec - 1.0) + jnp.sum(rhs * phi_traj.f * self.dx, axis=1)

        loss_val = self.total_cost(f_traj, control)
        terminal_ = self.terminal_cost(state_T)
        return loss_val, terminal_, grad_u, f_traj, phi_traj
    
    # ---------------------------------------------------------------------------------
    # Projected gradient descent
    # ---------------------------------------------------------------------------------
        
    @partial(jax.jit, static_argnums=(0, 3, 4))
    def pgd_adjoint(
        self,
        u_init: jnp.ndarray,
        lr: float,
        n_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> Tuple[jnp.ndarray, float, float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Returns:
            u_opt, loss_opt, terminal_cost_opt, pmp_final,
            pmp_hist (n_iter,), res_hist (n_iter,), loss_hist (n_iter,), terminal_hist (n_iter,)
        """

        # time weights must match the control grid (length T)
        T = self.solver.times[-1]

        # init
        u = jnp.clip(u_init, self.u_min, self.u_max)
        loss, terminal, grad, f_traj, phi_traj = self._loss_and_grad_adj(u)
        delta_control_rel, delta_H_rel = self.PMP_check_(u, f_traj, phi_traj)
        delta_control0 = delta_control_rel
        delta_H0 = delta_H_rel
        k0     = jnp.array(1, dtype=jnp.int32)
        done0  = jnp.array(False)
        if self.verbose:
                jax.debug.print(
                    "iter 000,  delta_control_rel={P:.3e}, delta_H_rel={Pr:.3e} "
                    "||g||={gn:.3e}, J={val:.6e} "
                    "terminal_cost = {s}",
                    gn=jnp.sqrt(jnp.sum(grad**2) * self.solver.dt), val=loss,
                    s = terminal, P=delta_control0, Pr=delta_H0
                )

        def _do_step(carry):
            u_c, loss_c, terminal_c, grad_c, delta_control_c, delta_H_c, k_c, _done_c = carry

            d = - grad_c
            step, ls_it = self._armijo(u_c, d, loss_c, lr)

            u_next = jnp.clip(u_c + step * d, self.u_min, self.u_max)
            u_next_residual = jnp.clip(u_c + 1.0 * d, self.u_min, self.u_max)
            loss_next, terminal_next, grad_next, f_traj, phi_traj = self._loss_and_grad_adj(u_next)

            # residual & stopping
            res = jnp.sqrt(jnp.sum((u_next_residual - u_c) ** 2) * self.solver.dt) 
            done_next = res < tolerance

            # PMP diagnostic
            delta_control_rel, delta_H_rel = self.PMP_check_(u_next, f_traj, phi_traj)

            if self.verbose:
                jax.debug.print(
                    "iter {i:03d}, ls={it}, delta_control_rel={P:.3e}, delta_H_rel={Pr:.3e} "
                    "res={r:.3e}, ||g||={gn:.3e}, J={val:.6e} "
                    "terminal_cost = {s}",
                    i=k_c, it=ls_it, P=delta_control_rel, Pr=delta_H_rel, r=res,
                    gn=jnp.sqrt(jnp.sum(grad_c**2) * self.solver.dt), val=loss_next,
                    s = terminal_next
                )

            new_state = (u_next, loss_next, terminal_next, grad_next, delta_control_rel, delta_H_rel, k_c + 1, done_next)
            outs = (delta_control_rel, delta_H_rel, res, grad_next, loss_next, terminal_next)
            return new_state, outs

        def _skip_step(carry):
            u_c, loss_c, terminal_c, grad_c, delta_control_c, delta_H_c, k_c, done_c = carry
            new_state = (u_c, loss_c, terminal_c, grad_c, delta_control_c, delta_H_c, k_c + 1, done_c)
            outs = (delta_control_c, delta_H_c, 0.0, grad_c, loss_c, terminal_c)
            return new_state, outs

        def _body(state):
            return jax.lax.cond(state[-1], _skip_step, _do_step, state)

        init_state = (u, loss, terminal, grad, delta_control0, delta_H0, k0, done0)

        final_state, (delta_control_hist, delta_H_hist, res_hist, grad_hist, loss_hist, terminal_hist) = jax.lax.scan(
            lambda s, _: _body(s), init_state, xs=None, length=n_iter
        )

        u_opt, loss_opt, terminal_opt, grad_final, delta_control_final, delta_H_final, k_final, done_final = final_state

        delta_H_hist = jnp.concatenate([jnp.array([delta_H0]), delta_H_hist])
        delta_control_hist = jnp.concatenate([jnp.array([delta_control0]), delta_control_hist])

        return u_opt, loss_opt, terminal_opt, grad_hist, delta_control_hist, delta_H_hist, res_hist, loss_hist, terminal_hist, k_final

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def pcg_adjoint(
        self,
        u_init: jnp.ndarray,
        lr: float,
        n_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> Tuple[jnp.ndarray, float, float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Nonlinear Conjugate Gradient (Polak–Ribiere+) with Armijo line search.

        Returns (same order/shape as your PGD version):
            u_opt, loss_opt, terminal_cost_opt, grad_hist,
            pmp_final, pmp_hist (n_iter,), res_hist (n_iter,),
            loss_hist (n_iter,), terminal_hist (n_iter,)
        """
        # weights / grid
        dt = jnp.asarray(self.solver.dt, dtype=u_init.dtype)

        # helper: time-weighted inner product and norms (consistent with your prints)
        def wdot(a, b):
            return jnp.sum(a * b) * dt

        def wnorm(a):
            return jnp.sqrt(wdot(a, a))

        # init
        u = jnp.clip(u_init, self.u_min, self.u_max)
        loss, terminal, grad = self._loss_and_grad_adj(u)

        # initial direction is steepest descent
        d0 = -grad
        pmp0   = jnp.inf
        k0     = jnp.array(1, dtype=jnp.int32)
        done0  = jnp.array(False)

        if self.verbose:
            jax.debug.print(
                "iter 000,  ||g||={gn:.3e}, J={val:.6e} terminal_cost={s}",
                gn=wnorm(grad), val=loss, s=terminal
            )

        # CG parameters
        cg_restart_every = jnp.array(50, dtype=jnp.int32)  # periodic restart

        def _do_step(carry):
            u_c, loss_c, terminal_c, grad_c, dir_c, pmp_c, k_c, _done_c = carry

            # Armijo along current conjugate direction
            step, ls_it = self._armijo_dir(u_c, dir_c, loss_c, grad_c, lr)
            u_next  = jnp.clip(u_c + step * dir_c, self.u_min, self.u_max)

            # evaluate next loss/grad
            loss_next, terminal_next, grad_next = self._loss_and_grad_adj(u_next)

            # residual (matching your PGD diagnostic convention)
            u_next_residual = jnp.clip(u_c + 1.0 * dir_c, self.u_min, self.u_max)
            res = wnorm(u_next_residual - u_c)
            done_next = jnp.logical_or(res < tolerance, jnp.abs(loss_next - loss_c) < tolerance)

            # Polak–Ribiere+ beta (time-weighted)
            eps = jnp.asarray(1e-12, dtype=grad_c.dtype)
            yk  = grad_next - grad_c
            num = wdot(grad_next, yk)
            den = jnp.maximum(wdot(grad_c, grad_c), eps)
            beta_raw = num / den
            beta_prp = jnp.maximum(beta_raw, 0.0)  # PR+

            # candidate conjugate direction
            dir_candidate = -grad_next + beta_prp * dir_c

            # restart criteria: periodic OR loss of conjugacy (dir ⋅ g >= 0)
            periodic_restart = (k_c % cg_restart_every) == 0
            lost_conjugacy   = wdot(dir_candidate, grad_next) >= 0.0
            do_restart       = jnp.logical_or(periodic_restart, lost_conjugacy)

            dir_next = jnp.where(do_restart, -grad_next, dir_candidate)

            # PMP diagnostic
            delta_control_rel, delta_H_rel = self.PMP_check(u_next)

            if self.verbose:
                jax.debug.print(
                    "iter {i:03d}, ls={it}, PMP_rel={Pr:.3e}, H_rel={d: 3e} "
                    "res={res:.3e}, ||g||={gn:.3e}, J={val:.6e} terminal_cost={s}",
                    i=k_c, it=ls_it, Pr=delta_H_rel,
                    res=res, gn=wnorm(grad_next), val=loss_next, s=terminal_next, d=delta_control_rel
                )

            new_state = (u_next, loss_next, terminal_next, grad_next, dir_next, delta_H_rel, k_c + 1, done_next)
            outs = (delta_H_rel, res, grad_next, loss_next, terminal_next)
            return new_state, outs

        def _skip_step(carry):
            # Keep structure/order EXACTLY the same as _do_step
            u_c, loss_c, terminal_c, grad_c, dir_c, pmp_c, k_c, done_c = carry
            new_state = (u_c, loss_c, terminal_c, grad_c, dir_c, pmp_c, k_c + 1, done_c)
            outs = (pmp_c, 0.0, grad_c, loss_c, terminal_c)
            return new_state, outs

        def _body(state):
            # branch on 'done' (index 7)
            return jax.lax.cond(state[7], _skip_step, _do_step, state)

        init_state = (u, loss, terminal, grad, d0, pmp0, k0, done0)

        final_state, (pmp_hist, res_hist, grad_hist, loss_hist, terminal_hist) = jax.lax.scan(
            lambda s, _: _body(s), init_state, xs=None, length=n_iter
        )

        u_opt, loss_opt, terminal_opt, grad_final, dir_final, pmp_final, k_final, done_final = final_state

        return (
            u_opt,               # jnp.ndarray, optimal control
            loss_opt,            # float, final objective
            terminal_opt,        # float, final terminal cost
            grad_hist,           # (n_iter, ...), gradient history
            pmp_final,           # float, last PMP mismatch (L2)
            pmp_hist,            # (n_iter,), PMP mismatch history
            res_hist,            # (n_iter,), residual history
            loss_hist,           # (n_iter,), loss history
            terminal_hist,       # (n_iter,), terminal cost history
        )


    # ------------------------------------------------------------------
    #  PMP‑optimality check for an existing control
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def PMP_check(self, control_vec: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        control = Control(control_vec, self.t)
        state_T, f_traj, _ = self.solver.solve(initial_state=self.state0, control=control)
        phi_T = self.terminal_cost_grad(state_T)
        phi_traj = self.adj_solver.integrate(f_traj, control, phi_T)
        u_min  = self.u_min
        u_max  = self.u_max
        w_control  = self.w_control

        @jax.jit
        def _pmp_control(f, phi):
            S = jnp.sum(self.solver._coag(f, 1.0) * phi * self.dx)
            def _bang_bang():
                return jnp.where(S >= 0.0, u_min, u_max)
            def _quadratic():
                denom = jnp.maximum(w_control, 1e-12)
                return jnp.clip(1-S / denom, u_min, u_max)
            return lax.cond(w_control <= 1e-12, _bang_bang, _quadratic)
        # optimal control mismatch
        optimal_control = jax.vmap(_pmp_control)(f_traj.f, phi_traj.f)
        optimal_control_L2 = jnp.sum(optimal_control ** 2 * self.solver.dt) ** 0.5
        control_mismatches = optimal_control - control_vec
        control_mismatch_L2 = jnp.sum(control_mismatches ** 2 * self.solver.dt) ** 0.5
        delta_control_L2_rel = control_mismatch_L2 / optimal_control_L2
        # Hamiltonian mismatch
        H_values = jax.vmap(self.H)(f_traj.f, phi_traj.f, control_vec)
        optimal_H_values = jax.vmap(self.H)(f_traj.f, phi_traj.f, optimal_control)
        delta_H_L2 = jnp.sum((optimal_H_values - H_values) ** 2 * self.solver.dt)  ** 0.5
        optimal_H_L2 = jnp.sum(optimal_H_values ** 2 * self.solver.dt) ** 0.5
        delta_H_L2_rel = delta_H_L2 / optimal_H_L2
        return  delta_control_L2_rel, delta_H_L2_rel
    
    @partial(jax.jit, static_argnums=0)
    def PMP_check_(self, control_vec, f_traj, phi_traj) -> Tuple[jnp.ndarray, float]:
        u_min  = self.u_min
        u_max  = self.u_max
        w_control  = self.w_control

        @jax.jit
        def _pmp_control(f, phi):
            S = jnp.sum(self.solver._coag(f, 1.0) * phi * self.dx)
            def _bang_bang():
                return jnp.where(S >= 0.0, u_min, u_max)
            def _quadratic():
                denom = jnp.maximum(w_control, 1e-12)
                return jnp.clip(1-S / denom, u_min, u_max)
            return lax.cond(w_control <= 1e-12, _bang_bang, _quadratic)
        # optimal control mismatch
        optimal_control = jax.vmap(_pmp_control)(f_traj.f, phi_traj.f)
        optimal_control_L2 = jnp.sum(optimal_control ** 2 * self.solver.dt) ** 0.5
        control_mismatches = optimal_control - control_vec
        control_mismatch_L2 = jnp.sum(control_mismatches ** 2 * self.solver.dt) ** 0.5
        delta_control_L2_rel = control_mismatch_L2 / optimal_control_L2
        # Hamiltonian mismatch
        H_values = jax.vmap(self.H)(f_traj.f, phi_traj.f, control_vec)
        optimal_H_values = jax.vmap(self.H)(f_traj.f, phi_traj.f, optimal_control)
        delta_H_L2 = jnp.sum((optimal_H_values - H_values) ** 2 * self.solver.dt)  ** 0.5
        optimal_H_L2 = jnp.sum(optimal_H_values ** 2 * self.solver.dt) ** 0.5
        delta_H_L2_rel = delta_H_L2 / optimal_H_L2
        return  delta_control_L2_rel, delta_H_L2_rel

    @partial(jax.jit, static_argnums=0)
    def H(self, f: jax.Array, phi: jax.Array, u: jax.Array) -> Tuple[jnp.ndarray, float]:
        return 0.5 * self.w_control * (u - 1.0) ** 2 +  u * jnp.sum(self.solver._coag(f, 1.0) * phi * self.dx)