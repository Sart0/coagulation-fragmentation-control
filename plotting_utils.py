import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax
import jax.numpy as jnp
from typing import List
import os

from data_classes import State, Trajectory, Control

# ----------------------------------------------
# 1. Objective plot
# ----------------------------------------------

def plot_objective(loss_hist, 
                   terminal_hist,
                   no_control = None,
                   *,
                   title="Total Cost vs Iteration",
                   markevery=5,
                   name: str = "loss.png",
                   output_dir = None):
    K = len(loss_hist)
    k = jnp.arange(K)
    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    ax.plot(k, loss_hist, marker='o', markevery=markevery, color = 'red', linewidth=1.8, label="Total cost")
    ax.plot(k, terminal_hist, marker='s', markevery=markevery, color ='blue', linewidth=1.8, label="Terminal cost")
    ax.set_xlim(-1, K)

    if no_control is not None:
        no_control = jnp.ones_like(k) * no_control
        ax.plot(k, no_control, marker='s', markevery=markevery, linewidth=1.8, label="No control loss")

    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss"); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)

# ----------------------------------------------
# 2. Residuals plot
# ----------------------------------------------

def plot_residuals(pmp_hist, proj_hist, *,
                         title="Residuals vs iteration",
                         tol=None,
                         mark_every=5,
                         output_dir = None,
                         name = "residuals.png"):
    K = len(pmp_hist)
    k = jnp.arange(len(pmp_hist))
    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    ax.plot(k, pmp_hist, linewidth=1.8, marker='o', color = 'red',  markevery=mark_every, label="PMP residual")
    ax.plot(k, proj_hist, linewidth=1.8, marker='s', color = 'blue', markevery=mark_every, linestyle=":", label="Projection residual")
    ax.set_xlim(-1,K)
    
    if tol is not None:
        ax.axhline(tol, linestyle=":", linewidth=1.0, label=f"tol = {tol:g}")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)

def plot_all_residuals(delta_H, delta_control, res, *,
                         title="Residuals vs iteration",
                         tol=None,
                         mark_every=5,
                         output_dir = None,
                         name = "residuals.png"):
    K = len(delta_H)
    k = jnp.arange(K)
    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    ax.plot(k, delta_control, linewidth=1.8, marker='o', color = 'red',  markevery=mark_every, label="relatve control error")
    ax.plot(k, delta_H, linewidth=1.8, marker='s', color = 'blue', markevery=mark_every, linestyle=":", label="Relative Hamiltonian error")
    ax.plot(k, res, linewidth=1.8, marker='s', color = 'green', markevery=mark_every, linestyle=":", label="Projection residual")
    ax.set_xlim(-1,K)
    
    if tol is not None:
        ax.axhline(tol, linestyle=":", linewidth=1.0, label=f"tol = {tol:g}")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)
# ----------------------------------------------
# 2. Densities plot
# ----------------------------------------------

def plot_densities(
        state0: State,
        stateT: State,
        stateT_no_control: State,
        lower_bound: float,
        upper_bound: float,
        title: str = "Final densities",
        name: str = "densities.png",
        output_dir: str = None):
    
    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    xmin, xmax = state0.centers[0], state0.centers[-1]
    # --- target region -----------------------------------------------------------
    ax.axvspan(lower_bound, upper_bound, color='gold', alpha=0.18, label="target region")
    ax.vlines([lower_bound, upper_bound], ymin=0, ymax=1, transform=ax.get_xaxis_transform(),
            linestyles=":", linewidth=1.0)  # subtle boundaries

    # --- curves ------------------------------------------------------------------
    ax.plot(state0.centers, state0.f,                color='k',        lw=1.0, label="initial density")
    ax.plot(stateT_no_control.centers, stateT_no_control.f, color='tab:blue', lw=2.5, ls='--',
            label="final density — no control")
    ax.plot(stateT.centers, stateT.f,       color='tab:orange', lw=2.5,
            label="final density — optimal control")

    # --- Domain delimiters
    for xi in (xmin, xmax):
        ax.axvline(xi, linestyle='-', linewidth=1.2, color='0.6')   # neutral gray
    # ax.annotate(r"$x_{\min}$", (xmin, ax.get_ylim()[1]*0.97), ha='center', va='top', fontsize=10)
    # ax.annotate(r"$x_{\max}$", (xmax, ax.get_ylim()[1]*0.97), ha='center', va='top', fontsize=10)

    # --- Positivity baseline
    ax.axhline(0.0, linewidth=1.5, color = '0.6')                 # thicker x-axis at y=0
    ymax = max(state0.f.max(), stateT.f.max(), stateT_no_control.f.max())
    ax.set_ylim(-0.03*ymax, 1.05*ymax)  

    # --- labels / ticks / grid ---------------------------------------------------
    ax.set_xlabel("particle size $x$", fontsize=12)
    ax.set_ylabel(r"density $f(x,T)$", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.15)
    ax.minorticks_on()

    # Optional: keep consistent y-range across figures
    # ax.set_ylim(bottom=0)

    # --- legend: compact and readable -------------------------------------------
    ax.legend(frameon=True, framealpha=0.9, fontsize=10, loc='upper right')

    # --- light spines ------------------------------------------------------------
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Title is optional in papers; keep if you like
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)

def plot_multiple_densities(
        state0: State,
        terminal_states: List[State],
        labels: List[str],
        lower_bound: float,
        upper_bound: float,
        legend_position: str,
        y_min: float,
        y_max: float,
        fontsize: int = 10,
        title: str = "Final densities",
        name: str = "densities.png",
        output_dir: str = None):
    
    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    xmin, xmax = state0.centers[0], state0.centers[-1]
    terminal_values = jnp.array([state.f for state in terminal_states])
    ymax = jnp.max(terminal_values)
    ymax0 = state0.f.max()
    ymax = max(ymax, ymax0)
    # --- target region -----------------------------------------------------------
    ax.axvspan(lower_bound, upper_bound, color='gold', alpha=0.18, label="target region")
    ax.vlines([lower_bound, upper_bound], ymin=0, ymax=1, transform=ax.get_xaxis_transform(),
            linestyles=":", linewidth=1.0)  # subtle boundaries

    # --- curves ------------------------------------------------------------------
    ax.plot(state0.centers, state0.f, color='k', lw=1.0, label="initial density")
    for stateT, label in zip(terminal_states, labels):
        ax.plot(stateT.centers, stateT.f, lw=2.5, label=label)

    # --- Domain delimiters
    for xi in (xmin, xmax):
        ax.axvline(xi, linestyle='-', linewidth=1.2, color='0.6')   # neutral gray

    # --- Positivity baseline
    ax.axhline(0.0, linewidth=1.5, color = '0.6')                 # thicker x-axis at y=0
    ax.set_ylim(y_min, y_max)  

    # --- labels / ticks / grid ---------------------------------------------------
    ax.set_xlabel("particle size $x$", fontsize=12)
    ax.set_ylabel(r"density $f(x,T)$", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.15)
    ax.minorticks_on()

    # --- legend: compact and readable -------------------------------------------
    ax.legend(frameon=True, framealpha=0.9, fontsize=fontsize, loc=legend_position)

    # --- light spines ------------------------------------------------------------
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Title is optional in papers; keep if you like
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)

# ----------------------------------------------
# 2. Controls plot
# ----------------------------------------------

def plot_controls(
        control0: Control,
        optimal_control: Control,
        u_min: float,
        u_max: float,
        title: str = "Optimal control",
        name: str = "controls.png",
        output_dir: str = None):

    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    t_min, t_max = control0.times[0], control0.times[-1]

    # --- admissible band and bounds ---------------------------------------------
    ax.axhspan(u_min, u_max, alpha=0.20, color='lightgreen', label="admissible band")
    ax.axhline(u_min, ls=':', lw=1.0, color='k')
    ax.axhline(u_max, ls=':', lw=1.0, color='k')

    # --- controls ----------------------------------------------------------------
    # If your control is piecewise-constant on the time grid, a step plot is clearer:
    ax.step(control0.times, control0.values, where='post', lw=2.0, ls='--', color='tab:blue',  label="baseline (no control)")
    ax.step(optimal_control.times, optimal_control.values, where='post', lw=2.4,            color='tab:orange', label="optimal control")

    # --- Domain delimiters
    for xi in (t_min, t_max):
        ax.axvline(xi, linestyle='-', linewidth=1.2, color='0.6')   # neutral gray

    # --- labels / ticks / grid ---------------------------------------------------
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel("control amplitude $u(t)$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_ylim(u_min - 1.0, u_max + 1.0)

    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, fontsize=10, loc='upper left')

    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)

def plot_multiple_controls(
        controls: List[Control],
        labels: List[str],
        u_min: float,
        u_max: float,
        legend_position: str,
        y_min: float,
        y_max: float,
        fontsize: int = 10,
        title: str = "Optimal control",
        name: str = "controls.png",
        output_dir: str = None):

    fig, ax = plt.subplots(figsize=(6.2, 3.8)) 
    control0 = controls[-1]
    t_min, t_max = control0.times[0], control0.times[-1]

    # --- admissible band and bounds ---------------------------------------------
    ax.axhspan(u_min, u_max, alpha=0.20, color='lightgreen', label="admissible band")
    ax.axhline(u_min, ls=':', lw=1.0, color='k')
    ax.axhline(u_max, ls=':', lw=1.0, color='k')

    # --- controls ----------------------------------------------------------------
    for control, label in zip(controls, labels):
        ax.step(control.times, control.values, where='post', lw=2.0, label=label)

    # --- Domain delimiters
    for xi in (t_min, t_max):
        ax.axvline(xi, linestyle='-', linewidth=1.2, color='0.6')   # neutral gray

    # --- labels / ticks / grid ---------------------------------------------------
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel("control amplitude $u(t)$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, fontsize=fontsize, loc='upper left')

    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, name), dpi=300)


def animate_evolution(
    f_traj: Trajectory,
    interval=100,     
    repeat=True,
    ylim=None,
    line_kwargs=None,     
    title="Time evolution",
    x_label="x",
    y_label="f(t,x)",
    show_time=True,
    filename=None,      
    dpi=150,
):
    f = f_traj.f
    t = f_traj.times
    x = f_traj.centers
    T = t.shape[0]

    line_kwargs = {} if line_kwargs is None else dict(line_kwargs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (line,) = ax.plot(x, f[0], **line_kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    xlim = (x.min(), x.max())
    ax.set_xlim(*xlim)

    if ylim is None:
        mn, mx = float(jnp.min(f)), float(jnp.max(f))
        pad = 0.05 * (mx - mn if mx > mn else 1.0)
        ylim = (mn - pad, mx + pad)
    ax.set_ylim(*ylim)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top"
    ) if show_time else None

    def init():
        line.set_ydata(f[0])
        if time_text is not None:
            time_text.set_text(f"t = {t[0]:.3g}")
        return (line,) if time_text is None else (line, time_text)

    def update(i):
        line.set_ydata(f[i])
        if time_text is not None:
            time_text.set_text(f"t = {t[i]:.3g}")
        return (line,) if time_text is None else (line, time_text)

    anim = FuncAnimation(
        fig, update, frames=T, init_func=init, interval=interval, blit=True, repeat=repeat
    )

    if filename is not None:
        if filename.lower().endswith(".gif"):
            anim.save(filename, writer=PillowWriter(fps=max(1, int(1000 / interval))), dpi=dpi)
        elif filename.lower().endswith(".mp4"):
            # Requires ffmpeg installed in your environment
            anim.save(filename, writer="ffmpeg", dpi=dpi, fps=max(1, int(1000 / interval)))
        else:
            raise ValueError("filename must end with .gif or .mp4")

    return fig, anim

def animate_evolutions(
    trajs: list[Trajectory],
    interval=100,     
    repeat=True,
    ylim=None,
    line_kwargs=None,     
    title="Time evolution",
    x_label="x",
    y_label="f(t,x)",
    show_time=True,
    filename=None,      
    dpi=150,
):

    # --- centers and time steps check ---
    t_shape0 = trajs[0].times.shape
    f0_centers = trajs[0].centers

    for i, traj in enumerate(trajs):
        if not jnp.array_equal(traj.centers, f0_centers):
            raise ValueError(f"Trajectory {i} has inconsistent centers")
        if traj.times.shape != t_shape0:
            raise ValueError(f"Trajectory {i} has inconsistent times")

    # stack trajectories
    f_trajs = jnp.stack([traj.f for traj in trajs], axis=0)      # (n_traj, T, ...)

    x = f0_centers
    t = trajs[0].times
    T = t.shape[0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    xlim = (x.min(), x.max())
    ax.set_xlim(*xlim)

    if ylim is None:
        mn, mx = float(jnp.min(f_trajs)), float(jnp.max(f_trajs))
        pad = 0.05 * (mx - mn if mx > mn else 1.0)
        ylim = (mn - pad, mx + pad)
    ax.set_ylim(*ylim)

    for f_traj in trajs:
        f = f_traj.f

        line_kwargs = {} if line_kwargs is None else dict(line_kwargs)

        (line,) = ax.plot(x, f[0], **line_kwargs)

        time_text = ax.text(
            0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top"
        ) if show_time else None

        def init():
            line.set_ydata(f[0])
            if time_text is not None:
                time_text.set_text(f"t = {t[0]:.3g}")
            return (line,) if time_text is None else (line, time_text)

        def update(i):
            line.set_ydata(f[i])
            if time_text is not None:
                time_text.set_text(f"t = {t[i]:.3g}")
            return (line,) if time_text is None else (line, time_text)

        anim = FuncAnimation(
            fig, update, frames=T, init_func=init, interval=interval, blit=True, repeat=repeat
        )

    if filename is not None:
        if filename.lower().endswith(".gif"):
            anim.save(filename, writer=PillowWriter(fps=max(1, int(1000 / interval))), dpi=dpi)
        elif filename.lower().endswith(".mp4"):
            # Requires ffmpeg installed in your environment
            anim.save(filename, writer="ffmpeg", dpi=dpi, fps=max(1, int(1000 / interval)))
        else:
            raise ValueError("filename must end with .gif or .mp4")

    return fig, anim

def animate_trajectories(
    trajs,
    interval=100,
    repeat=True,
    ylim=None,
    line_kwargs=None,
    title="Time evolution",
    x_label="x",
    y_label="f(t,x)",
    show_time=True,
    filename=None,
    dpi=150,
):
    if len(trajs) == 0:
        raise ValueError("trajs must be non-empty")

    # --- consistency checks ---
    t_shape0 = trajs[0].times.shape
    centers0 = trajs[0].centers

    for i, traj in enumerate(trajs):
        # centers: compare arrays
        if traj.centers.shape != centers0.shape or not jnp.allclose(traj.centers, centers0):
            raise ValueError(f"Trajectory {i} has inconsistent centers")
        # times
        if traj.times.shape != t_shape0 or not jnp.allclose(traj.times, trajs[0].times):
            raise ValueError(f"Trajectory {i} has inconsistent times")

    # stack f: (n_traj, T, N_x)
    f_trajs = jnp.stack([traj.f for traj in trajs], axis=0)
    n_traj, T, N_x = f_trajs.shape

    x = centers0
    t = trajs[0].times

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(x.min()), float(x.max()))

    # auto-ylim if not provided
    if ylim is None:
        mn = float(f_trajs.min())
        mx = float(f_trajs.max())
        pad = 0.05 * (mx - mn if mx > mn else 1.0)
        ylim = (mn - pad, mx + pad)
    ax.set_ylim(*ylim)

    # create one line per trajectory
    # line_kwargs can be shared; user can later pass colors themselves
    base_kwargs = {} if line_kwargs is None else dict(line_kwargs)
    lines = []
    for k in range(n_traj):
        (line,) = ax.plot(x, f_trajs[k, 0], **base_kwargs)
        lines.append(line)

    # optional time text
    if show_time:
        time_text = ax.text(
            0.02, 0.95, f"t = {t[0]:.3g}", transform=ax.transAxes, ha="left", va="top"
        )
    else:
        time_text = None

    def init():
        for k in range(n_traj):
            lines[k].set_ydata(f_trajs[k, 0])
        if time_text is not None:
            time_text.set_text(f"t = {t[0]:.3g}")
        # return everything that changes
        return (*lines, time_text) if time_text is not None else tuple(lines)

    def update(i):
        for k in range(n_traj):
            lines[k].set_ydata(f_trajs[k, i])
        if time_text is not None:
            time_text.set_text(f"t = {t[i]:.3g}")
        return (*lines, time_text) if time_text is not None else tuple(lines)

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=repeat,
    )

    if filename is not None:
        fps = max(1, int(1000 / interval))
        if filename.lower().endswith(".gif"):
            anim.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
        elif filename.lower().endswith(".mp4"):
            anim.save(filename, writer="ffmpeg", dpi=dpi, fps=fps)
        else:
            raise ValueError("filename must end with .gif or .mp4")

    return fig, anim

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_evolutions(
    trajs,
    interval=100,
    repeat=True,
    ylim=None,
    line_kwargs=None,
    title="Time evolution",
    x_label="x",
    y_label="f(t,x)",
    show_time=True,
    filename=None,             # just the file name, e.g. "evo.gif"
    dpi=150,
    labels=None,               # <- list of strings, len = len(trajs)
    stripe_range=None,         # <- (x_min, x_max) in data coords
    stripe_kwargs=None,        # <- dict for the stripe style
    output_dir=None,           # <- NEW: directory where to save
):
    if len(trajs) == 0:
        raise ValueError("trajs must be non-empty")

    # --- consistency checks ---
    t0 = trajs[0].times
    x0 = trajs[0].centers
    for i, traj in enumerate(trajs):
        if traj.times.shape != t0.shape or not jnp.allclose(traj.times, t0):
            raise ValueError(f"Trajectory {i} has inconsistent times")
        if traj.centers.shape != x0.shape or not jnp.allclose(traj.centers, x0):
            raise ValueError(f"Trajectory {i} has inconsistent centers")

    # stack f: (n_traj, T, N_x)
    f_trajs = jnp.stack([traj.f for traj in trajs], axis=0)
    n_traj, T, N_x = f_trajs.shape

    x = np.array(x0)
    t = np.array(t0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(x.min()), float(x.max()))

    # --- fixed vertical stripe in the background ---
    if stripe_range is not None:
        sx0, sx1 = stripe_range
        stripe_style = dict(color="gold", alpha=0.15, zorder=0)
        if stripe_kwargs is not None:
            stripe_style.update(stripe_kwargs)
        ax.axvspan(sx0, sx1, **stripe_style)

    # auto-ylim
    if ylim is None:
        mn = float(f_trajs.min())
        mx = float(f_trajs.max())
        pad = 0.05 * (mx - mn if mx > mn else 1.0)
        ylim = (mn - pad, mx + pad)
    ax.set_ylim(*ylim)

    base_kwargs = {} if line_kwargs is None else dict(line_kwargs)

    lines = []
    for k in range(n_traj):
        (line,) = ax.plot(x, np.array(f_trajs[k, 0]), **base_kwargs)
        lines.append(line)

    if labels is not None:
        if len(labels) != n_traj:
            raise ValueError("labels must have the same length as trajs")
        for line, lab in zip(lines, labels):
            line.set_label(lab)
        ax.legend(loc="upper right")

    time_text = None
    if show_time:
        time_text = ax.text(
            0.02, 0.95, f"t = {t[0]:.3g}", transform=ax.transAxes,
            ha="left", va="top"
        )

    def init():
        for k in range(n_traj):
            lines[k].set_ydata(np.array(f_trajs[k, 0]))
        if time_text is not None:
            time_text.set_text(f"t = {t[0]:.3g}")
        return lines if time_text is None else (*lines, time_text)

    def update(i):
        for k in range(n_traj):
            lines[k].set_ydata(np.array(f_trajs[k, i]))
        if time_text is not None:
            time_text.set_text(f"t = {t[i]:.3g}")
        return lines if time_text is None else (*lines, time_text)

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    # --- saving ---
    if filename is not None:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
        else:
            filepath = Path(filename)

        fps = max(1, int(1000 / interval))
        suffix = filepath.suffix.lower()
        if suffix == ".gif":
            anim.save(filepath, writer=PillowWriter(fps=fps), dpi=dpi)
        elif suffix == ".mp4":
            anim.save(filepath, writer="ffmpeg", dpi=dpi, fps=fps)
        else:
            raise ValueError("filename must end with .gif or .mp4")

    return fig, anim