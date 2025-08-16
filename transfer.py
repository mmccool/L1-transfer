"""
Sun–Earth L1 Shadowed Transfer Planner — Planar (x,y)
=====================================================

Plan a **planar (x,y)** transfer in the Sun–Earth CR3BP synodic frame from a
near‑Earth circular orbit (configurable altitude) to the Sun–Earth L1 region,
while staying inside the shadow corridor of a planetary‑scale solar shade at L1.
The vehicle performs a two‑burn profile with **smoothly ramped thrust** bounded
by a **max g‑limit** (default 3 g):

    accelerate (ramped to |a|max, direction θ₁) → coast → decelerate (ramped, direction θ₂)

You specify **time of flight**, **g‑limit**, **ramp fractions**, start altitude,
and arrival offset (stop just short of L1). The solver coarsely searches for
burn durations and **burn directions** (θ₁, θ₂ in the synodic plane) that meet
terminal constraints **[x≈x* , y≈0 , ẋ≈0 , ẏ≈0]** and penalizes excursions
outside the shadow corridor. Departure is chosen on the **midnight line**
(shadow‑compliant) which is the fuel‑optimal admissible departure in this 2‑D
model (ignoring the Moon as requested).

This is an educational trade‑study tool, not flight software.

Quick start
-----------
>>> from sun_earth_L1_shadow_transfer import ShadowTransferPlanner
>>> planner = ShadowTransferPlanner(
...     tof_s=6*24*3600,
...     g_limit_g=3.0,
...     alt0_m=600e3,
...     arrival_offset_m=20_000e3,
...     ramp_up_frac=0.1, ramp_down_frac=0.1,
...     angle_deg_max=10.0, angle_steps=9,
... )
>>> res = planner.plan()
>>> print(res.summary())
>>> # res.plot()     # optional matplotlib plots

API surface
----------
- ShadowTransferPlanner: main configuration + solver
  - plan() → TransferResult: searches (t₁,t₂,θ₁,θ₂)
  - simulate_given(t1_s, t2_s, theta1_deg=None, theta2_deg=None) → TransferResult
  - sweep(n=...) → list[TransferResult] (symmetric burns; uses θ₁=−θ₂=0° by default)
  - find_departure_windows() → list[(start_s, end_s, center_s)]
- TransferResult
  - summary(), plot(), export_csv(path)

Key assumptions / limitations
-----------------------------
- Planar CR3BP (x,y) in the synodic frame; Moon, SRP, non‑circularity ignored.
- Thrust magnitude uses **raised‑cosine ramps**; direction is constant during
  each burn but can differ between burns.
- LEO departure is reduced to **phase selection** (midnight line) and the model
  starts with synodic velocities ≈ 0; the cost to create that state is not
  included here.
- Shadow corridor is a cylinder of radius `corridor_radius_m` around the
  Sun–Earth line; penalty applied if |y| exceeds it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import numpy as np

# --------------------------- Physical constants ------------------------------
G = 6.67430e-11  # m^3 / (kg s^2)
M_SUN = 1.98847e30  # kg
M_EARTH = 5.9722e24  # kg
MU_SUN = G * M_SUN
MU_EARTH = G * M_EARTH
AU = 149_597_870_700.0  # m (IAU 2012)
R_EARTH = 6_371_000.0  # m
G0 = 9.80665  # m/s^2

# ----------------------------- Helper math -----------------------------------

def mean_motion(L: float = AU) -> float:
    """Return synodic-frame mean motion ω = sqrt(G(Ms+Me)/L^3) [rad/s]."""
    return math.sqrt((MU_SUN + MU_EARTH) / L**3)


def mass_parameter() -> float:
    """Return CR3BP mass ratio μ = μ_E/(μ_S+μ_E)."""
    return MU_EARTH / (MU_SUN + MU_EARTH)


def nondim_scales(L: float = AU) -> Tuple[float, float]:
    """Return (time_unit, accel_unit) for CR3BP nondimensionalization.

    time_unit = 1/ω  [s];  accel_unit = L * ω^2  [m/s^2].
    """
    w = mean_motion(L)
    time_unit = 1.0 / w
    accel_unit = L * (w**2)
    return time_unit, accel_unit


# ------------------------ CR3BP planar dynamics ------------------------------

def dU_dx_dy(x: float, y: float, mu: float) -> Tuple[float, float]:
    """Return ∂Ω/∂x and ∂Ω/∂y for planar CR3BP (nondimensional)."""
    rx1 = x + mu
    rx2 = x - (1 - mu)
    r1 = math.sqrt(rx1 * rx1 + y * y)
    r2 = math.sqrt(rx2 * rx2 + y * y)
    dUx = x - (1 - mu) * rx1 / (r1**3) - mu * rx2 / (r2**3)
    dUy = y - (1 - mu) * y / (r1**3) - mu * y / (r2**3)
    return dUx, dUy


def rk4_step_planar(state: np.ndarray, dt: float, mu: float, a_ctrl: Tuple[float, float]) -> np.ndarray:
    """One RK4 step for planar state s=[x,y,xd,yd] with control accel (ax,ay)."""
    ax_ctrl, ay_ctrl = a_ctrl

    def deriv(s: np.ndarray) -> np.ndarray:
        x, y, xd, yd = s
        dUx, dUy = dU_dx_dy(x, y, mu)
        xdd = 2*yd + dUx + ax_ctrl
        ydd = -2*xd + dUy + ay_ctrl
        return np.array([xd, yd, xdd, ydd])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------------------ Results type ---------------------------------

@dataclass
class TransferResult:
    ok: bool
    message: str
    # burn durations (seconds)
    t1_s: float
    t2_s: float
    tf_s: float
    # thrust accel (m/s^2)
    a_burn_mps2: float
    # 2D states over time (nondimensional)
    t_nd: np.ndarray
    x_nd: np.ndarray
    y_nd: np.ndarray
    xd_nd: np.ndarray
    yd_nd: np.ndarray
    # conversions & context
    L_m: float
    time_unit_s: float
    accel_unit_mps2: float
    mu: float
    x0_nd: float
    y0_nd: float
    xL1_nd: float
    x_target_nd: float
    # energy-ish quantities
    delta_v_mps: float
    # departure info
    t_depart_s: float
    depart_phase_deg: float
    # path metrics
    max_offaxis_m: float

    def summary(self) -> str:
        if not self.ok:
            return f"Solver failed: {self.message}"
        lines = []
        lines.append("Shadowed L1 transfer (planar CR3BP, ramped two-burn)")
        lines.append(f"  Time of flight: {self.tf_s/3600:.2f} h")
        lines.append(f"  Burns (s): t1={self.t1_s:.1f}, coast={self.tf_s-self.t1_s-self.t2_s:.1f}, t2={self.t2_s:.1f}")
        lines.append(f"  Burn g-limit: {self.a_burn_mps2/G0:.3f} g ({self.a_burn_mps2:.4f} m/s^2)")
        lines.append(f"  Δv (thrust-only, with ramps): {self.delta_v_mps:.1f} m/s")
        x_final_m = self.x_nd[-1] * self.L_m
        xL1_m = self.xL1_nd * self.L_m
        y_final_m = self.y_nd[-1] * self.L_m
        lines.append(f"  Final x from L1: {(x_final_m - xL1_m)/1e3:.1f} km; y={y_final_m/1e3:.2f} km")
        lines.append(f"  Final (ẋ,ẏ): ({self.xd_nd[-1]*self.L_m/self.time_unit_s:.4f}, {self.yd_nd[-1]*self.L_m/self.time_unit_s:.4f}) m/s")
        lines.append(f"  Max off-axis |y|: {self.max_offaxis_m/1e3:.2f} km")
        return "
".join(lines)

    # --- optional visualization ---
    def plot(self):  # pragma: no cover
        import matplotlib.pyplot as plt
        t_s = self.t_nd * self.time_unit_s
        x_m = self.x_nd * self.L_m
        y_m = self.y_nd * self.L_m
        vx = self.xd_nd * self.L_m / self.time_unit_s
        vy = self.yd_nd * self.L_m / self.time_unit_s

        # 1) Trajectory in (x,y)
        fig1 = plt.figure()
        plt.plot((x_m - self.xL1_nd*self.L_m)/1e6, y_m/1e6)
        plt.axvline(0, linestyle='--')
        plt.xlabel('x - x_L1  [Mm]')
        plt.ylabel('y  [Mm]')
        plt.title('Planar trajectory relative to L1 (Mm)')
        plt.grid(True)

        # 2) Velocities vs time
        fig2 = plt.figure()
        plt.plot(t_s/3600, vx, label='ẋ')
        plt.plot(t_s/3600, vy, label='ẏ')
        plt.axhline(0, linestyle='--')
        plt.xlabel('Time [h]')
        plt.ylabel('Velocity [m/s] (synodic)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def export_csv(self, path: str):
        t_s = self.t_nd * self.time_unit_s
        x_m = self.x_nd * self.L_m
        y_m = self.y_nd * self.L_m
        vx = self.xd_nd * self.L_m / self.time_unit_s
        vy = self.yd_nd * self.L_m / self.time_unit_s
        data = np.column_stack([t_s, x_m, y_m, vx, vy])
        header = 't_s,x_m,y_m,xdot_mps,ydot_mps (synodic)'
        np.savetxt(path, data, delimiter=',', header=header)


# --------------------------- Planner / solver --------------------------------

@dataclass
class ShadowTransferPlanner:
    """Configure and solve a shadow-constrained L1 transfer in planar CR3BP.

    Parameters
    ----------
    tof_s : float
        Total time of flight [s].
    g_limit_g : float, optional
        Maximum thrust g-load per burn [g]. If provided, overrides g_limit_mps2.
    g_limit_mps2 : float, optional
        Maximum thrust acceleration [m/s^2]. Used only if g_limit_g is None.
    alt0_m : float, default 400e3
        Initial Earth-centered circular-orbit altitude [m].
    arrival_offset_m : float, default 10e6
        Target x is x_L1 + arrival_offset toward Earth [m] (stop short of L1).
    corridor_radius_m : float, default 20e6
        Radius of the shadow corridor [m] around the Sun–Earth line.
    L_m : float, default AU
        Sun–Earth distance used for nondimensionalization [m].
    dt_s : float, default 60.0
        Integrator step in seconds.
    tolerances : tuple[float, float, float, float], default (5e6, 5e6, 0.02, 0.02)
        Terminal tolerances: (|x−x*| [m], |y| [m], |ẋ| [m/s], |ẏ| [m/s]).
    ramp_up_frac : float, default 0.1
        Fraction of each burn spent ramping up (raised‑cosine).
    ramp_down_frac : float, default 0.1
        Fraction of each burn spent ramping down (raised‑cosine).
    angle_deg_max : float, default 10.0
        Half-range for burn-direction angle search (degrees).
    angle_steps : int, default 9
        Number of samples for θ₁ and θ₂ in the grid search.
    corridor_penalty_weight : float, default 25.0
        Weight for normalized corridor violation in the cost function.
    """

    tof_s: float
    g_limit_g: Optional[float] = 3.0
    g_limit_mps2: Optional[float] = None
    alt0_m: float = 400e3
    arrival_offset_m: float = 10e6
    corridor_radius_m: float = 20e6
    L_m: float = AU
    dt_s: float = 60.0
    tolerances: Tuple[float, float, float, float] = (5e6, 5e6, 0.02, 0.02)
    ramp_up_frac: float = 0.1
    ramp_down_frac: float = 0.1
    angle_deg_max: float = 10.0
    angle_steps: int = 9
    corridor_penalty_weight: float = 25.0

    # --- derived (set in __post_init__) ---
    mu: float = None  # type: ignore
    time_unit_s: float = None  # type: ignore
    accel_unit_mps2: float = None  # type: ignore
    x_earth_nd: float = None  # type: ignore
    x0_nd: float = None  # type: ignore
    y0_nd: float = 0.0
    xL1_nd: float = None  # type: ignore
    x_target_nd: float = None  # type: ignore
    a_burn_nd: float = None  # type: ignore
    a_burn_mps2: float = None  # type: ignore

    def __post_init__(self):
        self.mu = mass_parameter()
        self.time_unit_s, self.accel_unit_mps2 = nondim_scales(self.L_m)
        self.x_earth_nd = 1.0 - self.mu
        # Departure: midnight-line point of a circular LEO
        r0 = R_EARTH + self.alt0_m
        self.x0_nd = self.x_earth_nd - r0 / self.L_m
        self.y0_nd = 0.0
        self.xL1_nd = l1_location_x(self.mu)
        self.x_target_nd = self.xL1_nd + self.arrival_offset_m / self.L_m
        a_mps2 = self.g_limit_mps2 if self.g_limit_mps2 is not None else (self.g_limit_g or 0.0) * G0
        self.a_burn_mps2 = float(a_mps2)
        self.a_burn_nd = self.a_burn_mps2 / self.accel_unit_mps2

    # -------------------- integration and controls --------------------
    def _thrust_scale(self, local_t: float, T: float) -> float:
        """Raised‑cosine up/down; return scale in [0,1]."""
        up = max(1e-9, self.ramp_up_frac * T)
        dn = max(1e-9, self.ramp_down_frac * T)
        if local_t < 0:
            return 0.0
        if local_t < up:
            return 0.5 * (1 - math.cos(math.pi * (local_t / up)))
        if local_t <= T - dn:
            return 1.0
        if local_t <= T:
            tau = (local_t - (T - dn)) / dn
            return 0.5 * (1 + math.cos(math.pi * tau))
        return 0.0

    def _control_vec(self, t_nd: float, t1_nd: float, t2_nd: float, tf_nd: float, th1: float, th2: float) -> Tuple[float, float]:
        """Return (ax, ay) in nondimensional units.

        First burn aims roughly sunward; second burn aims roughly Earthward.
        `th1` and `th2` are in **radians**, measured from the nominal axis
        (−x for burn 1, +x for burn 2). Positive angle rotates toward +y.
        """
        if t_nd < t1_nd:
            s = self._thrust_scale(t_nd, t1_nd)
            # Nominal axis is −x
            ux = -math.cos(th1)
            uy = +math.sin(th1)
            return (self.a_burn_nd * s * ux, self.a_burn_nd * s * uy)
        elif t_nd <= tf_nd - t2_nd:
            return (0.0, 0.0)
        else:
            local_t = t_nd - (tf_nd - t2_nd)
            s = self._thrust_scale(local_t, t2_nd)
            # Nominal axis is +x
            ux = +math.cos(th2)
            uy = +math.sin(th2)
            return (self.a_burn_nd * s * ux, self.a_burn_nd * s * uy)

    def _integrate(self, t1_nd: float, t2_nd: float, tf_nd: float, th1: float, th2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        dt_nd = self.dt_s / self.time_unit_s
        n_steps = max(2, int(math.ceil(tf_nd / dt_nd)))
        dt_nd = tf_nd / n_steps  # exact
        t_nd = np.linspace(0.0, tf_nd, n_steps + 1)
        state = np.array([self.x0_nd, self.y0_nd, 0.0, 0.0])
        x_hist = np.empty(n_steps + 1)
        y_hist = np.empty(n_steps + 1)
        xd_hist = np.empty(n_steps + 1)
        yd_hist = np.empty(n_steps + 1)
        max_offaxis = 0.0
        for k, tk in enumerate(t_nd):
            x_hist[k], y_hist[k], xd_hist[k], yd_hist[k] = state
            max_offaxis = max(max_offaxis, abs(y_hist[k]) * self.L_m)
            if k == n_steps:
                break
            a_ctrl = self._control_vec(tk, t1_nd, t2_nd, tf_nd, th1, th2)
            state = rk4_step_planar(state, dt_nd, self.mu, a_ctrl)
        return t_nd, x_hist, y_hist, xd_hist, yd_hist, max_offaxis

    # ------------------------ objective / solver -----------------------
    def _residuals(self, t1_nd: float, t2_nd: float, tf_nd: float, th1: float, th2: float) -> Tuple[float, float, float, float, float]:
        t_nd, x_nd, y_nd, xd_nd, yd_nd, max_offaxis = self._integrate(t1_nd, t2_nd, tf_nd, th1, th2)
        r_x = (x_nd[-1] - self.x_target_nd) * self.L_m  # m
        r_y = y_nd[-1] * self.L_m  # m
        r_vx = xd_nd[-1] * self.L_m / self.time_unit_s  # m/s
        r_vy = yd_nd[-1] * self.L_m / self.time_unit_s  # m/s
        # Corridor violation (m, ≥0)
        viol = max(0.0, max_offaxis - self.corridor_radius_m)
        return r_x, r_y, r_vx, r_vy, viol

    def _cost(self, t1_nd: float, t2_nd: float, tf_nd: float, th1: float, th2: float) -> float:
        r_x, r_y, r_vx, r_vy, viol = self._residuals(t1_nd, t2_nd, tf_nd, th1, th2)
        sx, sy, svx, svy = self.tolerances
        J_term = (r_x / sx) ** 2 + (r_y / sy) ** 2 + (r_vx / svx) ** 2 + (r_vy / svy) ** 2
        J_corr = self.corridor_penalty_weight * (viol / max(1.0, self.corridor_radius_m)) ** 2
        return J_term + J_corr

    def _grid_search(self, tf_nd: float, levels: int = 2, grid_t: Tuple[int, int] = (13, 13)) -> Tuple[float, float, float, float, float]:
        """Hierarchical grid search over (t1, t2, θ1, θ2).

        Returns best (t1_nd, t2_nd, th1_rad, th2_rad, Jmin).
        """
        g1, g2 = grid_t
        th_vals = np.deg2rad(np.linspace(-self.angle_deg_max, self.angle_deg_max, self.angle_steps))
        bounds_t = [ (0.0, 0.6*tf_nd), (0.0, 0.6*tf_nd) ]
        best = (0.0, 0.0, 0.0, 0.0, float('inf'))
        for _ in range(levels):
            t1_vals = np.linspace(bounds_t[0][0], bounds_t[0][1], g1)
            t2_vals = np.linspace(bounds_t[1][0], bounds_t[1][1], g2)
            for t1 in t1_vals:
                for t2 in t2_vals:
                    if t1 + t2 > 0.95 * tf_nd:
                        continue
                    for th1 in th_vals:
                        for th2 in th_vals:
                            J = self._cost(t1, t2, tf_nd, th1, th2)
                            if J < best[4]:
                                best = (t1, t2, th1, th2, J)
            # refine around best t1,t2
            (t1b, t2b, _, _, _) = best
            span1 = max(1e-6, 0.25*(bounds_t[0][1]-bounds_t[0][0]))
            span2 = max(1e-6, 0.25*(bounds_t[1][1]-bounds_t[1][0]))
            bounds_t = [ (max(0.0, t1b-span1), min(0.6*tf_nd, t1b+span1)),
                         (max(0.0, t2b-span2), min(0.6*tf_nd, t2b+span2)) ]
        return best

    # ----------------------- Departure phasing utilities ----------------------
    def find_departure_windows(self) -> List[Tuple[float, float, float]]:
        """Return shadow-compliant departure windows for a circular LEO.

        Each window is (start_s, end_s, center_s) relative to an arbitrary epoch
        where the midnight line is at phase 0. The corridor is a cylinder of
        radius `corridor_radius_m` about the Sun–Earth line, so the half-angle
        at orbital radius r0 is θ_c = arctan(R_corr / r0). Crossings repeat every
        half-orbit; we return the next two.
        """
        r0 = R_EARTH + self.alt0_m
        n = math.sqrt(MU_EARTH / r0**3)  # rad/s around Earth
        theta_c = math.atan2(self.corridor_radius_m, r0)  # radians
        windows = []
        for k in range(2):
            center_phase = k * math.pi  # midnights
            start = (center_phase - theta_c) / n
            end = (center_phase + theta_c) / n
            center = center_phase / n
            windows.append((start, end, center))
        return windows

    def _choose_departure(self) -> Tuple[float, float]:
        """Pick the fuel-optimal admissible departure (window center).

        For the shadow-constrained planar model, the midnight-line center is
        optimal w.r.t. lateral steering (within the model’s assumptions).
        Returns (t_depart_s, phase_deg). We set t=0 at the chosen center.
        """
        _start_s, _end_s, _center_s = self.find_departure_windows()[0]
        return 0.0, 0.0

    # ------------------------------- public -----------------------------------
    def plan(self) -> TransferResult:
        """Solve for departure phasing and burn schedule (t₁,t₂,θ₁,θ₂)."""
        t_dep_s, phase_deg = self._choose_departure()

        tf_nd = self.tof_s / self.time_unit_s
        t1_nd, t2_nd, th1, th2, _J = self._grid_search(tf_nd)
        t_nd, x_nd, y_nd, xd_nd, yd_nd, max_offaxis = self._integrate(t1_nd, t2_nd, tf_nd, th1, th2)
        r_x, r_y, r_vx, r_vy, viol = self._residuals(t1_nd, t2_nd, tf_nd, th1, th2)
        ok = (
            abs(r_x) <= self.tolerances[0]
            and abs(r_y) <= self.tolerances[1]
            and abs(r_vx) <= self.tolerances[2]
            and abs(r_vy) <= self.tolerances[3]
            and viol <= 1.0  # up to 1 m tolerance beyond corridor
        )
        msg = "ok" if ok else (
            f"terminal miss: |x−x*|={abs(r_x):.3e} m, |y|={abs(r_y):.3e} m, |v|={(abs(r_vx)+abs(r_vy)):.3e} m/s; corridor viol={viol:.3e} m"
        )
        # Δv = ∫|a(t)| dt with ramps
        dt_nd = (t_nd[1]-t_nd[0]) if len(t_nd) > 1 else 0.0
        dv_nd = 0.0
        for tk in t_nd[:-1]:
            ax_nd, ay_nd = self._control_vec(tk, t1_nd, t2_nd, tf_nd, th1, th2)
            a_mag = math.hypot(ax_nd, ay_nd)
            dv_nd += a_mag * dt_nd
        delta_v = dv_nd * self.accel_unit_mps2 * self.time_unit_s

        return TransferResult(
            ok=ok,
            message=msg,
            t1_s=t1_nd * self.time_unit_s,
            t2_s=t2_nd * self.time_unit_s,
            tf_s=self.tof_s,
            a_burn_mps2=self.a_burn_mps2,
            t_nd=t_nd,
            x_nd=x_nd,
            y_nd=y_nd,
            xd_nd=xd_nd,
            yd_nd=yd_nd,
            L_m=self.L_m,
            time_unit_s=self.time_unit_s,
            accel_unit_mps2=self.accel_unit_mps2,
            mu=self.mu,
            x0_nd=self.x0_nd,
            y0_nd=self.y0_nd,
            xL1_nd=self.xL1_nd,
            x_target_nd=self.x_target_nd,
            delta_v_mps=delta_v,
            t_depart_s=t_dep_s,
            depart_phase_deg=phase_deg,
            max_offaxis_m=max_offaxis,
        )

    def simulate_given(self, t1_s: float, t2_s: float, theta1_deg: Optional[float] = None, theta2_deg: Optional[float] = None) -> TransferResult:
        """Integrate with user-selected burns and (optionally) directions.

        If theta* are None, use 0° (pure ±x thrust).
        """
        th1 = math.radians(theta1_deg if theta1_deg is not None else 0.0)
        th2 = math.radians(theta2_deg if theta2_deg is not None else 0.0)
        t_dep_s, phase_deg = self._choose_departure()
        tf_nd = self.tof_s / self.time_unit_s
        t1_nd = t1_s / self.time_unit_s
        t2_nd = t2_s / self.time_unit_s
        t_nd, x_nd, y_nd, xd_nd, yd_nd, max_offaxis = self._integrate(t1_nd, t2_nd, tf_nd, th1, th2)
        r_x, r_y, r_vx, r_vy, viol = self._residuals(t1_nd, t2_nd, tf_nd, th1, th2)
        ok = (
            abs(r_x) <= self.tolerances[0]
            and abs(r_y) <= self.tolerances[1]
            and abs(r_vx) <= self.tolerances[2]
            and abs(r_vy) <= self.tolerances[3]
            and viol <= 1.0
        )
        msg = "ok" if ok else (
            f"terminal miss: |x−x*|={abs(r_x):.3e} m, |y|={abs(r_y):.3e} m, |v|={(abs(r_vx)+abs(r_vy)):.3e} m/s; corridor viol={viol:.3e} m"
        )
        # Δv with ramps
        dt_nd = (t_nd[1]-t_nd[0]) if len(t_nd) > 1 else 0.0
        dv_nd = 0.0
        for tk in t_nd[:-1]:
            ax_nd, ay_nd = self._control_vec(tk, t1_nd, t2_nd, tf_nd, th1, th2)
            dv_nd += math.hypot(ax_nd, ay_nd) * dt_nd
        delta_v = dv_nd * self.accel_unit_mps2 * self.time_unit_s

        return TransferResult(
            ok=ok,
            message=msg,
            t1_s=t1_s,
            t2_s=t2_s,
            tf_s=self.tof_s,
            a_burn_mps2=self.a_burn_mps2,
            t_nd=t_nd,
            x_nd=x_nd,
            y_nd=y_nd,
            xd_nd=xd_nd,
            yd_nd=yd_nd,
            L_m=self.L_m,
            time_unit_s=self.time_unit_s,
            accel_unit_mps2=self.accel_unit_mps2,
            mu=self.mu,
            x0_nd=self.x0_nd,
            y0_nd=self.y0_nd,
            xL1_nd=self.xL1_nd,
            x_target_nd=self.x_target_nd,
            delta_v_mps=delta_v,
            t_depart_s=t_dep_s,
            depart_phase_deg=phase_deg,
            max_offaxis_m=max_offaxis,
        )

    def sweep(self, n: int = 6) -> List[TransferResult]:
        """Coarsely explore symmetric burns across a range of durations.

        Uses θ₁=−θ₂=0° (pure ±x) for the sweep to keep it simple.
        """
        tf_nd = self.tof_s / self.time_unit_s
        t_vals = np.linspace(0.02*tf_nd, 0.4*tf_nd, n)
        out = []
        for t in t_vals:
            out.append(self.simulate_given(t1_s=t*self.time_unit_s, t2_s=t*self.time_unit_s, theta1_deg=0.0, theta2_deg=0.0))
        return out

# ------------------------------- CLI demo ------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Sun–Earth L1 shadow transfer (planar CR3BP)")
    p.add_argument('--tof-h', type=float, default=144.0, help='Time of flight in hours (default: 144 h)')
    p.add_argument('--g', type=float, default=3.0, help='Burn g-limit in g (default: 3 g)')
    p.add_argument('--alt-km', type=float, default=400.0, help='Start altitude in km (default: 400)')
    p.add_argument('--arrival-offset-km', type=float, default=10_000.0,
                   help='Stop short of L1 by this many km on the Earth side (default: 10,000 km)')
    p.add_argument('--ramp-up-frac', type=float, default=0.1, help='Raised-cosine ramp-up fraction (default: 0.1)')
    p.add_argument('--ramp-down-frac', type=float, default=0.1, help='Raised-cosine ramp-down fraction (default: 0.1)')
    p.add_argument('--angle-deg-max', type=float, default=10.0, help='Half-range for thrust-direction angle search (deg)')
    p.add_argument('--angle-steps', type=int, default=9, help='Number of samples for angle grid')
    p.add_argument('--dt-s', type=float, default=60.0, help='Integrator step (s)')
    args = p.parse_args()

    planner = ShadowTransferPlanner(
        tof_s=args.tof_h * 3600,
        g_limit_g=args.g,
        alt0_m=args.alt_km * 1e3,
        arrival_offset_m=args.arrival_offset_km * 1e3,
        ramp_up_frac=args.ramp_up_frac,
        ramp_down_frac=args.ramp_down_frac,
        angle_deg_max=args.angle_deg_max,
        angle_steps=args.angle_steps,
        dt_s=args.dt_s,
    )
    res = planner.plan()
    print(res.summary())
