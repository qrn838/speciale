import numpy as np
from scipy.stats import norm as scipy_norm
from EconModel import EconModelClass
from consav.grids import nonlinspace
import pandas as pd
from numba import njit, prange

# ─────────────────────────────────────────────────────────────────────────────
# Tauchen discretization of the logit-AR(1) health process
# ─────────────────────────────────────────────────────────────────────────────

def discretize_health_ar1(Nh, rho, sigma_eps, n_std=3.0, delta=0.0):
    """
    Discretize  z_{t+1} = rho*z_t + delta + eps,  eps ~ N(0, sigma_eps^2)  via Tauchen.
    Health:  h = exp(z) / (1 + exp(z)).
    delta > 0 shifts health upward (recovery); delta < 0 shifts downward (deterioration).
    Returns h_grid (Nh,) in (0,1) and transition matrix P_h (Nh x Nh).
    """
    sig_z  = sigma_eps / np.sqrt(max(1.0 - rho**2, 1e-12))
    z_grid = np.linspace(-n_std * sig_z, n_std * sig_z, Nh)
    step   = z_grid[1] - z_grid[0]

    P_h = np.zeros((Nh, Nh))
    for iz in range(Nh):
        mu = rho * z_grid[iz] + delta
        for jz in range(Nh):
            lo = z_grid[jz] - 0.5 * step
            hi = z_grid[jz] + 0.5 * step
            if jz == 0:
                P_h[iz, jz] = scipy_norm.cdf(hi, mu, sigma_eps)
            elif jz == Nh - 1:
                P_h[iz, jz] = 1.0 - scipy_norm.cdf(lo, mu, sigma_eps)
            else:
                P_h[iz, jz] = (scipy_norm.cdf(hi, mu, sigma_eps)
                                - scipy_norm.cdf(lo, mu, sigma_eps))

    h_grid = np.exp(z_grid) / (1.0 + np.exp(z_grid))
    return h_grid.astype(np.float64), P_h.astype(np.float64)


@njit(cache=True)
def _crra_u(c, mu, eta):
    """CRRA utility: η·c^(1+µ)/(1+µ).  Limit as µ→-1 is η·log(c)."""
    if c < 1e-10:
        c = 1e-10
    if abs(mu + 1.0) < 1e-10:
        return eta * np.log(c)
    else:
        return eta * c ** (1.0 + mu) / (1.0 + mu)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class FullLaborModelClass(EconModelClass):
    """
    Three-state life-cycle model: Employment (j=1), Unemployment (j=0),
    Sickness leave (j=2).  Hand-to-Mouth agents.

    State space:
      VE / VU  –  (T, Nh, Nk, NdU, [Nb,] Ntype)
      VS       –  (T, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype)

    Simplification: sick-leave duration (dS) always resets to 0 at the
    start of each new spell, so VE and VU carry no dS dimension.
    """

    def settings(self):
        pass

    # ── parameters ──────────────────────────────────────────────────────────
    def setup(self):
        par = self.par

        # horizon
        par.T = 100

        # discounting
        par.beta = 0.98

        # taxes (flat rate)
        par.tau = 0.30

        # ── health AR(1) (logit scale) ──────────────────────────────────────
        par.rho_h   = 0.915   # persistence
        par.sigma_h = 0.30    # innovation std on logit scale
        par.Nh      = 10       # health grid points

        # ── human capital ───────────────────────────────────────────────────
        par.alpha   = 0.10    # wage return: w_t = w*(1 + alpha*k_t)
        par.delta_k = 0.05    # depreciation rate

        # ── labor market ─────────────────────────────────────────────────────
        par.sigma_sep = 0.05  # exogenous separation rate
        par.psi       = 1.0   # job-finding scale: p_find = psi * s

        # ── unobserved heterogeneity ─────────────────────────────────────────
        par.Ntype        = 2
        par.type_shares  = np.array([0.60, 0.40])
        par.lambda_grid  = np.array([20.0, 50.0])  # search cost scale by type
        par.nu_grid      = np.array([2.0, 5.0])  # work disutility scale by type
        #   search cost (unemployed):  (1-h)*lambda_n * s^(1+gamma)/(1+gamma)
        #   work disutility (employed): nu_n * (1-h)^(1+iota)/(1+iota)

        # Initial health distribution: stationary distribution of the health AR(1),
        # shifted in logit z-space per type.  z=0 maps to h=0.5; negative z → lower h.
        # Type 0 (low cost):  centred at h≈0.5
        # Type 1 (high cost): shifted left → lower initial health mean
        par.h_init_mu = np.array([1.0, 0.5])

        # ── UI system ────────────────────────────────────────────────────────
        par.Ubar = 24    # max UI entitlement (periods)
        par.J    = 3    # beskæftigelsestillæg window (last J periods of spell)
        par.zeta = 2    # re-qualification increment per employment period

        par.b_wel  = 0.40   # social assistance (dU = 0)
        par.b_emp  = 1.0   # high UI benefit (beskæftigelsestillæg)
        par.bmax   = 0.90   # UI benefit cap

        # ── UI search requirement ─────────────────────────────────────────────
        # Workers on UI (dU > 0) must search at s ≥ s_bar to receive benefit.
        # Translates "2 job applications per week" to a minimum effort level.
        # Workers who find the constraint too costly can escape by going sick.
        par.s_bar  = 0.10   # minimum search effort to keep UI (calibrate to data)
        par.Nb     = 3      # grid size for pinned UI benefit

        # ── health-dependent fixed cost of labour market participation ────────
        # chi*(1-h) is paid when unemployed and required to search (idU > 0).
        # Represents the physical/mental burden of job searching while ill,
        # beyond the variable search effort cost. Calibrate to match U→S uptake.
        par.chi    = 1.0

        # ── wage during sick leave (E-origin, highest tier only) ─────────────
        # If True: E-origin workers at ir=0 receive repl_sick fraction of their
        # wage while on sick leave instead of the standard sick benefit.
        # If False: all sick workers receive u_b_S regardless of origin.
        par.wage_sick  = False
        par.repl_sick  = 1.0   # wage replacement rate on sick leave

        # ── job search while on sick leave (U-origin only) ───────────────────
        # If True: U-origin sick workers can search for a job at standard cost
        # (1-h)*lambda*s^(1+gamma)/(1+gamma) with no minimum effort requirement.
        # If found, they exit directly to E without returning to U first.
        par.search_on_sick = False

        # ── utility function ──────────────────────────────────────────────────
        # Consumption: η·c^(1+µ)/(1+µ)  [µ=-1 → log(c), requires η=1 for log]
        # Search cost (U): (1-h)·λ·s^(1+γ)/(1+γ)
        # Work disutility (E): ν·(1-h)^(1+ι)/(1+ι)
        par.mu    = -1.0   # CRRA curvature (mu=-1 → log utility)
        par.eta   =  1.0   # utility scale
        par.gamma =  0.9   # search cost curvature exponent
        par.iota  =  0.9   # work disutility curvature exponent

        # wage base
        par.w = 1.0

        # ── human capital grid ───────────────────────────────────────────────
        par.k_max = 20.0
        par.Nk    = 8

        # ── sickness leave ───────────────────────────────────────────────────
        # dS resets to 0 at every new sick spell (no carry-over across U spells).
        par.Sbar        = 24    # max sick duration tracked (capped after this)
        par.t_reassess  = 6     # reassessment at transition dS: t_reassess-1 → t_reassess
        # Sick leave benefit structure (ir tracks post-reassessment tier):
        #   ir=0:  full benefit  b_grid[ib]  — before and after reassessment (high outcome)
        #   ir=1:  low benefit   b_sick_low  — only possible post-reassessment
        #   ir=2:  welfare       b_wel       — only possible post-reassessment
        # Before reassessment only ir=0 exists; ir=1 and ir=2 are skipped by the solver.
        par.b_sick_low  = 0.50  # intermediate benefit at reassessment (ir=1)
        par.b_floor     = 0.01  # income received by U workers rejected at spell entry

        # Medical documentation check at spell entry (binary: accept / reject).
        # Rejected workers are kicked back to their origin state (never enter VS).
        # U-origin workers receive b_floor for the rejection period (income penalty).
        # Healthy workers (high h) face a higher rejection probability.
        # delta0_doc must be large enough so even low-health workers face meaningful
        # rejection risk — otherwise it has no effect on the gaming workers
        # (who are low-health and have the highest search-cost burden from s_bar).
        par.delta0_doc =  0.15
        par.delta1_doc =  0.9
        par.delta2_doc =  0.00

        # Benefit-reduction probabilities at reassessment — linear in health:
        #   P(low  | h) = delta0_low + delta1_low*h   [clipped to [0,1]]
        #   P(out  | h) = delta0_out + delta1_out*h   [clipped to [0, 1-P(low)]]
        #   P(high | h) = 1 - P(low) - P(out)
        # Healthy workers (high h) face larger benefit cuts (less genuinely sick).
        # Calibrated so that all health grid points have positive cut probabilities.
        par.delta0_low =  0.15
        par.delta1_low =  0.20
        par.delta2_low =  0.00
        par.delta0_out = -0.20
        par.delta1_out =  0.80
        par.delta2_out =  0.00

        # simulation
        par.simT = par.T

    # ── allocate arrays ──────────────────────────────────────────────────────
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # ── health ──────────────────────────────────────────────────────────
        par.h_grid, par.P_h = discretize_health_ar1(par.Nh, par.rho_h, par.sigma_h)

        # Health recovery on sick leave: positive drift on logit scale.
        # delta_h_S > 0 means health improves faster while on sick leave
        # than when working or searching.
        par.delta_h_S = 0.06
        _, par.P_h_S  = discretize_health_ar1(par.Nh, par.rho_h, par.sigma_h,
                                               delta=par.delta_h_S)

        # ── type-specific initial health distributions ────────────────────────
        # Stationary distribution of the health AR(1) is N(0, sigma_z^2) in logit
        # z-space.  Shift the mean by h_init_mu[itype] to give each type a
        # different initial health level, then discretise onto h_grid.
        sig_z  = par.sigma_h / np.sqrt(max(1.0 - par.rho_h**2, 1e-12))
        n_std  = 3.0
        z_grid = np.linspace(-n_std * sig_z, n_std * sig_z, par.Nh)
        step   = z_grid[1] - z_grid[0]

        par.h_init_dist = np.zeros((par.Ntype, par.Nh))
        for itype in range(par.Ntype):
            mu_z = par.h_init_mu[itype]
            for ih in range(par.Nh):
                lo = z_grid[ih] - 0.5 * step
                hi = z_grid[ih] + 0.5 * step
                if ih == 0:
                    par.h_init_dist[itype, ih] = scipy_norm.cdf(hi, mu_z, sig_z)
                elif ih == par.Nh - 1:
                    par.h_init_dist[itype, ih] = 1.0 - scipy_norm.cdf(lo, mu_z, sig_z)
                else:
                    par.h_init_dist[itype, ih] = (scipy_norm.cdf(hi, mu_z, sig_z)
                                                   - scipy_norm.cdf(lo, mu_z, sig_z))
            par.h_init_dist[itype] /= par.h_init_dist[itype].sum()

        # ── k grid ──────────────────────────────────────────────────────────
        par.k_grid = nonlinspace(0.0, par.k_max, par.Nk, 1.1)

        # ── UI eligibility ──────────────────────────────────────────────────
        par.NdU     = par.Ubar + 1
        par.dU_grid = np.arange(par.NdU, dtype=np.int32)

        # ── sickness duration (0 … Sbar) ─────────────────────────────────────
        par.NdS     = par.Sbar + 1
        par.dS_grid = np.arange(par.NdS, dtype=np.int32)

        # sick benefit state: 0=high, 1=low, 2=floor/out
        par.Nr = 3
        # spell origin:       0=from U, 1=from E
        par.No = 2

        # ── UI benefit pinning ────────────────────────────────────────────────
        b_reset_vals     = np.minimum(0.8 * par.w * (1.0 + par.alpha * par.k_grid), par.bmax)
        par.b_grid       = np.linspace(b_reset_vals.min(), b_reset_vals.max(), par.Nb)
        par.ib_sep_by_ik = (np.searchsorted(par.b_grid, b_reset_vals)
                             .clip(0, par.Nb - 1).astype(np.int32))

        # ── pre-computed flow utilities (after tax, CRRA) ────────────────────
        def _u(b):
            c = max(float(b), 1e-10) * (1.0 - par.tau)
            c = max(c, 1e-10)
            if abs(par.mu + 1.0) < 1e-10:
                return par.eta * np.log(c)
            else:
                return par.eta * c ** (1.0 + par.mu) / (1.0 + par.mu)

        # UI benefit utility by (dU, ib)
        par.u_b_U = np.empty((par.NdU, par.Nb))
        for dU in range(par.NdU):
            for ib in range(par.Nb):
                if dU == 0:
                    b = par.b_wel
                elif dU > par.Ubar - par.J:
                    b = par.b_emp
                else:
                    b = par.b_grid[ib]
                par.u_b_U[dU, ib] = _u(b)

        # Sick benefit utility by (dS, ir, ib):
        #   ir=0:  b_grid[ib]  — full UI entitlement (only active state pre-reassessment)
        #   ir=1:  b_sick_low  — low benefit; only populated post-reassessment
        #   ir=2:  b_wel       — welfare; only populated post-reassessment
        # Pre-reassessment entries for ir>0 are computed but never accessed.
        par.u_b_S = np.empty((par.NdS, par.Nr, par.Nb))
        for ids in range(par.NdS):
            for ir in range(par.Nr):
                for ib in range(par.Nb):
                    if ir == 0:
                        b = par.b_grid[ib]
                    elif ir == 1:
                        b = par.b_sick_low          # only meaningful post-reassessment
                    else:                            # ir == 2
                        b = par.b_wel if ids >= par.t_reassess else par.b_floor
                    par.u_b_S[ids, ir, ib] = _u(b)

        # ── medical documentation check probabilities ─────────────────────────
        # p_doc_out[ih]: probability of being rejected at spell entry (kick-back to origin).
        par.p_doc_out = np.empty(par.Nh)
        for ih in range(par.Nh):
            h  = par.h_grid[ih]
            po = par.delta0_doc + par.delta1_doc * h + par.delta2_doc * h**2
            par.p_doc_out[ih] = float(np.clip(po, 0.0, 1.0))

        # E[p_doc_out(h') | h]: expected rejection probability at next-period health.
        # Used for the 1-period zero-benefit penalty for rejected U→S attempts.
        par.E_p_doc_out = par.P_h @ par.p_doc_out

        # wage utility by ik (full wage for employment; replacement rate for sick leave)
        par.u_w      = np.array([_u(par.w * (1.0 + par.alpha * k))
                                  for k in par.k_grid], dtype=np.float64)
        par.u_w_sick = np.array([_u(par.w * (1.0 + par.alpha * k) * par.repl_sick)
                                  for k in par.k_grid], dtype=np.float64)

        # ── reassessment probabilities by health grid point ───────────────────
        par.p_low = np.empty(par.Nh)
        par.p_out = np.empty(par.Nh)
        for ih in range(par.Nh):
            h  = par.h_grid[ih]
            pl = par.delta0_low + par.delta1_low * h + par.delta2_low * h**2
            po = par.delta0_out + par.delta1_out * h + par.delta2_out * h**2
            pl = float(np.clip(pl, 0.0, 1.0))
            po = float(np.clip(po, 0.0, 1.0 - pl))
            par.p_low[ih] = pl
            par.p_out[ih] = po

        # ── k transitions (nearest-grid) ─────────────────────────────────────
        # Unemployed / sick: k' = (1-delta_k)*k  [Eq. 11, j≠1]
        par.ik_next_U = np.empty(par.Nk, dtype=np.int32)
        for ik in range(par.Nk):
            kn = (1.0 - par.delta_k) * par.k_grid[ik]
            par.ik_next_U[ik] = int(np.clip(np.searchsorted(par.k_grid, kn), 0, par.Nk - 1))

        # Employed: k' = (1-delta_k)*k + h  [Eq. 11, health-dependent]
        par.ik_next_EH = np.empty((par.Nk, par.Nh), dtype=np.int32)
        for ik in range(par.Nk):
            for ih in range(par.Nh):
                kn = (1.0 - par.delta_k) * par.k_grid[ik] + par.h_grid[ih]
                par.ik_next_EH[ik, ih] = int(
                    np.clip(np.searchsorted(par.k_grid, kn), 0, par.Nk - 1))

        # ── eligibility transitions ──────────────────────────────────────────
        par.dU_next_U = np.maximum(par.dU_grid - 1, 0).astype(np.int32)          # -1 while U
        par.dU_next_E = np.minimum(par.dU_grid + par.zeta, par.Ubar).astype(np.int32)  # +zeta while E
        # sick: dU frozen

        # ── sick duration transition ──────────────────────────────────────────
        par.dS_next_S = np.minimum(par.dS_grid + 1, par.Sbar).astype(np.int32)

        # ── solution arrays ───────────────────────────────────────────────────
        # VE: no ib (ib is freshly pinned on each E→U separation)
        # VU: with ib (pinned at spell start)
        # VS: full sick-leave state
        shapeE = (par.T, par.Nh, par.Nk, par.NdU,                                    par.Ntype)
        shapeU = (par.T, par.Nh, par.Nk, par.NdU, par.Nb,                            par.Ntype)
        shapeS = (par.T, par.Nh, par.Nk, par.NdU, par.NdS, par.Nr, par.No, par.Nb, par.Ntype)

        sol.VE = np.full(shapeE, -1e10, dtype=np.float64)
        sol.VU = np.full(shapeU, -1e10, dtype=np.float64)
        sol.VS = np.full(shapeS, -1e10, dtype=np.float64)

        sol.s   = np.zeros(shapeU, dtype=np.float64)  # search effort (U)
        sol.g_U = np.zeros(shapeU, dtype=np.float64)  # go-sick flag  (U → S)
        sol.q   = np.zeros(shapeE, dtype=np.float64)  # quit flag     (E → U)
        sol.g_E = np.zeros(shapeE, dtype=np.float64)  # go-sick flag  (E → S)
        sol.ret = np.zeros(shapeS, dtype=np.float64)  # return flag   (S → E/U)
        sol.s_S = np.zeros(shapeS, dtype=np.float64)  # search effort while sick (U-origin)

        # ── distribution arrays ───────────────────────────────────────────────
        sim.muE = np.zeros(shapeE, dtype=np.float64)
        sim.muU = np.zeros(shapeU, dtype=np.float64)
        sim.muS = np.zeros(shapeS, dtype=np.float64)

    # ── solve ────────────────────────────────────────────────────────────────
    def solve(self):
        par, sol = self.par, self.sol

        _b_floor_c = float(par.b_floor * (1.0 - par.tau))
        _b_floor_c = max(_b_floor_c, 1e-10)
        if abs(par.mu + 1.0) < 1e-10:
            _u_b_floor = np.float64(par.eta * np.log(_b_floor_c))
        else:
            _u_b_floor = np.float64(par.eta * _b_floor_c ** (1.0 + par.mu)
                                    / (1.0 + par.mu))

        _solve_all(
            np.int64(par.T), np.int64(par.Nh), np.int64(par.Nk),
            np.int64(par.NdU), np.int64(par.NdS),
            np.int64(par.Nr), np.int64(par.No), np.int64(par.Nb), np.int64(par.Ntype),
            np.float64(par.beta), np.float64(par.psi), np.float64(par.sigma_sep),
            np.int64(par.t_reassess), np.float64(par.s_bar), np.float64(par.chi),
            np.int64(par.wage_sick), np.int64(par.search_on_sick),
            np.float64(par.gamma), np.float64(par.iota),
            par.lambda_grid.astype(np.float64), par.nu_grid.astype(np.float64),
            par.h_grid,
            par.P_h, par.P_h_S,
            par.p_low, par.p_out, par.p_doc_out, par.E_p_doc_out,
            par.u_w, par.u_w_sick, par.u_b_U, par.u_b_S, _u_b_floor,
            par.ik_next_U, par.ik_next_EH,
            par.dU_next_U, par.dU_next_E, par.dS_next_S,
            par.ib_sep_by_ik,
            np.int64(par.dS_next_S[par.t_reassess - 1]),
            sol.VE, sol.VU, sol.VS,
            sol.s, sol.g_U, sol.q, sol.g_E, sol.ret, sol.s_S,
        )

    # ── forward distribution ─────────────────────────────────────────────────
    def simulate(self):
        par, sol, sim = self.par, self.sol, self.sim

        # initial distribution: each type distributed over health grid according
        # to their type-specific stationary distribution (h_init_dist), all at
        # lowest k and full UI entitlement, all employed.
        muE0 = np.zeros((par.Nh, par.Nk, par.NdU, par.Ntype))
        muU0 = np.zeros((par.Nh, par.Nk, par.NdU, par.Nb, par.Ntype))
        muS0 = np.zeros((par.Nh, par.Nk, par.NdU, par.NdS, par.Nr, par.No, par.Nb, par.Ntype))

        ik0, idU0 = 0, par.Ubar
        for itype in range(par.Ntype):
            for ih in range(par.Nh):
                muE0[ih, ik0, idU0, itype] = par.type_shares[itype] * par.h_init_dist[itype, ih]

        _forward_distribution(
            par.T, par.Nh, par.Nk, par.NdU, par.NdS,
            par.Nr, par.No, par.Nb, par.Ntype,
            par.psi, par.sigma_sep, par.t_reassess,
            par.P_h, par.P_h_S,
            par.ik_next_U, par.ik_next_EH,
            par.dU_next_U, par.dU_next_E,
            par.dS_next_S,
            par.ib_sep_by_ik,
            par.p_low, par.p_out, par.p_doc_out,
            sol.s, sol.g_U, sol.q, sol.g_E, sol.ret, sol.s_S,
            muE0, muU0, muS0,
            sim.muE, sim.muU, sim.muS,
        )

    # ── convenience aggregates ────────────────────────────────────────────────
    def aggregate_shares(self):
        """Return (T,) arrays of employment, unemployment, sick-leave shares."""
        muE = self.sim.muE.sum(axis=(1, 2, 3, 4))
        muU = self.sim.muU.sum(axis=(1, 2, 3, 4, 5))
        muS = self.sim.muS.sum(axis=(1, 2, 3, 4, 5, 6, 7, 8))
        return muE, muU, muS

    def hazard_u_to_e(self, t0=0, max_d=None):
        """
        Unemployment-to-employment hazard for the cohort entering U at calendar
        time t0.  Uses the population distribution from simulate().
        Returns a DataFrame with columns: duration, at_risk, exits, hazard.
        """
        par, sol, sim = self.par, self.sol, self.sim
        if max_d is None:
            max_d = min(par.T - 1 - t0, par.Ubar + 2)
        max_d = int(max_d)

        cohort0     = self._build_u_entry_cohort(t0)
        at_risk_out = np.empty(max_d)
        exits_E_out = np.empty(max_d)
        exits_S_out = np.empty(max_d)

        _hazard_cohort_u(
            max_d, par.Nh, par.Nk, par.NdU, par.Nb, par.Ntype,
            par.psi, par.P_h,
            par.ik_next_U, par.dU_next_U,
            par.p_doc_out,
            sol.s[t0:t0 + max_d],
            sol.g_U[t0:t0 + max_d],
            cohort0,
            at_risk_out, exits_E_out, exits_S_out,
        )
        hazard = np.where(at_risk_out > 0, exits_E_out / at_risk_out, 0.0)
        return pd.DataFrame({
            "duration": np.arange(1, max_d + 1),
            "at_risk":  at_risk_out,
            "exits":    exits_E_out,
            "hazard":   hazard,
        })

    # ── cohort builders (shared by multiple hazard methods) ───────────────────

    def _build_u_entry_cohort(self, t0):
        """Mass entering unemployment at t0 (from E separations/quits and S returns)."""
        par, sol, sim = self.par, self.sol, self.sim
        cohort0 = np.zeros((par.Nh, par.Nk, par.NdU, par.Nb, par.Ntype))

        # from E: quit or exogenous separation (not going sick)
        muE_t0 = sim.muE[t0]
        for ih in range(par.Nh):
            for ik in range(par.Nk):
                ik_E     = par.ik_next_EH[ik, ih]
                ib_sep_E = par.ib_sep_by_ik[ik_E]
                for idU in range(par.NdU):
                    idU_E = par.dU_next_E[idU]
                    for itype in range(par.Ntype):
                        m = muE_t0[ih, ik, idU, itype]
                        if m == 0.0:
                            continue
                        g_E = sol.g_E[t0, ih, ik, idU, itype]
                        q   = sol.q[t0, ih, ik, idU, itype]
                        if g_E < 0.5:
                            p_to_u = 1.0 if q >= 0.5 else par.sigma_sep
                            for ih2 in range(par.Nh):
                                cohort0[ih2, ik_E, idU_E, ib_sep_E, itype] += (
                                    par.P_h[ih, ih2] * m * p_to_u)

        # from S with origin=1: return attempt hits exogenous separation
        muS_t0 = sim.muS[t0]
        for ih in range(par.Nh):
            for ik in range(par.Nk):
                ik_S     = par.ik_next_U[ik]
                ib_sep_S = par.ib_sep_by_ik[ik_S]
                for idU in range(par.NdU):
                    for ids in range(par.NdS):
                        for ir in range(par.Nr):
                            for ib in range(par.Nb):
                                for itype in range(par.Ntype):
                                    m = muS_t0[ih, ik, idU, ids, ir, 1, ib, itype]
                                    if m == 0.0:
                                        continue
                                    if sol.ret[t0, ih, ik, idU, ids, ir, 1, ib, itype] >= 0.5:
                                        for ih2 in range(par.Nh):
                                            cohort0[ih2, ik_S, idU, ib_sep_S, itype] += (
                                                par.P_h[ih, ih2] * m * par.sigma_sep)

        # from S with origin=0: return to job search
        for ih in range(par.Nh):
            for ik in range(par.Nk):
                ik_S = par.ik_next_U[ik]
                for idU in range(par.NdU):
                    for ids in range(par.NdS):
                        for ir in range(par.Nr):
                            for ib in range(par.Nb):
                                for itype in range(par.Ntype):
                                    m = muS_t0[ih, ik, idU, ids, ir, 0, ib, itype]
                                    if m == 0.0:
                                        continue
                                    if sol.ret[t0, ih, ik, idU, ids, ir, 0, ib, itype] >= 0.5:
                                        for ih2 in range(par.Nh):
                                            cohort0[ih2, ik_S, idU, ib, itype] += (
                                                par.P_h[ih, ih2] * m)
        return cohort0

    def _build_s_entry_cohort(self, t0):
        """Mass entering sick leave at t0 (from E via g_E=1, from U via g_U=1)."""
        par, sol, sim = self.par, self.sol, self.sim
        cohort0 = np.zeros((par.Nh, par.Nk, par.NdU, par.NdS,
                            par.Nr, par.No, par.Nb, par.Ntype))

        # from E (g_E=1): enter S with dS=0, ir=0, io=1, ib=ib_sep at new k
        muE_t0 = sim.muE[t0]
        for ih in range(par.Nh):
            for ik in range(par.Nk):
                ik_E     = par.ik_next_EH[ik, ih]
                ib_sep_E = par.ib_sep_by_ik[ik_E]
                for idU in range(par.NdU):
                    idU_E = par.dU_next_E[idU]
                    for itype in range(par.Ntype):
                        m = muE_t0[ih, ik, idU, itype]
                        if m == 0.0:
                            continue
                        if sol.g_E[t0, ih, ik, idU, itype] >= 0.5:
                            for ih2 in range(par.Nh):
                                # only accepted workers (1-p_doc_out) enter the cohort
                                ph = par.P_h[ih, ih2] * m * (1.0 - par.p_doc_out[ih2])
                                cohort0[ih2, ik_E, idU_E, 0, 0, 1, ib_sep_E, itype] += ph

        # from U (g_U=1): enter S with dS=0, ir=0, io=0, ib=current ib
        muU_t0 = sim.muU[t0]
        for ih in range(par.Nh):
            for ik in range(par.Nk):
                ik_U = par.ik_next_U[ik]
                for idU in range(par.NdU):
                    idU_U = par.dU_next_U[idU]
                    for ib in range(par.Nb):
                        for itype in range(par.Ntype):
                            m = muU_t0[ih, ik, idU, ib, itype]
                            if m == 0.0:
                                continue
                            if sol.g_U[t0, ih, ik, idU, ib, itype] >= 0.5:
                                for ih2 in range(par.Nh):
                                    # only accepted workers (1-p_doc_out) enter the cohort
                                    ph = par.P_h[ih, ih2] * m * (1.0 - par.p_doc_out[ih2])
                                    cohort0[ih2, ik_U, idU_U, 0, 0, 0, ib, itype] += ph
        return cohort0

    # ── hazard: U → S ────────────────────────────────────────────────────────

    def hazard_u_to_s(self, t0=0, max_d=None):
        """
        U→S hazard by unemployment duration for cohort entering U at t0.
        Both exits to E (search) and to S (sick leave) remove workers from the
        at-risk pool.  Returns the rate of exiting specifically via sick leave.
        """
        par, sol = self.par, self.sol
        if max_d is None:
            max_d = min(par.T - 1 - t0, par.Ubar + 2)
        max_d = int(max_d)

        cohort0     = self._build_u_entry_cohort(t0)
        at_risk_out = np.empty(max_d)
        exits_E_out = np.empty(max_d)
        exits_S_out = np.empty(max_d)

        _hazard_cohort_u(
            max_d, par.Nh, par.Nk, par.NdU, par.Nb, par.Ntype,
            par.psi, par.P_h,
            par.ik_next_U, par.dU_next_U,
            par.p_doc_out,
            sol.s[t0:t0 + max_d],
            sol.g_U[t0:t0 + max_d],
            cohort0,
            at_risk_out, exits_E_out, exits_S_out,
        )
        hazard = np.where(at_risk_out > 0, exits_S_out / at_risk_out, 0.0)
        return pd.DataFrame({
            "duration": np.arange(1, max_d + 1),
            "at_risk":  at_risk_out,
            "exits":    exits_S_out,
            "hazard":   hazard,
        })

    # ── hazard: S → E ────────────────────────────────────────────────────────

    def hazard_s_to_e(self, t0=0, max_d=None):
        """
        S→E hazard by sick leave duration for cohort entering S at t0.
        Counts returns to employment: io=1 workers who choose ret=1 and the
        exogenous separation does NOT fire (prob 1 - sigma_sep).
        """
        par, sol = self.par, self.sol
        if max_d is None:
            max_d = min(par.T - 1 - t0, par.Sbar + 2)
        max_d = int(max_d)

        cohort0     = self._build_s_entry_cohort(t0)
        at_risk_out = np.empty(max_d)
        exits_E_out = np.empty(max_d)
        exits_U_out = np.empty(max_d)

        _hazard_cohort_s(
            max_d, par.Nh, par.Nk, par.NdU, par.NdS, par.Nr, par.No, par.Nb, par.Ntype,
            par.sigma_sep, par.t_reassess, par.psi,
            par.P_h, par.ik_next_U, par.dS_next_S,
            par.ib_sep_by_ik, par.p_low, par.p_out,
            sol.ret[t0:t0 + max_d], sol.s_S[t0:t0 + max_d],
            cohort0,
            at_risk_out, exits_E_out, exits_U_out,
        )
        hazard = np.where(at_risk_out > 0, exits_E_out / at_risk_out, 0.0)
        return pd.DataFrame({
            "duration": np.arange(1, max_d + 1),
            "at_risk":  at_risk_out,
            "exits":    exits_E_out,
            "hazard":   hazard,
        })

    # ── hazard: S → U ────────────────────────────────────────────────────────

    def hazard_s_to_u(self, t0=0, max_d=None):
        """
        S→U hazard by sick leave duration for cohort entering S at t0.
        Counts transitions to unemployment: io=0 workers who choose ret=1,
        plus io=1 workers who choose ret=1 but hit exogenous separation (sigma_sep).
        """
        par, sol = self.par, self.sol
        if max_d is None:
            max_d = min(par.T - 1 - t0, par.Sbar + 2)
        max_d = int(max_d)

        cohort0     = self._build_s_entry_cohort(t0)
        at_risk_out = np.empty(max_d)
        exits_E_out = np.empty(max_d)
        exits_U_out = np.empty(max_d)

        _hazard_cohort_s(
            max_d, par.Nh, par.Nk, par.NdU, par.NdS, par.Nr, par.No, par.Nb, par.Ntype,
            par.sigma_sep, par.t_reassess, par.psi,
            par.P_h, par.ik_next_U, par.dS_next_S,
            par.ib_sep_by_ik, par.p_low, par.p_out,
            sol.ret[t0:t0 + max_d], sol.s_S[t0:t0 + max_d],
            cohort0,
            at_risk_out, exits_E_out, exits_U_out,
        )
        hazard = np.where(at_risk_out > 0, exits_U_out / at_risk_out, 0.0)
        return pd.DataFrame({
            "duration": np.arange(1, max_d + 1),
            "at_risk":  at_risk_out,
            "exits":    exits_U_out,
            "hazard":   hazard,
        })

    # ── hazard: S pooled (all spells, all calendar times) ────────────────────

    def hazard_s_pooled(self, max_d=None):
        """
        Sick-leave hazard rates pooled across all calendar times and spell origins,
        plus origin-specific rates (E-origin and U-origin separately).

        At each duration d (= idS + 1), aggregates all mass currently at that
        sick-leave duration across every period t, then computes:
            hazard_E(d) = exits to E / at_risk
            hazard_U(d) = exits to U / at_risk

        This matches how empirical hazard rates are estimated from administrative
        data: every ongoing spell contributes to the at-risk count at its current
        duration, regardless of when the spell started.
        """
        par, sol, sim = self.par, self.sol, self.sim

        if max_d is None:
            max_d = par.NdS
        max_d = int(max_d)

        # Single-pass parallel Numba kernel — avoids large numpy temporaries.
        # Exclude last period (terminal ret=0 everywhere, no forward decision).
        muS = sim.muS[:-1, :, :, :, :max_d, :, :, :, :]
        ret = sol.ret[:-1, :, :, :, :max_d, :, :, :, :]
        s_S = sol.s_S[:-1, :, :, :, :max_d, :, :, :, :]

        at_risk, exits_E, exits_U, hazard_E, hazard_U = _pooled_hazard_kernel(
            muS, ret, s_S, par.psi, max_d, par.sigma_sep
        )

        # Origin-specific hazards via numpy.
        # muS/ret shape: (T, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype)
        # After selecting io, shape: (T, Nh, Nk, NdU, NdS, Nr, Nb, Ntype)
        # Sum over all axes except NdS (axis 4): axes (0,1,2,3,5,6,7)
        _sum = (0, 1, 2, 3, 5, 6, 7)
        muS_E = muS[:, :, :, :, :, :, 1, :, :]   # E-origin (io=1)
        muS_U = muS[:, :, :, :, :, :, 0, :, :]   # U-origin (io=0)
        ret_E = ret[:, :, :, :, :, :, 1, :, :]
        ret_U = ret[:, :, :, :, :, :, 0, :, :]

        s_S_U     = s_S[:, :, :, :, :, :, 0, :, :]   # io=0 search effort
        ret_U_inv = 1.0 - ret_U                        # non-returning U-origin

        ar_E        = muS_E.sum(axis=_sum)
        ar_U        = muS_U.sum(axis=_sum)
        retmass_E   = (muS_E * ret_E).sum(axis=_sum)
        retmass_U   = (muS_U * ret_U).sum(axis=_sum)
        findmass_U  = (muS_U * ret_U_inv * par.psi * s_S_U).sum(axis=_sum)

        ex_E_Eorig = retmass_E * (1.0 - par.sigma_sep)
        ex_U_Eorig = retmass_E * par.sigma_sep
        ex_U_Uorig = retmass_U
        ex_E_Uorig = findmass_U   # U-origin finding job via on-sick search

        hz_E_Eorig = np.where(ar_E > 0, ex_E_Eorig / ar_E, 0.0)
        hz_U_Eorig = np.where(ar_E > 0, ex_U_Eorig / ar_E, 0.0)
        hz_U_Uorig = np.where(ar_U > 0, ex_U_Uorig / ar_U, 0.0)
        hz_E_Uorig = np.where(ar_U > 0, ex_E_Uorig / ar_U, 0.0)

        return pd.DataFrame({
            "duration":       np.arange(1, max_d + 1),
            "at_risk":        at_risk,
            "at_risk_E":      ar_E,
            "at_risk_U":      ar_U,
            "exits_E":        exits_E,
            "exits_U":        exits_U,
            "hazard_E":       hazard_E,
            "hazard_U":       hazard_U,
            "hazard_E_Eorig": hz_E_Eorig,
            "hazard_U_Eorig": hz_U_Eorig,
            "hazard_U_Uorig": hz_U_Uorig,
            "hazard_E_Uorig": hz_E_Uorig,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Numba: full backward induction over all T periods.
# Expected values are computed on-the-fly inside the inner loops using
# Nh-sized dot products against the raw V[t+1] arrays, eliminating the
# large EV array allocations and Python→Numba transitions that existed
# when EVs were pre-computed in Python before each period call.
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _solve_all(
    T, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype,
    beta, psi, sigma_sep, t_reassess, s_bar, chi, wage_sick, search_on_sick,
    gamma, iota,
    lambda_grid, nu_grid, h_grid,
    P_h, P_h_S,
    p_low, p_out, p_doc_out, E_p_doc_out,
    u_w, u_w_sick, u_b_U, u_b_S, u_b_floor,
    ik_next_U, ik_next_EH,
    dU_next_U, dU_next_E, dS_next_S,
    ib_sep_by_ik,
    ids_nxt_ra,
    VE, VU, VS,
    s_pol, g_U_pol, q_pol, g_E_pol, ret_pol, s_S_pol,
):
    for t in range(T - 1, -1, -1):
        last = (t == T - 1)

        # ──────────────────────── V^S ────────────────────────────────────
        for idx_S in prange(Ntype * Nh * Nk):
            itype    = idx_S // (Nh * Nk)
            ih       = (idx_S % (Nh * Nk)) // Nk
            ik       = idx_S % Nk
            lam      = lambda_grid[itype]
            h        = h_grid[ih]
            omh      = max(1.0 - h, 1e-8)
            ik_S     = ik_next_U[ik]
            ib_sep_S = ib_sep_by_ik[ik_S]
            for idU in range(NdU):
                for idS in range(NdS):
                    idS_nxt = dS_next_S[idS]
                    for ir in range(Nr):
                        if idS < t_reassess and ir > 0:
                            continue
                        for io in range(No):
                            for ib in range(Nb):
                                if wage_sick and io == 1 and ir == 0:
                                    u_S = u_w_sick[ik]
                                else:
                                    u_S = u_b_S[idS, ir, ib]

                                if last:
                                    VS[t,ih,ik,idU,idS,ir,io,ib,itype]      = u_S
                                    ret_pol[t,ih,ik,idU,idS,ir,io,ib,itype] = 0.0
                                    s_S_pol[t,ih,ik,idU,idS,ir,io,ib,itype] = 0.0
                                    continue

                                # ── EV for return ──────────────────────
                                if io == 1:
                                    ev_E_S   = 0.0
                                    ev_U_sep = 0.0
                                    for ih2 in range(Nh):
                                        ev_E_S   += P_h_S[ih, ih2] * VE[t+1, ih2, ik_S, idU, itype]
                                        ev_U_sep += P_h_S[ih, ih2] * VU[t+1, ih2, ik_S, idU, ib_sep_S, itype]
                                    ev_ret = (1.0 - sigma_sep) * ev_E_S + sigma_sep * ev_U_sep
                                else:
                                    ev_ret = 0.0
                                    for ih2 in range(Nh):
                                        ev_ret += P_h_S[ih, ih2] * VU[t+1, ih2, ik_S, idU, ib, itype]

                                # ── EV for stay ────────────────────────
                                at_reassess  = (idS == t_reassess - 1) and (ir == 0)
                                ev_stay_base = 0.0
                                if at_reassess:
                                    for ih2 in range(Nh):
                                        p_hi2 = 1.0 - p_low[ih2] - p_out[ih2]
                                        v_ra  = (p_hi2          * VS[t+1, ih2, ik_S, idU, ids_nxt_ra, 0, io, ib, itype]
                                                + p_low[ih2]    * VS[t+1, ih2, ik_S, idU, ids_nxt_ra, 1, io, ib, itype]
                                                + p_out[ih2]    * VS[t+1, ih2, ik_S, idU, ids_nxt_ra, 2, io, ib, itype])
                                        ev_stay_base += P_h_S[ih, ih2] * v_ra
                                else:
                                    for ih2 in range(Nh):
                                        ev_stay_base += P_h_S[ih, ih2] * VS[t+1, ih2, ik_S, idU, idS_nxt, ir, io, ib, itype]

                                # ── Search while sick (U-origin only) ──
                                if search_on_sick and io == 0:
                                    ev_find_S = 0.0
                                    for ih2 in range(Nh):
                                        ev_find_S += P_h_S[ih, ih2] * VE[t+1, ih2, ik_S, idU, itype]
                                    dV_S = ev_find_S - ev_stay_base
                                    if dV_S > 0.0:
                                        s_S_star = (beta * psi * dV_S / (omh * lam)) ** (1.0 / gamma)
                                        if s_S_star > 1.0:
                                            s_S_star = 1.0
                                    else:
                                        s_S_star = 0.0
                                    pf_S        = psi * s_S_star
                                    srch_cost_S = omh * lam * s_S_star ** (1.0 + gamma) / (1.0 + gamma)
                                    ev_stay     = pf_S * ev_find_S + (1.0 - pf_S) * ev_stay_base
                                    v_stay      = u_S - srch_cost_S + beta * ev_stay
                                else:
                                    s_S_star = 0.0
                                    v_stay   = u_S + beta * ev_stay_base

                                v_ret_val = u_S + beta * ev_ret

                                if v_ret_val >= v_stay:
                                    VS[t,ih,ik,idU,idS,ir,io,ib,itype]      = v_ret_val
                                    ret_pol[t,ih,ik,idU,idS,ir,io,ib,itype] = 1.0
                                    s_S_pol[t,ih,ik,idU,idS,ir,io,ib,itype] = 0.0
                                else:
                                    VS[t,ih,ik,idU,idS,ir,io,ib,itype]      = v_stay
                                    ret_pol[t,ih,ik,idU,idS,ir,io,ib,itype] = 0.0
                                    s_S_pol[t,ih,ik,idU,idS,ir,io,ib,itype] = s_S_star

        # ──────────────────────── V^E ────────────────────────────────────
        for idx_E in prange(Ntype * Nh * Nk):
            itype    = idx_E // (Nh * Nk)
            ih       = (idx_E % (Nh * Nk)) // Nk
            ik       = idx_E % Nk
            nu       = nu_grid[itype]
            h        = h_grid[ih]
            dis_work = -nu * (1.0 - h) ** (1.0 + iota) / (1.0 + iota)
            ik_E     = ik_next_EH[ik, ih]
            ib_sep_E = ib_sep_by_ik[ik_E]
            u_E      = u_w[ik] + dis_work
            for idU in range(NdU):
                idU_E = dU_next_E[idU]

                if last:
                    VE[t,ih,ik,idU,itype]      = u_E
                    q_pol[t,ih,ik,idU,itype]   = 0.0
                    g_E_pol[t,ih,ik,idU,itype] = 0.0
                    continue

                # EV: quit → U  and  stay E (share common ev_U_sep)
                ev_E_nxt = 0.0
                ev_U_sep = 0.0
                for ih2 in range(Nh):
                    ev_E_nxt += P_h[ih, ih2] * VE[t+1, ih2, ik_E, idU_E, itype]
                    ev_U_sep += P_h[ih, ih2] * VU[t+1, ih2, ik_E, idU_E, ib_sep_E, itype]
                ev_quit = ev_U_sep
                ev_stay = (1.0 - sigma_sep) * ev_E_nxt + sigma_sep * ev_U_sep

                # EV: try going sick (accepted → S, rejected → E/U)
                ev_enter_E   = 0.0
                ev_doc_combo = 0.0
                for ih2 in range(Nh):
                    ev_enter_E   += P_h_S[ih, ih2] * (1.0 - p_doc_out[ih2]) * VS[t+1, ih2, ik_E, idU_E, 0, 0, 1, ib_sep_E, itype]
                    ev_doc_combo += P_h[ih, ih2]   * p_doc_out[ih2] * (
                                        (1.0 - sigma_sep) * VE[t+1, ih2, ik_E, idU_E, itype]
                                        + sigma_sep        * VU[t+1, ih2, ik_E, idU_E, ib_sep_E, itype])
                ev_sick = ev_enter_E + ev_doc_combo

                if ev_quit >= ev_stay:
                    best_ns = ev_quit
                    q_best  = 1.0
                else:
                    best_ns = ev_stay
                    q_best  = 0.0

                if ev_sick >= best_ns:
                    VE[t,ih,ik,idU,itype]      = u_E + beta * ev_sick
                    g_E_pol[t,ih,ik,idU,itype] = 1.0
                    q_pol[t,ih,ik,idU,itype]   = 0.0
                else:
                    VE[t,ih,ik,idU,itype]      = u_E + beta * best_ns
                    g_E_pol[t,ih,ik,idU,itype] = 0.0
                    q_pol[t,ih,ik,idU,itype]   = q_best

        # ──────────────────────── V^U ────────────────────────────────────
        for idx_U in prange(Ntype * Nh * Nk):
            itype = idx_U // (Nh * Nk)
            ih    = (idx_U % (Nh * Nk)) // Nk
            ik    = idx_U % Nk
            lam   = lambda_grid[itype]
            h     = h_grid[ih]
            omh   = max(1.0 - h, 1e-8)
            ik_U  = ik_next_U[ik]
            for idU in range(NdU):
                idU_U = dU_next_U[idU]
                for ib in range(Nb):
                    u_U = u_b_U[idU, ib]

                    if last:
                        VU[t,ih,ik,idU,ib,itype]      = u_U
                        s_pol[t,ih,ik,idU,ib,itype]   = 0.0
                        g_U_pol[t,ih,ik,idU,ib,itype] = 0.0
                        continue

                    ev_find    = 0.0
                    ev_nfind   = 0.0
                    ev_enter_U = 0.0
                    ev_doc_U   = 0.0
                    for ih2 in range(Nh):
                        ev_find    += P_h[ih, ih2]   * VE[t+1, ih2, ik_U, idU_U, itype]
                        ev_nfind   += P_h[ih, ih2]   * VU[t+1, ih2, ik_U, idU_U, ib, itype]
                        ev_enter_U += P_h_S[ih, ih2] * (1.0 - p_doc_out[ih2]) * VS[t+1, ih2, ik_U, idU_U, 0, 0, 0, ib, itype]
                        ev_doc_U   += P_h[ih, ih2]   * p_doc_out[ih2]          * VU[t+1, ih2, ik_U, idU_U, ib, itype]
                    ev_sick = ev_enter_U + ev_doc_U + (u_b_floor - u_b_U[idU_U, ib]) * E_p_doc_out[ih]

                    dV = ev_find - ev_nfind
                    if dV > 0.0:
                        s_star = (beta * psi * dV / (omh * lam)) ** (1.0 / gamma)
                        if s_star > 1.0:
                            s_star = 1.0
                    else:
                        s_star = 0.0

                    if idU > 0 and s_star < s_bar:
                        s_star = s_bar

                    pf        = psi * s_star
                    srch_cost = omh * lam * s_star ** (1.0 + gamma) / (1.0 + gamma)
                    part_cost = chi * omh if idU > 0 else 0.0
                    v_search  = u_U - srch_cost - part_cost + beta * (pf * ev_find + (1.0 - pf) * ev_nfind)
                    v_sick    = u_U + beta * ev_sick

                    if v_sick >= v_search:
                        VU[t,ih,ik,idU,ib,itype]      = v_sick
                        g_U_pol[t,ih,ik,idU,ib,itype] = 1.0
                        s_pol[t,ih,ik,idU,ib,itype]   = 0.0
                    else:
                        VU[t,ih,ik,idU,ib,itype]      = v_search
                        g_U_pol[t,ih,ik,idU,ib,itype] = 0.0
                        s_pol[t,ih,ik,idU,ib,itype]   = s_star


# ─────────────────────────────────────────────────────────────────────────────
# Numba: forward distribution
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _forward_distribution(
    T, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype,
    psi, sigma_sep, t_reassess,
    P_h, P_h_S,
    ik_next_U, ik_next_EH,
    dU_next_U, dU_next_E,
    dS_next_S,
    ib_sep_by_ik,
    p_low, p_out, p_doc_out,
    s_pol, g_U_pol, q_pol, g_E_pol, ret_pol, s_S_pol,
    muE0, muU0, muS0,
    muE_path, muU_path, muS_path,
):
    muE_path[0] = muE0
    muU_path[0] = muU0
    muS_path[0] = muS0

    for t in range(T - 1):
        # zero next-period mass
        muE_path[t+1, :, :, :, :]                   = 0.0
        muU_path[t+1, :, :, :, :, :]                = 0.0
        muS_path[t+1, :, :, :, :, :, :, :, :]      = 0.0

        # ── flows from E ──────────────────────────────────────────────────────
        for itype in range(Ntype):
            for ih in range(Nh):
                for ik in range(Nk):
                    ik_E     = ik_next_EH[ik, ih]
                    ib_sep_E = ib_sep_by_ik[ik_E]
                    for idU in range(NdU):
                        idU_E = dU_next_E[idU]
                        m = muE_path[t, ih, ik, idU, itype]
                        if m == 0.0:
                            continue
                        g_E = g_E_pol[t, ih, ik, idU, itype]
                        q   = q_pol[t, ih, ik, idU, itype]
                        for ih2 in range(Nh):
                            ph = P_h[ih, ih2] * m
                            if ph == 0.0:
                                continue
                            if g_E >= 0.5:
                                # try going sick: medical check fires at transition
                                # accepted → sick leave; rejected → stay in E (exog. sep applies)
                                pd_out = p_doc_out[ih2]
                                muS_path[t+1, ih2, ik_E, idU_E, 0, 0, 1, ib_sep_E, itype] += ph * (1.0 - pd_out)
                                muE_path[t+1, ih2, ik_E, idU_E, itype]           += ph * pd_out * (1.0 - sigma_sep)
                                muU_path[t+1, ih2, ik_E, idU_E, ib_sep_E, itype] += ph * pd_out * sigma_sep
                            elif q >= 0.5:
                                # quit → U
                                muU_path[t+1, ih2, ik_E, idU_E, ib_sep_E, itype] += ph
                            else:
                                # stay E with exogenous separation
                                muE_path[t+1, ih2, ik_E, idU_E, itype]           += ph * (1.0 - sigma_sep)
                                muU_path[t+1, ih2, ik_E, idU_E, ib_sep_E, itype] += ph * sigma_sep

        # ── flows from U ──────────────────────────────────────────────────────
        for itype in range(Ntype):
            for ih in range(Nh):
                for ik in range(Nk):
                    ik_U = ik_next_U[ik]
                    for idU in range(NdU):
                        idU_U = dU_next_U[idU]
                        for ib in range(Nb):
                            m = muU_path[t, ih, ik, idU, ib, itype]
                            if m == 0.0:
                                continue
                            g_U = g_U_pol[t, ih, ik, idU, ib, itype]
                            s   = s_pol[t, ih, ik, idU, ib, itype]
                            for ih2 in range(Nh):
                                ph = P_h[ih, ih2] * m
                                if ph == 0.0:
                                    continue
                                if g_U >= 0.5:
                                    # try going sick: medical check fires at transition
                                    # accepted → sick leave; rejected → stay in U
                                    pd_out = p_doc_out[ih2]
                                    muS_path[t+1, ih2, ik_U, idU_U, 0, 0, 0, ib, itype] += ph * (1.0 - pd_out)
                                    muU_path[t+1, ih2, ik_U, idU_U, ib, itype]           += ph * pd_out
                                else:
                                    pf = psi * s
                                    muE_path[t+1, ih2, ik_U, idU_U, itype]     += ph * pf
                                    muU_path[t+1, ih2, ik_U, idU_U, ib, itype] += ph * (1.0 - pf)

        # ── flows from S ──────────────────────────────────────────────────────
        for itype in range(Ntype):
            for ih in range(Nh):
                for ik in range(Nk):
                    ik_S = ik_next_U[ik]
                    for idU in range(NdU):
                        for idS in range(NdS):
                            idS_nxt = dS_next_S[idS]
                            for ir in range(Nr):
                                # Only ir=0 valid before reassessment (rejected workers
                                # never enter sick leave — they stay in E/U).
                                if idS < t_reassess and ir > 0:
                                    continue
                                for io in range(No):
                                    for ib in range(Nb):
                                        m = muS_path[t, ih, ik, idU, idS, ir, io, ib, itype]
                                        if m == 0.0:
                                            continue
                                        ret = ret_pol[t, ih, ik, idU, idS, ir, io, ib, itype]
                                        for ih2 in range(Nh):
                                            ph = P_h_S[ih, ih2] * m
                                            if ph == 0.0:
                                                continue
                                            if ret >= 0.5:
                                                # return from sick leave
                                                if io == 1:  # origin = E
                                                    ib_sep_S = ib_sep_by_ik[ik_S]
                                                    muE_path[t+1, ih2, ik_S, idU, itype]           += ph * (1.0 - sigma_sep)
                                                    muU_path[t+1, ih2, ik_S, idU, ib_sep_S, itype] += ph * sigma_sep
                                                else:  # origin = U
                                                    muU_path[t+1, ih2, ik_S, idU, ib, itype] += ph
                                            else:
                                                # stay sick; reassessment draw at threshold.
                                                # U-origin workers (io=0) may find a job via on-sick search.
                                                pf_S = psi * s_S_pol[t, ih, ik, idU, idS, ir, io, ib, itype]
                                                if io == 0 and pf_S > 0.0:
                                                    muE_path[t+1, ih2, ik_S, idU, itype] += ph * pf_S
                                                ph_stay = ph * (1.0 - pf_S)
                                                if ph_stay == 0.0:
                                                    continue
                                                at_reassess = (idS == t_reassess - 1) and (ir == 0)
                                                if at_reassess:
                                                    pl   = p_low[ih2]
                                                    po   = p_out[ih2]
                                                    ph_h = 1.0 - pl - po
                                                    muS_path[t+1, ih2, ik_S, idU, idS_nxt, 0, io, ib, itype] += ph_stay * ph_h
                                                    muS_path[t+1, ih2, ik_S, idU, idS_nxt, 1, io, ib, itype] += ph_stay * pl
                                                    muS_path[t+1, ih2, ik_S, idU, idS_nxt, 2, io, ib, itype] += ph_stay * po
                                                else:
                                                    muS_path[t+1, ih2, ik_S, idU, idS_nxt, ir, io, ib, itype] += ph_stay


# ─────────────────────────────────────────────────────────────────────────────
# Numba: unemployment cohort — tracks exits to E and exits to S separately
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _hazard_cohort_u(
    max_d, Nh, Nk, NdU, Nb, Ntype,
    psi, P_h,
    ik_next_U, dU_next_U,
    p_doc_out,  # (Nh,) — rejection probability at next-period health
    s_pol,      # (max_d, Nh, Nk, NdU, Nb, Ntype)
    g_U_pol,    # (max_d, Nh, Nk, NdU, Nb, Ntype)
    cohort0,    # (Nh, Nk, NdU, Nb, Ntype)
    at_risk_out, exits_E_out, exits_S_out,
):
    """
    Propagate a U cohort forward.  At each duration:
      - g_U=1: worker tries sick leave; accepted (1-p_doc_out) → exits_S,
        rejected (p_doc_out) → stays in U and continues next period.
      - g_U=0: searches; psi*s fraction exit to E (exits_E).
    """
    cur = cohort0.copy()
    for d in range(1, max_d + 1):
        at_risk = 0.0
        exits_E = 0.0
        exits_S = 0.0
        nxt = np.zeros_like(cur)
        for itype in range(Ntype):
            for ih in range(Nh):
                for ik in range(Nk):
                    ik_U = ik_next_U[ik]
                    for idU in range(NdU):
                        idU_U = dU_next_U[idU]
                        for ib in range(Nb):
                            m = cur[ih, ik, idU, ib, itype]
                            if m == 0.0:
                                continue
                            at_risk += m
                            g_U = g_U_pol[d-1, ih, ik, idU, ib, itype]
                            if g_U >= 0.5:
                                # try sick leave: accepted → exit, rejected → stay in U
                                for ih2 in range(Nh):
                                    ph = P_h[ih, ih2] * m
                                    exits_S += ph * (1.0 - p_doc_out[ih2])
                                    nxt[ih2, ik_U, idU_U, ib, itype] += ph * p_doc_out[ih2]
                            else:
                                s  = s_pol[d-1, ih, ik, idU, ib, itype]
                                pf = psi * s
                                exits_E += m * pf
                                # remaining fraction stays in U
                                for ih2 in range(Nh):
                                    nxt[ih2, ik_U, idU_U, ib, itype] += (
                                        P_h[ih, ih2] * m * (1.0 - pf))

        at_risk_out[d-1]  = at_risk
        exits_E_out[d-1]  = exits_E
        exits_S_out[d-1]  = exits_S
        cur = nxt


# ─────────────────────────────────────────────────────────────────────────────
# Numba: sick-leave cohort — tracks exits to E and exits to U separately
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _hazard_cohort_s(
    max_d, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype,
    sigma_sep, t_reassess,
    psi,
    P_h, ik_next_U, dS_next_S,
    ib_sep_by_ik, p_low, p_out,
    ret_pol,   # (max_d, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype)
    s_S_pol,   # (max_d, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype)
    cohort0,   # (Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype)
    at_risk_out, exits_E_out, exits_U_out,
):
    """
    Propagate a sick-leave cohort forward.  At each duration:
      - ret=1, io=1: exits split as (1-sigma_sep) → E, sigma_sep → U.
      - ret=1, io=0: exits entirely to U.
      - ret=0, io=0: fraction psi*s_S exits to E (on-sick search); rest stays.
      - ret=0, io=1: stays sick with dS transition and reassessment draw.
    Health is integrated over P_h for the staying mass.
    """
    cur = cohort0.copy()
    for d in range(1, max_d + 1):
        at_risk = 0.0
        exits_E = 0.0
        exits_U = 0.0
        nxt = np.zeros_like(cur)
        for itype in range(Ntype):
            for ih in range(Nh):
                for ik in range(Nk):
                    ik_S = ik_next_U[ik]
                    for idU in range(NdU):
                        for idS in range(NdS):
                            idS_nxt = dS_next_S[idS]
                            for ir in range(Nr):
                                # Only ir=0 valid before reassessment.
                                if idS < t_reassess and ir > 0:
                                    continue
                                for io in range(No):
                                    for ib in range(Nb):
                                        m = cur[ih, ik, idU, idS, ir, io, ib, itype]
                                        if m == 0.0:
                                            continue
                                        at_risk += m
                                        ret = ret_pol[d-1, ih, ik, idU, idS, ir, io, ib, itype]
                                        if ret >= 0.5:
                                            # exit sick leave — split by origin
                                            if io == 1:  # from E
                                                exits_E += m * (1.0 - sigma_sep)
                                                exits_U += m * sigma_sep
                                            else:        # from U
                                                exits_U += m
                                            # exits removed from cohort; no propagation
                                        else:
                                            # stay sick: U-origin may find job via on-sick search
                                            pf_S = psi * s_S_pol[d-1, ih, ik, idU, idS, ir, io, ib, itype]
                                            if io == 0 and pf_S > 0.0:
                                                exits_E += m * pf_S
                                            m_stay = m * (1.0 - pf_S)
                                            if m_stay == 0.0:
                                                continue
                                            # propagate staying mass with health transition
                                            at_reassess = (idS == t_reassess - 1) and (ir == 0)
                                            for ih2 in range(Nh):
                                                p = P_h[ih, ih2] * m_stay
                                                if p == 0.0:
                                                    continue
                                                if at_reassess:
                                                    pl = p_low[ih2]
                                                    po = p_out[ih2]
                                                    ph = 1.0 - pl - po
                                                    nxt[ih2, ik_S, idU, idS_nxt, 0, io, ib, itype] += p * ph
                                                    nxt[ih2, ik_S, idU, idS_nxt, 1, io, ib, itype] += p * pl
                                                    nxt[ih2, ik_S, idU, idS_nxt, 2, io, ib, itype] += p * po
                                                else:
                                                    nxt[ih2, ik_S, idU, idS_nxt, ir, io, ib, itype] += p

        at_risk_out[d-1]  = at_risk
        exits_E_out[d-1]  = exits_E
        exits_U_out[d-1]  = exits_U
        cur = nxt


# ─────────────────────────────────────────────────────────────────────────────
# Numba: pooled sick-leave hazard — single pass, parallel over idS
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _pooled_hazard_kernel(muS, ret, s_S, psi, max_d, sigma_sep):
    """
    Single-pass parallel reduction over the full muS / ret arrays.
    Each idS is handled by its own thread, accumulating local scalars.
    muS / ret shape: (T, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype)
    """
    T, Nh, Nk, NdU, NdS, Nr, No, Nb, Ntype = muS.shape
    at_risk    = np.zeros(max_d)
    from_U     = np.zeros(max_d)
    from_E     = np.zeros(max_d)
    from_U_srch= np.zeros(max_d)   # U-origin finding job via on-sick search

    for idS in prange(max_d):
        ar = 0.0
        fu = 0.0
        fe = 0.0
        fu_s = 0.0
        for t in range(T):
            for ih in range(Nh):
                for ik in range(Nk):
                    for idU in range(NdU):
                        for ir in range(Nr):
                            for io in range(No):
                                for ib in range(Nb):
                                    for itype in range(Ntype):
                                        m = muS[t, ih, ik, idU, idS, ir, io, ib, itype]
                                        if m == 0.0:
                                            continue
                                        r = ret[t, ih, ik, idU, idS, ir, io, ib, itype]
                                        ar += m
                                        if io == 0:
                                            fu += m * r
                                            # non-returning U-origin workers who search
                                            if r < 0.5:
                                                s = s_S[t, ih, ik, idU, idS, ir, io, ib, itype]
                                                fu_s += m * psi * s
                                        else:
                                            fe += m * r
        at_risk[idS]     = ar
        from_U[idS]      = fu
        from_E[idS]      = fe
        from_U_srch[idS] = fu_s

    exits_U  = from_U + from_E * sigma_sep
    exits_E  = from_E * (1.0 - sigma_sep) + from_U_srch
    hazard_E = np.where(at_risk > 0.0, exits_E / at_risk, 0.0)
    hazard_U = np.where(at_risk > 0.0, exits_U / at_risk, 0.0)
    return at_risk, exits_E, exits_U, hazard_E, hazard_U
