import numpy as np
from scipy.optimize import minimize_scalar
from EconModel import EconModelClass
from consav.grids import nonlinspace
import pandas as pd
from numba import njit

class UIEmploymentModelClass(EconModelClass):

    def settings(self):
        pass

    def setup(self):

        par = self.par

        # horizon
        par.T = 500

        # discounting
        par.beta = 0.98

        # human capital
        par.alpha = 0.10      # wage return to k, Eq (5)
        par.delta = 0.05      # depreciation in Eq (2)

        # labor market
        par.sigma = 0.05      # exogenous separation prob in Eq (10)
        par.psi = 1           # scaling: job-finding prob = psi*s (Eq 9)

        # UI rights / duration-like state
        par.Ubar = 24         # max UI rights/weeks (discrete)
        par.J = 3             # high-benefit window length
        par.zeta = 2          # re-qualification increment during employment in Eq (3)

        # benefits
        par.b_wel = 0.6
        par.b_emp = 1.5
        par.bmax  = 1.2

        # search costs, Eq (8)
        # par.lambda_1 = 5.0
        par.Ntype = 3
        par.type_shares = np.array([0.30, 0.50, 0.20])
        par.lambda_1_grid = np.array([2.0, 3.0, 4.0])  # example; you calibrate


        # wage level
        par.w = 1.0

        # grids for k
        par.k_max = 20.0
        par.Nk = 30

        # simulation
        # par.simN = 50_000
        par.simT = par.T

        # IMPLEMENTATION SWITCH:
        # if True: benefit is pinned at spell start -> add state b_ui0 grid
        par.benefit_pinned_at_spell_start = True

    def allocate(self):

        par = self.par
        sol = self.sol
        sim = self.sim

        # k grid (continuous)
        par.k_grid = nonlinspace(0.0, par.k_max, par.Nk, 1.1)

        # dU grid (discrete integer 0..Ubar)
        par.dU_grid = np.arange(par.Ubar + 1, dtype=np.int32)

        # optional pinned benefit state grid
        if par.benefit_pinned_at_spell_start:
            b_reset_vals = np.minimum(0.8 * par.w * (1.0 + par.alpha * par.k_grid), par.bmax)

            par.Nb = 10  # try 8–12
            par.b_grid = np.linspace(b_reset_vals.min(), b_reset_vals.max(), par.Nb)

            # map each ik to a bin index
            par.ib_sep_by_ik = np.searchsorted(par.b_grid, b_reset_vals).clip(0, par.Nb-1).astype(np.int32)

            shapeU = (par.T, par.Nk, par.Ubar+1, par.Nb, par.Ntype)
            shapeE = (par.T, par.Nk, par.Ubar+1, par.Nb, par.Ntype)
        else:
            shapeU = (par.T, par.Nk, par.Ubar+1)
            shapeE = (par.T, par.Nk, par.Ubar+1)

        # value functions
        sol.VU = np.full(shapeU, np.nan)
        sol.VE = np.full(shapeE, np.nan)

        # policy functions
        sol.s = np.full(shapeU, np.nan)    # unemployed search effort
        sol.q = np.full(shapeE, np.nan)    # employed quit (0/1)

        # distribution iteration storage (population mass)
        sim.muU = np.zeros((par.T, par.Nk, par.Ubar+1, par.Nb, par.Ntype))
        sim.muE = np.zeros((par.T, par.Nk, par.Ubar+1, par.Nb, par.Ntype))


    # ---------- solve ----------
    def ik_from_k(self, k):
        # nearest index on k-grid (fast and stable)
        return int(np.clip(np.searchsorted(self.par.k_grid, k), 1, self.par.Nk-1))

    def solve(self):
        par, sol = self.par, self.sol

        # --- transitions (same as before, but as arrays) ---
        ik_next_U = np.empty(par.Nk, dtype=np.int32)
        ik_next_E = np.empty(par.Nk, dtype=np.int32)
        for ik, k in enumerate(par.k_grid):
            ik_next_U[ik] = self.ik_from_k((1 - par.delta) * k)
            ik_next_E[ik] = self.ik_from_k((1 - par.delta) * k + 1.0)

        dU_next_U = np.maximum(par.dU_grid - 1, 0).astype(np.int32)
        dU_next_E = np.minimum(par.dU_grid + par.zeta, par.Ubar).astype(np.int32)

        # --- per-state flow utility arrays ---
        # log wage (employed)
        logw = np.log(par.w * (1.0 + par.alpha * par.k_grid)).astype(np.float64)

        # log benefit in unemployment given (dU, ib)
        # benefit depends on dU and ib (pinned level), not directly on k (given pinned)
        logbU = np.empty((par.Ubar + 1, par.Nb), dtype=np.float64)
        for dU in range(par.Ubar + 1):
            for ib in range(par.Nb):
                if dU == 0:
                    b = par.b_wel
                elif dU > (par.Ubar - par.J):
                    b = par.b_emp
                else:
                    b = par.b_grid[ib]  # pinned regular UI
                logbU[dU, ib] = np.log(b)

        # --- allocate solution arrays if not already (you already do this in allocate) ---
        # Ensure float64 arrays (Numba likes consistent dtypes)
        sol.VU = np.asarray(sol.VU, dtype=np.float64)
        sol.VE = np.asarray(sol.VE, dtype=np.float64)
        sol.s  = np.asarray(sol.s,  dtype=np.float64)
        sol.q  = np.asarray(sol.q,  dtype=np.float64)

        # --- call numba solver ---
        solve_numba_pinned_types(
            par.T, par.Nk, par.Ubar, par.Nb, par.Ntype,
            par.beta, par.psi, par.sigma,
            par.lambda_1_grid.astype(np.float64),
            logw, logbU,
            ik_next_U, ik_next_E,
            dU_next_U, dU_next_E,
            par.ib_sep_by_ik.astype(np.int32),
            sol.VU, sol.VE, sol.s, sol.q
        )


    # ---------- simulate ----------
    # small helper: nearest k-grid index (cheap discrete approximation)
    # you can replace with interpolation if you want smoother k transitions
    def _ik(self, k):
        par = self.par
        return int(np.clip(np.searchsorted(par.k_grid, k), 1, par.Nk-1))
    
    def simulate(self):
        par, sol, sim = self.par, self.sol, self.sim

        # precompute transitions (same as solve)
        ik_next_U = np.empty(par.Nk, dtype=np.int32)
        ik_next_E = np.empty(par.Nk, dtype=np.int32)
        for ik, k in enumerate(par.k_grid):
            ik_next_U[ik] = self.ik_from_k((1-par.delta)*k)
            ik_next_E[ik] = self.ik_from_k((1-par.delta)*k + 1)

        dU_next_U = np.maximum(par.dU_grid - 1, 0).astype(np.int32)
        dU_next_E = np.minimum(par.dU_grid + par.zeta, par.Ubar).astype(np.int32)

        # initial distribution:
        # everyone employed at k=0 index, dU=0, and ib arbitrary (choose 0)
        muU0 = np.zeros((par.Nk, par.Ubar+1, par.Nb, par.Ntype))
        muE0 = np.zeros((par.Nk, par.Ubar+1, par.Nb, par.Ntype))

        ik0 = self.ik_from_k(0.0)

        # split initial mass across types according to shares
        for itype in range(par.Ntype):
            muE0[ik0, par.Ubar, 0, itype] = par.type_shares[itype]

        forward_distribution_types(
            par.T, par.Nk, par.Ubar, par.Nb, par.Ntype,
            par.sigma, par.psi,
            ik_next_U, ik_next_E,
            dU_next_U, dU_next_E,
            par.ib_sep_by_ik,
            sol.s, sol.q,
            muU0, muE0,
            sim.muU, sim.muE
        )


    def hazard_from_entry_time(self, t0, max_d=None):
        """
        Hazard-by-duration for a cohort that ENTERS unemployment at calendar time t0.
        Exact, no micro simulation.

        Returns DataFrame with duration 1..max_d.
        """
        par, sol, sim = self.par, self.sol, self.sim

        if max_d is None:
            max_d = (par.T - 1) - t0

        max_d = int(max_d)   # <= 24
        if max_d <= 0:
            raise ValueError("No time left in horizon for hazards at this entry time.")


        # need distributions first
        if not hasattr(sim, "muU") or sim.muU.sum() == 0.0:
            raise RuntimeError("Run simulate_dist() first so we have population distributions.")


        # precompute U transitions
        ik_next_U = np.empty(par.Nk, dtype=np.int32)
        for ik, k in enumerate(par.k_grid):
            ik_next_U[ik] = self.ik_from_k((1-par.delta)*k)
        dU_next_U = np.maximum(par.dU_grid - 1, 0).astype(np.int32)

        # cohort entry distribution at t0:
        # inflow from employment into unemployment between t0 and t0+1.
        # In your timing, separation/quit happens end of period t0.
        # We approximate entry cohort as the mass that goes from E at t0 to U at t0+1.
        cohort0 = np.zeros((par.Nk, par.Ubar+1, par.Nb, par.Ntype))
        muE_t0 = sim.muE[t0]  # (Nk,Ubar+1,Nb,Ntype)

        ik_next_E = np.empty(par.Nk, dtype=np.int32)
        for ik in range(par.Nk):
            ik_next_E[ik] = self.ik_from_k((1-par.delta)*par.k_grid[ik] + 1.0)

        dU_next_E = np.minimum(par.dU_grid + par.zeta, par.Ubar).astype(np.int32)

        for ik in range(par.Nk):
            ib_sep = par.ib_sep_by_ik[ik]
            ike = ik_next_E[ik]
            for dU in range(par.Ubar+1):
                dEn = dU_next_E[dU]
                for ib in range(par.Nb):
                    m = 0.0
                    p_to_U_m = 0.0
                    for itype in range(par.Ntype):
                        m += muE_t0[ik, dU, ib, itype]
                        m_ty = muE_t0[ik, dU, ib, itype]
                        if m_ty == 0.0:
                            continue
                        q = sol.q[t0, ik, dU, ib, itype]
                        quit_prob = 1.0 if q >= 0.5 else 0.0
                        p_to_U = quit_prob + (1.0 - quit_prob) * par.sigma
                        p_to_U_m += m_ty * p_to_U

                    # entrants:
                    cohort0[ike, dEn, ib_sep] += p_to_U_m


        # Now propagate cohort forward for max_d durations using calendar time t0+d-1 policies
        hazard = np.empty(max_d)
        at_risk = np.empty(max_d)
        exits = np.empty(max_d)

        # We do a small numba loop but need s_pol[t0 + d - 1] inside.
        # Easiest: pass a view of policies from t0 onward.
        s_slice = sol.s[t0:t0+max_d, :, :, :, :]          # (max_d, Nk, Ubar+1, Nb, Ntype)

        hazard_from_entry_cohort_types(
            max_d,
            par.Nk, par.Ubar, par.Nb, par.Ntype,
            par.psi,
            ik_next_U, dU_next_U,
            s_slice,
            cohort0,
            hazard, at_risk, exits
        )


        df = pd.DataFrame({
            "duration": np.arange(1, max_d+1),
            "at_risk": at_risk,
            "exits": exits,
            "hazard": hazard
        })
        return df


def extract_ui_spells(j):
    """
    j: array (N,T) with 0=U, 1=E
    returns list of spells as tuples:
    (i, t_start, t_end, duration, exit_to_E)
    where t_end is the last period in U spell (inclusive).
    exit_to_E = 1 if spell ends with U->E, else 0 (censored at end of panel)
    """
    N, T = j.shape
    spells = []

    for i in range(N):
        in_spell = False
        t_start = None

        for t in range(T):
            if (not in_spell) and (j[i, t] == 0):
                in_spell = True
                t_start = t

            # spell ends if we are in U and next period is E OR end of panel
            if in_spell:
                is_last_period = (t == T - 1)
                next_is_E = (not is_last_period) and (j[i, t] == 0) and (j[i, t+1] == 1)
                end_of_panel = is_last_period

                if next_is_E or end_of_panel:
                    t_end = t
                    duration = t_end - t_start + 1
                    exit_to_E = 1 if next_is_E else 0
                    spells.append((i, t_start, t_end, duration, exit_to_E))
                    in_spell = False
                    t_start = None

    return spells

def hazard_by_duration(spells, max_d=None):
    """
    spells: list from extract_ui_spells
    returns dataframe with columns: duration, at_risk, exits, hazard
    duration is 1,2,3,... in spell-time
    """
    durations = np.array([s[3] for s in spells], dtype=int)
    exits = np.array([s[4] for s in spells], dtype=int)

    if max_d is None:
        max_d = durations.max()

    rows = []
    for d in range(1, max_d + 1):
        # at risk: spells with duration >= d
        at_risk = np.sum(durations >= d)

        # exits at d: duration == d AND exit_to_E==1
        exit_d = np.sum((durations == d) & (exits == 1))

        hazard = exit_d / at_risk if at_risk > 0 else np.nan
        rows.append((d, at_risk, exit_d, hazard))

    return pd.DataFrame(rows, columns=["duration", "at_risk", "exits", "hazard"])

@njit
def forward_distribution_types(
    T, Nk, Ubar, Nb, Ntype,
    sigma, psi,
    ik_next_U, ik_next_E,
    dU_next_U, dU_next_E,
    ib_sep_by_ik,
    s_pol, q_pol,             # (T,Nk,Ubar+1,Nb,Ntype)
    muU0, muE0,               # (Nk,Ubar+1,Nb,Ntype)
    muU_path, muE_path        # (T,Nk,Ubar+1,Nb,Ntype)
):
    muU_path[0, :, :, :, :] = muU0
    muE_path[0, :, :, :, :] = muE0

    for t in range(T-1):

        muU_next = muU_path[t+1]
        muE_next = muE_path[t+1]
        muU_next[:, :, :, :] = 0.0
        muE_next[:, :, :, :] = 0.0

        # ---------- from unemployment ----------
        for itype in range(Ntype):
            for ik in range(Nk):
                iku = ik_next_U[ik]
                for dU in range(Ubar+1):
                    dUn = dU_next_U[dU]
                    for ib in range(Nb):
                        m = muU_path[t, ik, dU, ib, itype]
                        if m == 0.0:
                            continue

                        s = s_pol[t, ik, dU, ib, itype]
                        p_find = psi * s
                        if p_find < 0.0:
                            p_find = 0.0
                        if p_find > 1.0:
                            p_find = 1.0

                        muE_next[iku, dUn, ib, itype] += m * p_find
                        muU_next[iku, dUn, ib, itype] += m * (1.0 - p_find)

        # ---------- from employment ----------
        for itype in range(Ntype):
            for ik in range(Nk):
                ike = ik_next_E[ik]
                ib_sep = ib_sep_by_ik[ik]
                for dU in range(Ubar+1):
                    dEn = dU_next_E[dU]
                    for ib in range(Nb):
                        m = muE_path[t, ik, dU, ib, itype]
                        if m == 0.0:
                            continue

                        q = q_pol[t, ik, dU, ib, itype]
                        quit_prob = 1.0 if q >= 0.5 else 0.0

                        p_stay_E = (1.0 - quit_prob) * (1.0 - sigma)
                        p_to_U = 1.0 - p_stay_E

                        muE_next[ike, dEn, ib, itype] += m * p_stay_E
                        muU_next[ike, dEn, ib_sep, itype] += m * p_to_U


@njit
def hazard_from_entry_cohort_types(
    max_d,
    Nk, Ubar, Nb, Ntype,
    psi,
    ik_next_U, dU_next_U,
    s_pol,             # (max_d, Nk, Ubar+1, Nb, Ntype)
    cohort0,           # (Nk, Ubar+1, Nb, Ntype)
    hazard_out,
    at_risk_out,
    exits_out
):
    cur = cohort0.copy()

    for d in range(1, max_d+1):
        at_risk = 0.0
        exits = 0.0
        nxt = np.zeros_like(cur)

        for itype in range(Ntype):
            for ik in range(Nk):
                iku = ik_next_U[ik]
                for dU in range(Ubar+1):
                    dUn = dU_next_U[dU]
                    for ib in range(Nb):
                        m = cur[ik, dU, ib, itype]
                        if m == 0.0:
                            continue

                        at_risk += m

                        s = s_pol[d-1, ik, dU, ib, itype]
                        p_find = psi * s
                        if p_find < 0.0:
                            p_find = 0.0
                        if p_find > 1.0:
                            p_find = 1.0

                        exits += m * p_find
                        nxt[iku, dUn, ib, itype] += m * (1.0 - p_find)

        hazard_out[d-1] = exits / at_risk if at_risk > 0 else np.nan
        at_risk_out[d-1] = at_risk
        exits_out[d-1] = exits
        cur = nxt

@njit
def solve_numba_pinned_types(
    T, Nk, Ubar, Nb, Ntype,
    beta, psi, sigma,
    lambda1_grid,       # (Ntype,)
    logw,               # (Nk,)
    logbU,              # (Ubar+1, Nb)
    ik_next_U, ik_next_E,
    dU_next_U, dU_next_E,
    ib_sep_by_ik,
    VU, VE, s_pol, q_pol   # (T, Nk, Ubar+1, Nb, Ntype)
):
    for t in range(T - 1, -1, -1):
        last = (t == T - 1)

        for itype in range(Ntype):
            lambda1 = lambda1_grid[itype]

            for ik in range(Nk):
                iku = ik_next_U[ik]
                ike = ik_next_E[ik]
                ib_sep = ib_sep_by_ik[ik]
                uE = logw[ik]

                for dU in range(Ubar + 1):
                    dUn = dU_next_U[dU]
                    dEn = dU_next_E[dU]

                    for ib in range(Nb):

                        # -------------------------
                        # Unemployed: choose s in [0,1]
                        # -------------------------
                        uU = logbU[dU, ib]

                        if last:
                            s = 0.0
                            VU[t, ik, dU, ib, itype] = uU
                            s_pol[t, ik, dU, ib, itype] = s
                        else:
                            VEn = VE[t + 1, iku, dUn, ib, itype]
                            VUn = VU[t + 1, iku, dUn, ib, itype]
                            dV = VEn - VUn

                            # C(s)=0.5*lambda1*s^2 => s* = beta*psi*dV/lambda1 clipped
                            gain = beta * psi * dV
                            s = gain / lambda1
                            if s < 0.0:
                                s = 0.0
                            elif s > 1.0:
                                s = 1.0

                            p_find = psi * s
                            if p_find < 0.0:
                                p_find = 0.0
                            elif p_find > 1.0:
                                p_find = 1.0

                            EV = p_find * VEn + (1.0 - p_find) * VUn
                            VU[t, ik, dU, ib, itype] = uU - 0.5 * lambda1 * s * s + beta * EV
                            s_pol[t, ik, dU, ib, itype] = s

                        # -------------------------
                        # Employed: choose q in {0,1}
                        # -------------------------
                        if last:
                            q_pol[t, ik, dU, ib, itype] = 0.0
                            VE[t, ik, dU, ib, itype] = uE
                        else:
                            # Quit => unemployed next period with reset pinned-benefit index
                            V_quit = VU[t + 1, ike, dEn, ib_sep, itype]

                            # Stay => employed next period with same ib, but sep shock sends to U with reset ib
                            V_keepE = VE[t + 1, ike, dEn, ib, itype]
                            V_sep   = VU[t + 1, ike, dEn, ib_sep, itype]
                            V_stay  = (1.0 - sigma) * V_keepE + sigma * V_sep

                            if V_quit > V_stay:
                                q_pol[t, ik, dU, ib, itype] = 1.0
                                VE[t, ik, dU, ib, itype] = uE + beta * V_quit
                            else:
                                q_pol[t, ik, dU, ib, itype] = 0.0
                                VE[t, ik, dU, ib, itype] = uE + beta * V_stay