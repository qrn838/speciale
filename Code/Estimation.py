"""
Estimation.py  –  SMM estimator for FullLaborModelClass
========================================================

Estimation via Nelder-Mead optimisation of the SMM objective.

Standard errors via numerical Jacobian of the moment function
  (delta method / sandwich formula for SMM).

Usage
-----
    from DynLaborSickModel import FullLaborModelClass
    from Estimation import SMMEstimator, PARAM_SPEC, make_data_moments

    # Load / compute your empirical moments
    data_moments = make_data_moments(...)        # or build dict by hand

    est = SMMEstimator(FullLaborModelClass)
    result = est.estimate(data_moments, theta0=theta0)
    print(result["table"])
"""

import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────────────────
# Parameter specification
#   (name, lower_bound, upper_bound, description)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_SPEC = [
    # Job-finding / search technology
    ("psi",          0.10,  5.00,  "job-finding scale"),
    ("gamma",        0.30,  2.50,  "search cost curvature"),
    ("iota",         0.30,  2.50,  "work disutility curvature"),
    # Health-dependent participation cost
    ("chi",          0.00,  6.00,  "participation cost coefficient"),
    # E-origin medical gate — linear in health: P = delta0 + delta1*h
    ("delta0_doc",  -0.20,  0.60,  "E-origin medical gate intercept"),
    ("delta1_doc",  -1.00,  2.00,  "E-origin medical gate slope (health)"),
    # U-origin medical gate (separate screening probabilities for U workers)
    ("delta0_doc_U", -0.20, 0.60,  "U-origin medical gate intercept"),
    ("delta1_doc_U", -1.00, 2.00,  "U-origin medical gate slope (health)"),
    # E-origin reassessment probabilities — linear in health
    ("delta0_low",  -0.30,  0.80,  "E-origin P(reduced benefit | h) intercept"),
    ("delta1_low",  -1.00,  1.50,  "E-origin P(reduced benefit | h) slope"),
    ("delta0_out",  -0.50,  0.80,  "E-origin P(kicked out | h) intercept"),
    ("delta1_out",   0.00,  2.50,  "E-origin P(kicked out | h) slope"),
    # U-origin reassessment probabilities (may differ from E-origin)
    ("delta0_low_U", -0.30, 0.80,  "U-origin P(reduced benefit | h) intercept"),
    ("delta1_low_U", -1.00, 1.50,  "U-origin P(reduced benefit | h) slope"),
    ("delta0_out_U", -0.50, 0.80,  "U-origin P(kicked out | h) intercept"),
    ("delta1_out_U",  0.00, 2.50,  "U-origin P(kicked out | h) slope"),
    # Unobserved heterogeneity — search cost scale
    ("lam0",         0.50, 100.0,  "search cost scale type 0"),
    ("lam1",         0.50, 100.0,  "search cost scale type 1"),
    # Unobserved heterogeneity — work disutility scale
    ("nu0",          0.01,  15.0,  "work disutility scale type 0"),
    ("nu1",          0.01,  15.0,  "work disutility scale type 1"),
    # Type distribution
    ("type_share1",  0.05,  0.95,  "population share of type 1"),
    # Initial health distribution (z-space mean per type; h = sigmoid(z))
    ("z_init_mu0",  -3.00,  3.00,  "initial health z-mean type 0 (z-space)"),
    ("z_init_mu1",  -3.00,  3.00,  "initial health z-mean type 1 (z-space)"),
    # Health AR(1) process
    ("rho_h",        0.50,  0.999, "health AR(1) persistence (monthly)"),
    ("sigma_h",      0.05,  2.00,  "health innovation std"),
    # Utility curvature (CRRA: mu=-1 → log, mu=-sigma where sigma=RRA coefficient)
    ("mu",          -3.00, -0.10,  "CRRA curvature"),
]

PARAM_NAMES  = [p[0] for p in PARAM_SPEC]
PARAM_BOUNDS = [(p[1], p[2]) for p in PARAM_SPEC]
N_PARAMS     = len(PARAM_SPEC)


def _mean_duration_from_hazard(hazard_array):
    """
    Compute mean duration from a hazard rate vector h_1, h_2, ..., h_T.
    Survivor function S(d) = prod_{j=1}^{d} (1 - h_j);  S(0) = 1.
    E[D] = sum_{d=0}^{T-1} S(d)   (using the discrete survival identity).
    """
    h = np.clip(np.asarray(hazard_array, dtype=float), 0.0, 1.0)
    S = np.cumprod(1.0 - h)
    return float(1.0 + S[:-1].sum())   # 1 + S(1) + S(2) + ...


# ─────────────────────────────────────────────────────────────────────────────
# Moment vector helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_moments(model):
    """
    Extract the full moment vector from a solved-and-simulated model.

    Returns
    -------
    dict  {moment_name: scalar}
    """
    hz_ue = model.hazard_u_to_e(t0=0)
    hz_us = model.hazard_u_to_s(t0=0)
    hz_s  = model.hazard_s_pooled()

    dur   = hz_ue["duration"].values
    dur_s = hz_s["duration"].values

    moments = {}

    # ── U → E hazard by duration ──────────────────────────────────────────────
    for d, v in zip(dur, hz_ue["hazard"].values):
        moments[f"hz_ue_d{int(d):02d}"] = float(v)

    # ── U → S hazard by duration ──────────────────────────────────────────────
    for d, v in zip(dur, hz_us["hazard"].values):
        moments[f"hz_us_d{int(d):02d}"] = float(v)

    # ── S → E hazard by duration (pooled) ────────────────────────────────────
    for d, v in zip(dur_s, hz_s["hazard_E"].values):
        moments[f"hz_se_d{int(d):02d}"] = float(v)

    # ── S → U hazard by duration (pooled) ────────────────────────────────────
    for d, v in zip(dur_s, hz_s["hazard_U"].values):
        moments[f"hz_su_d{int(d):02d}"] = float(v)

    # ── S → E by origin ──────────────────────────────────────────────────────
    for d, v in zip(dur_s, hz_s["hazard_E_Eorig"].values):
        moments[f"hz_se_Eorig_d{int(d):02d}"] = float(v)

    # ── S → U by origin ──────────────────────────────────────────────────────
    for d, v in zip(dur_s, hz_s["hazard_U_Uorig"].values):
        moments[f"hz_su_Uorig_d{int(d):02d}"] = float(v)

    # ── Average durations ─────────────────────────────────────────────────────
    moments["avg_u_dur"]    = _mean_duration_from_hazard(
                                  hz_ue["hazard"].values + hz_us["hazard"].values)
    moments["avg_s_dur"]    = _mean_duration_from_hazard(
                                  hz_s["hazard_E"].values + hz_s["hazard_U"].values)

    # ── State shares (time-averaged over the simulated path) ─────────────────
    muE, muU, muS = model.aggregate_shares()
    total = muE + muU + muS
    total = np.where(total > 0, total, 1.0)
    moments["share_E"] = float((muE / total).mean())
    moments["share_U"] = float((muU / total).mean())
    moments["share_S"] = float((muS / total).mean())

    # ── Sick-leave origin split (share of S stock that is E-origin) ──────────
    ar_E = hz_s["at_risk_E"].values.sum()
    ar_T = hz_s["at_risk"].values.sum()
    moments["share_S_Eorig"] = float(ar_E / ar_T) if ar_T > 0 else 0.0

    # ── Average compensation rate at UI entry ─────────────────────────────────
    # For each k level: comp_rate = min(repl_rate*wage, bmax) / wage
    # Weighted by the employment mass at each k level (time-averaged).
    par = model.par
    wages      = par.w * (1.0 + par.alpha * par.k_grid)        # (Nk,)
    benefits   = np.minimum(par.repl_rate * wages, par.bmax)   # (Nk,)
    comp_rates = benefits / wages                               # (Nk,)
    emp_by_k   = model.sim.muE.sum(axis=(0, 1, 3, 4))          # sum over T,Nh,NdU,Ntype → (Nk,)
    emp_total  = emp_by_k.sum()
    moments["avg_comp_rate"] = float(
        (comp_rates * emp_by_k).sum() / emp_total
    ) if emp_total > 1e-12 else np.nan

    # ── Exhaustion spike: ratio of U→E hazard at exhaustion vs 6 periods before
    #    (captures the incentive effect of UI cliff)                            ─
    ubar = model.par.Ubar
    h_all = hz_ue["hazard"].values
    if ubar <= len(h_all):
        h_before = h_all[max(0, ubar - 7): ubar - 1].mean()
        h_at     = h_all[ubar - 1]
        moments["exhaustion_spike"] = (float(h_at / h_before)
                                       if h_before > 1e-8 else np.nan)
    else:
        moments["exhaustion_spike"] = np.nan

    return moments


def moments_to_vector(moments_dict, keys=None):
    """Flatten a moments dict to a numpy array (dropping NaN entries)."""
    if keys is None:
        keys = sorted(moments_dict.keys())
    vals = np.array([moments_dict[k] for k in keys], dtype=float)
    return vals, keys


def make_data_moments(
    hz_ue_df=None,
    hz_us_df=None,
    hz_se_df=None,
    hz_su_df=None,
    hz_se_Eorig_df=None,
    hz_su_Uorig_df=None,
    avg_u_dur=None,
    avg_s_dur=None,
    share_E=None,
    share_U=None,
    share_S=None,
    share_S_Eorig=None,
    exhaustion_spike=None,
):
    """
    Build a data moments dict in the same format as _model_moments().

    Each hazard DataFrame should have columns 'duration' and 'hazard'.
    Scalar moments can be passed directly.

    Any moment left as None will be excluded from estimation (i.e. the
    corresponding model moment will be dropped too when computing the
    objective).
    """
    moments = {}

    def _add_hazard(df, key_prefix):
        if df is None:
            return
        for _, row in df.iterrows():
            d = int(round(float(row["duration"])))
            v = float(row["hazard"])
            if d < 1 or np.isnan(v) or v == 0:
                continue
            moments[f"{key_prefix}_d{d:02d}"] = v

    _add_hazard(hz_ue_df,       "hz_ue")
    _add_hazard(hz_us_df,       "hz_us")
    _add_hazard(hz_se_df,       "hz_se")
    _add_hazard(hz_su_df,       "hz_su")
    _add_hazard(hz_se_Eorig_df, "hz_se_Eorig")
    _add_hazard(hz_su_Uorig_df, "hz_su_Uorig")

    for name, val in [
        ("avg_u_dur",       avg_u_dur),
        ("avg_s_dur",       avg_s_dur),
        ("share_E",         share_E),
        ("share_U",         share_U),
        ("share_S",         share_S),
        ("share_S_Eorig",   share_S_Eorig),
        ("exhaustion_spike",exhaustion_spike),
    ]:
        if val is not None:
            moments[name] = float(val)

    return moments


def make_weight_matrix(se_dict, data_moments):
    """
    Build an inverse-variance diagonal weight matrix aligned to data_moments.

    Each moment is weighted by 1/se², then normalised so the mean weight
    equals 1 (keeping Q on the same scale as the identity-weighted case).
    Moments absent from se_dict receive weight 1.0 before normalisation.

    Parameters
    ----------
    se_dict : dict  {moment_key: standard_error}
        Standard errors from the empirical hazard estimation.
    data_moments : dict  {moment_key: value}
        Output of make_data_moments().

    Returns
    -------
    dict  {moment_key: weight}
        Pass directly to SMMEstimator.estimate() / .objective() as W.

    Example
    -------
    >>> se_dict = {f"hz_ue_d{d:02d}": se  for d, se in zip(durations, se_vals)}
    >>> W = make_weight_matrix(se_dict, data_moments)
    >>> result = est.estimate(data_moments, W=W, theta0=theta0)
    """
    raw = {}
    for k in data_moments:
        se = se_dict.get(k, None)
        if se is not None and se > 0:
            raw[k] = 1.0 / (se ** 2)
        else:
            raw[k] = 1.0

    # Normalise so mean weight = 1
    mean_w = np.mean(list(raw.values()))
    return {k: v / mean_w for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SMM Estimator
# ─────────────────────────────────────────────────────────────────────────────

class SMMEstimator:
    """
    SMM estimator for FullLaborModelClass.

    Parameters
    ----------
    ModelClass : class
        The model class to instantiate (e.g. FullLaborModelClass).
    calibrated : dict, optional
        Parameters held fixed at specified values, e.g.
        {"beta": 0.98, "tau": 0.30}.  Must be valid par attributes.
    param_spec : list, optional
        Override the default PARAM_SPEC list.
    """

    def __init__(self, ModelClass, calibrated=None, param_spec=None,
                 update_w_avg=False):
        self.ModelClass   = ModelClass
        self.calibrated   = calibrated or {}
        self.param_spec   = param_spec or PARAM_SPEC
        self.param_names  = [p[0] for p in self.param_spec]
        self.param_bounds = [(p[1], p[2]) for p in self.param_spec]
        self._moment_keys = None   # set on first evaluation
        self.update_w_avg = update_w_avg  # if True: re-calibrate w_avg at each eval

    # ── parameter handling ────────────────────────────────────────────────────

    def _build_model(self, theta):
        """
        Instantiate and fully set up a model from parameter vector theta.
        Returns the model instance (not yet solved).
        """
        model = self.ModelClass()
        model.setup()

        par = model.par

        # Apply calibrated overrides
        for name, val in self.calibrated.items():
            setattr(par, name, val)

        # Apply estimated parameters
        for name, val in zip(self.param_names, theta):
            if name == "lam0":
                par.lambda_grid[0] = val
            elif name == "lam1":
                par.lambda_grid[1] = val
            elif name == "nu0":
                par.nu_grid[0] = val
            elif name == "nu1":
                par.nu_grid[1] = val
            elif name == "type_share1":
                par.type_shares = np.array([1.0 - val, val])
            elif name == "z_init_mu0":
                par.z_init_mu[0] = val
            elif name == "z_init_mu1":
                par.z_init_mu[1] = val
            else:
                setattr(par, name, val)

        model.allocate()
        return model

    def _run_model(self, theta, override_par=None):
        """Build, solve, and simulate model. Returns model or None on failure."""
        try:
            model = self._build_model(theta)
            if override_par:
                for k, v in override_par.items():
                    setattr(model.par, k, v)
                model.allocate()   # recompute grids with overridden parameters
            model.solve()
            model.simulate()

            # Step-2 mode: re-calibrate w_avg from simulated model, then re-solve
            if self.update_w_avg and 'w_avg' in self.calibrated:
                w_avg_sim = model.avg_wage()
                if np.isfinite(w_avg_sim) and abs(w_avg_sim - self.calibrated['w_avg']) > 1e-6:
                    self.calibrated['w_avg'] = w_avg_sim
                    model = self._build_model(theta)
                    model.solve()
                    model.simulate()

            return model
        except Exception as e:
            warnings.warn(f"Model failed for theta={theta}: {e}")
            return None

    # ── moment computation ────────────────────────────────────────────────────

    def compute_moments(self, theta):
        """
        Evaluate the model at theta and return the moment dict.
        Returns None if the model fails.
        """
        model = self._run_model(theta)
        if model is None:
            return None
        return _model_moments(model)

    def _align_moments(self, model_moments, data_moments):
        """
        Return aligned (model_vec, data_vec, keys) for the intersection
        of moments present in both dicts and not NaN in either.
        """
        common_keys = sorted(
            k for k in data_moments
            if k in model_moments
            and not np.isnan(data_moments[k])
            and not np.isnan(model_moments.get(k, np.nan))
        )
        m_vec = np.array([model_moments[k] for k in common_keys])
        d_vec = np.array([data_moments[k]  for k in common_keys])
        return m_vec, d_vec, common_keys

    # ── objective function ────────────────────────────────────────────────────

    def objective(self, theta, data_moments, W=None, verbose=False):
        """
        SMM objective Q(θ) = (m(θ) - m_data)' W (m(θ) - m_data).

        Parameters
        ----------
        theta : array-like, length N_PARAMS
        data_moments : dict
        W : ndarray or None
            Weighting matrix.  None → identity (equal weights).
        verbose : bool
            Print each evaluation.

        Returns
        -------
        float  (1e10 on model failure)
        """
        model_moments = self.compute_moments(theta)
        if model_moments is None:
            return 1e10

        m_vec, d_vec, keys = self._align_moments(model_moments, data_moments)
        if self._moment_keys is None:
            self._moment_keys = keys

        dev = m_vec - d_vec
        if W is None:
            Q = float(dev @ dev)
        elif isinstance(W, dict):
            # Dict-based weights: extract 1-d weight vector aligned to keys
            w = np.array([W.get(k, 1.0) for k in keys])
            Q = float(dev * w @ dev)
        else:
            # Legacy: numpy diagonal matrix — subset to matched keys if needed
            if W.shape[0] == len(keys):
                Q = float(dev @ W @ dev)
            else:
                Q = float(dev @ dev)

        if verbose:
            print(f"  Q={Q:.6f}  theta={np.round(theta, 4)}")

        return Q

    # ── two-stage estimation ──────────────────────────────────────────────────

    def estimate(
        self,
        data_moments,
        W=None,
        theta0=None,
        nm_maxiter=5000,
        verbose=False,
        patience=100,
        patience_tol=1e-6,
    ):
        """
        Estimate parameters by minimising the SMM objective with Nelder-Mead.

        Parameters
        ----------
        data_moments : dict
            Output of make_data_moments() or a manually constructed dict.
        W : ndarray or None
            Weighting matrix (None → identity).
        theta0 : array-like
            Starting values (required).
        nm_maxiter : int
            Max Nelder-Mead iterations.
        verbose : bool
            Print Q at every single evaluation.
        patience : int
            Stop early if best Q has not improved by more than patience_tol
            over this many consecutive iterations (0 = disabled).
        patience_tol : float
            Minimum improvement in Q to reset the patience counter.

        Returns
        -------
        dict with keys:
            "theta"     : best parameter vector
            "Q"         : objective value at best theta
            "table"     : DataFrame with estimates and bounds
            "nm_result" : raw Nelder-Mead OptimizeResult
        """
        if theta0 is None:
            raise ValueError("theta0 must be provided")

        _n_iters        = [0]
        _best_Q         = [1e10]
        _best_Q_at_iter = [1e10]
        _no_improve     = [0]
        _theta_best     = [np.asarray(theta0, dtype=float).copy()]

        lo = np.array([b[0] for b in self.param_bounds])
        hi = np.array([b[1] for b in self.param_bounds])

        def _obj(theta):
            theta_clipped = np.clip(theta, lo, hi)
            Q = self.objective(theta_clipped, data_moments, W, verbose)
            if Q < _best_Q[0]:
                _best_Q[0] = Q
                _theta_best[0] = theta_clipped.copy()
            return Q

        def _nm_cb(xk):
            _n_iters[0] += 1
            print(f"  iter {_n_iters[0]:4d}  |  best Q = {_best_Q[0]:.6f}")
            # early stopping
            if patience > 0:
                if _best_Q_at_iter[0] - _best_Q[0] > patience_tol:
                    _no_improve[0] = 0
                    _best_Q_at_iter[0] = _best_Q[0]
                else:
                    _no_improve[0] += 1
                if _no_improve[0] >= patience:
                    print(f"  Early stop: no improvement in {patience} iterations.")
                    raise StopIteration

        # Build initial simplex: theta0 as first vertex, then N vertices each
        # perturbed by 20% of the parameter range — much larger than scipy's default 5%,
        # which is too small for 26 parameters on a flat landscape.
        n_p   = len(self.param_names)
        t0    = np.asarray(theta0, dtype=float)
        simplex = np.tile(t0, (n_p + 1, 1))
        for j in range(n_p):
            step = 0.20 * (hi[j] - lo[j])
            simplex[j + 1, j] = np.clip(t0[j] + step, lo[j], hi[j])

        print(f"Nelder-Mead  ({len(self.param_names)} parameters)"
              f"  patience={patience}")
        try:
            nm_result = minimize(
                _obj,
                x0       = t0,
                method   = "Nelder-Mead",
                callback = _nm_cb,
                options  = {
                    "maxiter":        nm_maxiter,
                    "xatol":          1e-5,
                    "fatol":          1e-6,
                    "disp":           False,
                    "adaptive":       True,
                    "initial_simplex": simplex,
                },
            )
            theta_best = np.clip(nm_result.x, lo, hi)
        except StopIteration:
            theta_best = _theta_best[0]
            nm_result  = None

        print(f"  done: Q = {_best_Q[0]:.6f}  ({_n_iters[0]} iterations)")

        table = self._make_table(theta_best)
        return {
            "theta":     theta_best,
            "Q":         float(_best_Q[0]),
            "table":     table,
            "nm_result": nm_result,
        }

    # ── standard errors ───────────────────────────────────────────────────────

    def standard_errors(self, theta_hat, data_moments, W=None, h=1e-4):
        """
        Asymptotic standard errors via the SMM sandwich formula.

        V(θ̂) = (G'WG)^{-1} G'W Ω W G (G'WG)^{-1}  / n

        Here we set n=1 (relative SEs) and use the numerical Jacobian G
        of the moment function.  Ω is estimated as diag(m_data) under
        the assumption that moments are sample averages.

        Parameters
        ----------
        theta_hat : array-like
        data_moments : dict
        W : ndarray or None
        h : float
            Step size for numerical Jacobian.

        Returns
        -------
        se : ndarray, shape (N_PARAMS,)
        """
        n_p = len(theta_hat)

        # Compute moment Jacobian G  (n_moments × n_params)
        m0 = self.compute_moments(theta_hat)
        if m0 is None:
            return np.full(n_p, np.nan)
        _, d_vec, keys = self._align_moments(m0, data_moments)
        n_m = len(keys)

        G = np.zeros((n_m, n_p))
        for j in range(n_p):
            theta_p = theta_hat.copy()
            theta_m = theta_hat.copy()
            lo, hi  = self.param_bounds[j]
            step    = h * max(abs(theta_hat[j]), 1e-4)
            theta_p[j] = min(theta_hat[j] + step, hi)
            theta_m[j] = max(theta_hat[j] - step, lo)
            mp = self.compute_moments(theta_p)
            mm = self.compute_moments(theta_m)
            if mp is None or mm is None:
                G[:, j] = 0.0
                continue
            mp_v = np.array([mp.get(k, np.nan) for k in keys])
            mm_v = np.array([mm.get(k, np.nan) for k in keys])
            G[:, j] = (mp_v - mm_v) / (theta_p[j] - theta_m[j] + 1e-16)

        # Weighting
        if W is None or W.shape[0] != n_m:
            W_use = np.eye(n_m)
        else:
            W_use = W

        # Moment variance: diagonal with data_moments values
        # (placeholder — replace with bootstrap estimates when available)
        Omega = np.diag(np.maximum(np.abs(d_vec), 1e-6))

        GWG   = G.T @ W_use @ G
        try:
            GWG_inv = np.linalg.inv(GWG + 1e-10 * np.eye(n_p))
        except np.linalg.LinAlgError:
            return np.full(n_p, np.nan)

        V   = GWG_inv @ G.T @ W_use @ Omega @ W_use @ G @ GWG_inv
        se  = np.sqrt(np.maximum(np.diag(V), 0.0))
        return se

    def optimal_weight_matrix(self, theta_hat, data_moments, n_bootstrap=200,
                               rng_seed=0):
        """
        Estimate the optimal weighting matrix via parametric bootstrap:
        resample simulated moments from a model at theta_hat and compute
        their covariance.

        This is a simple implementation: add small noise to theta and
        re-evaluate moments to approximate sampling variation.

        Returns W_opt = Omega^{-1}.
        """
        rng = np.random.default_rng(rng_seed)
        m0  = self.compute_moments(theta_hat)
        if m0 is None:
            return None
        _, _, keys = self._align_moments(m0, data_moments)
        n_m = len(keys)

        draws = []
        noise_scale = 0.02   # 2% perturbation of each parameter
        for _ in range(n_bootstrap):
            theta_b = theta_hat + rng.normal(0, noise_scale, size=len(theta_hat))
            # Clip to bounds
            for j, (lo, hi) in enumerate(self.param_bounds):
                theta_b[j] = np.clip(theta_b[j], lo, hi)
            mb = self.compute_moments(theta_b)
            if mb is None:
                continue
            draws.append(np.array([mb.get(k, np.nan) for k in keys]))

        if len(draws) < 10:
            warnings.warn("Too few bootstrap draws succeeded; returning identity.")
            return np.eye(n_m)

        draws = np.array(draws)
        Omega = np.cov(draws.T) + 1e-8 * np.eye(n_m)
        try:
            return np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            return np.eye(n_m)

    # ── results display ───────────────────────────────────────────────────────

    def _make_table(self, theta, se=None):
        rows = []
        desc = {p[0]: p[3] for p in self.param_spec}
        for j, name in enumerate(self.param_names):
            lo, hi = self.param_bounds[j]
            row = {
                "parameter":   name,
                "description": desc.get(name, ""),
                "estimate":    theta[j],
                "lower_bound": lo,
                "upper_bound": hi,
            }
            if se is not None:
                row["std_err"] = se[j]
                row["t_stat"]  = theta[j] / se[j] if se[j] > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows).set_index("parameter")

    def results_table(self, theta_hat, data_moments=None, W=None,
                      compute_se=False):
        """
        Pretty-print estimates, optionally with standard errors.

        Parameters
        ----------
        theta_hat : array-like
        data_moments : dict, optional  (required if compute_se=True)
        W : ndarray, optional
        compute_se : bool

        Returns
        -------
        pd.DataFrame
        """
        se = None
        if compute_se and data_moments is not None:
            print("Computing standard errors (numerical Jacobian)...")
            se = self.standard_errors(theta_hat, data_moments, W)
        table = self._make_table(np.asarray(theta_hat), se)
        print("\n" + "="*60)
        print("SMM Estimates")
        print("="*60)
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.max_colwidth", 40):
            print(table.to_string())
        return table

    # ── hazard fit plot ───────────────────────────────────────────────────────

    def plot_fit(self, theta, data_moments, figsize=None, override_par=None,
                 ci_dict=None, raw_data=None):
        """
        Plot simulated vs empirical hazard rates for every moment group
        present in data_moments.

        Parameters
        ----------
        theta : array-like
        data_moments : dict
        figsize : tuple or None  (auto-sized if None)
        ci_dict : dict or None
            {moment_key: (lower, upper)} confidence interval bounds.
        raw_data : dict or None
            {prefix: DataFrame} mapping e.g. 'hz_se' → original DataFrame with
            columns 'duration', 'hazard', 'hazard_lower', 'hazard_upper'.
            When provided, the empirical series and CI band are plotted using
            the original continuous duration values rather than the integer keys,
            giving smooth curves matching the data presentation notebook.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from collections import defaultdict

        model = self._run_model(theta, override_par=override_par)
        if model is None:
            print("Model failed — cannot plot fit.")
            return None

        model_mom = _model_moments(model)
        _, _, keys = self._align_moments(model_mom, data_moments)

        # Group keys by prefix (everything before the last '_d')
        groups = defaultdict(list)
        for k in keys:
            prefix = k.rsplit('_d', 1)[0] if '_d' in k else '__scalars__'
            groups[prefix].append(k)

        n_panels = len(groups)
        if figsize is None:
            figsize = (4.5 * n_panels, 4)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes[0]

        TITLES = {
            'hz_ue':          'U → E  (job finding)',
            'hz_us':          'U → S  (sick-leave entry)',
            'hz_se':          'S → E  (return to work, pooled)',
            'hz_su':          'S → U  (exit to unemp., pooled)',
            'hz_se_Eorig':    'S → E  (E-origin workers)',
            'hz_su_Uorig':    'S → U  (U-origin workers)',
            '__scalars__':    'Scalar moments',
        }

        for ax, (prefix, ks) in zip(axes, groups.items()):
            # Integer durations for the model line (always from moment keys)
            int_durs = [int(k.split('_d')[-1]) for k in ks]
            m_vals   = [model_mom[k] for k in ks]

            # Empirical series: use raw DataFrame if provided (continuous x-axis)
            # otherwise fall back to integer durations from moment keys
            df_raw = (raw_data or {}).get(prefix)
            if df_raw is not None:
                d_x  = df_raw['duration'].values
                d_vals = df_raw['hazard'].values
                lo   = df_raw['hazard_lower'].values if 'hazard_lower' in df_raw.columns else None
                hi   = df_raw['hazard_upper'].values if 'hazard_upper' in df_raw.columns else None
            else:
                d_x    = int_durs
                d_vals = [data_moments[k] for k in ks]
                if ci_dict is not None:
                    lo = [ci_dict[k][0] if k in ci_dict else np.nan for k in ks]
                    hi = [ci_dict[k][1] if k in ci_dict else np.nan for k in ks]
                else:
                    lo = hi = None

            if lo is not None and hi is not None:
                ax.fill_between(d_x, lo, hi, color='darkred', alpha=0.15, label='95% CI')
            ax.plot(d_x, d_vals, '-',  color='darkred', label='Data',  lw=1.8)
            # ci_dict fallback for when raw_data is not provided
            if df_raw is None and ci_dict is not None and lo is None:
                lo = [ci_dict[k][0] if k in ci_dict else np.nan for k in ks]
                hi = [ci_dict[k][1] if k in ci_dict else np.nan for k in ks]
                ax.fill_between(int_durs, lo, hi, color='darkred', alpha=0.15)
            ax.plot(int_durs, m_vals, '--', color='tan', label='Model', lw=1.8)
            ax.set_title(TITLES.get(prefix, prefix.replace('_', ' ')), fontsize=10)
            ax.set_xlabel('Duration (months)', fontsize=9)
            ax.set_ylabel('Hazard rate', fontsize=9)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
            ax.legend(fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        plt.close(fig)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: moment comparison table
# ─────────────────────────────────────────────────────────────────────────────

def moment_fit_table(model_moments, data_moments):
    """
    Side-by-side comparison of model vs data moments.

    Parameters
    ----------
    model_moments : dict   (output of _model_moments or SMMEstimator.compute_moments)
    data_moments  : dict   (output of make_data_moments)

    Returns
    -------
    pd.DataFrame
    """
    common = sorted(k for k in data_moments if k in model_moments)
    rows = []
    for k in common:
        d = data_moments[k]
        m = model_moments[k]
        rows.append({
            "moment":     k,
            "data":       d,
            "model":      m,
            "deviation":  m - d,
            "rel_dev_%":  100 * (m - d) / abs(d) if abs(d) > 1e-10 else np.nan,
        })
    return pd.DataFrame(rows).set_index("moment")
