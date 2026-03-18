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
    ("psi",          0.05,  1.50,  "job-finding scale"),
    ("gamma",        0.30,  2.50,  "search cost curvature"),
    ("iota",         0.30,  2.50,  "work disutility curvature"),
    # Health-dependent participation cost
    ("chi",          0.00,  6.00,  "participation cost coefficient"),
    # Health dynamics
    ("rho_h",        0.50,  0.995, "health AR(1) persistence"),
    ("sigma_h",      0.05,  1.20,  "health shock std dev"),
    ("delta_h_S",   -0.10,  1.00,  "health recovery drift on sick leave"),
    # Medical documentation gate
    ("delta0_doc",   0.00,  0.60,  "medical gate intercept"),
    ("delta1_doc",   0.00,  1.50,  "medical gate slope (health)"),
    # Sick-leave benefit structure
    ("b_sick_low",   0.20,  0.95,  "intermediate benefit post-reassessment"),
    # Reassessment probabilities — linear in health: P = delta0 + delta1 * h
    ("delta0_low",  -0.30,  0.80,  "P(reduced benefit | h) intercept"),
    ("delta1_low",  -1.00,  1.50,  "P(reduced benefit | h) slope"),
    ("delta0_out",  -0.50,  0.80,  "P(kicked out | h) intercept"),
    ("delta1_out",   0.00,  2.50,  "P(kicked out | h) slope"),
    # Unobserved heterogeneity — search cost scale
    ("lam0",         0.50,  30.0,  "search cost scale type 0"),
    ("lam1",         0.50,  30.0,  "search cost scale type 1"),
    # Unobserved heterogeneity — work disutility scale
    ("nu0",          0.01,  5.00,  "work disutility scale type 0"),
    ("nu1",          0.01,  5.00,  "work disutility scale type 1"),
    # Type distribution
    ("type_share1",  0.05,  0.95,  "population share of type 1"),
    # Initial health distribution (type 1 shift relative to type 0)
    ("h_init_mu1",  -3.00,  0.50,  "initial health z-shift type 1"),
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
            d = int(row["duration"])
            v = float(row["hazard"])
            if not np.isnan(v):
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

    def __init__(self, ModelClass, calibrated=None, param_spec=None):
        self.ModelClass  = ModelClass
        self.calibrated  = calibrated or {}
        self.param_spec  = param_spec or PARAM_SPEC
        self.param_names = [p[0] for p in self.param_spec]
        self.param_bounds= [(p[1], p[2]) for p in self.param_spec]
        self._moment_keys = None   # set on first evaluation

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
            elif name == "h_init_mu1":
                par.h_init_mu[1] = val
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
        else:
            # Subset W to matched keys if needed
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
        progress_every=5,
        verbose=False,
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
        progress_every : int
            Print best Q every this many function evaluations (0 = off).
        verbose : bool
            Print Q at every single evaluation.

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

        _n_evals = [0]
        _best_Q  = [1e10]

        def _obj(theta):
            _n_evals[0] += 1
            Q = self.objective(theta, data_moments, W, verbose)
            if Q < _best_Q[0]:
                _best_Q[0] = Q
            if progress_every and _n_evals[0] % progress_every == 0:
                print(f"  eval {_n_evals[0]:5d}  |  best Q = {_best_Q[0]:.6f}")
            return Q

        def _nm_cb(xk):
            print(f"  iter {_n_evals[0]:5d} evals  |  best Q = {_best_Q[0]:.6f}")

        print(f"Nelder-Mead  ({len(self.param_names)} parameters)")
        nm_result = minimize(
            _obj,
            x0       = np.asarray(theta0, dtype=float),
            method   = "Nelder-Mead",
            callback = _nm_cb,
            options  = {
                "maxiter":  nm_maxiter,
                "xatol":    1e-5,
                "fatol":    1e-6,
                "disp":     True,
                "adaptive": True,
            },
        )
        theta_best = nm_result.x
        print(f"  done: Q = {nm_result.fun:.6f}")

        table = self._make_table(theta_best)
        return {
            "theta":     theta_best,
            "Q":         float(nm_result.fun),
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

    def plot_fit(self, theta, data_moments, figsize=None, override_par=None):
        """
        Plot simulated vs empirical hazard rates for every moment group
        present in data_moments.

        Each distinct moment prefix (the part before '_bin') gets its own
        panel so it is easy to see where the model fits well or poorly.

        Parameters
        ----------
        theta : array-like
        data_moments : dict
        figsize : tuple or None  (auto-sized if None)

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt
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
            # Extract duration from key suffix (e.g. 'hz_se_d03' → 3)
            durations = [int(k.split('_d')[-1]) for k in ks]
            d_vals    = [data_moments[k] for k in ks]
            m_vals    = [model_mom[k]    for k in ks]

            ax.plot(durations, d_vals, 'o-',  color='darkred',   label='Data',  lw=1.8, ms=5)
            ax.plot(durations, m_vals, 's--', color='steelblue', label='Model', lw=1.8, ms=5)
            ax.set_title(TITLES.get(prefix, prefix.replace('_', ' ')), fontsize=10)
            ax.set_xlabel('Duration (months)', fontsize=9)
            ax.set_ylabel('Hazard rate', fontsize=9)
            ax.legend(fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        plt.show()
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
