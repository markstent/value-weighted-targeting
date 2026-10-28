"""Data-generating processes.

Every DGP is split into `draw_features`, a `structural_<name>` step (latent
churn, tau, baseline and any DGP-specific latents) and a `generate_<name>`
wrapper that then draws treatment and potential outcomes.  The RNG draw order
inside each generator is identical to the original monolithic generators, so a
given RandomState seed reproduces the same population.

corr_gamma and heavytail are legacy exceptions: their LTV noise is drawn AFTER
the treatment/outcome draws (as in the original code), so `structural_*` for
those two omits `true_ltv` and the wrapper adds it last.
"""

import math

import numpy as np

TREATMENT_PROB = 0.50

# --- Population moments (verified via N=2M draw, fixed for reproducibility) ---
# These replace within-sample standardisation to prevent train/test leakage.
POP_MEAN = {
    "account_age": 24.01,
    "weekly_spend": 45.56,
    "login_freq": 4.02,
    "bonus_history": 3.01,
}
POP_SD = {
    "account_age": 7.98,
    "weekly_spend": 42.51,
    "login_freq": 1.97,
    "bonus_history": 1.47,
}

FEATURE_NAMES = ["account_age", "weekly_spend", "login_freq", "bonus_history"]

# --- DGP Specification ---
DGP_SPEC = {
    "features": {
        "account_age": "N(24, 8), clipped [1, 60]",
        "weekly_spend": "exp(N(3.5, 0.8)), clipped [5, 500]",
        "login_freq": "Poisson(4), clipped [1, 15]",
        "bonus_history": "N(3, 1.5), clipped [0, 10]",
    },
    "standardisation": "Population parameters (fixed across replications)",
    "churn": (
        "logit = -1.0 + 0.04*(30-age) - 0.25*login + 0.01*(40-spend) + 0.1*z_bonus + N(0,0.4); "
        "sigmoid, clip [0.01, 0.99]"
    ),
    "ltv": "10*spend + 40*login + 5*age + N(0,80), clip [50, 8000]",
    "ltv_proxy": "12*spend^0.9 + 35*login + N(0,200) -- misspecified form for estimation",
    "cate": (
        "tau_base = 0.06 + 0.10*z_login - 0.02*z_login^2 - 0.04*z_spend; churn_mod = 4*p*(1-p); "
        "tau = tau_base * churn_mod + N(0,0.03), clip [-0.20, 0.40]"
    ),
    "baseline": "0.6 + 0.15*z_login + 0.05*z_spend - 0.08*z_age + N(0,0.05), clip [0.05, 0.95]",
    "treatment": "Bernoulli(0.5)",
    "note": (
        "The CATE-churn moderation (4*p*(1-p)) creates an inverse-U relationship: treatment effects "
        "peak at moderate churn (p=0.5) and are attenuated at extremes."
    ),
}

# Normalising constant for the asymmetric moderation p(1-p)^3 (max at p=0.25).
_ASYM_PEAK = 0.25 * (1 - 0.25) ** 3  # = 0.10546875

V_CORR_SCALE = 180.0  # maps standardised tau_base into LTV units (moderate corr ~+/-0.35)


def _standardise(values, feature_name):
    """Standardise using fixed population parameters."""
    return (values - POP_MEAN[feature_name]) / POP_SD[feature_name]


def churn_moderation_sym(p):
    """Symmetric inverse-U moderation 4 p (1 - p)."""
    return 4.0 * p * (1.0 - p)


def churn_moderation_asym(p):
    """Asymmetric inverse-U: peaks near p=0.25, decays faster at high churn."""
    return np.clip(p * (1.0 - p) ** 3 / _ASYM_PEAK, 0.0, 1.0)


# --- Features ---


def draw_features(n, rng):
    """Independent-feature draw: age, spend, login, bonus (in that RNG order)."""
    account_age = rng.normal(24, 8, n).clip(1, 60)
    weekly_spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login_freq = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus_history = rng.normal(3, 1.5, n).clip(0, 10)
    return {
        "account_age": account_age,
        "weekly_spend": weekly_spend,
        "login_freq": login_freq,
        "bonus_history": bonus_history,
    }


def draw_features_correlated(n, rng):
    """Correlated-feature draw via Gaussian copula [M9]."""
    corr = np.array(
        [
            [1.0, 0.3, 0.4, 0.0],  # age
            [0.3, 1.0, 0.0, 0.0],  # spend (underlying normal)
            [0.4, 0.0, 1.0, 0.3],  # login (underlying normal)
            [0.0, 0.0, 0.3, 1.0],  # bonus
        ]
    )
    L = np.linalg.cholesky(corr)
    Z = rng.normal(0, 1, (n, 4))
    Z_corr = Z @ L.T

    account_age = (24 + 8 * Z_corr[:, 0]).clip(1, 60)
    weekly_spend = np.exp(3.5 + 0.8 * Z_corr[:, 1]).clip(5, 500)
    # Map correlated normal to Poisson-like via rounding of exp transform
    login_freq = np.round(np.exp(np.log(4) + 0.45 * Z_corr[:, 2])).clip(1, 15).astype(float)
    bonus_history = (3 + 1.5 * Z_corr[:, 3]).clip(0, 10)
    return {
        "account_age": account_age,
        "weekly_spend": weekly_spend,
        "login_freq": login_freq,
        "bonus_history": bonus_history,
    }


# --- Harm ---

# harm_logit = -0.3 + A + N(0, 0.5) with A = 1.1*z_spend + 0.4*z_login.  The
# features are independent with (approximately) unit standardised variance, so
# sd(A) ~ sqrt(1.1^2 + 0.4^2) = sqrt(1.37).  HARM_SCALE is that constant, derived
# rather than hard-coded; L_std = A / HARM_SCALE is the standardised value-linked
# driver (spend and login are the LTV drivers).
HARM_SCALE = math.sqrt(1.1**2 + 0.4**2)
C_HARM_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


def draw_harm(f, rng, c_harm=1.0, noise_scale=1.0):
    """True harm in [0,1] with tunable alignment c_harm in [-1, 1] to the value driver.

    harm_logit = -0.3 + s * (c * L_std + sqrt(1 - c^2) * Z) + N(0, 0.5), Z ~ N(0,1)
    independent, s = HARM_SCALE.  c = 1 reproduces the original harm DGP (harm
    rises with spend, hence with V); c = 0 is harm independent of V; c < 0 is
    harm falling with V.  The marginal logit variance is constant in c.
    RNG order: Z, then the logit noise (both drawn after the whole population)."""
    if not -1.0 <= c_harm <= 1.0:
        raise ValueError(f"c_harm must be in [-1, 1], got {c_harm}")
    n = len(f["weekly_spend"])
    z_spend = _standardise(f["weekly_spend"], "weekly_spend")
    z_login = _standardise(f["login_freq"], "login_freq")
    l_std = (1.1 * z_spend + 0.4 * z_login) / HARM_SCALE
    z = rng.normal(0, 1, n) * noise_scale
    eps = rng.normal(0, 0.5 * noise_scale, n)
    logit = -0.3 + HARM_SCALE * (c_harm * l_std + math.sqrt(1.0 - c_harm**2) * z) + eps
    return (1.0 / (1.0 + np.exp(-logit))).clip(0.0, 1.0)


# Noise-free structural value V_bar = 10 spend + 40 login + 5 age (the LTV formula without its
# N(0, 80) noise and clip), standardised with population moments computed ONCE here from
# POP_MEAN / POP_SD (the features are independent, so the variance is the sum of squares).
V_BAR_MEAN = 10 * POP_MEAN["weekly_spend"] + 40 * POP_MEAN["login_freq"] + 5 * POP_MEAN["account_age"]
V_BAR_SD = math.sqrt(
    (10 * POP_SD["weekly_spend"]) ** 2 + (40 * POP_SD["login_freq"]) ** 2 + (5 * POP_SD["account_age"]) ** 2
)


def z_vbar(f):
    """Standardised noise-free structural value of the features (population moments)."""
    v_bar = 10 * f["weekly_spend"] + 40 * f["login_freq"] + 5 * f["account_age"]
    return (v_bar - V_BAR_MEAN) / V_BAR_SD


def draw_harm_decoupled(f, rng, c_harm=1.0, noise_scale=1.0):
    """True harm that depends on X ONLY through V_bar (the decoupled harm DGP).

    harm_logit = -0.3 + s * (c * z_Vbar + sqrt(1 - c^2) * Z) + N(0, 0.5), with the same s
    (HARM_SCALE), the same Z / logit-noise draws and the same RNG order as draw_harm; only
    the value-linked driver differs: z_Vbar replaces L_std = (1.1 z_spend + 0.4 z_login)/s.
    By construction harm is a function of (V_bar, independent noise), so it is independent
    of tau given V_bar (tau is unchanged from the main DGP, including its -0.04 z_spend
    term).  In the decoupled DGP harm depends on X only through V_bar, so harm carries no
    information about tau beyond V_bar.  A soft-gate gain that persists there therefore
    operates through V_bar: harm_hat proxies the structural value that the misspecified V_hat
    misses, and, because tau is itself correlated with V_bar (about -0.11), part of that proxy
    information also bears on tau.  It is not a direct harm-tau link beyond V_bar."""
    if not -1.0 <= c_harm <= 1.0:
        raise ValueError(f"c_harm must be in [-1, 1], got {c_harm}")
    n = len(f["weekly_spend"])
    z = rng.normal(0, 1, n) * noise_scale
    eps = rng.normal(0, 0.5 * noise_scale, n)
    logit = -0.3 + HARM_SCALE * (c_harm * z_vbar(f) + math.sqrt(1.0 - c_harm**2) * z) + eps
    return (1.0 / (1.0 + np.exp(-logit))).clip(0.0, 1.0)


# --- Structural steps ---


def _structural(f, rng, *, moderation, bonus_in_tau=False, with_ltv=True, noise_scale=1.0):
    """Shared latent structure. RNG order: churn noise, [ltv noise], tau noise,
    baseline noise.

    noise_scale multiplies every noise sd (1.0 reproduces the original draws
    bit-for-bit; 0.0 gives the noise-free structure used by the feasible oracle
    tests)."""
    account_age = f["account_age"]
    weekly_spend = f["weekly_spend"]
    login_freq = f["login_freq"]
    n = len(account_age)

    z_spend = _standardise(weekly_spend, "weekly_spend")
    z_login = _standardise(login_freq, "login_freq")
    z_age = _standardise(account_age, "account_age")
    z_bonus = _standardise(f["bonus_history"], "bonus_history")

    churn_logit = (
        -1.0
        + 0.04 * (30 - account_age)
        - 0.25 * login_freq
        + 0.01 * (40 - weekly_spend)
        + 0.1 * z_bonus
        + rng.normal(0, 0.4 * noise_scale, n)
    )
    true_churn = (1.0 / (1.0 + np.exp(-churn_logit))).clip(0.01, 0.99)

    out = {}
    if with_ltv:
        out["true_ltv"] = (
            weekly_spend * 10 + login_freq * 40 + account_age * 5 + rng.normal(0, 80 * noise_scale, n)
        ).clip(50, 8000)

    if bonus_in_tau:
        tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend + 0.10 * z_bonus
    else:
        tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
    if moderation is None:
        true_tau = (tau_base + rng.normal(0, 0.03 * noise_scale, n)).clip(-0.20, 0.40)
    else:
        true_tau = (tau_base * moderation(true_churn) + rng.normal(0, 0.03 * noise_scale, n)).clip(-0.20, 0.40)

    baseline = (0.6 + 0.15 * z_login + 0.05 * z_spend - 0.08 * z_age + rng.normal(0, 0.05 * noise_scale, n)).clip(
        0.05, 0.95
    )
    out.update(
        {
            "true_tau": true_tau,
            "true_churn": true_churn,
            "y0_prob": baseline,
            "y1_prob": (baseline + true_tau).clip(0.01, 0.99),
            "tau_base": tau_base,
        }
    )
    return out


def structural_main(f, rng, noise_scale=1.0):
    return _structural(f, rng, moderation=churn_moderation_sym, noise_scale=noise_scale)


def structural_null(f, rng, noise_scale=1.0):
    """No CATE-churn interaction (m=1)."""
    return _structural(f, rng, moderation=None, noise_scale=noise_scale)


def structural_correlated(f, rng, noise_scale=1.0):
    return _structural(f, rng, moderation=churn_moderation_sym, noise_scale=noise_scale)


def structural_asym(f, rng, noise_scale=1.0):
    return _structural(f, rng, moderation=churn_moderation_asym, noise_scale=noise_scale)


def structural_harm(f, rng, noise_scale=1.0, c_harm=1.0, decoupled=False):
    """Main structure plus true_harm, drawn LAST from the same rng (so the rest
    of the population is identical to structural_main for the same stream).
    decoupled=True draws harm with draw_harm_decoupled instead of draw_harm."""
    out = structural_main(f, rng, noise_scale=noise_scale)
    draw = draw_harm_decoupled if decoupled else draw_harm
    out["true_harm"] = draw(f, rng, c_harm, noise_scale)
    return out


def structural_harm_decoupled(f, rng, noise_scale=1.0, c_harm=1.0):
    return structural_harm(f, rng, noise_scale, c_harm, decoupled=True)


def structural_adv(f, rng, noise_scale=1.0):
    """tau also depends on bonus_history, which is absent from V."""
    return _structural(f, rng, moderation=churn_moderation_sym, bonus_in_tau=True, noise_scale=noise_scale)


def structural_corr_gamma(f, rng, noise_scale=1.0):
    """Legacy order: no LTV here; LTV noise is drawn after the outcome draws."""
    return _structural(f, rng, moderation=churn_moderation_sym, with_ltv=False, noise_scale=noise_scale)


def structural_heavytail(f, rng, noise_scale=1.0):
    """Legacy order: no LTV here; LTV noise is drawn after the outcome draws."""
    return _structural(f, rng, moderation=churn_moderation_sym, with_ltv=False, noise_scale=noise_scale)


# --- Assembly ---


def _assemble(f, s, rng):
    """Draw treatment and potential outcomes (original order) and build the dict."""
    n = len(f["account_age"])
    treatment = rng.binomial(1, TREATMENT_PROB, n)
    y0 = rng.binomial(1, s["y0_prob"])
    y1 = rng.binomial(1, s["y1_prob"])
    y_obs = treatment * y1 + (1 - treatment) * y0
    features = np.column_stack([f["account_age"], f["weekly_spend"], f["login_freq"], f["bonus_history"]])
    data = {
        "features": features,
        "feature_names": list(FEATURE_NAMES),
        "treatment": treatment,
        "y_obs": y_obs,
        "y0": y0,
        "y1": y1,
    }
    for key in ("true_tau", "true_ltv", "true_churn", "true_harm", "y0_prob", "y1_prob"):
        if key in s:
            data[key] = s[key]
    return data


def _generate(n, rng, draw, structural):
    f = draw(n, rng)
    s = structural(f, rng)
    return _assemble(f, s, rng)


def generate_main(n, rng):
    """Player population with CATE-churn inverse-U interaction."""
    return _generate(n, rng, draw_features, structural_main)


def generate_null(n, rng):
    """Population with NO CATE-churn interaction (m=1)."""
    return _generate(n, rng, draw_features, structural_null)


def generate_correlated(n, rng):
    """Population with correlated features via Gaussian copula [M9]."""
    return _generate(n, rng, draw_features_correlated, structural_correlated)


def generate_asym(n, rng):
    """Main DGP but with asymmetric churn moderation (Ascarza-style)."""
    return _generate(n, rng, draw_features, structural_asym)


def generate_harm(n, rng, harm_rng=None, c_harm=1.0, decoupled=False):
    """Main DGP plus true_harm.  The population, treatment and outcomes are
    exactly generate_main's; harm is drawn LAST, from harm_rng (default: rng
    itself, after everything else), so it never perturbs anything else.
    decoupled=True draws harm with draw_harm_decoupled (population, tau, V unchanged)."""
    data = _generate(n, rng, draw_features, structural_main)
    f = {name: data["features"][:, i] for i, name in enumerate(FEATURE_NAMES)}
    draw = draw_harm_decoupled if decoupled else draw_harm
    data["true_harm"] = draw(f, rng if harm_rng is None else harm_rng, c_harm)
    return data


def generate_adv(n, rng):
    """Main DGP, but tau also depends on bonus_history, which is ABSENT from V."""
    return _generate(n, rng, draw_features, structural_adv)


def _late_ltv_corr_gamma(f, s, rng, n, gamma=0.0, noise_scale=1.0, tau_base_moments=None):
    """LTV with gamma * standardised tau_base (tunes corr(tau, V))."""
    tau_base = s["tau_base"]
    mean, sd = (tau_base.mean(), tau_base.std()) if tau_base_moments is None else tau_base_moments
    z_taubase = (tau_base - mean) / (sd + 1e-12)
    base_v = 10 * f["weekly_spend"] + 40 * f["login_freq"] + 5 * f["account_age"]
    return (base_v + gamma * V_CORR_SCALE * z_taubase + rng.normal(0, 80 * noise_scale, n)).clip(50, 8000)


def _late_ltv_heavytail(f, s, rng, n, noise_scale=1.0):
    """Multiplicative lognormal heavy-tailed LTV."""
    base_v = 10 * f["weekly_spend"] + 40 * f["login_freq"] + 5 * f["account_age"]
    return (base_v * np.exp(rng.normal(0, 1.0 * noise_scale, n))).clip(50, 100000)


def _generate_late_ltv(n, rng, structural, ltv_fn):
    f = draw_features(n, rng)
    s = structural(f, rng)
    data = _assemble(f, s, rng)
    data["true_ltv"] = ltv_fn(f, s, rng, n)
    return data


def generate_corr_gamma(n, rng, gamma=0.0):
    """LTV carries gamma * standardised tau_base (tunes corr(tau, V))."""
    return _generate_late_ltv(
        n, rng, structural_corr_gamma, lambda f, s, rng, n: _late_ltv_corr_gamma(f, s, rng, n, gamma=gamma)
    )


def generate_heavytail(n, rng):
    """Multiplicative lognormal heavy-tailed LTV."""
    return _generate_late_ltv(n, rng, structural_heavytail, _late_ltv_heavytail)


def make_corr_dgp(gamma):
    """Bind gamma, returning a generator with the (n, rng) signature.

    The closure carries dgp_name / gamma so the feasible oracle can recover them.
    """

    def dgp(n, rng):
        return generate_corr_gamma(n, rng, gamma)

    dgp.dgp_name = "corr_gamma"
    dgp.gamma = gamma
    return dgp


def dgp_spec(dgp_func):
    """Return (dgp_name, kwargs for conditional_means) for a generator.

    Raises ValueError for generators it cannot identify (no silent fallback).
    """
    name = getattr(dgp_func, "dgp_name", None)
    if name is not None:
        return name, {"gamma": dgp_func.gamma} if name == "corr_gamma" else {}
    for key, f in DGPS.items():
        if f is dgp_func:
            return key, {}
    raise ValueError(
        f"unknown DGP generator {dgp_func!r}: cannot compute the feasible oracle; pass dgp_name explicitly"
    )


def tau_base_stats(features):
    """(mean, sd) of tau_base over the given features (the DGP's corr_gamma
    standardisation is over the generating sample, so pass the full sample)."""
    tb = structural_corr_gamma(features, np.random.RandomState(0), noise_scale=0.0)["tau_base"]
    return tb.mean(), tb.std()


DGPS = {
    "main": generate_main,
    "null": generate_null,
    "correlated": generate_correlated,
    "asym": generate_asym,
    "harm": generate_harm,
    "adv": generate_adv,
    "corr_gamma": generate_corr_gamma,
    "heavytail": generate_heavytail,
}


# --- Conditional means E[. | X] for the feasible oracle ---

ORACLE_MC = 400  # noise draws per unit for the feasible oracle; the one value used by every study

_STRUCTURAL = {
    "main": structural_main,
    "null": structural_null,
    "correlated": structural_correlated,
    "asym": structural_asym,
    "harm": structural_harm,
    "adv": structural_adv,
    "harm_decoupled": structural_harm_decoupled,
    "corr_gamma": structural_corr_gamma,
    "heavytail": structural_heavytail,
}

# Rough float64 temporaries per simulated unit inside one structural call.
_BYTES_PER_UNIT = 8 * 24
DEFAULT_CHUNK_BYTES = 200 * 2**20


class _CommonNoiseRng:
    """RandomState stand-in whose normal() returns the SAME n_mc noise vector for
    every unit (common random numbers): call k draws n_mc standard normals from a
    stream seeded identically per chunk, tiled across the unit-major repeat
    layout.  Ranking is then free of unit-specific MC noise and chunk-invariant."""

    def __init__(self, seed, n_mc):
        self._rs = np.random.RandomState(seed)
        self._n_mc = n_mc

    def normal(self, loc, scale, size):
        assert size % self._n_mc == 0, (
            f"_CommonNoiseRng.normal: size {size} is not a multiple of n_mc={self._n_mc}; "
            "the unit-major repeat layout requires whole blocks of n_mc draws"
        )
        z = self._rs.normal(0.0, 1.0, self._n_mc)
        return loc + scale * np.tile(z, size // self._n_mc)


def conditional_means(
    dgp_name,
    features,
    n_mc=ORACLE_MC,
    *,
    seed,
    gamma=0.0,
    noise_scale=1.0,
    tau_base_moments=None,
    max_units_per_chunk=None,
):
    """Monte Carlo E[tau|X], E[V|X], E[churn|X] (and E[harm|X] for 'harm' and
    'harm_decoupled', whose harm depends on X only through V_bar).

    Features are held fixed; the structural step is re-run n_mc times per unit
    with fresh noise (churn-logit, tau, LTV and harm noise).  Clipping happens
    inside each draw, so the average is the true conditional mean of the clipped
    variable.  Uses its own RandomState(seed), so it never touches the
    simulation's main RNG stream.  Common random numbers: every unit sees the same
    n_mc noise vectors (identical in every chunk), so ranking differences across
    units reflect X, not MC noise.  Units are processed in chunks so peak memory
    stays near DEFAULT_CHUNK_BYTES.

    features: dict of 1-D arrays keyed by FEATURE_NAMES.
    gamma / tau_base_moments: corr_gamma only.  tau_base_moments=(mean, sd) must be the
    moments of tau_base over the generating sample (use tau_base_stats(full
    features)); it is REQUIRED for corr_gamma (ValueError when missing: the moments of
    the features passed in would silently differ from the DGP's own standardisation).
    Returns dict of arrays; '*_se' entries are Monte Carlo standard errors.
    """
    structural = _STRUCTURAL[dgp_name]
    n = len(features["account_age"])
    if max_units_per_chunk is None:
        max_units_per_chunk = DEFAULT_CHUNK_BYTES // _BYTES_PER_UNIT
    step = max(1, max_units_per_chunk // n_mc)

    if dgp_name == "corr_gamma" and tau_base_moments is None:
        raise ValueError(
            "conditional_means('corr_gamma') needs tau_base_moments=(mean, sd) "
            "of tau_base over the generating sample; pass tau_base_stats(features)"
        )

    names = ["tau", "ltv", "churn"] + (["harm"] if dgp_name in ("harm", "harm_decoupled") else [])
    mean = {k: np.empty(n) for k in names}
    se = {k: np.empty(n) for k in names}

    for lo in range(0, n, step):
        hi = min(n, lo + step)
        m = hi - lo
        rep = {k: np.repeat(np.asarray(v)[lo:hi], n_mc) for k, v in features.items()}
        rng = _CommonNoiseRng(seed, n_mc)
        s = structural(rep, rng, noise_scale=noise_scale)
        draws = {"tau": s["true_tau"], "churn": s["true_churn"]}
        if "true_harm" in s:
            draws["harm"] = s["true_harm"]
        if "true_ltv" in s:
            draws["ltv"] = s["true_ltv"]
        elif dgp_name == "corr_gamma":
            draws["ltv"] = _late_ltv_corr_gamma(
                rep, s, rng, m * n_mc, gamma=gamma, noise_scale=noise_scale, tau_base_moments=tau_base_moments
            )
        else:
            draws["ltv"] = _late_ltv_heavytail(rep, s, rng, m * n_mc, noise_scale=noise_scale)
        for k in names:
            d = draws[k].reshape(m, n_mc)
            mean[k][lo:hi] = d.mean(axis=1)
            se[k][lo:hi] = d.std(axis=1, ddof=1) / np.sqrt(n_mc) if n_mc > 1 else 0.0

    out = {f"{k}_bar": mean[k] for k in names}
    out.update({f"{k}_se": se[k] for k in names})
    return out
