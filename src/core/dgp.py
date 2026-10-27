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

import numpy as np

TREATMENT_PROB = 0.50

# --- Population moments (verified via N=2M draw, fixed for reproducibility) ---
# These replace within-sample standardisation to prevent train/test leakage.
POP_MEAN = {
    'account_age': 24.01,
    'weekly_spend': 45.56,
    'login_freq': 4.02,
    'bonus_history': 3.01,
}
POP_SD = {
    'account_age': 7.98,
    'weekly_spend': 42.51,
    'login_freq': 1.97,
    'bonus_history': 1.47,
}

FEATURE_NAMES = ['account_age', 'weekly_spend', 'login_freq', 'bonus_history']

# --- DGP Specification ---
DGP_SPEC = {
    'features': {
        'account_age': 'N(24, 8), clipped [1, 60]',
        'weekly_spend': 'exp(N(3.5, 0.8)), clipped [5, 500]',
        'login_freq': 'Poisson(4), clipped [1, 15]',
        'bonus_history': 'N(3, 1.5), clipped [0, 10]',
    },
    'standardisation': 'Population parameters (fixed across replications)',
    'churn': 'logit = -1.0 + 0.04*(30-age) - 0.25*login + 0.01*(40-spend) + 0.1*z_bonus + N(0,0.4); sigmoid, clip [0.01, 0.99]',
    'ltv': '10*spend + 40*login + 5*age + N(0,80), clip [50, 8000]',
    'ltv_proxy': '12*spend^0.9 + 35*login + N(0,200) -- misspecified form for estimation',
    'cate': 'tau_base = 0.06 + 0.10*z_login - 0.02*z_login^2 - 0.04*z_spend; churn_mod = 4*p*(1-p); tau = tau_base * churn_mod + N(0,0.03), clip [-0.20, 0.40]',
    'baseline': '0.6 + 0.15*z_login + 0.05*z_spend - 0.08*z_age + N(0,0.05), clip [0.05, 0.95]',
    'treatment': 'Bernoulli(0.5)',
    'note': 'The CATE-churn moderation (4*p*(1-p)) creates an inverse-U relationship: treatment effects peak at moderate churn (p=0.5) and are attenuated at extremes.',
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
    return {'account_age': account_age, 'weekly_spend': weekly_spend,
            'login_freq': login_freq, 'bonus_history': bonus_history}


def draw_features_correlated(n, rng):
    """Correlated-feature draw via Gaussian copula [M9]."""
    corr = np.array([
        [1.0,  0.3,  0.4,  0.0],   # age
        [0.3,  1.0,  0.0,  0.0],   # spend (underlying normal)
        [0.4,  0.0,  1.0,  0.3],   # login (underlying normal)
        [0.0,  0.0,  0.3,  1.0],   # bonus
    ])
    L = np.linalg.cholesky(corr)
    Z = rng.normal(0, 1, (n, 4))
    Z_corr = Z @ L.T

    account_age = (24 + 8 * Z_corr[:, 0]).clip(1, 60)
    weekly_spend = np.exp(3.5 + 0.8 * Z_corr[:, 1]).clip(5, 500)
    # Map correlated normal to Poisson-like via rounding of exp transform
    login_freq = np.round(np.exp(np.log(4) + 0.45 * Z_corr[:, 2])).clip(1, 15).astype(float)
    bonus_history = (3 + 1.5 * Z_corr[:, 3]).clip(0, 10)
    return {'account_age': account_age, 'weekly_spend': weekly_spend,
            'login_freq': login_freq, 'bonus_history': bonus_history}


# --- Structural steps ---

def _structural(f, rng, *, moderation, bonus_in_tau=False, with_harm=False,
                with_ltv=True):
    """Shared latent structure. RNG order: churn noise, [ltv noise], tau noise,
    [harm noise], baseline noise."""
    account_age = f['account_age']
    weekly_spend = f['weekly_spend']
    login_freq = f['login_freq']
    n = len(account_age)

    z_spend = _standardise(weekly_spend, 'weekly_spend')
    z_login = _standardise(login_freq, 'login_freq')
    z_age = _standardise(account_age, 'account_age')
    z_bonus = _standardise(f['bonus_history'], 'bonus_history')

    churn_logit = (-1.0 + 0.04 * (30 - account_age) - 0.25 * login_freq
                   + 0.01 * (40 - weekly_spend) + 0.1 * z_bonus
                   + rng.normal(0, 0.4, n))
    true_churn = (1.0 / (1.0 + np.exp(-churn_logit))).clip(0.01, 0.99)

    out = {}
    if with_ltv:
        out['true_ltv'] = (weekly_spend * 10 + login_freq * 40 + account_age * 5
                           + rng.normal(0, 80, n)).clip(50, 8000)

    if bonus_in_tau:
        tau_base = (0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
                    + 0.10 * z_bonus)
    else:
        tau_base = 0.06 + 0.10 * z_login - 0.02 * z_login**2 - 0.04 * z_spend
    if moderation is None:
        true_tau = (tau_base + rng.normal(0, 0.03, n)).clip(-0.20, 0.40)
    else:
        true_tau = (tau_base * moderation(true_churn)
                    + rng.normal(0, 0.03, n)).clip(-0.20, 0.40)

    if with_harm:
        # Harm score in [0,1]: rises with spend (the LTV driver), so harm and value
        # are positively correlated -- value-weighting will chase harmful players.
        harm_logit = 1.1 * z_spend + 0.4 * z_login - 0.3 + rng.normal(0, 0.5, n)
        out['true_harm'] = (1.0 / (1.0 + np.exp(-harm_logit))).clip(0.0, 1.0)

    baseline = (0.6 + 0.15 * z_login + 0.05 * z_spend - 0.08 * z_age
                + rng.normal(0, 0.05, n)).clip(0.05, 0.95)
    out.update({
        'true_tau': true_tau, 'true_churn': true_churn,
        'y0_prob': baseline,
        'y1_prob': (baseline + true_tau).clip(0.01, 0.99),
        'tau_base': tau_base,
    })
    return out


def structural_main(f, rng):
    return _structural(f, rng, moderation=churn_moderation_sym)


def structural_null(f, rng):
    """No CATE-churn interaction (m=1)."""
    return _structural(f, rng, moderation=None)


def structural_correlated(f, rng):
    return _structural(f, rng, moderation=churn_moderation_sym)


def structural_asym(f, rng):
    return _structural(f, rng, moderation=churn_moderation_asym)


def structural_harm(f, rng):
    return _structural(f, rng, moderation=churn_moderation_sym, with_harm=True)


def structural_adv(f, rng):
    """tau also depends on bonus_history, which is absent from V."""
    return _structural(f, rng, moderation=churn_moderation_sym, bonus_in_tau=True)


def structural_corr_gamma(f, rng):
    """Legacy order: no LTV here; LTV noise is drawn after the outcome draws."""
    return _structural(f, rng, moderation=churn_moderation_sym, with_ltv=False)


def structural_heavytail(f, rng):
    """Legacy order: no LTV here; LTV noise is drawn after the outcome draws."""
    return _structural(f, rng, moderation=churn_moderation_sym, with_ltv=False)


# --- Assembly ---

def _assemble(f, s, rng):
    """Draw treatment and potential outcomes (original order) and build the dict."""
    n = len(f['account_age'])
    treatment = rng.binomial(1, TREATMENT_PROB, n)
    y0 = rng.binomial(1, s['y0_prob'])
    y1 = rng.binomial(1, s['y1_prob'])
    y_obs = treatment * y1 + (1 - treatment) * y0
    features = np.column_stack([f['account_age'], f['weekly_spend'],
                                f['login_freq'], f['bonus_history']])
    data = {
        'features': features,
        'feature_names': list(FEATURE_NAMES),
        'treatment': treatment, 'y_obs': y_obs, 'y0': y0, 'y1': y1,
    }
    for key in ('true_tau', 'true_ltv', 'true_churn', 'true_harm',
                'y0_prob', 'y1_prob'):
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


def generate_harm(n, rng):
    """Main DGP plus a harm covariate correlated with LTV (via spend)."""
    return _generate(n, rng, draw_features, structural_harm)


def generate_adv(n, rng):
    """Main DGP, but tau also depends on bonus_history, which is ABSENT from V."""
    return _generate(n, rng, draw_features, structural_adv)


def _generate_late_ltv(n, rng, structural, ltv_fn):
    f = draw_features(n, rng)
    s = structural(f, rng)
    data = _assemble(f, s, rng)
    data['true_ltv'] = ltv_fn(f, s, rng, n)
    return data


def generate_corr_gamma(n, rng, gamma):
    """LTV carries gamma * standardised tau_base (tunes corr(tau, V))."""
    def ltv(f, s, rng, n):
        tau_base = s['tau_base']
        z_taubase = (tau_base - tau_base.mean()) / (tau_base.std() + 1e-12)
        base_v = (10 * f['weekly_spend'] + 40 * f['login_freq']
                  + 5 * f['account_age'])
        return (base_v + gamma * V_CORR_SCALE * z_taubase
                + rng.normal(0, 80, n)).clip(50, 8000)
    return _generate_late_ltv(n, rng, structural_corr_gamma, ltv)


def generate_heavytail(n, rng):
    """Multiplicative lognormal heavy-tailed LTV."""
    def ltv(f, s, rng, n):
        base_v = (10 * f['weekly_spend'] + 40 * f['login_freq']
                  + 5 * f['account_age'])
        return (base_v * np.exp(rng.normal(0, 1.0, n))).clip(50, 100000)
    return _generate_late_ltv(n, rng, structural_heavytail, ltv)


def make_corr_dgp(gamma):
    """Bind gamma, returning a generator with the (n, rng) signature."""
    def dgp(n, rng):
        return generate_corr_gamma(n, rng, gamma)
    return dgp


DGPS = {
    'main': generate_main,
    'null': generate_null,
    'correlated': generate_correlated,
    'asym': generate_asym,
    'harm': generate_harm,
    'adv': generate_adv,
    'corr_gamma': generate_corr_gamma,
    'heavytail': generate_heavytail,
}
