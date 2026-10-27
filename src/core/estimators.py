"""CATE, LTV and churn estimators plus held-out theta calibration."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold

from core.policy import PROPENSITY, aipw_scores

# Independent RNG streams per replication seed (see seed_from).
LTV_STREAM = 0        # LTV-proxy noise for the deployed estimator
CALIB_STREAM = 1      # LTV-proxy noise for the calibration (validation) fit
VALUE_STREAM_V42 = 3  # well-specified value model (component_update in study_robustness)
VALUE_STREAM_V43 = 4  # richer reference value model (obs_diagnostic in study_robustness)
ORACLE_STREAM = 5     # common-random-number noise for the feasible oracle MC
CROSSFIT_STREAM = 6   # validation-fold cross-fit split for AIPW calibration
HARM_STREAM = 7       # harm DGP noise (Z and logit noise), drawn after everything else
HARM_ESTIMATE_STREAM = 8  # noise of the estimated harm score (estimate_harm)
BOOT_STREAM = 9       # bootstrap resamples over reps for the harm sweep CIs


def seed_from(seed, stream):
    """Deterministic child seed for (replication seed, stream); replaces hash()."""
    return (seed * 1_000_003 + stream) % 2**31


def _estimate_ltv_proxy(X_train, X_test, seed, stream=LTV_STREAM):
    """Misspecified LTV proxy (drops age, nonlinear spend, different coefficients).

    The proxy noise is seeded by seed_from(seed, stream), where seed is the
    replication seed."""
    rng_ltv = np.random.RandomState(seed_from(seed, stream))
    ltv_proxy = (12 * X_train[:, 1]**0.9 + 35 * X_train[:, 2]
                 + rng_ltv.normal(0, 200, len(X_train)))
    ltv_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    ltv_model.fit(X_train, ltv_proxy)
    return ltv_model.predict(X_test).clip(50, 8000)


def _estimate_ltv_churn(X_train, X_test, W_train, Y_train, seed, stream=LTV_STREAM):
    """Shared LTV and churn estimation (used by all CATE methods)."""
    ltv_hat = _estimate_ltv_proxy(X_train, X_test, seed, stream)

    # Churn: binary control non-retention as proxy
    ctrl_mask = W_train == 0
    churn_proxy = 1 - Y_train[ctrl_mask]
    churn_model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    churn_model.fit(X_train[ctrl_mask], churn_proxy)
    churn_hat = churn_model.predict_proba(X_test)[:, 1].clip(0.01, 0.99)

    return ltv_hat, churn_hat


def _gb_kwargs(params, **legacy):
    """GradientBoosting kwargs: the legacy values when params is None, else the
    tuned (max_depth, n_estimators, learning_rate) dict."""
    if params is None:
        return dict(legacy, random_state=0)
    return {'max_depth': params['max_depth'], 'n_estimators': params['n_estimators'],
            'learning_rate': params['learning_rate'], 'random_state': 0}


def _estimate_cate_tlearner(X_train, X_test, W_train, Y_train, params=None):
    """T-Learner CATE estimation (params=None: the legacy depth-3, 100-tree fit)."""
    treat_mask = W_train == 1
    ctrl_mask = W_train == 0
    kw = _gb_kwargs(params, n_estimators=100, max_depth=3, learning_rate=0.1)
    model_t = GradientBoostingClassifier(**kw)
    model_c = GradientBoostingClassifier(**kw)
    model_t.fit(X_train[treat_mask], Y_train[treat_mask])
    model_c.fit(X_train[ctrl_mask], Y_train[ctrl_mask])
    tau_hat = (model_t.predict_proba(X_test)[:, 1]
               - model_c.predict_proba(X_test)[:, 1])
    return tau_hat


def _estimate_cate_xlearner(X_train, X_test, W_train, Y_train, params=None):
    """X-Learner CATE estimation (Kunzel et al. 2019).  params=None keeps the legacy
    fit (100-tree base classifiers, 80-tree effect regressors, depth 3); a params
    dict sets depth, trees and rate of every stage."""
    treat_mask = W_train == 1
    ctrl_mask = W_train == 0

    # Step 1: Base response models (same as T-Learner)
    mu1 = GradientBoostingClassifier(**_gb_kwargs(
        params, n_estimators=100, max_depth=3, learning_rate=0.1))
    mu0 = GradientBoostingClassifier(**_gb_kwargs(
        params, n_estimators=100, max_depth=3, learning_rate=0.1))
    mu1.fit(X_train[treat_mask], Y_train[treat_mask])
    mu0.fit(X_train[ctrl_mask], Y_train[ctrl_mask])

    # Step 2: Imputed treatment effects
    D1 = Y_train[treat_mask] - mu0.predict_proba(X_train[treat_mask])[:, 1]
    D0 = mu1.predict_proba(X_train[ctrl_mask])[:, 1] - Y_train[ctrl_mask]

    # Step 3: CATE models on imputed effects
    tau1_model = GradientBoostingRegressor(**_gb_kwargs(
        params, n_estimators=80, max_depth=3, learning_rate=0.1))
    tau0_model = GradientBoostingRegressor(**_gb_kwargs(
        params, n_estimators=80, max_depth=3, learning_rate=0.1))
    tau1_model.fit(X_train[treat_mask], D1)
    tau0_model.fit(X_train[ctrl_mask], D0)

    # Step 4: Combine with known propensity e(x) = 0.5
    tau_hat = 0.5 * tau0_model.predict(X_test) + 0.5 * tau1_model.predict(X_test)
    return tau_hat


def estimate_components(data, train_idx, test_idx, cate_method='tlearner', *,
                        seed, stream=LTV_STREAM):
    """Estimate CATE, LTV, and churn from training data."""
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']
    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    if cate_method == 'xlearner':
        tau_hat = _estimate_cate_xlearner(X_train, X_test, W_train, Y_train)
    else:
        tau_hat = _estimate_cate_tlearner(X_train, X_test, W_train, Y_train)

    ltv_hat, churn_hat = _estimate_ltv_churn(X_train, X_test, W_train, Y_train,
                                              seed, stream)
    return tau_hat, ltv_hat, churn_hat


def estimate_components_misspec(data, train_idx, test_idx, cate_method='tlearner', *,
                                seed, stream=LTV_STREAM):
    """Estimate with MISSPECIFIED CATE model: login_freq dropped."""
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']
    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    # CATE model uses only 3 features (drops login_freq at index 2)
    cate_cols = [0, 1, 3]
    X_train_cate = X_train[:, cate_cols]
    X_test_cate = X_test[:, cate_cols]

    if cate_method == 'xlearner':
        tau_hat = _estimate_cate_xlearner(X_train_cate, X_test_cate, W_train, Y_train)
    else:
        tau_hat = _estimate_cate_tlearner(X_train_cate, X_test_cate, W_train, Y_train)

    # LTV and churn use ALL features
    ltv_hat, churn_hat = _estimate_ltv_churn(X_train, X_test, W_train, Y_train,
                                              seed, stream)
    return tau_hat, ltv_hat, churn_hat


def _estimate_cate_slearner(X_train, X_test, W_train, Y_train, params=None):
    """S-Learner CATE: predicted mu(x, 1) - mu(x, 0) from one model with treatment as a
    feature (params=None: the legacy depth-4, 100-tree fit)."""
    X_aug = np.column_stack([X_train, W_train])
    slearner = GradientBoostingRegressor(**_gb_kwargs(
        params, n_estimators=100, max_depth=4, learning_rate=0.1))
    slearner.fit(X_aug, Y_train)
    n_test = len(X_test)
    pred_t1 = slearner.predict(np.column_stack([X_test, np.ones(n_test)]))
    pred_t0 = slearner.predict(np.column_stack([X_test, np.zeros(n_test)]))
    return pred_t1 - pred_t0


def _estimate_slearner(X_train, X_test, W_train, Y_train, ltv_hat):
    """S-Learner: single model with treatment as feature [C3]; returns tau_sl * ltv_hat."""
    return _estimate_cate_slearner(X_train, X_test, W_train, Y_train) * ltv_hat


def _wellspec_value_model(Xtr, Xte, seed):
    """A correctly-specified value model: trained on the true LTV functional form."""
    rng_v = np.random.RandomState(seed_from(seed, VALUE_STREAM_V42))
    # True DGP LTV form: 10*spend + 40*login + 5*age (cf. study_main eq for V).
    target = (10 * Xtr[:, 1] + 40 * Xtr[:, 2] + 5 * Xtr[:, 0]
              + rng_v.normal(0, 80, len(Xtr))).clip(50, 8000)
    m = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    m.fit(Xtr, target)
    return m.predict(Xte).clip(50, 8000)


# ---------------------------------------------------------------------------
# DR and R learners, params plumbing, dispatcher
# ---------------------------------------------------------------------------
CROSSFIT_FOLDS = 2

LEARNERS = ('T', 'X', 'S', 'DR', 'R')
DEFAULT_PARAMS = {
    # Nominal "default" settings.  T, X and S are the legacy fits (passing
    # params=None reproduces them exactly; for X the legacy effect regressors use
    # 80 trees, a params dict sets every stage).  DR and R have no legacy fit.
    'T': {'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1},
    'X': {'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1},
    'S': {'max_depth': 4, 'n_estimators': 100, 'learning_rate': 0.1},
    'DR': {'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1},
    'R': {'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1},
}
# Learners whose legacy fit is reproduced by params=None.
LEGACY_LEARNERS = ('T', 'X', 'S')


def _clf(params):
    return GradientBoostingClassifier(**_gb_kwargs(params))


def _reg(params):
    return GradientBoostingRegressor(**_gb_kwargs(params))


def _folds(n, n_folds, seed):
    return KFold(n_splits=n_folds, shuffle=True,
                 random_state=seed_from(seed, CROSSFIT_STREAM)).split(np.arange(n))


def _crossfit_m(X, Y, params, n_folds, seed):
    """Cross-fitted m_hat(x) = E[Y|X] (each row predicted by a model that never saw it)."""
    m = np.empty(len(Y))
    for fit_idx, pred_idx in _folds(len(Y), n_folds, seed):
        model = _clf(params).fit(X[fit_idx], Y[fit_idx])
        m[pred_idx] = model.predict_proba(X[pred_idx])[:, 1]
    return m


def estimate_cate_dr(X_tr, X_te, W_tr, Y_tr, params, n_folds=CROSSFIT_FOLDS, *,
                     seed, e=PROPENSITY):
    """DR-learner: the cross-fitted AIPW pseudo-outcome regressed on X.

    The pseudo-outcome is core.policy.aipw_scores (outcome nuisances mu0, mu1
    cross-fitted over n_folds folds, propensity e=0.5 known), regressed on X with a
    gradient-boosted regressor using the same params as the nuisances."""
    pseudo = aipw_scores(X_tr, W_tr, Y_tr, seed, n_folds, e, params)
    return _reg(params).fit(X_tr, pseudo).predict(X_te)


def estimate_cate_r(X_tr, X_te, W_tr, Y_tr, params, n_folds=CROSSFIT_FOLDS, *,
                    seed, e=PROPENSITY):
    """R-learner (Robinson residual-on-residual).

    m_hat(x) = E[Y|X] is cross-fitted; with e known, tau(x) minimises
    sum (W-e)^2 * ((Y - m_hat)/(W - e) - tau(x))^2, so the learner regresses
    (Y - m_hat)/(W - e) on X with sample weights (W - e)^2."""
    m_hat = _crossfit_m(X_tr, Y_tr, params, n_folds, seed)
    resid_w = W_tr - e
    target = (Y_tr - m_hat) / resid_w
    return _reg(params).fit(X_tr, target, sample_weight=resid_w ** 2).predict(X_te)


def estimate_cate(learner, X_tr, X_te, W_tr, Y_tr, params=None, *, seed):
    """Dispatch to a CATE learner in LEARNERS.

    params: None for the default fit (the legacy fit for T/X/S, DEFAULT_PARAMS for
    DR/R) or a {max_depth, n_estimators, learning_rate} dict.  seed drives only the
    cross-fitting folds of DR and R."""
    if learner == 'T':
        return _estimate_cate_tlearner(X_tr, X_te, W_tr, Y_tr, params)
    if learner == 'X':
        return _estimate_cate_xlearner(X_tr, X_te, W_tr, Y_tr, params)
    if learner == 'S':
        return _estimate_cate_slearner(X_tr, X_te, W_tr, Y_tr, params)
    p = DEFAULT_PARAMS[learner] if params is None else params
    if learner == 'DR':
        return estimate_cate_dr(X_tr, X_te, W_tr, Y_tr, p, seed=seed)
    if learner == 'R':
        return estimate_cate_r(X_tr, X_te, W_tr, Y_tr, p, seed=seed)
    raise ValueError(f'unknown CATE learner {learner!r}')
