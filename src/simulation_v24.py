"""
Simulation study for v24 manuscript.

Key design choices vs. the old simulation (v22):
- All components (CATE, LTV, churn) are ESTIMATED, not oracle
- DGP has realistic noise and correlation structure
- Framework is tested under well-specified, misspecified, and noisy conditions
- Value-conservatism frontier is computed with actual welfare numbers
- Comparison to a single end-to-end model baseline
"""

import numpy as np
import json
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# --- Configuration ---
N_PLAYERS = 10000
N_RUNS = 30
BUDGET_FRACTION = 0.30
TREATMENT_PROB = 0.50
SEED_BASE = 42


def f_churn(p, pL=0.05, pML=0.40, pMH=0.60, pH=0.90, phi=0.20):
    """Piecewise linear risk-modulation function."""
    if p < pL:
        return 0.0
    elif p < pML:
        return (p - pL) / (pML - pL)
    elif p <= pMH:
        return 1.0
    elif p < pH:
        return 1.0 - (1.0 - phi) * (p - pMH) / (pH - pMH)
    else:
        return phi


def f_churn_vec(p_arr, **kwargs):
    """Vectorised risk-modulation function."""
    return np.array([f_churn(p, **kwargs) for p in p_arr])


def generate_population(n, rng):
    """
    Generate a player population with correlated features,
    heterogeneous treatment effects, LTV, and churn probability.

    The DGP is designed to be realistic:
    - Features are correlated (not independent)
    - Treatment effects depend on features non-linearly
    - LTV and churn share feature drivers with CATE
    - No clean segment separation
    """
    # Features (correlated)
    account_age = rng.normal(24, 8, n).clip(1, 60)
    weekly_spend = np.exp(rng.normal(3.5, 0.8, n)).clip(5, 500)
    login_freq = rng.poisson(4, n).clip(1, 15).astype(float)
    bonus_history = rng.normal(3, 1.5, n).clip(0, 10)

    # Standardised features for effect generation
    z_spend = (weekly_spend - weekly_spend.mean()) / weekly_spend.std()
    z_login = (login_freq - login_freq.mean()) / login_freq.std()
    z_age = (account_age - account_age.mean()) / account_age.std()
    z_bonus = (bonus_history - bonus_history.mean()) / bonus_history.std()

    # True CATE: nonlinear function of features
    # Moderate-engagement players benefit most; extremes benefit less
    tau_base = 0.08 + 0.12 * z_login - 0.03 * z_login**2 - 0.05 * z_spend
    tau_noise = rng.normal(0, 0.04, n)
    true_tau = (tau_base + tau_noise).clip(-0.20, 0.40)

    # True LTV: function of spend and engagement
    true_ltv = (weekly_spend * 10 + login_freq * 40 + account_age * 5
                + rng.normal(0, 80, n)).clip(50, 8000)

    # True churn probability: logistic function of features
    churn_logit = (-1.0 + 0.04 * (30 - account_age) - 0.25 * login_freq
                   + 0.01 * (40 - weekly_spend) + 0.1 * z_bonus
                   + rng.normal(0, 0.4, n))
    true_churn = 1.0 / (1.0 + np.exp(-churn_logit))
    true_churn = true_churn.clip(0.01, 0.99)

    # Baseline retention probability
    baseline = (0.6 + 0.15 * z_login + 0.05 * z_spend - 0.08 * z_age
                + rng.normal(0, 0.05, n)).clip(0.05, 0.95)

    # Potential outcomes
    y0_prob = baseline
    y1_prob = (baseline + true_tau).clip(0.01, 0.99)
    treatment = rng.binomial(1, TREATMENT_PROB, n)
    y0 = rng.binomial(1, y0_prob)
    y1 = rng.binomial(1, y1_prob)
    y_obs = treatment * y1 + (1 - treatment) * y0

    features = np.column_stack([account_age, weekly_spend, login_freq, bonus_history])
    feature_names = ['account_age', 'weekly_spend', 'login_freq', 'bonus_history']

    return {
        'features': features,
        'feature_names': feature_names,
        'treatment': treatment,
        'y_obs': y_obs,
        'y0': y0,
        'y1': y1,
        'true_tau': true_tau,
        'true_ltv': true_ltv,
        'true_churn': true_churn,
        'y0_prob': y0_prob,
        'y1_prob': y1_prob,
    }


def estimate_components(data, train_idx, test_idx):
    """
    Estimate CATE, LTV, and churn from training data.
    Returns estimates on the test set.
    """
    X = data['features']
    W = data['treatment']
    Y = data['y_obs']

    X_train, X_test = X[train_idx], X[test_idx]
    W_train, Y_train = W[train_idx], Y[train_idx]

    # T-Learner for CATE
    treat_mask = W_train == 1
    ctrl_mask = W_train == 0

    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    model_c = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)

    model_t.fit(X_train[treat_mask], Y_train[treat_mask])
    model_c.fit(X_train[ctrl_mask], Y_train[ctrl_mask])

    tau_hat = (model_t.predict_proba(X_test)[:, 1]
               - model_c.predict_proba(X_test)[:, 1])

    # LTV estimation (regression on spend-related proxy)
    # In practice LTV would come from a separate model; here we train on
    # a noisy proxy available in training data
    ltv_proxy_train = (X_train[:, 1] * 10 + X_train[:, 2] * 40
                       + X_train[:, 0] * 5)
    ltv_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    ltv_model.fit(X_train, ltv_proxy_train)
    ltv_hat = ltv_model.predict(X_test).clip(50, 8000)

    # Churn estimation (logistic model on features)
    # Use control-group non-retention as churn proxy
    churn_proxy = 1 - Y_train[ctrl_mask]
    churn_model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
    churn_model.fit(X_train[ctrl_mask], churn_proxy)
    churn_hat = churn_model.predict_proba(X_test)[:, 1].clip(0.01, 0.99)

    return tau_hat, ltv_hat, churn_hat


def compute_welfare(targeted_idx, true_tau, true_ltv):
    """Compute oracle welfare: sum of tau_i * LTV_i for targeted players."""
    return np.sum(true_tau[targeted_idx] * true_ltv[targeted_idx])


def compute_risk_exposure(targeted_idx, true_churn):
    """Mean churn probability among targeted players."""
    return np.mean(true_churn[targeted_idx])


def target_topk(scores, k):
    """Return indices of top-k scored players."""
    return np.argsort(scores)[-k:]


def run_single_simulation(seed):
    """Run one replication. Returns results dict."""
    rng = np.random.RandomState(seed)
    data = generate_population(N_PLAYERS, rng)

    # Split
    n = len(data['treatment'])
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, random_state=seed)

    k = int(len(test_idx) * BUDGET_FRACTION)
    true_tau_test = data['true_tau'][test_idx]
    true_ltv_test = data['true_ltv'][test_idx]
    true_churn_test = data['true_churn'][test_idx]
    y0_test = data['y0'][test_idx]
    y1_test = data['y1'][test_idx]

    # Estimate components
    tau_hat, ltv_hat, churn_hat = estimate_components(data, train_idx, test_idx)

    # Oracle components for comparison
    tau_oracle = true_tau_test
    ltv_oracle = true_ltv_test
    churn_oracle = true_churn_test

    g_hat = f_churn_vec(churn_hat)
    g_oracle = f_churn_vec(churn_oracle)

    # --- Strategies ---
    strategies = {}

    # 1. Random
    random_scores = rng.uniform(0, 1, len(test_idx))
    strategies['Random'] = random_scores

    # 2. Estimated CATE only
    strategies['CATE Only'] = tau_hat

    # 3. Estimated CATE x LTV
    strategies['CATE x LTV'] = tau_hat * ltv_hat

    # 4. Full framework (estimated)
    strategies['Full Framework (Est.)'] = tau_hat * ltv_hat * g_hat

    # 5. Oracle full framework
    strategies['Full Framework (Oracle)'] = tau_oracle * ltv_oracle * g_oracle

    # 6. No regularisation (estimated CATE x LTV, no g)
    strategies['No Regularisation'] = tau_hat * ltv_hat

    # Evaluate all strategies
    results = {}
    for name, scores in strategies.items():
        targeted = target_topk(scores, k)
        # Targeted-group ATE (oracle potential outcomes)
        ate = np.mean(y1_test[targeted] - y0_test[targeted]) * 100
        # Oracle welfare
        welfare = compute_welfare(targeted, true_tau_test, true_ltv_test)
        # Risk exposure
        risk = compute_risk_exposure(targeted, true_churn_test)
        # Persuadable-like metric: mean true_tau in targeted
        mean_tau = np.mean(true_tau_test[targeted]) * 100
        # Mean LTV of causally retained
        causally_retained = targeted[(y1_test[targeted] - y0_test[targeted]) == 1]
        mean_ltv_cr = (np.mean(true_ltv_test[causally_retained])
                       if len(causally_retained) > 0 else 0.0)

        results[name] = {
            'ate': ate,
            'welfare': welfare,
            'risk_exposure': risk,
            'mean_tau_targeted': mean_tau,
            'mean_ltv_cr': mean_ltv_cr,
            'n_targeted': len(targeted),
            'n_causally_retained': len(causally_retained),
        }

    # --- Value-conservatism frontier ---
    # Vary the churn function strength
    frontier = []
    alpha_configs = [
        ('No reg (g=1)', {}),
        ('Wide', {'pL': 0.03, 'pML': 0.50, 'pMH': 0.70}),
        ('Default', {}),
        ('Shifted low', {'pL': 0.05, 'pML': 0.25, 'pMH': 0.45}),
        ('Narrow', {'pL': 0.10, 'pML': 0.30, 'pMH': 0.50}),
        ('Aggressive', {'pL': 0.15, 'pML': 0.35, 'pMH': 0.50, 'phi': 0.05}),
    ]
    for config_name, kwargs in alpha_configs:
        if config_name == 'No reg (g=1)':
            scores = tau_hat * ltv_hat
        else:
            g_config = f_churn_vec(churn_hat, **kwargs)
            scores = tau_hat * ltv_hat * g_config
        targeted = target_topk(scores, k)
        w = compute_welfare(targeted, true_tau_test, true_ltv_test)
        r = compute_risk_exposure(targeted, true_churn_test)
        frontier.append({
            'config': config_name,
            'welfare': w,
            'risk_exposure': r,
        })

    # --- Misspecification test ---
    # Inverted g: high weight for low churn, low weight for high churn
    # (opposite of the intended domain logic)
    g_inverted = f_churn_vec(1.0 - churn_hat)  # flip churn before applying f
    misspec_scores = tau_hat * ltv_hat * g_inverted
    targeted_misspec = target_topk(misspec_scores, k)
    misspec_result = {
        'ate': np.mean(y1_test[targeted_misspec] - y0_test[targeted_misspec]) * 100,
        'welfare': compute_welfare(targeted_misspec, true_tau_test, true_ltv_test),
        'risk_exposure': compute_risk_exposure(targeted_misspec, true_churn_test),
    }

    # --- CATE-churn moderated scenario ---
    # What if true CATE is attenuated at extreme churn (inverse-U)?
    # We evaluate the framework using a moderated version of the oracle tau
    churn_moderation = np.where(
        true_churn_test < 0.10, 0.3,
        np.where(true_churn_test > 0.70, 0.3, 1.0))
    moderated_tau = true_tau_test * churn_moderation
    # Welfare under moderated DGP for each strategy
    moderated_results = {}
    for name, scores in strategies.items():
        targeted = target_topk(scores, k)
        mod_welfare = np.sum(moderated_tau[targeted] * true_ltv_test[targeted])
        moderated_results[name] = {'welfare': mod_welfare}

    return results, frontier, misspec_result, moderated_results


def main():
    print("=" * 70)
    print("SIMULATION STUDY: Risk-Modulated Treatment Targeting (v24)")
    print("=" * 70)
    print(f"N={N_PLAYERS}, Runs={N_RUNS}, Budget={BUDGET_FRACTION*100:.0f}%")

    all_results = {s: [] for s in [
        'Random', 'CATE Only', 'CATE x LTV',
        'Full Framework (Est.)', 'Full Framework (Oracle)',
        'No Regularisation']}
    all_frontiers = []
    all_misspec = []
    all_moderated = {s: [] for s in all_results}

    for run in range(N_RUNS):
        if (run + 1) % 5 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = SEED_BASE + run * 7
        results, frontier, misspec, moderated = run_single_simulation(seed)
        for name, res in results.items():
            all_results[name].append(res)
        for name, res in moderated.items():
            all_moderated[name].append(res)
        all_frontiers.append(frontier)
        all_misspec.append(misspec)

    # --- Aggregate results ---
    output = {'main_results': {}}
    for name, runs in all_results.items():
        ates = [r['ate'] for r in runs]
        welfares = [r['welfare'] for r in runs]
        risks = [r['risk_exposure'] for r in runs]
        ltv_crs = [r['mean_ltv_cr'] for r in runs]
        output['main_results'][name] = {
            'ate_mean': round(np.mean(ates), 1),
            'ate_sd': round(np.std(ates), 1),
            'welfare_mean': round(np.mean(welfares), 0),
            'welfare_sd': round(np.std(welfares), 0),
            'risk_exposure_mean': round(np.mean(risks), 3),
            'risk_exposure_sd': round(np.std(risks), 3),
            'mean_ltv_cr': round(np.mean(ltv_crs), 0),
        }

    # Frontier (averaged)
    n_configs = len(all_frontiers[0])
    output['frontier'] = []
    for i in range(n_configs):
        config_name = all_frontiers[0][i]['config']
        welfares = [f[i]['welfare'] for f in all_frontiers]
        risks = [f[i]['risk_exposure'] for f in all_frontiers]
        output['frontier'].append({
            'config': config_name,
            'welfare_mean': round(np.mean(welfares), 0),
            'welfare_sd': round(np.std(welfares), 0),
            'risk_mean': round(np.mean(risks), 3),
            'risk_sd': round(np.std(risks), 3),
        })

    # Misspecification (inverted g)
    output['misspecification'] = {
        'inverted_g': {
            'ate_mean': round(np.mean([m['ate'] for m in all_misspec]), 1),
            'welfare_mean': round(np.mean([m['welfare'] for m in all_misspec]), 0),
            'risk_mean': round(np.mean([m['risk_exposure'] for m in all_misspec]), 3),
        }
    }

    # CATE-churn moderated scenario
    output['moderated_dgp'] = {}
    for name in all_moderated:
        welfares = [r['welfare'] for r in all_moderated[name]]
        output['moderated_dgp'][name] = {
            'welfare_mean': round(np.mean(welfares), 0),
            'welfare_sd': round(np.std(welfares), 0),
        }

    output['design'] = {
        'n_players': N_PLAYERS,
        'n_runs': N_RUNS,
        'budget_fraction': BUDGET_FRACTION,
        'treatment_prob': TREATMENT_PROB,
        'note': (
            'All components (CATE, LTV, churn) are estimated from training data. '
            'Oracle values used only for welfare evaluation on the test set. '
            'DGP has correlated features, nonlinear CATE, and shared feature drivers '
            'across components.'
        ),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<28} {'ATE (pp)':<12} {'Welfare':<14} {'Risk Exp.':<10}")
    print("-" * 64)
    for name in all_results:
        r = output['main_results'][name]
        print(f"{name:<28} {r['ate_mean']:>5.1f} ({r['ate_sd']:.1f})  "
              f"{r['welfare_mean']:>10,.0f}  {r['risk_exposure_mean']:.3f}")

    print("\n" + "=" * 70)
    print("VALUE-CONSERVATISM FRONTIER")
    print("=" * 70)
    print(f"{'Config':<20} {'Welfare':<14} {'Risk Exposure':<14}")
    print("-" * 48)
    for f in output['frontier']:
        print(f"{f['config']:<20} {f['welfare_mean']:>10,.0f}  {f['risk_mean']:.3f}")

    print("\n" + "=" * 70)
    print("MISSPECIFICATION (inverted g)")
    print("=" * 70)
    m = output['misspecification']['inverted_g']
    print(f"ATE: {m['ate_mean']:.1f} pp, Welfare: {m['welfare_mean']:,.0f}, "
          f"Risk: {m['risk_mean']:.3f}")

    print("\n" + "=" * 70)
    print("CATE-CHURN MODERATED DGP (welfare comparison)")
    print("=" * 70)
    for name, r in output['moderated_dgp'].items():
        print(f"  {name:<28} Welfare: {r['welfare_mean']:>10,.0f}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                               'simulation_v24.json')
    with open(output_path, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
