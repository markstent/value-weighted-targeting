"""
Supplementary analyses for the bonus allocation paper.
Extends the main simulation with churn-moderated DGP, estimated components,
hyperparameter sensitivity, X-Learner, churn distribution diagnostics,
S-Learner calibration, and Cohen's d effect sizes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Reuse constants and helpers from main simulation
from src.simulation_analysis import (
    N_PLAYERS, N_RUNS, BUDGET_FRACTION, TREATMENT_PROB,
    BOOTSTRAP_SAMPLES, ALPHA, SEGMENTS, STRATEGIES,
    generate_population, f_churn, f_churn_parameterized,
    train_t_learner, score_uplift, train_response_model,
    train_s_learner, score_s_learner,
    evaluate_strategy, bootstrap_ci, paired_permutation_test,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ['account_age', 'weekly_spend', 'login_freq', 'bonus_redeem']


# ---------------------------------------------------------------------------
# Analysis 1: Churn-Moderated DGP
# ---------------------------------------------------------------------------

def generate_population_churn_moderated(n, seed=None):
    """
    Generate synthetic player population identical to generate_population()
    but with churn-moderated treatment effects: low-churn and high-churn
    players are less treatment-responsive (inverse-U moderation).
    """
    rng = np.random.RandomState(seed)

    # Step 1: Generate raw features
    account_age = rng.uniform(1, 36, n)
    weekly_spend = rng.lognormal(mean=3.9, sigma=0.8, size=n)
    login_freq = rng.poisson(lam=4, size=n).astype(float)
    bonus_redeem = rng.beta(2, 5, size=n)

    # Step 2: Assign segments by direct multinomial sampling
    segment_names = list(SEGMENTS.keys())
    segment_probs = [SEGMENTS[s][0] for s in segment_names]
    segment_idx = rng.choice(len(segment_names), size=n, p=segment_probs)
    segments = [segment_names[idx] for idx in segment_idx]

    # Step 3: Modulate features by segment
    seg_array = np.array(segment_idx)

    mask_st = seg_array == 0
    login_freq[mask_st] += rng.poisson(3, mask_st.sum())
    weekly_spend[mask_st] *= rng.uniform(1.2, 1.8, mask_st.sum())
    bonus_redeem[mask_st] += rng.normal(0.05, 0.02, mask_st.sum())

    mask_p = seg_array == 1
    bonus_redeem[mask_p] += rng.normal(0.20, 0.05, mask_p.sum())
    login_freq[mask_p] += rng.poisson(1, mask_p.sum())
    weekly_spend[mask_p] *= rng.uniform(0.7, 1.0, mask_p.sum())

    mask_lc = seg_array == 2
    login_freq[mask_lc] = np.maximum(0, login_freq[mask_lc] - rng.poisson(2, mask_lc.sum()))
    weekly_spend[mask_lc] *= rng.uniform(0.4, 0.7, mask_lc.sum())
    bonus_redeem[mask_lc] -= rng.normal(0.08, 0.03, mask_lc.sum())

    mask_sd = seg_array == 3
    weekly_spend[mask_sd] *= rng.uniform(1.3, 2.0, mask_sd.sum())
    bonus_redeem[mask_sd] -= rng.normal(0.12, 0.04, mask_sd.sum())
    account_age[mask_sd] += rng.uniform(5, 15, mask_sd.sum())

    bonus_redeem = np.clip(bonus_redeem, 0, 1)
    login_freq = np.maximum(0, login_freq)

    z_login = (login_freq - 4.0) / 2.0

    ltv = weekly_spend * 12 + login_freq * 50 + rng.normal(0, 100, n)
    ltv = np.clip(ltv, 50, 10000)

    churn_logit = -1.5 + 0.05 * (36 - account_age) - 0.3 * login_freq + 0.01 * (50 - weekly_spend)
    churn_prob = 1 / (1 + np.exp(-churn_logit + rng.normal(0, 0.5, n)))
    churn_prob = np.clip(churn_prob, 0.01, 0.99)

    treatment = rng.binomial(1, TREATMENT_PROB, n)

    baselines = np.array([SEGMENTS[s][1] for s in segments])
    taus = np.array([SEGMENTS[s][2] for s in segments])

    tau_noise = rng.normal(0, 0.10, n)
    effective_tau = taus + tau_noise

    # Inverse-U moderation: treatment effects peak at moderate churn
    churn_mod = np.where(churn_prob < 0.10, 0.3,
                np.where(churn_prob < 0.30, 0.3 + 0.7 * (churn_prob - 0.10) / 0.20,
                np.where(churn_prob <= 0.60, 1.0,
                np.where(churn_prob < 0.85, 1.0 - 0.7 * (churn_prob - 0.60) / 0.25,
                0.3))))
    effective_tau = effective_tau * churn_mod

    y0_prob = np.clip(baselines + 0.1 * z_login, 0.01, 0.99)
    y1_prob = np.clip(y0_prob + effective_tau, 0.01, 0.99)

    y0 = rng.binomial(1, y0_prob)
    y1 = rng.binomial(1, y1_prob)
    y_obs = treatment * y1 + (1 - treatment) * y0

    spend_90th = np.percentile(weekly_spend, 90)
    login_90th = np.percentile(login_freq, 90)
    rg_flag = ((weekly_spend > spend_90th) & (login_freq > login_90th)).astype(int)

    df = pd.DataFrame({
        'account_age': account_age,
        'weekly_spend': weekly_spend,
        'login_freq': login_freq,
        'bonus_redeem': bonus_redeem,
        'ltv': ltv,
        'churn_prob': churn_prob,
        'treatment': treatment,
        'y_obs': y_obs,
        'segment': segments,
        'true_tau': effective_tau,
        'y0': y0,
        'y1': y1,
        'responsible_gaming_flag': rg_flag,
    })
    return df


def run_churn_moderated_simulation(seed):
    """Run one simulation iteration using the churn-moderated DGP."""
    df = generate_population_churn_moderated(N_PLAYERS, seed=seed)
    df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

    model_t, model_c, features = train_t_learner(df_train)
    response_model, resp_features = train_response_model(df_train)

    uplift_scores = score_uplift(model_t, model_c, df_test, features)
    X_test = df_test[resp_features]
    response_scores = response_model.predict_proba(X_test)[:, 1]

    f_churn_scores = np.array([f_churn(p) for p in df_test['churn_prob'].values])
    full_scores = uplift_scores * df_test['ltv'].values * f_churn_scores

    s_model = train_s_learner(df_train, features)
    s_learner_scores = score_s_learner(s_model, df_test, features)
    s_learner_ltv_scores = s_learner_scores * df_test['ltv'].values
    s_learner_full_scores = s_learner_scores * df_test['ltv'].values * f_churn_scores
    uplift_ltv_scores = uplift_scores * df_test['ltv'].values

    rng = np.random.RandomState(seed)
    random_scores = rng.uniform(0, 1, len(df_test))

    results = []
    results.append(evaluate_strategy(df_test, random_scores, BUDGET_FRACTION, 'Random'))
    results.append(evaluate_strategy(df_test, response_scores, BUDGET_FRACTION, 'Response Model'))
    results.append(evaluate_strategy(df_test, s_learner_scores, BUDGET_FRACTION, 'S-Learner'))
    results.append(evaluate_strategy(df_test, uplift_scores, BUDGET_FRACTION, 'Uplift Only'))
    results.append(evaluate_strategy(df_test, uplift_ltv_scores, BUDGET_FRACTION, 'Uplift x LTV'))
    results.append(evaluate_strategy(df_test, full_scores, BUDGET_FRACTION, 'Full Framework'))
    results.append(evaluate_strategy(df_test, s_learner_ltv_scores, BUDGET_FRACTION, 'S-Learner x LTV'))
    results.append(evaluate_strategy(df_test, s_learner_full_scores, BUDGET_FRACTION, 'S-Learner Full'))

    return results


def run_analysis_1():
    """Analysis 1: 8-strategy comparison under churn-moderated DGP."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Churn-Moderated DGP")
    print("=" * 70)

    all_results = {s: [] for s in STRATEGIES}
    for run in range(N_RUNS):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        results = run_churn_moderated_simulation(seed=run * 7 + 13)
        for r in results:
            all_results[r['strategy']].append(r)

    summary = {}
    print(f"\n{'Strategy':<20} {'Inc. Revenue':<20} {'ATE (pp)':<18}")
    print("-" * 58)
    for strategy in STRATEGIES:
        ates = np.array([r['ate_pp'] for r in all_results[strategy]])
        revs = np.array([r['incremental_revenue'] for r in all_results[strategy]])
        summary[strategy] = {
            'ate_mean': round(float(np.mean(ates)), 1),
            'ate_sd': round(float(np.std(ates)), 1),
            'rev_mean': round(float(np.mean(revs)), 0),
            'rev_sd': round(float(np.std(revs)), 0),
        }
        print(f"{strategy:<20} ${np.mean(revs):>9,.0f} ({np.std(revs):,.0f})  "
              f"{np.mean(ates):>5.1f} ({np.std(ates):.1f})")

    return summary


# ---------------------------------------------------------------------------
# Analysis 2: Estimated Components (LTV and churn models)
# ---------------------------------------------------------------------------

def train_ltv_model(df_train, features):
    """Train a GradientBoostingRegressor to predict LTV from features."""
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(df_train[features], df_train['ltv'])
    return model


def train_churn_model(df_train, features):
    """Train a GradientBoostingRegressor to predict churn_prob from features."""
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(df_train[features], df_train['churn_prob'])
    return model, None


def run_analysis_2():
    """Analysis 2: Oracle vs estimated LTV/churn components."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Oracle vs Estimated Components")
    print("=" * 70)

    oracle_revs = []
    estimated_revs = []
    oracle_ates = []
    estimated_ates = []

    for run in range(N_RUNS):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = run * 7 + 13
        df = generate_population(N_PLAYERS, seed=seed)
        df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

        # Train uplift model
        model_t, model_c, features = train_t_learner(df_train)
        uplift_scores = score_uplift(model_t, model_c, df_test, features)

        # Oracle Full Framework
        f_churn_scores = np.array([f_churn(p) for p in df_test['churn_prob'].values])
        oracle_scores = uplift_scores * df_test['ltv'].values * f_churn_scores
        r_oracle = evaluate_strategy(df_test, oracle_scores, BUDGET_FRACTION, 'Oracle Full')
        oracle_revs.append(r_oracle['incremental_revenue'])
        oracle_ates.append(r_oracle['ate_pp'])

        # Estimated components
        ltv_model = train_ltv_model(df_train, features)
        churn_model, churn_median = train_churn_model(df_train, features)

        pred_ltv = ltv_model.predict(df_test[features])
        # Predict continuous churn probability directly
        pred_churn_prob = churn_model.predict(df_test[features])
        pred_churn_prob = np.clip(pred_churn_prob, 0.01, 0.99)
        f_churn_estimated = np.array([f_churn(p) for p in pred_churn_prob])

        estimated_scores = uplift_scores * pred_ltv * f_churn_estimated
        r_est = evaluate_strategy(df_test, estimated_scores, BUDGET_FRACTION, 'Estimated Full')
        estimated_revs.append(r_est['incremental_revenue'])
        estimated_ates.append(r_est['ate_pp'])

    oracle_revs = np.array(oracle_revs)
    estimated_revs = np.array(estimated_revs)
    oracle_ates = np.array(oracle_ates)
    estimated_ates = np.array(estimated_ates)

    rev_degradation = float(np.mean(oracle_revs) - np.mean(estimated_revs))
    rev_degradation_pct = rev_degradation / np.mean(oracle_revs) * 100 if np.mean(oracle_revs) != 0 else 0

    result = {
        'oracle_rev_mean': round(float(np.mean(oracle_revs)), 0),
        'oracle_rev_sd': round(float(np.std(oracle_revs)), 0),
        'oracle_ate_mean': round(float(np.mean(oracle_ates)), 1),
        'estimated_rev_mean': round(float(np.mean(estimated_revs)), 0),
        'estimated_rev_sd': round(float(np.std(estimated_revs)), 0),
        'estimated_ate_mean': round(float(np.mean(estimated_ates)), 1),
        'rev_degradation': round(rev_degradation, 0),
        'rev_degradation_pct': round(rev_degradation_pct, 1),
    }

    print(f"  Oracle Full:    Rev=${np.mean(oracle_revs):,.0f} (SD={np.std(oracle_revs):,.0f}), "
          f"ATE={np.mean(oracle_ates):.1f}pp")
    print(f"  Estimated Full: Rev=${np.mean(estimated_revs):,.0f} (SD={np.std(estimated_revs):,.0f}), "
          f"ATE={np.mean(estimated_ates):.1f}pp")
    print(f"  Revenue degradation: ${rev_degradation:,.0f} ({rev_degradation_pct:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Analysis 3: Hyperparameter Sensitivity
# ---------------------------------------------------------------------------

def train_t_learner_custom(df_train, features, n_estimators=100, max_depth=4, learning_rate=0.1):
    """Train T-Learner with custom hyperparameters."""
    treated = df_train[df_train['treatment'] == 1]
    control = df_train[df_train['treatment'] == 0]

    model_t = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42
    )
    model_c = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42
    )
    model_t.fit(treated[features], treated['y_obs'])
    model_c.fit(control[features], control['y_obs'])
    return model_t, model_c


def train_s_learner_custom(df_train, base_features, n_estimators=100, max_depth=4, learning_rate=0.1):
    """Train S-Learner with custom hyperparameters."""
    features = base_features + ['treatment']
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42
    )
    model.fit(df_train[features], df_train['y_obs'])
    return model


def run_analysis_3():
    """Analysis 3: Hyperparameter sensitivity across 3 configurations."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Hyperparameter Sensitivity")
    print("=" * 70)

    configs = {
        'Default (100/4/0.1)': dict(n_estimators=100, max_depth=4, learning_rate=0.1),
        'High capacity (200/6/0.05)': dict(n_estimators=200, max_depth=6, learning_rate=0.05),
        'Low capacity (50/3/0.2)': dict(n_estimators=50, max_depth=3, learning_rate=0.2),
    }

    results = {}
    for config_name, params in configs.items():
        print(f"\n  Config: {config_name}")
        config_results = {s: [] for s in STRATEGIES}

        for run in range(N_RUNS):
            if (run + 1) % 10 == 0:
                print(f"    Run {run + 1}/{N_RUNS}...")
            seed = run * 7 + 13
            df = generate_population(N_PLAYERS, seed=seed)
            df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

            features = FEATURES
            model_t, model_c = train_t_learner_custom(df_train, features, **params)
            response_model, _ = train_response_model(df_train)  # always default for response

            uplift_scores = score_uplift(model_t, model_c, df_test, features)
            response_scores = response_model.predict_proba(df_test[features])[:, 1]

            f_churn_scores = np.array([f_churn(p) for p in df_test['churn_prob'].values])
            full_scores = uplift_scores * df_test['ltv'].values * f_churn_scores

            s_model = train_s_learner_custom(df_train, features, **params)
            s_learner_scores = score_s_learner(s_model, df_test, features)
            s_learner_ltv_scores = s_learner_scores * df_test['ltv'].values
            s_learner_full_scores = s_learner_scores * df_test['ltv'].values * f_churn_scores
            uplift_ltv_scores = uplift_scores * df_test['ltv'].values

            rng = np.random.RandomState(seed)
            random_scores = rng.uniform(0, 1, len(df_test))

            run_results = []
            run_results.append(evaluate_strategy(df_test, random_scores, BUDGET_FRACTION, 'Random'))
            run_results.append(evaluate_strategy(df_test, response_scores, BUDGET_FRACTION, 'Response Model'))
            run_results.append(evaluate_strategy(df_test, s_learner_scores, BUDGET_FRACTION, 'S-Learner'))
            run_results.append(evaluate_strategy(df_test, uplift_scores, BUDGET_FRACTION, 'Uplift Only'))
            run_results.append(evaluate_strategy(df_test, uplift_ltv_scores, BUDGET_FRACTION, 'Uplift x LTV'))
            run_results.append(evaluate_strategy(df_test, full_scores, BUDGET_FRACTION, 'Full Framework'))
            run_results.append(evaluate_strategy(df_test, s_learner_ltv_scores, BUDGET_FRACTION, 'S-Learner x LTV'))
            run_results.append(evaluate_strategy(df_test, s_learner_full_scores, BUDGET_FRACTION, 'S-Learner Full'))

            for r in run_results:
                config_results[r['strategy']].append(r)

        config_summary = {}
        for strategy in STRATEGIES:
            ates = [r['ate_pp'] for r in config_results[strategy]]
            revs = [r['incremental_revenue'] for r in config_results[strategy]]
            config_summary[strategy] = {
                'ate_mean': round(float(np.mean(ates)), 1),
                'ate_sd': round(float(np.std(ates)), 1),
                'rev_mean': round(float(np.mean(revs)), 0),
                'rev_sd': round(float(np.std(revs)), 0),
            }

        results[config_name] = config_summary

        # Print S-Learner vs T-Learner ranking
        s_rev = config_summary['S-Learner']['rev_mean']
        t_rev = config_summary['Uplift Only']['rev_mean']
        sf_rev = config_summary['S-Learner Full']['rev_mean']
        ff_rev = config_summary['Full Framework']['rev_mean']
        print(f"    S-Learner Rev: ${s_rev:,.0f}, T-Learner Rev: ${t_rev:,.0f} "
              f"-> {'S-Learner' if s_rev > t_rev else 'T-Learner'} wins (uplift only)")
        print(f"    S-Learner Full Rev: ${sf_rev:,.0f}, Full Framework Rev: ${ff_rev:,.0f} "
              f"-> {'S-Learner Full' if sf_rev > ff_rev else 'Full Framework'} wins (full)")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: X-Learner
# ---------------------------------------------------------------------------

def train_x_learner(df_train, features):
    """Train X-Learner: two-stage approach with imputed treatment effects."""
    treated = df_train[df_train['treatment'] == 1]
    control = df_train[df_train['treatment'] == 0]

    # Step 1: fit outcome models (same as T-Learner)
    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model_c = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model_t.fit(treated[features], treated['y_obs'])
    model_c.fit(control[features], control['y_obs'])

    # Step 2: compute imputed treatment effects
    tau_treated = treated['y_obs'].values - model_c.predict_proba(treated[features])[:, 1]
    tau_control = model_t.predict_proba(control[features])[:, 1] - control['y_obs'].values

    # Step 3: fit CATE models on imputed effects
    cate_t = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    cate_c = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    cate_t.fit(treated[features], tau_treated)
    cate_c.fit(control[features], tau_control)

    return cate_t, cate_c, model_t, model_c, features


def score_x_learner(cate_t, cate_c, df, features, propensity=0.5):
    """Score X-Learner uplift as weighted combination of CATE estimates."""
    tau_t = cate_t.predict(df[features])
    tau_c = cate_c.predict(df[features])
    return propensity * tau_t + (1 - propensity) * tau_c


def run_analysis_4():
    """Analysis 4: X-Learner standalone and composite variants."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: X-Learner Implementation")
    print("=" * 70)

    x_strategies = ['X-Learner', 'X-Learner x LTV', 'X-Learner Full']
    all_results = {s: [] for s in x_strategies + ['Full Framework', 'Uplift Only']}

    for run in range(N_RUNS):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        seed = run * 7 + 13
        df = generate_population(N_PLAYERS, seed=seed)
        df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

        features = FEATURES

        # X-Learner
        cate_t, cate_c, _, _, _ = train_x_learner(df_train, features)
        x_scores = score_x_learner(cate_t, cate_c, df_test, features)

        f_churn_scores = np.array([f_churn(p) for p in df_test['churn_prob'].values])
        x_ltv_scores = x_scores * df_test['ltv'].values
        x_full_scores = x_scores * df_test['ltv'].values * f_churn_scores

        all_results['X-Learner'].append(
            evaluate_strategy(df_test, x_scores, BUDGET_FRACTION, 'X-Learner'))
        all_results['X-Learner x LTV'].append(
            evaluate_strategy(df_test, x_ltv_scores, BUDGET_FRACTION, 'X-Learner x LTV'))
        all_results['X-Learner Full'].append(
            evaluate_strategy(df_test, x_full_scores, BUDGET_FRACTION, 'X-Learner Full'))

        # Also run T-Learner variants for comparison
        model_t, model_c, _ = train_t_learner(df_train)
        uplift_scores = score_uplift(model_t, model_c, df_test, features)
        full_scores = uplift_scores * df_test['ltv'].values * f_churn_scores

        all_results['Uplift Only'].append(
            evaluate_strategy(df_test, uplift_scores, BUDGET_FRACTION, 'Uplift Only'))
        all_results['Full Framework'].append(
            evaluate_strategy(df_test, full_scores, BUDGET_FRACTION, 'Full Framework'))

    summary = {}
    print(f"\n{'Strategy':<22} {'Inc. Revenue':<20} {'ATE (pp)':<18}")
    print("-" * 60)
    for strategy in x_strategies + ['Uplift Only', 'Full Framework']:
        ates = [r['ate_pp'] for r in all_results[strategy]]
        revs = [r['incremental_revenue'] for r in all_results[strategy]]
        summary[strategy] = {
            'ate_mean': round(float(np.mean(ates)), 1),
            'ate_sd': round(float(np.std(ates)), 1),
            'rev_mean': round(float(np.mean(revs)), 0),
            'rev_sd': round(float(np.std(revs)), 0),
        }
        print(f"{strategy:<22} ${np.mean(revs):>9,.0f} ({np.std(revs):,.0f})  "
              f"{np.mean(ates):>5.1f} ({np.std(ates):.1f})")

    return summary


# ---------------------------------------------------------------------------
# Analysis 5: Churn Distribution and Function Plot
# ---------------------------------------------------------------------------

def run_analysis_5():
    """Analysis 5: Churn distribution across regimes and f(p) plot."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Churn Distribution and Function Plot")
    print("=" * 70)

    df = generate_population(N_PLAYERS, seed=42)
    churn = df['churn_prob'].values

    regimes = {
        'p < 0.05 (zero zone)': churn < 0.05,
        '0.05 <= p < 0.40 (ramp)': (churn >= 0.05) & (churn < 0.40),
        '0.40 <= p <= 0.60 (plateau)': (churn >= 0.40) & (churn <= 0.60),
        '0.60 < p < 0.90 (descent)': (churn > 0.60) & (churn < 0.90),
        'p >= 0.90 (floor)': churn >= 0.90,
    }

    distribution = {}
    print(f"\n{'Regime':<35} {'Count':<10} {'Pct':<10} {'Mean churn_prob':<15}")
    print("-" * 70)
    for name, mask in regimes.items():
        count = int(mask.sum())
        pct = round(count / len(churn) * 100, 1)
        mean_p = round(float(churn[mask].mean()), 4) if count > 0 else 0
        distribution[name] = {
            'count': count,
            'pct': pct,
            'mean_churn_prob': mean_p,
        }
        print(f"{name:<35} {count:<10} {pct:<10.1f} {mean_p:<15.4f}")

    # Generate f(p; theta_0) plot
    p_vals = np.linspace(0, 1, 1000)
    f_vals = np.array([f_churn(p) for p in p_vals])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_vals, f_vals, 'b-', linewidth=2)
    ax.set_xlabel('Churn probability (p)', fontsize=12)
    ax.set_ylabel(r'$f(p; \theta_0)$', fontsize=12)
    ax.set_title(r'Churn Regularization Function $f(p; \theta_0)$', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.10)

    # Shade regimes
    ax.axvspan(0, 0.05, alpha=0.1, color='red', label='Zero zone')
    ax.axvspan(0.05, 0.40, alpha=0.1, color='orange', label='Ramp')
    ax.axvspan(0.40, 0.60, alpha=0.1, color='green', label='Plateau')
    ax.axvspan(0.60, 0.90, alpha=0.1, color='orange')
    ax.axvspan(0.90, 1.0, alpha=0.1, color='gray', label='Floor')

    # Mark thresholds
    for thresh in [0.05, 0.40, 0.60, 0.90]:
        ax.axvline(thresh, color='gray', linestyle='--', alpha=0.5)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(BASE_DIR, 'figures', 'fig4_churn_function.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved to {fig_path}")

    return distribution


# ---------------------------------------------------------------------------
# Analysis 6: S-Learner Calibration
# ---------------------------------------------------------------------------

def run_analysis_6():
    """Analysis 6: S-Learner calibration (predicted uplift vs latent by decile)."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: S-Learner Calibration")
    print("=" * 70)

    calibration_data = []
    for d in range(10):
        pred_uplifts = []
        latent_uplifts = []

        for run in range(N_RUNS):
            seed = run * 7 + 13
            df = generate_population(N_PLAYERS, seed=seed)
            df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

            features = FEATURES
            s_model = train_s_learner(df_train, features)
            s_scores = score_s_learner(s_model, df_test, features)

            df_t = df_test.copy()
            df_t['pred_uplift'] = s_scores
            df_t['decile'] = pd.qcut(df_t['pred_uplift'], 10, labels=False, duplicates='drop')
            dec = df_t[df_t['decile'] == d]
            if len(dec) > 0:
                pred_uplifts.append(dec['pred_uplift'].mean())
                latent_uplifts.append(dec['true_tau'].mean())

        calibration_data.append({
            'decile': d + 1,
            'mean_predicted_uplift': round(float(np.mean(pred_uplifts)), 4),
            'mean_latent_uplift': round(float(np.mean(latent_uplifts)), 4),
        })

    print(f"\n{'Decile':<10} {'Pred Uplift':<18} {'Latent Uplift':<18}")
    print("-" * 46)
    for row in calibration_data:
        print(f"{row['decile']:<10} {row['mean_predicted_uplift']:<18.4f} "
              f"{row['mean_latent_uplift']:<18.4f}")

    return calibration_data


# ---------------------------------------------------------------------------
# Analysis 7: Cohen's d Effect Sizes
# ---------------------------------------------------------------------------

def run_analysis_7():
    """Analysis 7: Cohen's d for pairwise strategy comparisons."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Cohen's d Effect Sizes")
    print("=" * 70)

    # First collect per-run revenue for all strategies
    all_revs = {s: [] for s in STRATEGIES}
    for run in range(N_RUNS):
        seed = run * 7 + 13
        df = generate_population(N_PLAYERS, seed=seed)
        df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

        features = FEATURES
        model_t, model_c, _ = train_t_learner(df_train)
        response_model, _ = train_response_model(df_train)

        uplift_scores = score_uplift(model_t, model_c, df_test, features)
        response_scores = response_model.predict_proba(df_test[features])[:, 1]

        f_churn_scores = np.array([f_churn(p) for p in df_test['churn_prob'].values])
        full_scores = uplift_scores * df_test['ltv'].values * f_churn_scores

        s_model = train_s_learner(df_train, features)
        s_learner_scores = score_s_learner(s_model, df_test, features)
        s_learner_ltv_scores = s_learner_scores * df_test['ltv'].values
        s_learner_full_scores = s_learner_scores * df_test['ltv'].values * f_churn_scores
        uplift_ltv_scores = uplift_scores * df_test['ltv'].values

        rng = np.random.RandomState(seed)
        random_scores = rng.uniform(0, 1, len(df_test))

        score_map = {
            'Random': random_scores,
            'Response Model': response_scores,
            'S-Learner': s_learner_scores,
            'Uplift Only': uplift_scores,
            'Uplift x LTV': uplift_ltv_scores,
            'Full Framework': full_scores,
            'S-Learner x LTV': s_learner_ltv_scores,
            'S-Learner Full': s_learner_full_scores,
        }

        for strategy in STRATEGIES:
            r = evaluate_strategy(df_test, score_map[strategy], BUDGET_FRACTION, strategy)
            all_revs[strategy].append(r['incremental_revenue'])

    # Compute Cohen's d for key pairwise comparisons
    comparisons = [
        ('Full Framework', 'Random'),
        ('Full Framework', 'Uplift Only'),
        ('Full Framework', 'Uplift x LTV'),
        ('Full Framework', 'Response Model'),
        ('S-Learner Full', 'S-Learner'),
        ('S-Learner Full', 'Full Framework'),
        ('S-Learner x LTV', 'Uplift x LTV'),
        ('Uplift Only', 'Random'),
        ('S-Learner', 'Random'),
        ('Uplift x LTV', 'Random'),
        ('S-Learner x LTV', 'Random'),
    ]

    cohens_d_results = {}
    print(f"\n{'Comparison':<45} {'Cohen d':<12} {'Mean Diff':<15} {'p-value':<10}")
    print("-" * 82)

    for strat_a, strat_b in comparisons:
        a = np.array(all_revs[strat_a])
        b = np.array(all_revs[strat_b])
        diffs = a - b
        d_val = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else 0.0
        _, p_val = paired_permutation_test(a, b, seed=42)

        key = f"{strat_a} vs {strat_b}"
        cohens_d_results[key] = {
            'cohens_d': round(d_val, 3),
            'mean_diff': round(float(np.mean(diffs)), 0),
            'p_value': round(float(p_val), 4),
        }

        # Interpret effect size
        abs_d = abs(d_val)
        if abs_d < 0.2:
            interp = "negligible"
        elif abs_d < 0.5:
            interp = "small"
        elif abs_d < 0.8:
            interp = "medium"
        else:
            interp = "large"

        print(f"{key:<45} {d_val:<12.3f} ${np.mean(diffs):>10,.0f}     {p_val:<10.4f} ({interp})")

    return cohens_d_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SUPPLEMENTARY ANALYSES: Bonus Allocation Paper")
    print("=" * 70)

    output = {}

    # Analysis 1: Churn-moderated DGP
    output['churn_moderated_dgp'] = run_analysis_1()

    # Analysis 2: Estimated components
    output['estimated_components'] = run_analysis_2()

    # Analysis 3: Hyperparameter sensitivity
    output['hyperparameter_sensitivity'] = run_analysis_3()

    # Analysis 4: X-Learner
    output['x_learner'] = run_analysis_4()

    # Analysis 5: Churn distribution
    output['churn_distribution'] = run_analysis_5()

    # Analysis 6: S-Learner calibration
    output['s_learner_calibration'] = run_analysis_6()

    # Analysis 7: Cohen's d
    output['cohens_d'] = run_analysis_7()

    # Save results
    output_path = os.path.join(BASE_DIR, 'outputs', 'supplementary_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print(f"All supplementary results saved to {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
