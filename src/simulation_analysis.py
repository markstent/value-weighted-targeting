"""
Enhanced simulation analysis for the bonus allocation paper.
Generates synthetic data with known treatment effects, trains uplift
baselines, and compares targeting strategies with bootstrap CIs and
permutation tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import json
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# --- Configuration ---
N_PLAYERS = 10000
N_RUNS = 50
BUDGET_FRACTION = 0.30
TREATMENT_PROB = 0.50
BOOTSTRAP_SAMPLES = 1000
ALPHA = 0.05
STRATEGIES = ['Random', 'Response Model', 'S-Learner', 'Uplift Only', 'Uplift x LTV', 'Full Framework']

# Segment parameters: (prevalence, P0_baseline, tau_treatment_effect)
SEGMENTS = {
    'Sure Things':  (0.40, 0.85, +0.03),
    'Persuadables': (0.15, 0.40, +0.35),
    'Lost Causes':  (0.35, 0.15, +0.05),
    'Sleeping Dogs': (0.10, 0.70, -0.15),
}


def generate_population(n, seed=None):
    """
    Generate synthetic player population with known causal structure.

    Key design: segments are assigned by direct multinomial sampling at stated
    prevalences, then features are modulated conditional on segment. This
    preserves exact prevalences while giving the T-Learner predictive signal.
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

    # Step 3: Modulate features by segment to preserve predictive signal.
    # Each segment gets shifts on MULTIPLE features so the T-Learner can
    # distinguish segments from observables.
    seg_array = np.array(segment_idx)

    # Sure Things (idx=0): high login, high spend, moderate redeem
    mask_st = seg_array == 0
    login_freq[mask_st] += rng.poisson(3, mask_st.sum())
    weekly_spend[mask_st] *= rng.uniform(1.2, 1.8, mask_st.sum())
    bonus_redeem[mask_st] += rng.normal(0.05, 0.02, mask_st.sum())

    # Persuadables (idx=1): high bonus_redeem, moderate login, lower spend
    mask_p = seg_array == 1
    bonus_redeem[mask_p] += rng.normal(0.20, 0.05, mask_p.sum())
    login_freq[mask_p] += rng.poisson(1, mask_p.sum())
    weekly_spend[mask_p] *= rng.uniform(0.7, 1.0, mask_p.sum())

    # Lost Causes (idx=2): low login, low spend, low redeem
    mask_lc = seg_array == 2
    login_freq[mask_lc] = np.maximum(0, login_freq[mask_lc] - rng.poisson(2, mask_lc.sum()))
    weekly_spend[mask_lc] *= rng.uniform(0.4, 0.7, mask_lc.sum())
    bonus_redeem[mask_lc] -= rng.normal(0.08, 0.03, mask_lc.sum())

    # Sleeping Dogs (idx=3): high spend, low redeem, longer tenure
    mask_sd = seg_array == 3
    weekly_spend[mask_sd] *= rng.uniform(1.3, 2.0, mask_sd.sum())
    bonus_redeem[mask_sd] -= rng.normal(0.12, 0.04, mask_sd.sum())
    account_age[mask_sd] += rng.uniform(5, 15, mask_sd.sum())

    bonus_redeem = np.clip(bonus_redeem, 0, 1)
    login_freq = np.maximum(0, login_freq)

    # Recompute z_login after feature modulation (used for baseline modulation)
    z_login = (login_freq - 4.0) / 2.0

    # NOTE: LTV and churn_prob are generated directly from the DGP, not
    # predicted by trained models. The simulation validates the scoring
    # rule (uplift x LTV x f(churn)) given known values, not end-to-end
    # model pipeline performance.
    # Step 3: Generate LTV (correlated with spend and login)
    ltv = weekly_spend * 12 + login_freq * 50 + rng.normal(0, 100, n)
    ltv = np.clip(ltv, 50, 10000)

    # Step 4: Generate churn probability
    churn_logit = -1.5 + 0.05 * (36 - account_age) - 0.3 * login_freq + 0.01 * (50 - weekly_spend)
    churn_prob = 1 / (1 + np.exp(-churn_logit + rng.normal(0, 0.5, n)))
    churn_prob = np.clip(churn_prob, 0.01, 0.99)

    # Step 5: Treatment assignment (randomized)
    treatment = rng.binomial(1, TREATMENT_PROB, n)

    # Step 6: Generate potential outcomes from segment parameters
    baselines = np.array([SEGMENTS[s][1] for s in segments])
    taus = np.array([SEGMENTS[s][2] for s in segments])

    # Add within-segment heterogeneity (SD=0.10 makes uplift estimation realistic)
    tau_noise = rng.normal(0, 0.10, n)
    effective_tau = taus + tau_noise

    # Baseline varies with features (meaningful modulation)
    y0_prob = np.clip(baselines + 0.1 * z_login, 0.01, 0.99)
    y1_prob = np.clip(y0_prob + effective_tau, 0.01, 0.99)

    y0 = rng.binomial(1, y0_prob)
    y1 = rng.binomial(1, y1_prob)
    y_obs = treatment * y1 + (1 - treatment) * y0

    # Responsible gaming flag: synthetic proxy for demonstration only.
    # Production systems require validated vulnerability models, not
    # simple percentile thresholds.
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


def f_churn(p):
    """Churn-based regularization function (inverse-U shape)."""
    if p < 0.05:
        return 0.0
    elif p < 0.40:
        return (p - 0.05) / 0.35
    elif p <= 0.60:
        return 1.0
    elif p < 0.90:
        return 1.0 - 0.8 * (p - 0.60) / 0.30
    else:
        return 0.2


def f_churn_parameterized(p, low=0.05, mid_low=0.40, mid_high=0.60, high=0.90, floor=0.2):
    """Churn regularization with configurable thresholds."""
    if p < low:
        return 0.0
    elif p < mid_low:
        return (p - low) / (mid_low - low)
    elif p <= mid_high:
        return 1.0
    elif p < high:
        return 1.0 - (1.0 - floor) * (p - mid_high) / (high - mid_high)
    else:
        return floor


def train_t_learner(df_train):
    """Train T-Learner: separate models for treatment and control."""
    features = ['account_age', 'weekly_spend', 'login_freq', 'bonus_redeem']

    treated = df_train[df_train['treatment'] == 1]
    control = df_train[df_train['treatment'] == 0]

    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model_c = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )

    model_t.fit(treated[features], treated['y_obs'])
    model_c.fit(control[features], control['y_obs'])

    return model_t, model_c, features


def score_uplift(model_t, model_c, df, features):
    """Score uplift as difference between treatment and control predictions."""
    X = df[features]
    pred_t = model_t.predict_proba(X)[:, 1]
    pred_c = model_c.predict_proba(X)[:, 1]
    return pred_t - pred_c


def train_response_model(df_train):
    """Train traditional response model (ignores causal structure)."""
    features = ['account_age', 'weekly_spend', 'login_freq', 'bonus_redeem']
    treated = df_train[df_train['treatment'] == 1]

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(treated[features], treated['y_obs'])
    return model, features


def train_s_learner(df_train, base_features):
    """Train S-Learner: single model with treatment as a feature."""
    features = base_features + ['treatment']
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(df_train[features], df_train['y_obs'])
    return model


def score_s_learner(model, df, base_features):
    """Score S-Learner uplift by differencing predictions with treatment=1 vs treatment=0."""
    X = df[base_features].copy()
    X1 = X.copy()
    X1['treatment'] = 1
    X0 = X.copy()
    X0['treatment'] = 0
    pred_t = model.predict_proba(X1)[:, 1]
    pred_c = model.predict_proba(X0)[:, 1]
    return pred_t - pred_c


def evaluate_strategy(df_test, scores, budget_frac, strategy_name):
    """
    Evaluate a targeting strategy using true potential outcomes.
    Excludes responsible-gaming-flagged players before targeting.
    Returns incremental conversions and ATE for treated group.
    """
    # Exclude responsible gaming flagged players
    eligible_mask = df_test['responsible_gaming_flag'] == 0
    df_eligible = df_test[eligible_mask]
    eligible_scores = scores[eligible_mask.values]

    n_treat = int(len(df_eligible) * budget_frac)
    top_idx = np.argsort(-eligible_scores)[:n_treat]

    # Using true potential outcomes (available in simulation)
    targeted = df_eligible.iloc[top_idx]
    incremental = targeted['y1'].sum() - targeted['y0'].sum()
    ate = (targeted['y1'].mean() - targeted['y0'].mean()) * 100  # percentage points

    # Incremental revenue (using LTV)
    inc_revenue = ((targeted['y1'] - targeted['y0']) * targeted['ltv']).sum()

    # Segment composition
    seg_counts = targeted['segment'].value_counts()
    persuadable_frac = seg_counts.get('Persuadables', 0) / len(targeted)
    sleeping_dog_frac = seg_counts.get('Sleeping Dogs', 0) / len(targeted)

    return {
        'strategy': strategy_name,
        'incremental_conversions': incremental,
        'ate_pp': ate,
        'incremental_revenue': inc_revenue,
        'persuadable_frac': persuadable_frac,
        'sleeping_dog_frac': sleeping_dog_frac,
        'n_targeted': n_treat,
        'n_rg_excluded': int((~eligible_mask).sum()),
    }


def run_single_simulation(seed):
    """Run one complete simulation iteration."""
    df = generate_population(N_PLAYERS, seed=seed)

    # Train/test split
    df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

    # Train models
    model_t, model_c, features = train_t_learner(df_train)
    response_model, resp_features = train_response_model(df_train)

    # Score test set
    uplift_scores = score_uplift(model_t, model_c, df_test, features)

    # Response model scores
    X_test = df_test[resp_features]
    response_scores = response_model.predict_proba(X_test)[:, 1]

    # Full framework: uplift x LTV x f(churn)
    f_churn_scores = np.array([f_churn(p) for p in df_test['churn_prob'].values])
    full_scores = uplift_scores * df_test['ltv'].values * f_churn_scores

    # S-Learner
    s_model = train_s_learner(df_train, features)
    s_learner_scores = score_s_learner(s_model, df_test, features)

    # Uplift x LTV (no churn regularization) - central ablation
    uplift_ltv_scores = uplift_scores * df_test['ltv'].values

    # Random scores
    rng = np.random.RandomState(seed)
    random_scores = rng.uniform(0, 1, len(df_test))

    # Evaluate all strategies
    results = []
    results.append(evaluate_strategy(df_test, random_scores, BUDGET_FRACTION, 'Random'))
    results.append(evaluate_strategy(df_test, response_scores, BUDGET_FRACTION, 'Response Model'))
    results.append(evaluate_strategy(df_test, s_learner_scores, BUDGET_FRACTION, 'S-Learner'))
    results.append(evaluate_strategy(df_test, uplift_scores, BUDGET_FRACTION, 'Uplift Only'))
    results.append(evaluate_strategy(df_test, uplift_ltv_scores, BUDGET_FRACTION, 'Uplift x LTV'))
    results.append(evaluate_strategy(df_test, full_scores, BUDGET_FRACTION, 'Full Framework'))

    # Also compute cumulative gain curve for uplift model
    gain_data = compute_cumulative_gains(df_test, uplift_scores)

    # Feature importance
    feat_imp_t = dict(zip(features, model_t.feature_importances_))
    feat_imp_c = dict(zip(features, model_c.feature_importances_))

    # Base learner AUC on test set
    treated_test = df_test[df_test['treatment'] == 1]
    control_test = df_test[df_test['treatment'] == 0]
    auc_t = roc_auc_score(treated_test['y_obs'], model_t.predict_proba(treated_test[features])[:, 1])
    auc_c = roc_auc_score(control_test['y_obs'], model_c.predict_proba(control_test[features])[:, 1])

    run_artifacts = {
        'df_test': df_test,
        'uplift_scores': uplift_scores,
        'random_scores': random_scores,
        'feat_imp_t': feat_imp_t,
        'feat_imp_c': feat_imp_c,
        'auc_t': auc_t,
        'auc_c': auc_c,
    }

    return results, gain_data, run_artifacts


def compute_cumulative_gains(df_test, uplift_scores):
    """Compute cumulative gain curve data by decile."""
    df_eval = df_test.copy()
    df_eval['uplift_score'] = uplift_scores
    df_eval['decile'] = pd.qcut(df_eval['uplift_score'], 10, labels=False, duplicates='drop')

    gains = []
    n_total = len(df_eval)

    for d in range(10):
        top_deciles = df_eval[df_eval['decile'] >= (9 - d)]
        p = len(top_deciles) / n_total

        # True incremental conversions in this group
        inc = top_deciles['y1'].sum() - top_deciles['y0'].sum()
        # ATE for this group
        ate = (top_deciles['y1'].mean() - top_deciles['y0'].mean()) * 100

        gains.append({
            'decile': d + 1,
            'proportion': round(p, 2),
            'cumulative_incremental': inc,
            'decile_ate_pp': round(ate, 1),
        })

    return gains


def bootstrap_ci(data, statistic_func, n_bootstrap=BOOTSTRAP_SAMPLES, alpha=ALPHA):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(42)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic_func(sample))
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lower, upper


def permutation_test(group_a, group_b, n_perms=5000):
    """
    Permutation test for difference in means.
    H0: group_a and group_b come from same distribution.
    """
    rng = np.random.RandomState(42)
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(n_perms):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / n_perms
    return observed_diff, p_value


def run_threshold_sensitivity(all_run_data):
    """Test sensitivity of Full Framework to churn threshold choices."""
    configs = {
        'Default (0.05/0.40/0.60)': dict(low=0.05, mid_low=0.40, mid_high=0.60),
        'Narrow (0.10/0.30/0.50)': dict(low=0.10, mid_low=0.30, mid_high=0.50),
        'Wide (0.03/0.50/0.70)': dict(low=0.03, mid_low=0.50, mid_high=0.70),
        'Shifted low (0.05/0.25/0.45)': dict(low=0.05, mid_low=0.25, mid_high=0.45),
        'Shifted high (0.10/0.55/0.75)': dict(low=0.10, mid_low=0.55, mid_high=0.75),
        'Lower ceiling (high=0.80)': dict(low=0.05, mid_low=0.40, mid_high=0.60, high=0.80),
        'No floor (floor=0.0)': dict(low=0.05, mid_low=0.40, mid_high=0.60, floor=0.0),
        'No regularization (f=1)': None,
    }

    results = {}

    for config_name, params in configs.items():
        ates = []
        revs = []
        pers = []
        for rd in all_run_data:
            df_test = rd['df_test']
            uplift_scores = rd['uplift_scores']

            if params is None:
                f_scores = np.ones(len(df_test))
            else:
                f_scores = np.array([
                    f_churn_parameterized(p, **params)
                    for p in df_test['churn_prob'].values
                ])

            full_scores = uplift_scores * df_test['ltv'].values * f_scores
            r = evaluate_strategy(df_test, full_scores, BUDGET_FRACTION, config_name)
            ates.append(r['ate_pp'])
            revs.append(r['incremental_revenue'])
            pers.append(r['persuadable_frac'])

        results[config_name] = {
            'ate_mean': round(np.mean(ates), 1),
            'ate_sd': round(np.std(ates), 1),
            'rev_mean': round(np.mean(revs), 0),
            'rev_sd': round(np.std(revs), 0),
            'persuadable_pct': round(np.mean(pers) * 100, 1),
        }

    return results


def run_misspecification_test(all_run_data, n_runs=20):
    """
    Stress test: train T-Learner with only 2 of 4 features (account_age, weekly_spend).
    This simulates a misspecified model where key predictive features are unavailable.
    Uses the same generated data as main runs for comparability.
    """
    reduced_features = ['account_age', 'weekly_spend']
    results_full = []
    results_reduced = []

    for run in range(min(n_runs, len(all_run_data))):
        seed = run * 7 + 13
        df = generate_population(N_PLAYERS, seed=seed)
        df_train, df_test = train_test_split(df, test_size=0.30, random_state=seed)

        # Full feature set (recomputed from same seed for consistency)
        model_t, model_c, features = train_t_learner(df_train)
        uplift_full = score_uplift(model_t, model_c, df_test, features)
        r_full = evaluate_strategy(df_test, uplift_full, BUDGET_FRACTION, 'Full Features')

        # Reduced feature set
        treated = df_train[df_train['treatment'] == 1]
        control = df_train[df_train['treatment'] == 0]
        model_t_r = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model_c_r = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model_t_r.fit(treated[reduced_features], treated['y_obs'])
        model_c_r.fit(control[reduced_features], control['y_obs'])
        uplift_reduced = score_uplift(model_t_r, model_c_r, df_test, reduced_features)
        r_reduced = evaluate_strategy(df_test, uplift_reduced, BUDGET_FRACTION, 'Reduced Features')

        results_full.append(r_full)
        results_reduced.append(r_reduced)

    return {
        'full_features': {
            'ate_mean': round(np.mean([r['ate_pp'] for r in results_full]), 1),
            'rev_mean': round(np.mean([r['incremental_revenue'] for r in results_full]), 0),
        },
        'reduced_features': {
            'ate_mean': round(np.mean([r['ate_pp'] for r in results_reduced]), 1),
            'rev_mean': round(np.mean([r['incremental_revenue'] for r in results_reduced]), 0),
            'features_used': reduced_features,
        },
        'n_runs': min(n_runs, len(all_run_data)),
        'note': (
            'Stress test with 2 of 4 features removed (login_freq, bonus_redeem dropped). '
            'Uses same seeds as main simulation runs for comparability. '
            'Shows degradation when the model is misspecified.'
        ),
    }


def expected_segment_mean_ate_pp():
    """Return the population ATE implied by the mean segment effects."""
    mean_ate = sum(prevalence * tau for prevalence, _, tau in SEGMENTS.values())
    return round(mean_ate * 100, 2)


def main():
    print("=" * 70)
    print("SIMULATION ANALYSIS: Uplift-Based Bonus Allocation Framework")
    print("=" * 70)
    print(f"\nConfiguration: N={N_PLAYERS}, Runs={N_RUNS}, Budget={BUDGET_FRACTION*100:.0f}%")
    print(f"Bootstrap samples={BOOTSTRAP_SAMPLES}, alpha={ALPHA}")

    # Run simulations - store artifacts for budget sensitivity reuse
    all_results = {s: [] for s in STRATEGIES}
    all_gains = []
    all_run_data = []
    all_seg_shares = {}

    for run in range(N_RUNS):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        df_run = generate_population(N_PLAYERS, seed=run * 7 + 13)
        for seg in SEGMENTS:
            all_seg_shares.setdefault(seg, []).append((df_run['segment'] == seg).mean())
        results, gains, run_artifacts = run_single_simulation(seed=run * 7 + 13)
        for r in results:
            all_results[r['strategy']].append(r)
        all_gains.append(gains)
        all_run_data.append(run_artifacts)

    # --- Main Results ---
    print("\n" + "=" * 70)
    print("TABLE: Simulation Results by Targeting Strategy")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Inc. Revenue':<20} {'ATE (pp)':<18} {'Inc. Conv.':<18} {'Persuadable %':<15}")
    print("-" * 91)

    summary = {}
    for strategy in STRATEGIES:
        ates = np.array([r['ate_pp'] for r in all_results[strategy]])
        incs = np.array([r['incremental_conversions'] for r in all_results[strategy]])
        revs = np.array([r['incremental_revenue'] for r in all_results[strategy]])
        pers = np.array([r['persuadable_frac'] for r in all_results[strategy]])

        ate_ci = bootstrap_ci(ates, np.mean)
        inc_ci = bootstrap_ci(incs, np.mean)
        rev_ci = bootstrap_ci(revs, np.mean)

        summary[strategy] = {
            'ate_mean': np.mean(ates),
            'ate_sd': np.std(ates),
            'ate_ci': ate_ci,
            'inc_mean': np.mean(incs),
            'inc_sd': np.std(incs),
            'inc_ci': inc_ci,
            'rev_mean': np.mean(revs),
            'rev_sd': np.std(revs),
            'rev_ci': rev_ci,
            'pers_mean': np.mean(pers),
            'ates_raw': ates,
            'revs_raw': revs,
        }

        print(f"{strategy:<20} "
              f"${np.mean(revs):>9,.0f} ({np.std(revs):,.0f})  "
              f"{np.mean(ates):>5.1f} ({np.std(ates):.1f})   "
              f"{np.mean(incs):>6.0f} ({np.std(incs):.0f})    "
              f"{np.mean(pers)*100:>5.1f}%")

    # --- Bootstrap CIs ---
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 70)
    for strategy in STRATEGIES:
        s = summary[strategy]
        print(f"\n{strategy}:")
        print(f"  ATE:          {s['ate_mean']:>6.1f} pp  [{s['ate_ci'][0]:.1f}, {s['ate_ci'][1]:.1f}]")
        print(f"  Inc. Conv.:   {s['inc_mean']:>6.0f}      [{s['inc_ci'][0]:.0f}, {s['inc_ci'][1]:.0f}]")
        print(f"  Inc. Revenue: ${s['rev_mean']:>9,.0f}  [${s['rev_ci'][0]:,.0f}, ${s['rev_ci'][1]:,.0f}]")

    # --- Permutation Tests ---
    print("\n" + "=" * 70)
    print("PERMUTATION TESTS (one-sided: strategy > random)")
    print("=" * 70)

    perm_results = {}
    for strategy in [s for s in STRATEGIES if s != 'Random']:
        diff, p_val = permutation_test(
            summary[strategy]['ates_raw'],
            summary['Random']['ates_raw']
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {strategy} vs Random (ATE): diff={diff:+.1f}pp, p={p_val:.4f} {sig}")
        perm_results[f'{strategy}_vs_Random_ATE'] = {
            'diff': round(diff, 1),
            'p_value': round(p_val, 4),
        }

    # Revenue permutation tests
    for strategy in ['Uplift Only', 'Full Framework']:
        comparator = 'Uplift Only' if strategy == 'Full Framework' else 'Random'
        diff, p_val = permutation_test(
            summary[strategy]['revs_raw'],
            summary[comparator]['revs_raw']
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {strategy} vs {comparator} (Revenue): diff=${diff:+,.0f}, p={p_val:.4f} {sig}")
        perm_results[f'{strategy}_vs_{comparator}_Revenue'] = {
            'diff': round(diff, 0),
            'p_value': round(p_val, 4),
        }

    # Key ablation: does churn regularization add value over Uplift x LTV?
    diff, p_val = permutation_test(
        summary['Full Framework']['revs_raw'],
        summary['Uplift x LTV']['revs_raw']
    )
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"  Full Framework vs Uplift x LTV (Revenue): diff=${diff:+,.0f}, p={p_val:.4f} {sig}")
    perm_results['Full Framework_vs_Uplift x LTV_Revenue'] = {
        'diff': round(diff, 0),
        'p_value': round(p_val, 4),
    }

    # Two-sided test for Response Model (is it significantly different from random?)
    resp_ates = summary['Response Model']['ates_raw']
    rand_ates = summary['Random']['ates_raw']
    diff_resp = np.mean(resp_ates) - np.mean(rand_ates)
    combined = np.concatenate([resp_ates, rand_ates])
    rng_perm = np.random.RandomState(42)
    count_extreme_2s = 0
    for _ in range(5000):
        rng_perm.shuffle(combined)
        perm_diff = np.mean(combined[:len(resp_ates)]) - np.mean(combined[len(resp_ates):])
        if abs(perm_diff) >= abs(diff_resp):
            count_extreme_2s += 1
    p_2sided = count_extreme_2s / 5000
    print(f"  Response Model vs Random (two-sided): diff={diff_resp:+.1f}pp, p={p_2sided:.4f}")
    perm_results['Response Model_vs_Random_ATE_twosided'] = {
        'diff': round(diff_resp, 1),
        'p_value': round(p_2sided, 4),
    }

    # --- Cumulative Gain Curve (averaged) ---
    print("\n" + "=" * 70)
    print("CUMULATIVE GAIN CURVE (averaged over runs)")
    print("=" * 70)
    print(f"{'Decile':<10} {'Proportion':<15} {'Cum. Inc. Conv':<20} {'Decile ATE (pp)':<15}")
    print("-" * 60)

    for d in range(10):
        cum_incs = [run_gains[d]['cumulative_incremental'] for run_gains in all_gains]
        decile_ates = [run_gains[d]['decile_ate_pp'] for run_gains in all_gains]
        prop = all_gains[0][d]['proportion']
        print(f"{d+1:<10} {prop:<15.2f} {np.mean(cum_incs):<20.0f} {np.mean(decile_ates):<15.1f}")

    # --- Budget Sensitivity (reusing same run data for consistency) ---
    print("\n" + "=" * 70)
    print("SENSITIVITY: Varying Budget Constraint")
    print("=" * 70)

    budget_sensitivity = {}
    for budget in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]:
        ates_uplift = []
        ates_random = []
        for rd in all_run_data:
            r_uplift = evaluate_strategy(rd['df_test'], rd['uplift_scores'], budget, 'Uplift')
            r_random = evaluate_strategy(rd['df_test'], rd['random_scores'], budget, 'Random')
            ates_uplift.append(r_uplift['ate_pp'])
            ates_random.append(r_random['ate_pp'])

        uplift_mean = round(np.mean(ates_uplift), 1)
        random_mean = round(np.mean(ates_random), 1)
        if random_mean > 0:
            improvement_pct = round((uplift_mean - random_mean) / random_mean * 100)
        else:
            improvement_pct = 0
        budget_sensitivity[f'{int(budget*100)}%'] = {
            'uplift_ate_mean': uplift_mean,
            'random_ate_mean': random_mean,
            'improvement_pct': improvement_pct,
        }
        print(f"  Budget={budget*100:.0f}%: Uplift ATE={uplift_mean}pp, "
              f"Random ATE={random_mean}pp, Improvement=+{improvement_pct}%")

    # --- Threshold Sensitivity ---
    print("\n" + "=" * 70)
    print("SENSITIVITY: Churn Threshold Parameters")
    print("=" * 70)

    threshold_results = run_threshold_sensitivity(all_run_data)
    for config, res in threshold_results.items():
        print(f"  {config}: ATE={res['ate_mean']:.1f}pp (SD={res['ate_sd']:.1f}), "
              f"Revenue=${res['rev_mean']:,.0f}, Persuadable={res['persuadable_pct']:.1f}%")

    # --- RG exclusion stats ---
    avg_rg_excluded = np.mean([r['n_rg_excluded'] for r in all_results['Uplift Only']])
    avg_test_size = np.mean([len(rd['df_test']) for rd in all_run_data])
    avg_rg_excluded_pct_test = (avg_rg_excluded / avg_test_size) * 100
    print(
        f"\nResponsible gaming: avg {avg_rg_excluded:.0f} players excluded per run "
        f"({avg_rg_excluded_pct_test:.2f}% of the held-out sample)"
    )

    # Save summary for paper generation
    output = {
        'main_results': {},
        'cumulative_gains': [],
    }
    for strategy in STRATEGIES:
        s = summary[strategy]
        output['main_results'][strategy] = {
            'ate_mean': round(s['ate_mean'], 1),
            'ate_sd': round(s['ate_sd'], 1),
            'ate_ci_lower': round(s['ate_ci'][0], 1),
            'ate_ci_upper': round(s['ate_ci'][1], 1),
            'inc_mean': round(s['inc_mean'], 0),
            'inc_sd': round(s['inc_sd'], 0),
            'inc_ci_lower': round(s['inc_ci'][0], 0),
            'inc_ci_upper': round(s['inc_ci'][1], 0),
            'rev_mean': round(s['rev_mean'], 0),
            'rev_sd': round(s['rev_sd'], 0),
            'rev_ci_lower': round(s['rev_ci'][0], 0),
            'rev_ci_upper': round(s['rev_ci'][1], 0),
            'pers_mean': round(s['pers_mean'] * 100, 1),
            'per_run_se': round(s['ate_sd'] / np.sqrt(N_RUNS), 2),
        }

    for d in range(10):
        cum_incs = [run_gains[d]['cumulative_incremental'] for run_gains in all_gains]
        decile_ates = [run_gains[d]['decile_ate_pp'] for run_gains in all_gains]
        output['cumulative_gains'].append({
            'decile': d + 1,
            'proportion': all_gains[0][d]['proportion'],
            'cum_inc_mean': round(np.mean(cum_incs), 0),
            'decile_ate_mean': round(np.mean(decile_ates), 1),
        })

    output['threshold_sensitivity'] = threshold_results
    output['budget_sensitivity'] = budget_sensitivity
    output['permutation_tests'] = perm_results
    output['responsible_gaming'] = {
        'avg_excluded_per_run': round(avg_rg_excluded, 0),
        'avg_excluded_pct_test': round(avg_rg_excluded_pct_test, 2),
        'exclusion_criteria': 'Players with weekly_spend > 90th percentile AND login_freq > 90th percentile',
    }
    output['simulation_design'] = {
        'segment_parameters': {
            name: {
                'prevalence': prevalence,
                'baseline_prob': baseline,
                'mean_treatment_effect': tau,
            }
            for name, (prevalence, baseline, tau) in SEGMENTS.items()
        },
        'actual_segment_shares': {
            seg: round(float(np.mean(shares)), 4)
            for seg, shares in all_seg_shares.items()
        },
        'expected_segment_mean_ate_pp': expected_segment_mean_ate_pp(),
        'note': (
            'This DGP is a favourable proof-of-concept design rather than a direct '
            'computational instantiation of the pedagogical four-segment table. '
            'Sure Things and Lost Causes have small positive mean effects, so the '
            'population ATE is positive across most score deciles.'
        ),
    }
    output['methodology_notes'] = {
        'bootstrap_method': 'CIs computed by bootstrapping over run-level means. This captures simulation-to-simulation variability.',
        'n_runs': N_RUNS,
        'n_players_per_run': N_PLAYERS,
        'n_bootstrap': BOOTSTRAP_SAMPLES,
        'alpha': ALPHA,
        'budget_sensitivity_note': 'Budget sensitivity uses same run data as main results for consistency.',
        'primary_metric': 'incremental_revenue',
        'primary_metric_note': (
            'The optimization objective is incremental revenue (uplift x LTV). '
            'ATE is reported as a secondary diagnostic. Churn regularization '
            'trades revenue for risk reduction; this trade-off is shown in '
            'threshold_sensitivity results.'
        ),
    }

    # Feature importance (averaged over runs)
    avg_feat_imp_t = {}
    avg_feat_imp_c = {}
    features = ['account_age', 'weekly_spend', 'login_freq', 'bonus_redeem']
    for f in features:
        avg_feat_imp_t[f] = round(np.mean([rd['feat_imp_t'][f] for rd in all_run_data]), 3)
        avg_feat_imp_c[f] = round(np.mean([rd['feat_imp_c'][f] for rd in all_run_data]), 3)
    output['feature_importance'] = {
        'treatment_model': avg_feat_imp_t,
        'control_model': avg_feat_imp_c,
    }

    # Base learner diagnostics
    avg_auc_t = round(np.mean([rd['auc_t'] for rd in all_run_data]), 3)
    avg_auc_c = round(np.mean([rd['auc_c'] for rd in all_run_data]), 3)
    output['base_learner_diagnostics'] = {
        'treatment_model_auc': avg_auc_t,
        'control_model_auc': avg_auc_c,
    }

    # Uplift calibration (predicted vs observed by decile, averaged over runs)
    calibration_data = []
    for d in range(10):
        pred_uplifts = []
        obs_uplifts = []
        for rd in all_run_data:
            df_t = rd['df_test'].copy()
            df_t['pred_uplift'] = rd['uplift_scores']
            df_t['decile'] = pd.qcut(df_t['pred_uplift'], 10, labels=False, duplicates='drop')
            dec = df_t[df_t['decile'] == d]
            if len(dec) > 0:
                pred_uplifts.append(dec['pred_uplift'].mean())
                obs_uplifts.append(dec['true_tau'].mean())
        calibration_data.append({
            'decile': d + 1,
            'mean_predicted_uplift': round(np.mean(pred_uplifts), 4),
            'mean_observed_uplift': round(np.mean(obs_uplifts), 4),
        })
    output['uplift_calibration'] = calibration_data

    # --- Misspecification Stress Test ---
    print("\n" + "=" * 70)
    print("STRESS TEST: Model Misspecification")
    print("=" * 70)
    misspec = run_misspecification_test(all_run_data, n_runs=20)
    print(f"  Full features:    ATE={misspec['full_features']['ate_mean']}pp, "
          f"Revenue=${misspec['full_features']['rev_mean']:,.0f}")
    print(f"  Reduced features: ATE={misspec['reduced_features']['ate_mean']}pp, "
          f"Revenue=${misspec['reduced_features']['rev_mean']:,.0f}")

    output['misspecification_test'] = misspec

    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to {output_path}")
    print("Done.")


if __name__ == '__main__':
    main()
