"""
Enhanced simulation analysis for the bonus allocation paper.
Generates synthetic data with known treatment effects, trains T-Learner,
and compares targeting strategies with bootstrap CIs and hypothesis tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
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

    Key design: features are PREDICTIVE of segment membership, so the
    T-Learner can learn to identify persuadable players from observables.
    Segment assignment uses a softmax over feature-based propensity scores.
    """
    rng = np.random.RandomState(seed)

    # Step 1: Generate raw features
    account_age = rng.uniform(1, 36, n)
    weekly_spend = rng.lognormal(mean=3.9, sigma=0.8, size=n)
    login_freq = rng.poisson(lam=4, size=n).astype(float)
    bonus_redeem = rng.beta(2, 5, size=n)

    # Step 2: Compute segment propensity scores from features
    # Each segment has a different feature profile
    z_spend = (np.log1p(weekly_spend) - 3.9) / 0.8
    z_login = (login_freq - 4.0) / 2.0
    z_redeem = (bonus_redeem - 0.28) / 0.15
    z_age = (account_age - 18) / 10.0

    # Sure Things: high login, high spend, high engagement
    logit_st = 0.8 + 1.2 * z_login + 0.8 * z_spend + 0.3 * z_age
    # Persuadables: moderate engagement, higher bonus redemption, some recency
    logit_p = -0.8 + 1.5 * z_redeem + 0.5 * z_login - 0.3 * z_spend
    # Lost Causes: low engagement, low spend, low redemption
    logit_lc = 0.3 - 1.0 * z_login - 0.8 * z_spend - 0.5 * z_redeem
    # Sleeping Dogs: high spend but low redemption, longer tenure
    logit_sd = -1.2 + 0.6 * z_spend - 1.0 * z_redeem + 0.4 * z_age

    # Softmax to get segment probabilities
    logits = np.column_stack([logit_st, logit_p, logit_lc, logit_sd])
    # Apply target prevalence weighting
    prevalence_weights = np.log([0.40, 0.15, 0.35, 0.10])
    logits += prevalence_weights
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    segment_names = ['Sure Things', 'Persuadables', 'Lost Causes', 'Sleeping Dogs']
    segment_idx = np.array([rng.choice(4, p=probs[i]) for i in range(n)])
    segments = [segment_names[idx] for idx in segment_idx]

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
    seg_params = {
        'Sure Things':   (0.85, +0.03),
        'Persuadables':  (0.40, +0.35),
        'Lost Causes':   (0.15, +0.05),
        'Sleeping Dogs': (0.70, -0.15),
    }

    baselines = np.array([seg_params[s][0] for s in segments])
    taus = np.array([seg_params[s][1] for s in segments])

    # Add some within-segment heterogeneity
    tau_noise = rng.normal(0, 0.03, n)
    effective_tau = taus + tau_noise

    # Baseline also varies slightly with features
    y0_prob = np.clip(baselines + 0.05 * z_login * 0.1, 0.01, 0.99)
    y1_prob = np.clip(y0_prob + effective_tau, 0.01, 0.99)

    y0 = rng.binomial(1, y0_prob)
    y1 = rng.binomial(1, y1_prob)
    y_obs = treatment * y1 + (1 - treatment) * y0

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


def evaluate_strategy(df_test, scores, budget_frac, strategy_name):
    """
    Evaluate a targeting strategy using true potential outcomes.
    Returns incremental conversions and ATE for treated group.
    """
    n_treat = int(len(df_test) * budget_frac)
    top_idx = np.argsort(-scores)[:n_treat]

    # Using true potential outcomes (available in simulation)
    targeted = df_test.iloc[top_idx]
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

    # Random scores
    rng = np.random.RandomState(seed)
    random_scores = rng.uniform(0, 1, len(df_test))

    # Evaluate all strategies
    results = []
    results.append(evaluate_strategy(df_test, random_scores, BUDGET_FRACTION, 'Random'))
    results.append(evaluate_strategy(df_test, response_scores, BUDGET_FRACTION, 'Response Model'))
    results.append(evaluate_strategy(df_test, uplift_scores, BUDGET_FRACTION, 'Uplift Only'))
    results.append(evaluate_strategy(df_test, full_scores, BUDGET_FRACTION, 'Full Framework'))

    # Also compute cumulative gain curve for uplift model
    gain_data = compute_cumulative_gains(df_test, uplift_scores)

    return results, gain_data


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


def main():
    print("=" * 70)
    print("SIMULATION ANALYSIS: Uplift-Based Bonus Allocation Framework")
    print("=" * 70)
    print(f"\nConfiguration: N={N_PLAYERS}, Runs={N_RUNS}, Budget={BUDGET_FRACTION*100:.0f}%")
    print(f"Bootstrap samples={BOOTSTRAP_SAMPLES}, alpha={ALPHA}")

    # Run simulations
    all_results = {s: [] for s in ['Random', 'Response Model', 'Uplift Only', 'Full Framework']}
    all_gains = []

    for run in range(N_RUNS):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{N_RUNS}...")
        results, gains = run_single_simulation(seed=run * 7 + 13)
        for r in results:
            all_results[r['strategy']].append(r)
        all_gains.append(gains)

    # --- Table 8: Main Results ---
    print("\n" + "=" * 70)
    print("TABLE: Simulation Results by Targeting Strategy")
    print("=" * 70)
    print(f"{'Strategy':<20} {'ATE (pp)':<18} {'Inc. Conv.':<18} {'Inc. Revenue':<20} {'Persuadable %':<15}")
    print("-" * 91)

    summary = {}
    for strategy in ['Random', 'Response Model', 'Uplift Only', 'Full Framework']:
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
              f"{np.mean(ates):>5.1f} ({np.std(ates):.1f})   "
              f"{np.mean(incs):>6.0f} ({np.std(incs):.0f})    "
              f"${np.mean(revs):>9,.0f} ({np.std(revs):,.0f})  "
              f"{np.mean(pers)*100:>5.1f}%")

    # --- Bootstrap CIs ---
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 70)
    for strategy in ['Random', 'Response Model', 'Uplift Only', 'Full Framework']:
        s = summary[strategy]
        print(f"\n{strategy}:")
        print(f"  ATE:          {s['ate_mean']:>6.1f} pp  [{s['ate_ci'][0]:.1f}, {s['ate_ci'][1]:.1f}]")
        print(f"  Inc. Conv.:   {s['inc_mean']:>6.0f}      [{s['inc_ci'][0]:.0f}, {s['inc_ci'][1]:.0f}]")
        print(f"  Inc. Revenue: ${s['rev_mean']:>9,.0f}  [${s['rev_ci'][0]:,.0f}, ${s['rev_ci'][1]:,.0f}]")

    # --- Permutation Tests ---
    print("\n" + "=" * 70)
    print("PERMUTATION TESTS (one-sided: strategy > random)")
    print("=" * 70)

    for strategy in ['Response Model', 'Uplift Only', 'Full Framework']:
        diff, p_val = permutation_test(
            summary[strategy]['ates_raw'],
            summary['Random']['ates_raw']
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {strategy} vs Random (ATE): diff={diff:+.1f}pp, p={p_val:.4f} {sig}")

    # Revenue permutation tests
    for strategy in ['Uplift Only', 'Full Framework']:
        diff, p_val = permutation_test(
            summary[strategy]['revs_raw'],
            summary['Uplift Only' if strategy == 'Full Framework' else 'Random']['revs_raw']
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        comparator = 'Uplift Only' if strategy == 'Full Framework' else 'Random'
        print(f"  {strategy} vs {comparator} (Revenue): diff=${diff:+,.0f}, p={p_val:.4f} {sig}")

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

    # --- Sensitivity: Budget fraction ---
    print("\n" + "=" * 70)
    print("SENSITIVITY: Varying Budget Constraint")
    print("=" * 70)

    for budget in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]:
        ates_uplift = []
        ates_random = []
        for run in range(10):
            df = generate_population(N_PLAYERS, seed=run * 7 + 200)
            df_train, df_test = train_test_split(df, test_size=0.30, random_state=run)
            model_t, model_c, features = train_t_learner(df_train)
            uplift_scores = score_uplift(model_t, model_c, df_test, features)
            rng = np.random.RandomState(run)
            random_scores = rng.uniform(0, 1, len(df_test))

            r_uplift = evaluate_strategy(df_test, uplift_scores, budget, 'Uplift')
            r_random = evaluate_strategy(df_test, random_scores, budget, 'Random')
            ates_uplift.append(r_uplift['ate_pp'])
            ates_random.append(r_random['ate_pp'])

        advantage = np.mean(ates_uplift) - np.mean(ates_random)
        print(f"  Budget={budget*100:.0f}%: Uplift ATE={np.mean(ates_uplift):.1f}pp, "
              f"Random ATE={np.mean(ates_random):.1f}pp, Advantage={advantage:+.1f}pp")

    # Save summary for paper generation
    output = {
        'main_results': {},
        'cumulative_gains': [],
    }
    for strategy in ['Random', 'Response Model', 'Uplift Only', 'Full Framework']:
        s = summary[strategy]
        output['main_results'][strategy] = {
            'ate_mean': round(s['ate_mean'], 1),
            'ate_sd': round(s['ate_sd'], 1),
            'ate_ci_lower': round(s['ate_ci'][0], 1),
            'ate_ci_upper': round(s['ate_ci'][1], 1),
            'inc_mean': round(s['inc_mean'], 0),
            'inc_sd': round(s['inc_sd'], 0),
            'rev_mean': round(s['rev_mean'], 0),
            'rev_sd': round(s['rev_sd'], 0),
            'rev_ci_lower': round(s['rev_ci'][0], 0),
            'rev_ci_upper': round(s['rev_ci'][1], 0),
            'pers_mean': round(s['pers_mean'] * 100, 1),
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

    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n\nResults saved to simulation_results.json")
    print("Done.")


if __name__ == '__main__':
    main()
