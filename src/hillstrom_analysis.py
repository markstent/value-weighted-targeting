"""
Hillstrom Email Dataset Analysis for Appendix A.
Downloads the public dataset, applies T-Learner uplift framework,
and generates cumulative gain curves and AUUC metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import hashlib
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
BOOTSTRAP_SAMPLES = 1000
ALPHA = 0.05
REPEATED_SPLIT_RUNS = 50
HILLSTROM_SHA256 = "00a6a868e05a9ffe7382da51629f6d6dce88c5acfc945e79d314ebc78fd3a2c0"


def load_hillstrom_data():
    """
    Load Hillstrom email marketing dataset.
    Source: Kevin Hillstrom's MineThatData blog.
    The dataset contains ~64k customers from an email marketing campaign.
    """
    csv_path = os.path.join(DATA_DIR, 'data', 'hillstrom.csv')

    if not os.path.exists(csv_path):
        print("Downloading Hillstrom dataset...")
        url = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        try:
            df = pd.read_csv(url)
            df.to_csv(csv_path, index=False)
            print(f"Downloaded {len(df)} records")
        except Exception as e:
            raise RuntimeError(
                f"Could not download Hillstrom dataset. "
                f"Please manually download from MineThatData blog and save as {csv_path}"
            ) from e
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {csv_path}")

    # Verify checksum
    with open(csv_path, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != HILLSTROM_SHA256:
        print(f"WARNING: Hillstrom CSV checksum mismatch. "
              f"Expected {HILLSTROM_SHA256[:16]}..., got {actual[:16]}... "
              f"Results may differ from published values.")

    return df


def prepare_data(df):
    """
    Prepare Hillstrom data for uplift analysis.
    Focus on email vs no-email comparison using 'visit' as outcome.
    """
    # The dataset has three segments: Mens Email, Womens Email, No Email
    # For uplift analysis: combine email segments as treatment, No Email as control
    df = df.copy()

    # Create binary treatment: any email vs no email
    df['treatment'] = (df['segment'] != 'No E-Mail').astype(int)

    # Outcome: 'visit' (binary) - did the customer visit the website?
    # Alternative outcomes: 'conversion', 'spend'
    df['outcome_visit'] = df['visit'].astype(int)
    df['outcome_conversion'] = df['conversion'].astype(int)
    df['outcome_spend'] = df['spend'].astype(float)

    # Features (all pre-treatment)
    df['recency_log'] = np.log1p(df['recency'])
    df['is_mens'] = (df['mens'] == 1).astype(int)
    df['is_womens'] = (df['womens'] == 1).astype(int)
    df['channel_phone'] = (df['channel'] == 'Phone').astype(int)
    df['channel_web'] = (df['channel'] == 'Web').astype(int)
    df['channel_multi'] = (df['channel'] == 'Multichannel').astype(int)
    df['is_newbie'] = (df['newbie'] == 1).astype(int)

    feature_cols = [
        'recency', 'recency_log', 'history', 'is_mens', 'is_womens',
        'channel_phone', 'channel_web', 'channel_multi', 'is_newbie'
    ]

    return df, feature_cols


def train_t_learner(df_train, feature_cols, outcome_col='outcome_visit'):
    """Train T-Learner on Hillstrom data."""
    treated = df_train[df_train['treatment'] == 1]
    control = df_train[df_train['treatment'] == 0]

    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=50, random_state=42
    )
    model_c = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=50, random_state=42
    )

    model_t.fit(treated[feature_cols], treated[outcome_col])
    model_c.fit(control[feature_cols], control[outcome_col])

    return model_t, model_c


def train_s_learner(df_train, feature_cols, outcome_col='outcome_visit'):
    """Train S-Learner: single model with treatment as an additional feature."""
    s_features = feature_cols + ['treatment']
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=50, random_state=42
    )
    model.fit(df_train[s_features], df_train[outcome_col])
    return model


def score_s_learner(model, df, feature_cols):
    """Score S-Learner uplift: P(Y|X, T=1) - P(Y|X, T=0)."""
    s_features = feature_cols + ['treatment']
    df_t1 = df[feature_cols].copy()
    df_t1['treatment'] = 1
    df_t0 = df[feature_cols].copy()
    df_t0['treatment'] = 0
    pred_t1 = model.predict_proba(df_t1[s_features])[:, 1]
    pred_t0 = model.predict_proba(df_t0[s_features])[:, 1]
    return pred_t1 - pred_t0


def score_uplift(model_t, model_c, df, feature_cols):
    """Compute uplift scores."""
    X = df[feature_cols]
    pred_t = model_t.predict_proba(X)[:, 1]
    pred_c = model_c.predict_proba(X)[:, 1]
    return pred_t - pred_c


def compute_qini_curve(df_test, scores, outcome_col='outcome_visit'):
    """
    Compute cumulative uplift curve and AUUC.

    Uses the standard cumulative gain formulation:
    CumulativeGain(p) = N*p * [Y_bar_t(top_p) - Y_bar_c(top_p)]

    AUUC is the area between the model curve and the random diagonal,
    computed via trapezoidal integration. This is NOT normalized by
    the perfect model area - it is an absolute measure in units of
    cumulative incremental conversions.
    """
    df_eval = df_test.copy()
    df_eval['score'] = scores
    df_eval = df_eval.sort_values('score', ascending=False)

    n_total = len(df_eval)
    n_deciles = 10
    decile_size = n_total // n_deciles

    # Overall ATE for random baseline
    all_treated = df_eval[df_eval['treatment'] == 1]
    all_control = df_eval[df_eval['treatment'] == 0]
    overall_ate = all_treated[outcome_col].mean() - all_control[outcome_col].mean()

    curve_points = []

    for d in range(1, n_deciles + 1):
        top_n = d * decile_size
        top = df_eval.iloc[:top_n]

        treated = top[top['treatment'] == 1]
        control = top[top['treatment'] == 0]

        n_t = len(treated)
        n_c = len(control)

        if n_t > 0 and n_c > 0:
            y_bar_t = treated[outcome_col].mean()
            y_bar_c = control[outcome_col].mean()
            observed_uplift = y_bar_t - y_bar_c

            # Cumulative incremental conversions
            cum_gain = top_n * observed_uplift
        else:
            observed_uplift = 0
            cum_gain = 0

        proportion = d / n_deciles

        # Random baseline: cumulative gain if targeting randomly
        random_gain = top_n * overall_ate

        curve_points.append({
            'decile': d,
            'proportion': proportion,
            'n_targeted': top_n,
            'n_treated': n_t,
            'n_control': n_c,
            'observed_uplift_pp': round(observed_uplift * 100, 2),
            'cumulative_gain': round(cum_gain, 1),
            'random_gain': round(random_gain, 1),
            'gain_over_random': round(cum_gain - random_gain, 1),
        })

    # AUUC: absolute area between model curve and random diagonal.
    # Using trapezoidal integration.
    model_area = 0
    random_area = 0
    for i in range(len(curve_points)):
        if i == 0:
            model_area += curve_points[i]['cumulative_gain'] * (1 / n_deciles) / 2
            random_area += curve_points[i]['random_gain'] * (1 / n_deciles) / 2
        else:
            model_area += (curve_points[i]['cumulative_gain'] + curve_points[i-1]['cumulative_gain']) * (1 / n_deciles) / 2
            random_area += (curve_points[i]['random_gain'] + curve_points[i-1]['random_gain']) * (1 / n_deciles) / 2

    auuc = model_area - random_area

    return curve_points, auuc, overall_ate


def summarize_repeated_splits(df, feature_cols, outcome_col='outcome_visit', n_runs=REPEATED_SPLIT_RUNS):
    """Estimate split sensitivity for AUUC and top-decile uplift."""
    auucs = []
    s_learner_auucs = []
    top_deciles = []
    strategy_uplifts = []

    for run in range(n_runs):
        df_train, df_test = train_test_split(
            df,
            test_size=0.30,
            random_state=42 + run,
            stratify=df['treatment'],
        )
        model_t, model_c = train_t_learner(df_train, feature_cols, outcome_col)
        uplift_scores = score_uplift(model_t, model_c, df_test, feature_cols)
        qini_points, auuc, _ = compute_qini_curve(df_test, uplift_scores, outcome_col)

        # S-Learner
        s_model = train_s_learner(df_train, feature_cols, outcome_col)
        s_scores = score_s_learner(s_model, df_test, feature_cols)
        _, s_auuc, _ = compute_qini_curve(df_test, s_scores, outcome_col)
        s_learner_auucs.append(s_auuc)

        strategies = compare_strategies(
            df_train, df_test, uplift_scores, feature_cols, outcome_col,
            s_learner_scores=s_scores,
        )

        auucs.append(auuc)
        top_deciles.append(qini_points[0]['observed_uplift_pp'])
        strategy_uplifts.append(strategies['Uplift Model']['uplift_pp'])

    return {
        'n_runs': n_runs,
        'auuc_mean': round(float(np.mean(auucs)), 1),
        'auuc_sd': round(float(np.std(auucs)), 1),
        'auuc_ci_lower': round(float(np.percentile(auucs, 2.5)), 1),
        'auuc_ci_upper': round(float(np.percentile(auucs, 97.5)), 1),
        's_learner_auuc_mean': round(float(np.mean(s_learner_auucs)), 1),
        's_learner_auuc_sd': round(float(np.std(s_learner_auucs)), 1),
        'top_decile_uplift_pp_mean': round(float(np.mean(top_deciles)), 2),
        'top_decile_uplift_pp_sd': round(float(np.std(top_deciles)), 2),
        'top_decile_uplift_pp_ci_lower': round(float(np.percentile(top_deciles, 2.5)), 2),
        'top_decile_uplift_pp_ci_upper': round(float(np.percentile(top_deciles, 97.5)), 2),
        'targeted_policy_uplift_pp_mean': round(float(np.mean(strategy_uplifts)), 2),
        'targeted_policy_uplift_pp_sd': round(float(np.std(strategy_uplifts)), 2),
    }


def compare_strategies(df_train, df_test, uplift_scores, feature_cols,
                       outcome_col='outcome_visit', s_learner_scores=None):
    """Compare uplift, S-Learner, response model, and random targeting."""
    n_total = len(df_test)
    budget_frac = 0.30
    n_target = int(n_total * budget_frac)

    results = {}

    def _eval_targeted(targeted):
        t = targeted[targeted['treatment'] == 1]
        c = targeted[targeted['treatment'] == 0]
        if len(t) > 0 and len(c) > 0:
            return {
                'uplift_pp': round((t[outcome_col].mean() - c[outcome_col].mean()) * 100, 2),
                'n_targeted': n_target,
                'n_treated_in_target': len(t),
                'n_control_in_target': len(c),
            }
        return None

    # Strategy 1: Random
    rng = np.random.RandomState(42)
    random_idx = rng.choice(n_total, n_target, replace=False)
    r = _eval_targeted(df_test.iloc[random_idx])
    if r:
        results['Random'] = r

    # Strategy 2: T-Learner uplift targeting
    top_uplift_idx = np.argsort(-uplift_scores)[:n_target]
    r = _eval_targeted(df_test.iloc[top_uplift_idx])
    if r:
        results['Uplift Model'] = r

    # Strategy 3: S-Learner targeting
    if s_learner_scores is not None:
        top_s_idx = np.argsort(-s_learner_scores)[:n_target]
        r = _eval_targeted(df_test.iloc[top_s_idx])
        if r:
            results['S-Learner'] = r

    # Strategy 4: Response model (predict P(visit | treated))
    treated_train = df_train[df_train['treatment'] == 1]
    resp_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=50, random_state=42
    )
    resp_model.fit(treated_train[feature_cols], treated_train[outcome_col])
    response_scores = resp_model.predict_proba(df_test[feature_cols])[:, 1]

    top_resp_idx = np.argsort(-response_scores)[:n_target]
    r = _eval_targeted(df_test.iloc[top_resp_idx])
    if r:
        results['Response Model'] = r

    return results


def bootstrap_uplift_ci(df_test, scores, outcome_col='outcome_visit',
                        n_bootstrap=500, alpha=0.05):
    """Bootstrap CI for top-decile observed uplift."""
    n_total = len(df_test)
    n_top = n_total // 10

    df_eval = df_test.copy()
    df_eval['score'] = scores
    df_eval = df_eval.sort_values('score', ascending=False)
    top_decile = df_eval.iloc[:n_top]

    rng = np.random.RandomState(42)
    boot_uplifts = []

    for _ in range(n_bootstrap):
        sample = top_decile.sample(n=len(top_decile), replace=True, random_state=rng.randint(1e9))
        t = sample[sample['treatment'] == 1]
        c = sample[sample['treatment'] == 0]
        if len(t) > 0 and len(c) > 0:
            boot_uplifts.append(t[outcome_col].mean() - c[outcome_col].mean())

    boot_uplifts = np.array(boot_uplifts)
    ci_lower = np.percentile(boot_uplifts, 100 * alpha / 2) * 100
    ci_upper = np.percentile(boot_uplifts, 100 * (1 - alpha / 2)) * 100
    return round(ci_lower, 2), round(ci_upper, 2)


def main():
    print("=" * 70)
    print("HILLSTROM EMAIL DATASET ANALYSIS")
    print("=" * 70)

    # Load data
    df = load_hillstrom_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Prepare
    df, feature_cols = prepare_data(df)

    print(f"\nSegment distribution:")
    print(df['segment'].value_counts())
    print(f"\nTreatment distribution: {df['treatment'].value_counts().to_dict()}")
    print(f"\nOutcome rates:")
    print(f"  Visit rate (treated): {df[df['treatment']==1]['outcome_visit'].mean():.3f}")
    print(f"  Visit rate (control): {df[df['treatment']==0]['outcome_visit'].mean():.3f}")
    print(f"  Overall ATE (visit): {df[df['treatment']==1]['outcome_visit'].mean() - df[df['treatment']==0]['outcome_visit'].mean():.4f}")

    # Train/test split
    df_train, df_test = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['treatment'],
    )
    print(f"\nTrain: {len(df_train)}, Test: {len(df_test)}")

    # Train T-Learner
    print("\nTraining T-Learner...")
    model_t, model_c = train_t_learner(df_train, feature_cols, 'outcome_visit')

    # Train S-Learner
    print("Training S-Learner...")
    s_model = train_s_learner(df_train, feature_cols, 'outcome_visit')

    # Score
    uplift_scores = score_uplift(model_t, model_c, df_test, feature_cols)
    s_learner_scores = score_s_learner(s_model, df_test, feature_cols)

    # Feature importance
    feat_imp_t = dict(zip(feature_cols, model_t.feature_importances_))
    feat_imp_c = dict(zip(feature_cols, model_c.feature_importances_))

    print(f"\nUplift score distribution:")
    print(f"  Mean: {np.mean(uplift_scores):.4f}")
    print(f"  Std:  {np.std(uplift_scores):.4f}")
    print(f"  Min:  {np.min(uplift_scores):.4f}")
    print(f"  Max:  {np.max(uplift_scores):.4f}")

    # Qini curve
    print("\n" + "=" * 70)
    print("QINI CURVE (Visit Outcome)")
    print("=" * 70)

    qini_points, auuc, overall_ate = compute_qini_curve(df_test, uplift_scores, 'outcome_visit')

    print(f"{'Decile':<10} {'Prop.':<10} {'Obs. Uplift (pp)':<20} {'Cum. Gain':<15} {'vs Random':<15}")
    print("-" * 70)
    for p in qini_points:
        print(f"{p['decile']:<10} {p['proportion']:<10.1f} {p['observed_uplift_pp']:<20.2f} {p['cumulative_gain']:<15.1f} {p['gain_over_random']:<15.1f}")

    print(f"\nOverall ATE: {overall_ate*100:.2f} pp")
    print(f"AUUC: {auuc:.1f}")

    # S-Learner qini curve
    s_qini_points, s_auuc, _ = compute_qini_curve(df_test, s_learner_scores, 'outcome_visit')
    print(f"\nS-Learner AUUC: {s_auuc:.1f}")

    # Bootstrap CI for top decile
    ci_low, ci_high = bootstrap_uplift_ci(df_test, uplift_scores, 'outcome_visit')
    print(f"Top decile uplift 95% CI: [{ci_low:.2f}, {ci_high:.2f}] pp")

    repeated_summary = summarize_repeated_splits(df, feature_cols, 'outcome_visit')
    print(
        "Repeated split summary "
        f"(n={repeated_summary['n_runs']}): AUUC mean={repeated_summary['auuc_mean']:.1f} "
        f"(SD={repeated_summary['auuc_sd']:.1f}), top decile mean="
        f"{repeated_summary['top_decile_uplift_pp_mean']:.2f} pp"
    )

    # Strategy comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON (30% budget, Visit outcome)")
    print("=" * 70)

    strategies = compare_strategies(
        df_train, df_test, uplift_scores, feature_cols, 'outcome_visit',
        s_learner_scores=s_learner_scores,
    )
    for name, result in strategies.items():
        print(f"  {name}: {result['uplift_pp']:.2f} pp observed uplift in targeted group")

    # Also analyze conversion outcome
    print("\n" + "=" * 70)
    print("CONVERSION OUTCOME ANALYSIS")
    print("=" * 70)

    model_t_conv, model_c_conv = train_t_learner(df_train, feature_cols, 'outcome_conversion')
    uplift_conv = score_uplift(model_t_conv, model_c_conv, df_test, feature_cols)
    qini_conv, auuc_conv, ate_conv = compute_qini_curve(df_test, uplift_conv, 'outcome_conversion')

    print(f"Overall ATE (conversion): {ate_conv*100:.2f} pp")
    print(f"AUUC (conversion): {auuc_conv:.1f}")
    print(f"\nTop 3 decile uplifts (conversion):")
    for p in qini_conv[:3]:
        print(f"  Decile {p['decile']}: {p['observed_uplift_pp']:.2f} pp")

    # Save results
    output = {
        'dataset': {
            'n_records': len(df),
            'n_treated': int(df['treatment'].sum()),
            'n_control': int((1 - df['treatment']).sum()),
            'treatment_control_ratio': round(df['treatment'].sum() / (1 - df['treatment']).sum(), 1),
            'visit_rate_treated': round(df[df['treatment']==1]['outcome_visit'].mean(), 4),
            'visit_rate_control': round(df[df['treatment']==0]['outcome_visit'].mean(), 4),
            'overall_ate_visit': round(overall_ate * 100, 2),
            'treatment_definition': 'Any campaign email (mens or womens) vs no email control',
        },
        'feature_importance': {
            'treatment_model': {k: round(v, 3) for k, v in feat_imp_t.items()},
            'control_model': {k: round(v, 3) for k, v in feat_imp_c.items()},
        },
        'uplift_curve_visit': qini_points,
        'auuc_visit': round(auuc, 1),
        's_learner_uplift_curve_visit': s_qini_points,
        's_learner_auuc_visit': round(s_auuc, 1),
        'top_decile_ci': {'lower': ci_low, 'upper': ci_high},
        'repeated_split_visit': repeated_summary,
        'strategy_comparison': strategies,
        'strategy_comparison_note': (
            'Strategy comparison uses observed uplift in the targeted subgroup. '
            'With a 2:1 treatment-to-control ratio, the control group in any targeted '
            'subset may be small, leading to noisier uplift estimates.'
        ),
        'conversion_analysis': {
            'auuc': round(auuc_conv, 1),
            'overall_ate_pp': round(ate_conv * 100, 2),
            'note': (
                'Conversion is much rarer than visits (base rate ~0.9% vs ~15%), making '
                'uplift estimation substantially noisier. The conversion AUUC should '
                'therefore be treated as split-sensitive and illustrative rather than a '
                'stable production-quality estimate.'
            ),
            'top_deciles': [
                {'decile': p['decile'], 'observed_uplift_pp': p['observed_uplift_pp']}
                for p in qini_conv[:3]
            ],
        },
    }

    output['top_decile_ci_note'] = (
        'Bootstrap CI conditions on a single train/test split. It captures '
        'outcome sampling variability but not model-fitting or split uncertainty. '
        'See repeated_split_visit for split-level variability.'
    )

    output_path = os.path.join(DATA_DIR, 'outputs', 'hillstrom_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == '__main__':
    main()
