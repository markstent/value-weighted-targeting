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
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
BOOTSTRAP_SAMPLES = 1000
ALPHA = 0.05


def load_hillstrom_data():
    """
    Load Hillstrom email marketing dataset.
    Source: Kevin Hillstrom's MineThatData blog.
    The dataset contains ~64k customers from an email marketing campaign.
    """
    csv_path = os.path.join(DATA_DIR, 'data', 'hillstrom.csv')

    if not os.path.exists(csv_path):
        print("Downloading Hillstrom dataset...")
        # The dataset is publicly available
        url = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        try:
            df = pd.read_csv(url)
            df.to_csv(csv_path, index=False)
            print(f"Downloaded {len(df)} records")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Trying alternative source...")
            # Alternative: create from known structure
            url2 = "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
            raise RuntimeError(
                f"Could not download Hillstrom dataset. "
                f"Please manually download from MineThatData blog and save as {csv_path}"
            )
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {csv_path}")

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
    normalized by the area of the perfect model.
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

    # AUUC: normalized area between model curve and random diagonal
    # Using trapezoidal integration
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


def compare_strategies(df_test, uplift_scores, feature_cols, outcome_col='outcome_visit'):
    """Compare uplift, response model, and random targeting."""
    n_total = len(df_test)
    budget_frac = 0.30
    n_target = int(n_total * budget_frac)

    results = {}

    # Strategy 1: Random
    rng = np.random.RandomState(42)
    random_idx = rng.choice(n_total, n_target, replace=False)
    random_targeted = df_test.iloc[random_idx]
    t_r = random_targeted[random_targeted['treatment'] == 1]
    c_r = random_targeted[random_targeted['treatment'] == 0]
    if len(t_r) > 0 and len(c_r) > 0:
        results['Random'] = {
            'uplift_pp': round((t_r[outcome_col].mean() - c_r[outcome_col].mean()) * 100, 2),
            'n_targeted': n_target,
        }

    # Strategy 2: Uplift targeting
    top_uplift_idx = np.argsort(-uplift_scores)[:n_target]
    uplift_targeted = df_test.iloc[top_uplift_idx]
    t_u = uplift_targeted[uplift_targeted['treatment'] == 1]
    c_u = uplift_targeted[uplift_targeted['treatment'] == 0]
    if len(t_u) > 0 and len(c_u) > 0:
        results['Uplift Model'] = {
            'uplift_pp': round((t_u[outcome_col].mean() - c_u[outcome_col].mean()) * 100, 2),
            'n_targeted': n_target,
        }

    # Strategy 3: Response model (predict P(visit | treated))
    treated_train = df_test[df_test['treatment'] == 1]
    resp_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    # Use a subset for training response model to avoid data leakage
    resp_train, resp_eval = train_test_split(treated_train, test_size=0.5, random_state=42)
    resp_model.fit(resp_train[feature_cols], resp_train[outcome_col])
    response_scores = resp_model.predict_proba(df_test[feature_cols])[:, 1]

    top_resp_idx = np.argsort(-response_scores)[:n_target]
    resp_targeted = df_test.iloc[top_resp_idx]
    t_resp = resp_targeted[resp_targeted['treatment'] == 1]
    c_resp = resp_targeted[resp_targeted['treatment'] == 0]
    if len(t_resp) > 0 and len(c_resp) > 0:
        results['Response Model'] = {
            'uplift_pp': round((t_resp[outcome_col].mean() - c_resp[outcome_col].mean()) * 100, 2),
            'n_targeted': n_target,
        }

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
    df_train, df_test = train_test_split(df, test_size=0.30, random_state=42)
    print(f"\nTrain: {len(df_train)}, Test: {len(df_test)}")

    # Train T-Learner
    print("\nTraining T-Learner...")
    model_t, model_c = train_t_learner(df_train, feature_cols, 'outcome_visit')

    # Score
    uplift_scores = score_uplift(model_t, model_c, df_test, feature_cols)

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

    # Bootstrap CI for top decile
    ci_low, ci_high = bootstrap_uplift_ci(df_test, uplift_scores, 'outcome_visit')
    print(f"Top decile uplift 95% CI: [{ci_low:.2f}, {ci_high:.2f}] pp")

    # Strategy comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON (30% budget, Visit outcome)")
    print("=" * 70)

    strategies = compare_strategies(df_test, uplift_scores, feature_cols, 'outcome_visit')
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
            'visit_rate_treated': round(df[df['treatment']==1]['outcome_visit'].mean(), 4),
            'visit_rate_control': round(df[df['treatment']==0]['outcome_visit'].mean(), 4),
            'overall_ate_visit': round(overall_ate * 100, 2),
        },
        'uplift_curve_visit': qini_points,
        'auuc_visit': round(auuc, 1),
        'top_decile_ci': {'lower': ci_low, 'upper': ci_high},
        'strategy_comparison': strategies,
        'conversion_auuc': round(auuc_conv, 1),
        'conversion_ate': round(ate_conv * 100, 2),
    }

    output_path = os.path.join(DATA_DIR, 'outputs', 'hillstrom_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == '__main__':
    main()
