"""Harm study: does value-weighted targeting concentrate on harmed players, and
does that depend on the data-generating process?

The harm DGP has a tunable alignment c_harm in {-1, -0.5, 0, 0.5, 1} between harm
and the value driver (see core.dgp.draw_harm); c_harm = 1 is the original DGP in
which harm rises with spend and therefore with V.  The operator never sees true
harm: every rule ranks on ESTIMATED components only (tau_hat, ltv_hat and
harm_hat = estimate_harm(true_harm, sigma_h)), and every rule is evaluated on the
TRUE tau, V and harm.

Rules (all rank on estimated quantities):
  tauV       tau_hat * ltv_hat
  soft gate  tau_hat * ltv_hat * h_protection(harm_hat)
  hard cap   tau_hat * ltv_hat - mu * harm_hat, with the cap set to the soft
             gate's mean harm_hat among its treated (equal ESTIMATED harm), mu
             solved by solve_mu
  CATE only  tau_hat

Oracle-component variants ("... (oracle tau,V)"): the same three rules ranking on
the feasible oracle tau_bar * ltv_bar = E[tau|X] E[V|X] from core.dgp.conditional_means
(main DGP structure; tau and V do not depend on c_harm, so it is computed once per
rep) combined with the same harm_hat as the estimated variants.  They adjudicate
whether differences between rules come from the rule or from a poor tau_hat * ltv_hat.

harm_hat is a NOISY ORACLE, not an estimator: at every sigma_h it is built from
true harm, which contains the Z and eps components that no feature-based estimator
could recover.  oracle_harm is True in the output for sigma_h == 0 (harm_hat equals
true harm); sigma_h > 0 degrades that oracle but never turns it into an estimator.

Per replication the population, outcomes and fitted tau_hat / ltv_hat are
identical for every (c_harm, sigma_h, kappa): harm noise is drawn last from its
own stream, the estimators are fitted once, and the 30 cells reuse them.

Cap violation: the cap is expressed in estimated harm.  A replication violates it
when the mean TRUE harm among the treated exceeds the cap (+1e-6).  The binary rate
is reported for the hard cap rules only (for the soft gate the cap is its own
estimated exposure, so the sign test is tautological for sigma_h > 0).  For every
capped rule (soft gate and hard cap, estimated and oracle-component) the mean
overshoot = true exposure - estimated exposure among the treated is reported with a
CI.  The overshoot is a selection effect (winner's curse: selecting low harm_hat
selects negative noise) but not purely regression to the mean: the noise is on the
logit scale, so harm_hat is also pulled toward 0.5 (Jensen), which alone moves the
mean even without selection.  When the cap cannot be met by any top-k (solve_mu
infeasible) the hard cap falls back to the k lowest-harm_hat players and the
replication is counted in infeasible_rate.

Population shape: only the variance of the harm logit is constant across c_harm.
Spend is lognormal, so the harm distribution is not the same shape across c: the
population mean true harm drifts (about 0.43 to 0.46).  Per c_harm the output
reports pop_mean_harm and share_harm_gt_0_6 (population share above the soft gate's
h_lo), so exposure differences can be read against them.

Decoupled DGP variant (W1).  In the main DGP the harm logit loads on z_spend and z_login
(L_std), which also drive tau_base (and, through churn, the moderation 4p(1-p)), so harm
and harm_hat carry information about tau beyond V_bar = 10 spend + 40 login + 5 age (the
noise-free structural value).  Partial corr(tau, harm logit | V_bar) is about 0.23 there.
The "decoupled" variant keeps the population, tau, V, treatment and outcomes EXACTLY as in
the main harm DGP and changes only the harm draw (core.dgp.draw_harm_decoupled): harm
depends on X only through V_bar,
  harm_logit = -0.3 + s (c z_Vbar + sqrt(1 - c^2) Z) + N(0, 0.5),
with the same s, Z and RNG discipline.  Harm is then a function of (V_bar, independent
noise), hence independent of tau given V_bar by construction.  What the design establishes
(and the tests check, on a 200k draw; values in _meta['coupling']): partial
corr(tau, harm logit | V_bar) is about 0 for decoupled (|r| < 0.01) and clearly nonzero for
main.  Decoupled harm therefore carries no tau information beyond V_bar, so a soft-gate gain
that REMAINS in the decoupled cells cannot come from a tau channel: it must come through
value information (harm_hat informative about V beyond the misspecified proxy V_hat).  A
gain that VANISHES in the decoupled cells is attributable to harm's information about tau.
_meta['coupling'] also reports corr(harm, V) and the partial correlation given the
realised V per DGP at c = 1.  The decoupled variant is run at c_harm = 1 for every sigma_h
and kappa, same rules, estimated and feasible-oracle components, same metrics; cells carry
dgp = "main" or "decoupled" (the oracle E[tau|X], E[V|X] is the main DGP's, tau and V being
unchanged).

Soft gate versus CATE only: every cell (both DGPs) adds, for each component source, one
paired welfare test of the soft gate minus CATE-only ranking (estimated: tau_hat; oracle:
tau_bar), with a CI and the delta-method paired_ratio_ci percentage relative to CATE-only
welfare.  Both join the cell's Holm family, which is therefore EIGHT tests (the six of
PAIRED_COMPARISONS x COMPONENT_SOURCES plus these two).

Usage:  python study_harm.py [--reps R] [--n-players N] [--smoke]
"""

import argparse

import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split

from core.dgp import (
    C_HARM_GRID,
    FEATURE_NAMES,
    ORACLE_MC,
    conditional_means,
    draw_harm,
    draw_harm_decoupled,
    generate_harm,
    z_vbar,
)
from core.estimators import (
    BOOT_STREAM,
    HARM_ESTIMATE_STREAM,
    HARM_STREAM,
    ORACLE_STREAM,
    estimate_components,
    seed_from,
)
from core.io import add_common_args, resolve_run_size, save_json
from core.policy import compute_welfare, h_protection, target_topk
from core.stats import holm_bonferroni, paired_test_raw, ratio_entry
from study_main import N_PLAYERS, N_RUNS, SEED_BASE

SIGMA_H_GRID = (0.0, 0.5, 1.0)
KAPPA_GRID = (0.10, 0.30)
ORACLE_SUFFIX = ' (oracle tau,V)'
BASE_RULES = ('tauV', 'soft gate', 'hard cap')
RULES = (BASE_RULES + ('CATE only',) + tuple(r + ORACLE_SUFFIX for r in BASE_RULES)
         + ('CATE only' + ORACLE_SUFFIX,))
SOFT_RULES = ('soft gate', 'soft gate' + ORACLE_SUFFIX)
HARD_RULES = ('hard cap', 'hard cap' + ORACLE_SUFFIX)
CAPPED_RULES = SOFT_RULES + HARD_RULES
# Paired welfare tests per (c_harm, sigma_h, kappa) cell, for each component source:
# (label, minuend rule, subtrahend rule).  One Holm family per (dgp, c_harm, sigma_h, kappa) cell:
# these three x two component sources, plus CATE_ONLY_COMPARISON for each source = eight tests.
PAIRED_COMPARISONS = (('hard cap vs soft gate', 'hard cap', 'soft gate'),
                      ('soft gate vs tauV', 'soft gate', 'tauV'),
                      ('hard cap vs tauV', 'hard cap', 'tauV'))
CATE_ONLY_COMPARISON = ('soft gate vs CATE only', 'soft gate', 'CATE only')
HARM_DGPS = ('main', 'decoupled')
DECOUPLED_C_HARMS = (1.0,)          # the decoupled variant is run at c_harm = 1 only
COMPONENT_SOURCES = (('estimated', ''), ('oracle', ORACLE_SUFFIX))
H_LO = 0.60
N_BOOT = 2000
CAP_TOL = 1e-6
_HARM_CLIP = 1e-6
COUPLING_N = 200_000


def estimate_harm(true_harm, rng, sigma_h):
    """harm_hat = sigmoid(logit(true_harm) + N(0, sigma_h)); true_harm is clipped
    away from 0 and 1 first so the logit is finite."""
    h = np.clip(np.asarray(true_harm, dtype=float), _HARM_CLIP, 1.0 - _HARM_CLIP)
    noise = rng.normal(0.0, 1.0, len(h)) * sigma_h
    return 1.0 / (1.0 + np.exp(-(np.log(h / (1.0 - h)) + noise)))


def solve_mu(score, harm_hat, k, cap):
    """Smallest mu >= 0 such that the top-k by (score - mu * harm_hat) has mean
    harm_hat <= cap.  Returns (mu, feasible).

    feasible is False when even the k lowest-harm players (the mu -> inf limit)
    exceed the cap; mu is then inf, never a silently wrong finite value."""
    score = np.asarray(score, dtype=float)
    harm_hat = np.asarray(harm_hat, dtype=float)
    if np.sort(harm_hat)[:k].mean() > cap + 1e-12:
        return float('inf'), False

    def meets(mu):
        return harm_hat[target_topk(score - mu * harm_hat, k)].mean() <= cap + 1e-12

    if meets(0.0):
        return 0.0, True
    lo, hi = 0.0, max(1.0, float(np.ptp(score)))
    while not meets(hi):            # mean harm is weakly decreasing in mu
        lo, hi = hi, hi * 2.0
        if hi > 1e18:               # ties / float limit: the cap is met only at mu -> inf
            return float('inf'), False
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if meets(mid):
            hi = mid
        else:
            lo = mid
    return hi, True


def _new_cell():
    return {k: [] for k in ('welfare', 'exposure', 'diff', 'cap', 'violation',
                            'infeasible', 'corr', 'overshoot', 'pop_harm',
                            'share_high')}


def select_rules(tau_hat, ltv_hat, tau_bar, ltv_bar, harm_hat, k):
    """Selected index set and hard-cap info per rule.

    Takes ONLY estimated or oracle-conditional components plus harm_hat: true harm
    is deliberately not an argument, so no rule can see it.
    Returns (selections, caps, feasible): caps[rule] is the estimated-harm cap of
    each capped rule (the matching soft gate's mean harm_hat)."""
    sel, caps, feasible = {}, {}, {}
    sel['CATE only'] = target_topk(tau_hat, k)
    sel['CATE only' + ORACLE_SUFFIX] = target_topk(tau_bar, k)
    for suffix, tau_c, ltv_c in (('', tau_hat, ltv_hat), (ORACLE_SUFFIX, tau_bar, ltv_bar)):
        base = tau_c * ltv_c
        sel['tauV' + suffix] = target_topk(base, k)
        sel['soft gate' + suffix] = target_topk(base * h_protection(harm_hat), k)
        cap = float(harm_hat[sel['soft gate' + suffix]].mean())
        caps['soft gate' + suffix] = caps['hard cap' + suffix] = cap
        mu, ok = solve_mu(base, harm_hat, k, cap)
        feasible['hard cap' + suffix] = ok
        sel['hard cap' + suffix] = (target_topk(base - mu * harm_hat, k) if ok
                                    else np.argsort(harm_hat)[:k])
    return sel, caps, feasible


def simulate(reps, n_players, c_harms=C_HARM_GRID, sigma_hs=SIGMA_H_GRID,
             kappas=KAPPA_GRID, dgp='main', _eval_harm_transform=None, _record=None):
    """Run the sweep for one DGP variant (dgp in HARM_DGPS).  Returns
    {(c_harm, sigma_h, kappa, rule): per-rep lists}.

    Test seams (not for production use): _eval_harm_transform(harm) is applied to
    the true harm only AFTER harm_hat has been computed from it and REBINDS the
    variable `harm`, so every later use of true harm sees it (a rule that leaked
    true harm would change its selection); _record, if a dict, receives every
    rule's selected index set under (rep, c_harm, sigma_h, kappa, rule)."""
    if dgp not in HARM_DGPS:
        raise ValueError(f'unknown harm DGP variant {dgp!r}; expected one of {HARM_DGPS}')
    decoupled = dgp == 'decoupled'
    draw = draw_harm_decoupled if decoupled else draw_harm
    raw = {(c, s, kp, r): _new_cell() for c in c_harms for s in sigma_hs
           for kp in kappas for r in RULES}
    for rep in range(reps):
        seed = SEED_BASE + rep * 7
        data = generate_harm(n_players, np.random.RandomState(seed),
                             harm_rng=np.random.RandomState(seed_from(seed, HARM_STREAM)),
                             decoupled=decoupled)
        idx = np.arange(len(data['treatment']))
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        tau = data['true_tau'][test_idx]
        ltv = data['true_ltv'][test_idx]
        tau_hat, ltv_hat, _ = estimate_components(data, train_idx, test_idx, seed=seed)
        # Feasible oracle E[tau|X], E[V|X] on the test rows: once per rep.
        feats_test = {name: data['features'][test_idx, i]
                      for i, name in enumerate(FEATURE_NAMES)}
        cm = conditional_means('main', feats_test, n_mc=ORACLE_MC,
                               seed=seed_from(seed, ORACLE_STREAM))
        tau_bar, ltv_bar = cm['tau_bar'], cm['ltv_bar']
        feats = {name: data['features'][:, i] for i, name in enumerate(FEATURE_NAMES)}

        for c in c_harms:
            # Same harm stream for every c: only the alignment changes.
            harm_all = draw(feats, np.random.RandomState(seed_from(seed, HARM_STREAM)), c)
            harm_c = harm_all[test_idx]
            for sigma in sigma_hs:
                harm_hat = estimate_harm(
                    harm_c, np.random.RandomState(seed_from(seed, HARM_ESTIMATE_STREAM)), sigma)
                # Test seam: rebind true harm itself once harm_hat is frozen, so any
                # later use of `harm` (selection, evaluation, stats) sees the transform.
                harm = harm_c if _eval_harm_transform is None else _eval_harm_transform(harm_c)
                pop_harm = float(harm.mean())
                share_high = float(np.mean(harm > H_LO))
                corr = float(np.corrcoef(harm, ltv)[0, 1])
                for kappa in kappas:
                    k = int(len(test_idx) * kappa)
                    sel, caps, feasible = select_rules(
                        tau_hat, ltv_hat, tau_bar, ltv_bar, harm_hat, k)
                    for rule in RULES:
                        t = sel[rule]
                        if _record is not None:
                            _record[(rep, c, sigma, kappa, rule)] = np.sort(t)
                        cell = raw[(c, sigma, kappa, rule)]
                        exposure = float(harm[t].mean())
                        capped = rule in CAPPED_RULES
                        cell['welfare'].append(compute_welfare(t, tau, ltv))
                        cell['exposure'].append(exposure)
                        cell['diff'].append(exposure - pop_harm)
                        cell['cap'].append(caps[rule] if capped else np.nan)
                        cell['overshoot'].append(
                            exposure - float(harm_hat[t].mean()) if capped else np.nan)
                        cell['violation'].append(
                            exposure > caps[rule] + CAP_TOL if rule in HARD_RULES else False)
                        cell['infeasible'].append(
                            rule in HARD_RULES and not feasible[rule])
                        cell['corr'].append(corr)
                        cell['pop_harm'].append(pop_harm)
                        cell['share_high'].append(share_high)
    return raw


def _partial_corr(x, y, z):
    """Correlation of the linear residuals of x and y after regressing each on z."""
    zc = np.asarray(z, dtype=float) - np.mean(z)

    def resid(v):
        v = np.asarray(v, dtype=float)
        return (v - v.mean()) - zc * (zc @ (v - v.mean())) / (zc @ zc)
    return float(np.corrcoef(resid(x), resid(y))[0, 1])


def coupling_diagnostics(n=COUPLING_N, seed=SEED_BASE, c_harm=1.0):
    """Per DGP variant at c_harm: corr(harm, V), partial corr(tau, harm logit | V_bar) and
    partial corr(tau, harm logit | V), on one large fixed-seed draw (n players).

    V is the realised LTV; V_bar = 10 spend + 40 login + 5 age is its noise-free structural
    part (core.dgp.z_vbar); the harm logit is logit(true harm); partial correlations are of
    linear residuals.  The decoupled harm is a function of (V_bar, independent noise), so its
    partial correlation with tau given V_bar is 0 up to sampling error (SE about 1/sqrt(n))."""
    out = {'n': n, 'seed': seed, 'c_harm': c_harm}
    for name in HARM_DGPS:
        data = generate_harm(n, np.random.RandomState(seed),
                             harm_rng=np.random.RandomState(seed_from(seed, HARM_STREAM)),
                             c_harm=c_harm, decoupled=name == 'decoupled')
        f = {nm: data['features'][:, i] for i, nm in enumerate(FEATURE_NAMES)}
        h = np.clip(data['true_harm'], _HARM_CLIP, 1.0 - _HARM_CLIP)
        logit = np.log(h / (1.0 - h))
        v_bar = z_vbar(f)
        out[name] = {
            'corr_harm_V': float(np.corrcoef(data['true_harm'], data['true_ltv'])[0, 1]),
            'partial_corr_tau_harm_given_Vbar': _partial_corr(data['true_tau'], logit, v_bar),
            'partial_corr_tau_harm_given_V': _partial_corr(data['true_tau'], logit, data['true_ltv'])}
    return out


def _t_ci(x):
    """Mean and 95% t interval over reps (ddof=1); interval is None for one rep."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return {'mean': m, 'ci95': None}
    half = scipy_stats.t.ppf(0.975, df=len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': m, 'ci95': [m - half, m + half]}


def _boot_ci(x, n_boot, seed):
    """Mean and percentile bootstrap 95% CI over reps (resampling whole reps)."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return {'mean': m, 'ci95': None}
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'mean': m, 'ci95': [float(lo), float(hi)]}


def summarise(raw, n_boot=N_BOOT, dgp='main'):
    """Turn per-rep lists into one entry per (c_harm, sigma_h, kappa, rule), tagged dgp."""
    boot_seed = seed_from(SEED_BASE, BOOT_STREAM)
    cells = []
    for (c, sigma, kappa, rule), v in raw.items():
        capped = rule in CAPPED_RULES
        cells.append({
            'dgp': dgp, 'c_harm': c, 'sigma_h': sigma, 'kappa': kappa, 'rule': rule,
            'n_reps': len(v['welfare']),
            'welfare': _t_ci(v['welfare']),
            'harm_exposure': _t_ci(v['exposure']),
            'exposure_minus_pop': _boot_ci(v['diff'], n_boot, boot_seed),
            'oracle_harm': sigma == 0.0,
            'cap_violation_rate': (float(np.mean(v['violation']))
                                   if rule in HARD_RULES else None),
            'overshoot': _t_ci(v['overshoot']) if capped else None,
            'infeasible_rate': (float(np.mean(v['infeasible']))
                                if rule in HARD_RULES else None),
            'pop_mean_harm': float(np.mean(v['pop_harm'])),
            'share_harm_gt_0_6': float(np.mean(v['share_high'])),
            'corr_harm_V': float(np.mean(v['corr'])),
        })
    return {'cells': cells, 'paired_tests': paired_tests(raw, dgp)}


def paired_tests(raw, dgp='main'):
    """Per-replication paired welfare tests (core.stats.paired_test_raw, unrounded) in every
    (c_harm, sigma_h, kappa) cell, for estimated and feasible-oracle components:
    hard cap minus soft gate, soft gate minus the harm-blind tauV, hard cap minus tauV.
    Each cell also gets the soft gate minus CATE-only test for each component source, with its
    ratio to CATE-only welfare in 'ratio_to_cate_only' (those rows only).
    Holm (core.stats.holm_bonferroni) runs within each cell: the family is that cell's eight
    tests (family_size).  Empty when there is a single replication (no variance)."""
    out = []
    for c, sigma, kappa in sorted({k[:3] for k in raw}):
        rows = []
        specs = [(source, suffix, cmp_) for source, suffix in COMPONENT_SOURCES
                 for cmp_ in PAIRED_COMPARISONS]
        specs += [(source, suffix, CATE_ONLY_COMPARISON) for source, suffix in COMPONENT_SOURCES]
        for source, suffix, (label, a, b) in specs:
            wa = raw[(c, sigma, kappa, a + suffix)]['welfare']
            wb = raw[(c, sigma, kappa, b + suffix)]['welfare']
            if len(wa) < 2:
                return []
            r = paired_test_raw(wa, wb)
            diffs = np.array(wa) - np.array(wb)
            # mean(a - b) / mean(harm-blind tauV welfare), delta-method CI from the
            # per-replication pairs (core.stats.paired_ratio_ci)
            blind = np.array(raw[(c, sigma, kappa, 'tauV' + suffix)]['welfare'])
            row = {'dgp': dgp, 'c_harm': c, 'sigma_h': sigma, 'kappa': kappa,
                   'comparison': label, 'components': source,
                   'a': a + suffix, 'b': b + suffix,
                   'mean_diff': r['mean_diff'], 'ci95': r['ci_95'],
                   'se_diff': r['se_diff'], 't_stat': r['t_stat'],
                   'cohens_d': r['cohens_d'], 'p_raw': r['p_value'],
                   'ratio_to_blind': ratio_entry(diffs, blind)}
            if label == CATE_ONLY_COMPARISON[0]:
                # mean(soft - CATE only) / mean(CATE-only welfare), delta-method CI
                row['ratio_to_cate_only'] = ratio_entry(diffs, np.array(wb))
            rows.append(row)
        for row, p in zip(rows, holm_bonferroni([r['p_raw'] for r in rows])):
            row['p_holm'] = float(p)
            row['family_size'] = len(rows)
        out.extend(rows)
    return out


def _print_table(summary, kappa=0.30):
    print(f"kappa={kappa}  diff = treated mean true harm - population mean; viol = "
          "hard-cap violation rate on true harm; over = mean (true - estimated) "
          "exposure; orc = sigma_h=0 (harm_hat is the true harm)")
    print(f"{'dgp':<9} {'c':>5} {'sig':>4}{'orc':>4} {'rule':<26} {'welfare':>8} {'expo':>6} "
          f"{'diff':>7} {'diff 95% CI':>17} {'viol':>5} {'over':>7} {'popH':>5} {'>.6':>5} {'corr':>5}")
    for e in summary['cells']:
        if e['kappa'] != kappa:
            continue
        ci = e['exposure_minus_pop']['ci95']
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else 'n/a'
        viol = '-' if e['cap_violation_rate'] is None else f"{e['cap_violation_rate']:.2f}"
        over = '-' if e['overshoot'] is None else f"{e['overshoot']['mean']:+.3f}"
        print(f"{e['dgp']:<9} {e['c_harm']:>5.1f} {e['sigma_h']:>4.1f}{'*' if e['oracle_harm'] else '':>4} "
              f"{e['rule']:<26} {e['welfare']['mean']:>8.0f} "
              f"{e['harm_exposure']['mean']:>6.3f} {e['exposure_minus_pop']['mean']:>+7.3f} "
              f"{ci_s:>17} {viol:>5} {over:>7} {e['pop_mean_harm']:>5.3f} "
              f"{e['share_harm_gt_0_6']:>5.3f} {e['corr_harm_V']:>+5.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(parser)
    args = parser.parse_args(argv)
    reps, n_players = resolve_run_size(args, N_RUNS, N_PLAYERS)
    n_cells = (len(C_HARM_GRID) + len(DECOUPLED_C_HARMS)) * len(SIGMA_H_GRID) * len(KAPPA_GRID)
    print(f"Harm sweep | reps={reps}, n={n_players}, {n_cells} cells x {len(RULES)} rules "
          f"(main + decoupled DGP)")
    raws = {'main': simulate(reps, n_players, dgp='main'),
            'decoupled': simulate(reps, n_players, c_harms=DECOUPLED_C_HARMS, dgp='decoupled')}
    parts = [summarise(raw, dgp=name) for name, raw in raws.items()]
    summary = {'cells': [c for p in parts for c in p['cells']],
               'paired_tests': [t for p in parts for t in p['paired_tests']]}
    summary['_meta'] = {'n_runs': reps, 'n_players': n_players, 'seeds_base': SEED_BASE,
                        'n_boot': N_BOOT, 'c_harm': list(C_HARM_GRID),
                        'sigma_h': list(SIGMA_H_GRID), 'kappa': list(KAPPA_GRID),
                        'dgps': list(HARM_DGPS),
                        'decoupled_c_harm': list(DECOUPLED_C_HARMS),
                        'coupling': coupling_diagnostics()}
    _print_table(summary)
    print(f"Saved {save_json(summary, 'harm_sweep_smoke.json' if args.smoke else 'harm_sweep.json')}")


if __name__ == '__main__':
    main()
