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

Per replication the population, outcomes and fitted tau_hat / ltv_hat are
identical for every (c_harm, sigma_h, kappa): harm noise is drawn last from its
own stream, the estimators are fitted once, and the 30 cells reuse them.

Cap violation: the cap is expressed in estimated harm.  A replication violates it
when the mean TRUE harm among the treated exceeds the cap (+1e-6).  It is
reported for the hard cap and the soft gate only.  When the cap cannot be met by
any top-k (solve_mu infeasible) the hard cap falls back to the k lowest-harm_hat
players and the replication is counted in infeasible_rate.

Usage:  python study_harm.py [--reps R] [--n-players N] [--smoke]
"""

import argparse

import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split

from simulation_v38 import N_PLAYERS, N_RUNS, SEED_BASE
from core.dgp import C_HARM_GRID, draw_harm, generate_harm, FEATURE_NAMES
from core.estimators import (
    BOOT_STREAM, HARM_ESTIMATE_STREAM, HARM_STREAM, estimate_components, seed_from,
)
from core.io import add_common_args, resolve_run_size, save_json
from core.policy import compute_welfare, h_protection, target_topk

SIGMA_H_GRID = (0.0, 0.5, 1.0)
KAPPA_GRID = (0.10, 0.30)
RULES = ('tauV', 'soft gate', 'hard cap', 'CATE only')
N_BOOT = 2000
CAP_TOL = 1e-6
_HARM_CLIP = 1e-6


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
                            'infeasible', 'corr')}


def simulate(reps, n_players, c_harms=C_HARM_GRID, sigma_hs=SIGMA_H_GRID,
             kappas=KAPPA_GRID):
    """Run the sweep.  Returns {(c_harm, sigma_h, kappa, rule): per-rep lists}."""
    raw = {(c, s, kp, r): _new_cell() for c in c_harms for s in sigma_hs
           for kp in kappas for r in RULES}
    for rep in range(reps):
        seed = SEED_BASE + rep * 7
        data = generate_harm(n_players, np.random.RandomState(seed),
                             harm_rng=np.random.RandomState(seed_from(seed, HARM_STREAM)))
        idx = np.arange(len(data['treatment']))
        train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        tau = data['true_tau'][test_idx]
        ltv = data['true_ltv'][test_idx]
        tau_hat, ltv_hat, _ = estimate_components(data, train_idx, test_idx, seed=seed)
        base = tau_hat * ltv_hat
        feats = {name: data['features'][:, i] for i, name in enumerate(FEATURE_NAMES)}

        for c in c_harms:
            # Same harm stream for every c: only the alignment changes.
            harm_all = draw_harm(feats, np.random.RandomState(seed_from(seed, HARM_STREAM)), c)
            harm = harm_all[test_idx]
            pop_harm = float(harm.mean())
            corr = float(np.corrcoef(harm, ltv)[0, 1])
            for sigma in sigma_hs:
                harm_hat = estimate_harm(
                    harm, np.random.RandomState(seed_from(seed, HARM_ESTIMATE_STREAM)), sigma)
                soft_score = base * h_protection(harm_hat)
                for kappa in kappas:
                    k = int(len(test_idx) * kappa)
                    sel = {'tauV': target_topk(base, k),
                           'soft gate': target_topk(soft_score, k),
                           'CATE only': target_topk(tau_hat, k)}
                    cap = float(harm_hat[sel['soft gate']].mean())
                    mu, feasible = solve_mu(base, harm_hat, k, cap)
                    sel['hard cap'] = (target_topk(base - mu * harm_hat, k) if feasible
                                       else np.argsort(harm_hat)[:k])
                    for rule in RULES:
                        t = sel[rule]
                        cell = raw[(c, sigma, kappa, rule)]
                        exposure = float(harm[t].mean())
                        capped = rule in ('hard cap', 'soft gate')
                        cell['welfare'].append(compute_welfare(t, tau, ltv))
                        cell['exposure'].append(exposure)
                        cell['diff'].append(exposure - pop_harm)
                        cell['cap'].append(cap if capped else np.nan)
                        cell['violation'].append(exposure > cap + CAP_TOL if capped else False)
                        cell['infeasible'].append(rule == 'hard cap' and not feasible)
                        cell['corr'].append(corr)
    return raw


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


def summarise(raw, n_boot=N_BOOT):
    """Turn per-rep lists into one entry per (c_harm, sigma_h, kappa, rule)."""
    boot_seed = seed_from(SEED_BASE, BOOT_STREAM)
    cells = []
    for (c, sigma, kappa, rule), v in raw.items():
        capped = rule in ('hard cap', 'soft gate')
        cells.append({
            'c_harm': c, 'sigma_h': sigma, 'kappa': kappa, 'rule': rule,
            'n_reps': len(v['welfare']),
            'welfare': _t_ci(v['welfare']),
            'harm_exposure': _t_ci(v['exposure']),
            'exposure_minus_pop': _boot_ci(v['diff'], n_boot, boot_seed),
            'cap_violation_rate': float(np.mean(v['violation'])) if capped else None,
            'infeasible_rate': float(np.mean(v['infeasible'])) if rule == 'hard cap' else None,
            'corr_harm_V': float(np.mean(v['corr'])),
        })
    return {'cells': cells}


def _print_table(summary, kappa=0.30):
    print(f"kappa={kappa}  (exposure diff = treated mean true harm - population mean; "
          "viol = cap-violation rate on true harm)")
    print(f"{'c':>5} {'sig':>4} {'rule':<10} {'welfare':>9} {'exposure':>9} "
          f"{'diff':>8} {'diff 95% CI':>20} {'viol':>5} {'corr':>6}")
    for e in summary['cells']:
        if e['kappa'] != kappa:
            continue
        ci = e['exposure_minus_pop']['ci95']
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else 'n/a'
        viol = '-' if e['cap_violation_rate'] is None else f"{e['cap_violation_rate']:.2f}"
        print(f"{e['c_harm']:>5.1f} {e['sigma_h']:>4.1f} {e['rule']:<10} "
              f"{e['welfare']['mean']:>9.0f} {e['harm_exposure']['mean']:>9.3f} "
              f"{e['exposure_minus_pop']['mean']:>+8.3f} {ci_s:>20} {viol:>5} "
              f"{e['corr_harm_V']:>+6.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(parser)
    args = parser.parse_args(argv)
    reps, n_players = resolve_run_size(args, N_RUNS, N_PLAYERS)
    print(f"Harm sweep | reps={reps}, n={n_players}, "
          f"{len(C_HARM_GRID) * len(SIGMA_H_GRID) * len(KAPPA_GRID)} cells x {len(RULES)} rules")
    summary = summarise(simulate(reps, n_players))
    summary['_meta'] = {'n_runs': reps, 'n_players': n_players, 'seeds_base': SEED_BASE,
                        'n_boot': N_BOOT, 'c_harm': list(C_HARM_GRID),
                        'sigma_h': list(SIGMA_H_GRID), 'kappa': list(KAPPA_GRID)}
    _print_table(summary)
    print(f"Saved {save_json(summary, 'harm_sweep.json')}")


if __name__ == '__main__':
    main()
