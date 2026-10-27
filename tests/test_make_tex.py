"""make_tex: formatters, macro naming, generated-file contract, determinism, manuscript checks."""
import re
from pathlib import Path

import pytest

import make_tex as mt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
TEX = DOCS / 'stent-2026-value-weighted-targeting.tex'
SMOKE = [ROOT / 'outputs' / f'{b}_smoke.json' for b in mt.INPUT_BASENAMES]


def smoke_data():
    if not all(p.exists() for p in SMOKE):
        pytest.skip('smoke outputs missing; run src/run_all.py --smoke')
    return mt.load(ROOT / 'outputs', True)


# ---- formatters

def test_thousands_separator():
    assert mt.fmt_welfare(1234567.4) == '1{,}234{,}567'
    assert mt.fmt_welfare(999.6) == '1{,}000'
    assert mt.fmt_welfare(-1234.2) == '-1{,}234'
    assert mt.fmt_welfare(12) == '12'
    assert mt.fmt_welfare(-0.2) == '0'


def test_precision_per_quantity():
    assert mt.fmt_harm(0.12345) == '0.123'
    assert mt.fmt_kappa(0.5) == '0.500'
    assert mt.fmt_d(1.4949) == '1.49'
    assert mt.fmt_pct(0.12345) == r'12.3\%'
    assert mt.fmt_pct(1.0291) == r'102.9\%'


def test_p_values():
    assert mt.fmt_p(0.0004) == '<0.001'
    assert mt.fmt_p(0.0) == '<0.001'
    assert mt.fmt_p(0.001) == '0.001'
    assert mt.fmt_p(0.0496) == '0.050'
    assert mt.fmt_p(1.0) == '1.000'


def test_tex_makes_signs_safe():
    assert mt.tex('-1{,}234') == r'\ensuremath{-}1{,}234'
    assert mt.tex('<0.001') == r'\ensuremath{<}0.001'
    assert mt.tex('[-0.1, >0.500]') == r'[\ensuremath{-}0.1, \ensuremath{>}0.500]'
    assert mt.tex('T-learner') == 'T-learner'


def test_signed_and_ci():
    assert mt.fmt_signed(5.2, mt.fmt_harm) == '+5.200'
    assert mt.fmt_signed(-5.2, mt.fmt_harm) == '-5.200'
    assert mt.fmt_ci(-1, 2000, mt.fmt_welfare) == '[-1, 2{,}000]'


def test_kappa_tags_spell_digits():
    assert mt.kappa_tag(0.1) == 'Kten'
    assert mt.kappa_tag(0.3) == 'Kthirty'
    assert mt.kappa_tag(0.5) == 'Kfifty'
    assert mt.kappa_tag(0.25) == 'Ktwentyfive'
    assert mt.kappa_tag(0.05) == 'Kfive'
    assert mt.check_macro_name('numX' + mt.kappa_tag(0.075))   # fallback is letters only


def test_macro_name_validation():
    mt.check_macro_name('numMainWelfareCateOnlyKthirty')
    for bad in ('numMain2', 'main', 'num_x', 'numlower', 'numX1'):
        with pytest.raises(ValueError):
            mt.check_macro_name(bad)
    m = mt.Macros()
    m.add('numAbc', '1', 'm', 's')
    with pytest.raises(ValueError):
        m.add('numAbc', '2', 'm', 's')


def test_kappa_star_censoring():
    grid = [0.05, 0.5]
    base = {'point': None, 'point_outcome': 'above_grid', 'ci95': [0.475, 0.5],
            'ci95_labels': [None, '>max']}
    assert mt.kappa_star_text(base, grid, 'point') == '>0.500'
    assert mt.kappa_star_text(base, grid, 'lo') == '0.475'
    assert mt.kappa_star_text(base, grid, 'hi') == '>0.500'
    below = dict(base, point_outcome='below_grid', ci95_labels=['<min', '<min'])
    assert mt.kappa_star_text(below, grid, 'point') == '<0.050'
    assert mt.kappa_star_text(below, grid, 'lo') == '<0.050'
    assert mt.kappa_star_text(dict(base, point=0.3), grid, 'point') == '0.300'


# ---- generated files

def test_generated_files_contract():
    files = mt.build(smoke_data())
    assert set(files) == set(mt.generated_names())
    for name, text in files.items():
        assert text.startswith(mt.HEADER + '\n'), name
        assert '\u2014' not in text, name
    for name, text in files.items():
        if name.startswith('tab_'):
            assert 'begin{table' not in text and 'begin{tabular' not in text, name
            rows = [ln for ln in text.splitlines()[1:] if ln != r'\midrule']
            assert rows and all(ln.endswith(r'\\') for ln in rows), name
            assert len({ln.count('&') for ln in rows}) == 1, f'{name}: ragged rows'
            assert not re.search(r'\bnan\b|None', text), name


def test_numbers_tex_documents_and_defines_every_macro():
    text = mt.build(smoke_data())['numbers.tex']
    defined = re.findall(r'\\newcommand\{\\(num[A-Za-z]+)\}', text)
    documented = re.findall(r'^%   \\(num[A-Za-z]+): .+ \[.+\]$', text, flags=re.M)
    assert defined and defined == documented
    assert len(set(defined)) == len(defined)
    assert all(mt.MACRO_RE.match(n) for n in defined)


def test_headline_macros_present_and_oracles_labelled():
    text = mt.build(smoke_data())['numbers.tex']
    names = set(re.findall(r'\\newcommand\{\\(num[A-Za-z]+)\}', text))
    for need in ('numHarmExposureDiff', 'numBudgetKappaStarPointTMisspec',
                 'numMainBestEstimatedOracleFeasibleEfficiency', 'numMainOracleHindsightWelfare',
                 'numMainOracleFeasibleWelfare', 'numLearnersSMinusTTunedKthirty',
                 'numLearnersShapleyCateShareTDefaultMisspec', 'numRobustnessIdentityRateAipw',
                 'numMainWelfareCateOnlyKthirty'):
        assert need in names, need
    assert all('Oracle' in n for n in names if 'Hindsight' in n or 'Feasible' in n)


def test_percentages_computed_not_typed():
    data = smoke_data()
    text = mt.build(data)['numbers.tex']
    feas = data['main']['oracle_main']['strategies']['Oracle feasible (g=1)']['welfare_mean']
    hind = data['main']['oracle_main']['strategies']['Oracle hindsight (g=1)']['welfare_mean']
    want = mt.fmt_pct(feas / hind)
    assert f'\\newcommand{{\\numMainOracleFeasibleShareOfHindsight}}{{{want}}}' in text


def test_make_tex_is_deterministic(tmp_path):
    data = smoke_data()
    assert mt.build(data) == mt.build(data)
    a, b = tmp_path / 'a', tmp_path / 'b'
    pa = mt.write(ROOT / 'outputs', a, True)
    pb = mt.write(ROOT / 'outputs', b, True)
    assert [p.name for p in pa] == [p.name for p in pb]
    for x, y in zip(pa, pb):
        assert x.read_bytes() == y.read_bytes()


def test_smoke_never_targets_tracked_dir():
    assert mt.SMOKE_GENERATED_DIR != mt.GENERATED_DIR
    assert mt.SMOKE_GENERATED_DIR.name == 'generated_smoke'


# ---- manuscript checks

def _generated_numbers():
    p = DOCS / 'generated' / 'numbers.tex'
    return p.read_text() if p.exists() else ''


def test_all_macros_defined():
    used = set(re.findall(r'\\(num[A-Z][A-Za-z]*)', TEX.read_text()))
    defined = set(re.findall(r'\\newcommand\{\\(num[A-Za-z]+)\}', _generated_numbers()))
    assert used <= defined, f'undefined macros: {sorted(used - defined)}'


@pytest.mark.xfail(strict=True, reason='enabled by Task 10 rewrite')
def test_no_hardcoded_table_numbers():
    text = TEX.read_text()
    envs = re.findall(r'\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}', text, flags=re.S)
    bodies = [m.group(0) for m in re.finditer(r'\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}', text, flags=re.S)]
    assert envs and not [b for b in bodies if re.search(r'\d{1,2}\{,\}\d{3}', b)]
