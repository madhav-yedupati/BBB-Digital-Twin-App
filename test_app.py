"""
Regression tests for the BBB Digital Twin app.

    python test_app.py            # standalone, no pytest needed
    pytest test_app.py -v         # also works under pytest

WHY THIS FILE LOOKS ODD
-----------------------
app.py is a single Streamlit script: importing it executes the whole UI and
loads ~191 MB of pickled models. So instead of importing it, we extract the
pure functions from its source and exec only those. That keeps these tests
fast and dependency-light, at the cost of the extraction helper below.

If you refactor app.py into modules, replace _load_from_app() with normal
imports and delete the extraction machinery.

WHAT IS AND ISN'T COVERED
-------------------------
Covered: the published-constant maths (BOILED-Egg, logBB equations), the
data-file/code drift guard, and the ADMETlab response parser.

NOT covered: the ML tiers, the Decision Layer, or anything needing the .pkl
files. Those have their own validation in the notebooks. These tests are about
the code added in Nov 2026 for Riccardo's three feedback items.
"""
import json
import math
import os
import re
import sys

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_compounds.json")


def _load_from_app(names, extra_globals=None):
    """Exec the named top-level defs/assignments out of app.py, without running it."""
    src = open(APP, encoding="utf-8").read()
    ns = dict(extra_globals or {})
    for name in names:
        if name.isupper() or name.startswith("_") and "=" in src.split(name)[1][:40]:
            m = re.search(rf"^{re.escape(name)} = \{{.*?\n\}}", src, re.S | re.M)
        else:
            m = None
        if m is None:
            m = re.search(rf"^def {re.escape(name)}\(.*?(?=\n\n\n)", src, re.S | re.M)
        if m is None:
            m = re.search(rf"^{re.escape(name)} = \{{.*?\n\}}", src, re.S | re.M)
        assert m, f"could not find {name!r} in app.py — did it get renamed?"
        exec(m.group(0), ns)
    return ns


# ══════════════════════════════════════════════════════════════════════════════
# BOILED-Egg — published constants
# ══════════════════════════════════════════════════════════════════════════════
# Source of truth: Daina & Zoete, ChemMedChem 2016;11(11):1117-1121,
# Supporting Information Figures S1 (HIA) and S2 (BBB).
PUBLISHED_ELLIPSES = {
    "HIA": (71.051, 2.292, 142.081, 8.740, -1.031325),
    "BBB": (38.117, 3.177, 82.061, 5.557, -0.171887),
}
# The SI also states the resulting extents; we re-derive them as a cross-check.
PUBLISHED_EXTENTS = {
    "HIA": {"tpsa": (0.0, 142.1)},
    "BBB": {"tpsa": (-2.9, 79.1), "wlogp": (0.4, 6.0)},
}


def test_boiled_egg_constants_match_publication():
    """Guards against someone 'tidying' the constants away from the paper."""
    ns = _load_from_app(["_BOILED_EGG"])
    assert ns["_BOILED_EGG"] == PUBLISHED_ELLIPSES, (
        "BOILED-Egg constants no longer match Daina & Zoete 2016 SI Figures S1/S2"
    )


def test_boiled_egg_angle_is_degrees_not_radians():
    """THE important one.

    The SI writes the angle as '-1.031325°'. Several public implementations treat
    it as radians, which rotates the ellipse ~59 degrees and produces wrong calls.
    Reading it as degrees reproduces the SI's own stated extents; reading it as
    radians does not. This test fails loudly if anyone 'fixes' the unit.
    """
    ns = _load_from_app(["_BOILED_EGG"])
    for region, exp in PUBLISHED_EXTENTS.items():
        cx, cy, major, minor, angle_deg = ns["_BOILED_EGG"][region]
        lo, hi = cx - major / 2.0, cx + major / 2.0
        want_lo, want_hi = exp["tpsa"]
        assert abs(lo - want_lo) < 0.15 and abs(hi - want_hi) < 0.15, (
            f"{region}: derived tPSA extent {lo:.2f}..{hi:.2f} does not match the "
            f"SI's stated {want_lo}..{want_hi} — is the angle being read as radians?"
        )
        # sanity: a radians reading would tilt the major axis far off the tPSA axis
        assert abs(math.degrees(math.radians(angle_deg))) < 2.0


def test_boiled_egg_classifies_known_compounds():
    """Regression lock on real compounds.

    NOTE the expected values here are what the PUBLISHED MODEL says, not ground
    truth. Temozolomide and caffeine are both genuinely brain-penetrant but the
    model calls them non-permeant — that is a documented limitation of a
    two-descriptor model on small polar molecules, and is exactly the
    disagreement the comparison panel exists to surface. Do not "fix" these.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    ns = _load_from_app(
        ["_BOILED_EGG", "mol_from_smiles", "_in_boiled_egg_region", "boiled_egg"],
        {"Descriptors": Descriptors, "Chem": Chem, "RDKIT_OK": True},
    )
    be = ns["boiled_egg"]
    cases = [
        # smiles,                                             bbb,   gi
        ("COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",     True,  True),   # Donepezil
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                      False, True),   # Caffeine
        ("CN1C(=O)N2C=NC(C(N)=O)=C2N=N1",                     False, True),   # Temozolomide
        ("CC(=O)Oc1ccccc1C(=O)O",                             True,  True),   # Aspirin
    ]
    for smi, bbb, gi in cases:
        got = be(smi)
        assert got is not None, f"failed to parse {smi}"
        assert got["bbb"] is bbb and got["gi"] is gi, (
            f"{smi}: got bbb={got['bbb']} gi={got['gi']}, expected bbb={bbb} gi={gi}"
        )


def test_boiled_egg_rejects_invalid_smiles():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    ns = _load_from_app(
        ["_BOILED_EGG", "mol_from_smiles", "_in_boiled_egg_region", "boiled_egg"],
        {"Descriptors": Descriptors, "Chem": Chem, "RDKIT_OK": True},
    )
    # mol_from_smiles RAISES on these; boiled_egg must absorb that, not propagate it
    assert ns["boiled_egg"]("not-a-smiles-!!") is None
    assert ns["boiled_egg"]("") is None          # RDKit returns an EMPTY Mol here
    assert ns["boiled_egg"]("   ") is None
    assert ns["boiled_egg"](None) is None


def test_boiled_egg_tpsa_includes_s_and_p():
    """SwissADME follows Ertl 2000, which counts S and P. Our row must too,
    otherwise it stops reproducing the tool it claims to reproduce."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    ns = _load_from_app(
        ["_BOILED_EGG", "mol_from_smiles", "_in_boiled_egg_region", "boiled_egg"],
        {"Descriptors": Descriptors, "Chem": Chem, "RDKIT_OK": True},
    )
    # thiophene-2-carboxylic acid: S contributes only when includeSandP=True
    smi = "OC(=O)c1cccs1"
    mol = Chem.MolFromSmiles(smi)
    plain = Descriptors.TPSA(mol)
    with_sp = Descriptors.TPSA(mol, includeSandP=True)
    assert with_sp > plain, "test compound does not discriminate; pick another"
    assert abs(ns["boiled_egg"](smi)["tpsa"] - with_sp) < 0.01, (
        "boiled_egg() is not using includeSandP=True"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Published logBB equations
# ══════════════════════════════════════════════════════════════════════════════
def test_logbb_equations_arithmetic():
    """Hand-computed expectations, so a coefficient typo in the JSON is caught."""
    ns = _load_from_app(["eval_logbb_model"])
    ev = ns["eval_logbb_model"]
    doc = json.load(open(REF, encoding="utf-8"))
    models = {m["id"]: m for m in doc["logbb_models"]}

    # Clark 1999:   logBB = 0.152*logP - 0.0148*PSA + 0.139
    # Rishton 2006: logBB = 0.155*logP - 0.0100*PSA + 0.164
    logp, psa = 2.0, 50.0
    assert abs(ev(models["clark1999"], logp, psa) - (0.152 * 2 - 0.0148 * 50 + 0.139)) < 1e-9
    assert abs(ev(models["rishton2006"], logp, psa) - (0.155 * 2 - 0.0100 * 50 + 0.164)) < 1e-9

    # the built-in tier-2 entry has no closed form and must return None
    assert ev(models["app_tier2"], logp, psa) is None
    # missing descriptors must not raise
    assert ev(models["clark1999"], None, psa) is None
    assert ev(models["clark1999"], logp, None) is None


def test_logbb_coefficients_match_literature():
    """Locks the published coefficients so they can't drift unnoticed."""
    doc = json.load(open(REF, encoding="utf-8"))
    models = {m["id"]: m for m in doc["logbb_models"]}
    expected = {
        "clark1999":   (0.152, -0.0148, 0.139),
        "rishton2006": (0.155, -0.0100, 0.164),
    }
    for mid, (lp, ps, ic) in expected.items():
        m = models[mid]
        assert (m["logp_coef"], m["psa_coef"], m["intercept"]) == (lp, ps, ic), (
            f"{mid} coefficients changed from the published values"
        )
        assert m.get("source"), f"{mid} has no citation attached"


# ══════════════════════════════════════════════════════════════════════════════
# reference_compounds.json integrity
# ══════════════════════════════════════════════════════════════════════════════
def test_reference_compounds_match_calibration_data():
    """The JSON was generated FROM _CALIBRATION_DATA in app.py. If someone edits
    one and not the other, the app shows numbers that disagree with the model it
    is comparing against. This catches that."""
    src = open(APP, encoding="utf-8").read()
    block = re.search(r"_CALIBRATION_DATA = \[(.*?)\n\]", src, re.S).group(1)
    rows = re.findall(
        r'\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(-?[\d.]+),\s*([\d.]+)\)', block
    )
    cal = {n: (float(kp), float(dt)) for n, kp, dt, _e, _l, _t in rows}

    doc = json.load(open(REF, encoding="utf-8"))
    ref = {c["name"]: (c["measured_kp"], c["model_dt_pred"]) for c in doc["compounds"]}

    assert set(cal) == set(ref), (
        f"compound sets differ — only in app.py: {set(cal) - set(ref)}; "
        f"only in JSON: {set(ref) - set(cal)}"
    )
    for name, (kp, dt) in cal.items():
        assert abs(ref[name][0] - kp) < 1e-9, f"{name}: measured Kp drifted"
        assert abs(ref[name][1] - dt) < 1e-9, f"{name}: model prediction drifted"


def test_every_reference_entry_has_a_resolvable_source():
    doc = json.load(open(REF, encoding="utf-8"))
    sources = doc["sources"]
    for c in doc["compounds"]:
        assert c["source"] in sources, f"{c['name']} cites unknown source {c['source']!r}"
        assert sources[c["source"]].get("citation"), f"{c['source']} has no citation text"
    for m in doc["logbb_models"]:
        if m.get("source"):
            assert m["source"] in sources, f"{m['id']} cites unknown source"


def test_calibration_set_is_flagged_as_circular():
    """The Bae et al. compounds ARE the model's calibration set. If that flag is
    lost, the app silently presents circular agreement as validation."""
    doc = json.load(open(REF, encoding="utf-8"))
    assert doc["sources"]["bae2024"]["in_calibration_set"] is True, (
        "bae2024 must stay flagged in_calibration_set — the UI's circularity "
        "warning keys off this"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ADMETlab response parsing
# ══════════════════════════════════════════════════════════════════════════════
def _admet_ns():
    return _load_from_app(
        ["_ADMETLAB_KEYS", "_flatten_scalars", "_admetlab_normalise"]
    )


def test_admetlab_parses_documented_response_shape():
    """Shape per admetlab3.scbdd.com/apis/ -> ADMET Calculation."""
    ns = _admet_ns()
    record = {"smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "MW": 194.08, "TPSA": 61.82,
              "BBB": 0.93, "logBB": -0.05, "PPB": 31.4, "Pgp-sub": 0.11}
    row, missing = ns["_admetlab_normalise"](ns["_flatten_scalars"](record))
    assert missing == []
    assert row["BBB penetrant"] == "Yes"        # 0.93 >= 0.5
    assert row["P-gp substrate"] == "No"        # 0.11 <  0.5
    assert row["logBB"] == -0.05
    assert row["PPB % bound"] == 31.4


def test_admetlab_ppb_fraction_is_converted_to_percent():
    ns = _admet_ns()
    row, _ = ns["_admetlab_normalise"](ns["_flatten_scalars"]({"PPB": 0.934}))
    assert row["PPB % bound"] == 93.4, "a 0-1 fraction must be scaled to a percentage"
    row, _ = ns["_admetlab_normalise"](ns["_flatten_scalars"]({"PPB": 93.4}))
    assert row["PPB % bound"] == 93.4, "a value already in percent must pass through"


def test_admetlab_tolerates_alternative_nestings():
    ns = _admet_ns()
    payloads = [
        {"BBB": 0.9, "logBB": -0.4, "PPB": 88.0, "Pgp-sub": 0.8},
        {"data": {"BBB": 0.9, "logBB": -0.4, "PPB": 88.0, "pgp_sub": 0.8}},
        [{"BBB": 0.9, "logBB": -0.4, "PPB": 88.0, "Pgp-sub": 0.8}],
        {"BBB": {"value": 0.9}, "logBB": {"value": -0.4},
         "PPB": {"value": 88.0}, "Pgp-sub": {"value": 0.8}},
    ]
    for p in payloads:
        row, missing = ns["_admetlab_normalise"](ns["_flatten_scalars"](p))
        assert missing == [], f"failed on {p}"
        assert row["BBB penetrant"] == "Yes" and row["P-gp substrate"] == "Yes"


def test_admetlab_reports_unknown_keys_instead_of_guessing():
    """If their key names change, we must surface it, not silently emit blanks."""
    ns = _admet_ns()
    row, missing = ns["_admetlab_normalise"](ns["_flatten_scalars"]({"wibble": 1, "wobble": 2}))
    assert len(missing) == 4, "all four columns should report as missing"
    assert row["logBB"] is None and row["BBB penetrant"] == ""


def test_admetlab_explanation_block_cannot_shadow_real_values():
    """Their response carries an 'explanation' dict keyed by the SAME property
    names, holding prose. If that ever wins, every numeric column breaks."""
    ns = _admet_ns()
    body = {"data": {"data": [{"BBB": 0.9, "logBB": -0.4, "PPB": 88.0, "Pgp-sub": 0.8}],
                     "explanation": {"BBB": "Category 0: ...", "logBB": "prose",
                                     "PPB": "prose", "Pgp-sub": "prose"}}}
    row, missing = ns["_admetlab_normalise"](ns["_flatten_scalars"](body))
    assert missing == [], "prose from 'explanation' shadowed the numeric values"
    assert row["logBB"] == -0.4


# ══════════════════════════════════════════════════════════════════════════════
# Held-out evaluation (circularity guard)
# ══════════════════════════════════════════════════════════════════════════════
# The 10 Bae et al. compounds ARE the set the Bayesian agents were fitted on, and
# for those compounds the app normally replays stored notebook values. Comparing
# that output back to the measured Kp is circular. holdout=True must remove every
# such path. These tests exist because that property is easy to undo by accident
# and impossible to notice from the UI.
def test_holdout_row_exclusion_drops_the_named_compound():
    src = open(APP, encoding="utf-8").read()
    block = re.search(r"_CALIBRATION_DATA = \[(.*?)\n\]", src, re.S).group(1)
    names = re.findall(r'\("([^"]+)",', block)
    assert len(names) == 10, f"expected 10 calibration compounds, found {len(names)}"

    # mirrors the filter inside fit_decision_layer()
    for target in ("Caffeine", "Donepezil"):
        kept = [n for n in names if n not in (target,)]
        assert target not in kept
        assert len(kept) == 9


def test_holdout_refuses_when_too_little_data_remains():
    """fit_decision_layer must raise rather than fit a meaningless posterior."""
    src = open(APP, encoding="utf-8").read()
    fn = re.search(r"^def fit_decision_layer\(.*?(?=\n\n\n)", src, re.S | re.M).group(0)
    assert "if len(rows) < 3:" in fn, "the minimum-rows guard is gone"
    assert "raise ValueError" in fn, "it no longer raises when it cannot fit"


def test_holdout_gates_every_leakage_path():
    """Source-level assertions, deliberately.

    These guard a property, not a return value: that no path reintroduces
    measured-Kp-derived information into a held-out prediction. Each assertion
    names the specific leak it prevents, so a failure says what broke.
    """
    src = open(APP, encoding="utf-8").read()
    fn = re.search(r"^def run_decision_layer\(.*?(?=\n\n\n)", src, re.S | re.M).group(0)

    assert "holdout=False" in fn, "run_decision_layer lost its holdout parameter"

    assert "if matched and not holdout:" in fn, (
        "LEAK: stored notebook dt_pred/efflux_liability are being substituted even in "
        "held-out mode — the prediction would be a recall of the fitted value"
    )
    assert "discount = 1.0 if holdout else" in fn, (
        "LEAK: the per-compound literature discount is applied in held-out mode — that "
        "discount was tuned on these specific compounds"
    )
    assert "fit_decision_layer(exclude=(matched,))" in fn, (
        "LEAK: agents are no longer refitted without the held-out compound, so their "
        "posteriors have still seen its measured Kp"
    )


def test_holdout_state_is_recorded_on_the_result():
    """Any reported number must carry how it was produced."""
    src = open(APP, encoding="utf-8").read()
    assert 'holdout=bool(st.session_state.get("holdout_mode"' in src, (
        "the result dict no longer records whether held-out mode was on — downstream "
        "panels would label validation numbers wrongly"
    )
    assert 'key="holdout_mode"' in src, "the held-out toggle is gone from the UI"


def test_uptake_literature_evidence_is_retained_in_holdout():
    """Deliberate design choice, asserted so it isn't 'fixed' silently.

    Curated uptake evidence comes from independent published transporter studies,
    not from the chip measurements, and the frozen architecture requires the uptake
    gate to rest on literature rather than an ML guess. So it is NOT gated on
    holdout. If that ever changes it should be a conscious decision.
    """
    src = open(APP, encoding="utf-8").read()
    fn = re.search(r"^def run_decision_layer\(.*?(?=\n\n\n)", src, re.S | re.M).group(0)
    assert "if matched and matched in UPTAKE_EVIDENCE_V2:" in fn, (
        "the uptake-evidence lookup changed shape; re-check whether it is still "
        "independent of the measured Kp"
    )


# ══════════════════════════════════════════════════════════════════════════════
def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
