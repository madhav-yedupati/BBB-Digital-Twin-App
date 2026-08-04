# Barrile Lab - CNS Drug Penetration Predictor
# Multi-tier GBM Drug Discovery Pipeline
# app.py  |  C:\Users\madha\Desktop\GBM\Model\app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import requests
import time
import matplotlib.pyplot as plt
from urllib.parse import quote

# ── RDKit ──────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, BRICS, QED
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CNS Drug Penetration Predictor | Barrile Lab",
    page_icon="🧠",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ABC classifier predicts one of 3 classes; efflux_liability = P(Substrate) + P(Both)
ABC_LABEL_FALLBACK = {0: "Inhibitor", 1: "Substrate", 2: "Both"}

# ── Physchem descriptor sets ────────────────────────────────────────────────
# ABC (rf_final_v4 / xgb_final_v4): order taken verbatim from model_meta_v4.pkl
# ("physchem_features": MW, LogP, TPSA, HBD, HBA, RotBonds) → 6 + 2048 Morgan = 2054.
ABC_PHYSCHEM_DESCS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]

# Tier 1/2/3 (bbb_tier1_consensus / bbb_tier2_stacked / tier3_ppb_model): these
# pickles were fit on plain numpy arrays, so no feature_names_in_ survived to
# recover the exact physchem set/order. n_features_in_ = 2056 = 8 + 2048 Morgan.
# Best-inferred 8 = the ABC 6 plus RingCount + NumAromaticRings, same order
# convention (physchem then Morgan) as the ABC model and the pre-existing
# featurize() in this file. NOT VERIFIED against the original training
# script -- confirm this against whatever trained these three .pkl files
# before trusting Tier 1/2/3 outputs for the manuscript.
TIER123_PHYSCHEM_DESCS = ABC_PHYSCHEM_DESCS + ["RingCount", "NumAromaticRings"]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    models = {}
    paths = {
        "tier1":    os.path.join(MODEL_DIR, "bbb_tier1_consensus.pkl"),
        "tier2":    os.path.join(MODEL_DIR, "bbb_tier2_stacked.pkl"),
        "tier3":    os.path.join(MODEL_DIR, "tier3_ppb_model.pkl"),
        "abc_rf":   os.path.join(MODEL_DIR, "rf_final_v4.pkl"),
        "abc_xgb":  os.path.join(MODEL_DIR, "xgb_final_v4.pkl"),
        "abc_meta": os.path.join(MODEL_DIR, "model_meta_v4.pkl"),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        else:
            models[key] = None
    return models

# ══════════════════════════════════════════════════════════════════════════════
# PUBCHEM INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class PubChemNotFound(Exception):
    """Compound name genuinely doesn't resolve on PubChem. Safe to cache."""
    pass


class PubChemLookupError(Exception):
    """Network/timeout/unexpected-response failure. NOT safe to cache -- a
    transient blip shouldn't poison every future lookup for the cache TTL."""
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def _pubchem_lookup_cached(name: str):
    """Only ever called (and cached) for genuine successes or genuine
    not-found results. Network errors raise PubChemLookupError, which
    st.cache_data does NOT cache (only successful returns are cached)."""
    # No custom User-Agent: a bare requests.get() with the default UA was
    # confirmed reachable from this network; a prior custom UA string here
    # may have been getting flagged/blocked by a corporate content filter.
    headers = {}

    try:
        cid_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(name)}/cids/JSON"
        )
        r = requests.get(cid_url, headers=headers, timeout=10)
    except requests.RequestException as e:
        raise PubChemLookupError(f"network error resolving CID: {e}")

    if r.status_code != 200:
        raise PubChemLookupError(
            f"CID lookup returned HTTP {r.status_code} | "
            f"Content-Type: {r.headers.get('Content-Type', 'unknown')} | "
            f"Server: {r.headers.get('Server', 'unknown')} | "
            f"final URL: {r.url} | "
            f"body: {r.text[:300]!r}"
        )

    try:
        cid = r.json()["IdentifierList"]["CID"][0]
    except (KeyError, IndexError, ValueError) as e:
        raise PubChemLookupError(f"unexpected CID response shape: {e}")

    try:
        prop_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/ConnectivitySMILES,CanonicalSMILES,IUPACName/JSON"
        )
        r2 = requests.get(prop_url, headers=headers, timeout=10)
    except requests.RequestException as e:
        raise PubChemLookupError(f"network error fetching properties: {e}")

    if r2.status_code != 200:
        raise PubChemLookupError(
            f"property lookup returned HTTP {r2.status_code} | "
            f"Content-Type: {r2.headers.get('Content-Type', 'unknown')} | "
            f"body: {r2.text[:300]!r}"
        )

    try:
        props = r2.json()["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, ValueError) as e:
        raise PubChemLookupError(f"unexpected property response shape: {e}")

    smiles = props.get("ConnectivitySMILES") or props.get("CanonicalSMILES") or ""
    if not smiles:
        raise PubChemLookupError(
            f"got a valid PubChem response but no SMILES property in it -- "
            f"available keys: {list(props.keys())}"
        )
    return {
        "smiles":     smiles,
        "iupac_name": props.get("IUPACName", ""),
        "cid":        cid,
    }


def pubchem_lookup(drug_name: str):
    """
    Fetch CanonicalSMILES + IUPACName + CID from PubChem PUG REST.
    Returns (result_dict_or_None, error_message_or_None).
    result=None, error=None  -> genuinely not found on PubChem
    result=None, error=str   -> network/transient failure -- NOT cached, retry-able
    result=dict, error=None  -> success
    """
    name = drug_name.strip()
    if not name:
        return None, None
    try:
        return _pubchem_lookup_cached(name), None
    except PubChemNotFound:
        return None, None
    except PubChemLookupError as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURIZATION
# ══════════════════════════════════════════════════════════════════════════════
# Full descriptor panel shown in the UI "Physicochemical Properties" expander
# (superset of what any single model consumes -- display only).
PHYSCHEM_DESCS = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "HeavyAtomCount", "FractionCSP3",
]

def mol_from_smiles(smiles: str):
    if not RDKIT_OK:
        raise ValueError("RDKit not installed.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def _features_for(mol, physchem_list, n_bits=2048, radius=2):
    fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))
    phys = [getattr(Descriptors, d)(mol) for d in physchem_list]
    return np.array(phys + fp, dtype=float).reshape(1, -1)


def featurize(smiles: str):
    """Returns (tier123_feats, abc_feats, mol) or raises ValueError.
    Tier 1/2/3 and the ABC classifier were trained with different physchem
    panels (2056 vs 2054 features) -- see TIER123_PHYSCHEM_DESCS / ABC_PHYSCHEM_DESCS."""
    mol = mol_from_smiles(smiles)
    tier123_feats = _features_for(mol, TIER123_PHYSCHEM_DESCS)
    abc_feats     = _features_for(mol, ABC_PHYSCHEM_DESCS)
    return tier123_feats, abc_feats, mol


def get_physchem(mol):
    return {d: getattr(Descriptors, d)(mol) for d in PHYSCHEM_DESCS}


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def run_tier1(models, feats):
    m = models["tier1"]
    if m is None:
        return None, None
    proba = m.predict_proba(feats)[0]
    pred  = int(m.predict(feats)[0])
    conf  = proba[pred]
    return pred, conf


def run_tier2(models, feats):
    m = models["tier2"]
    if m is None:
        return None
    return float(m.predict(feats)[0])


def run_tier3(models, feats):
    m = models["tier3"]
    if m is None:
        return None
    fu = float(m.predict(feats)[0])
    fu = max(0.001, min(1.0, fu))        # clamp to [0.001, 1.0]
    return fu


def run_abc(models, abc_feats):
    """ABC transporter classifier: rf_final_v4 + xgb_final_v4 soft-vote ensemble
    over 3 classes (Inhibitor / Substrate / Both). Returns dict with the
    predicted label, per-class probabilities, and efflux_liability =
    P(Substrate) + P(Both) -- the scalar the Decision Layer's efflux gate uses."""
    rf, xgb = models.get("abc_rf"), models.get("abc_xgb")
    if rf is None or xgb is None:
        return None
    proba = (rf.predict_proba(abc_feats)[0] + xgb.predict_proba(abc_feats)[0]) / 2.0
    meta = models.get("abc_meta") or {}
    inv_label = meta.get("inv_label", ABC_LABEL_FALLBACK)
    pred_idx = int(np.argmax(proba))
    label = inv_label.get(pred_idx, inv_label.get(str(pred_idx), str(pred_idx)))
    efflux_liability = float(proba[1] + proba[2]) if len(proba) >= 3 else float(proba[-1])
    return {
        "label": label,
        "proba": {inv_label.get(i, inv_label.get(str(i), str(i))): float(p) for i, p in enumerate(proba)},
        "efflux_liability": efflux_liability,
    }


def digital_twin_papp(logBB: float, fu_plasma: float, efflux_ratio: float = 1.0,
                      alpha: float = 1.99, beta: float = 0.357) -> float:
    """
    Papp,free = 10^((logBB - beta) / alpha)
    Papp,plasma = Papp,free * fu_plasma / efflux_ratio
    Returns Papp in units of ×10⁻⁶ cm/s (the standard Caco-2/permeability
    assay convention -- NOT literal nm/s; that would need a ×1e7 factor).
    """
    papp_free   = 10 ** ((logBB - beta) / alpha)
    papp_plasma = papp_free * fu_plasma / efflux_ratio
    return papp_plasma * 1e6   # convert cm/s -> ×10⁻⁶ cm/s display convention


# ══════════════════════════════════════════════════════════════════════════════
# DECISION LAYER  (frozen architecture: Digital Twin -> Decision Layer ->
# Biological Agents [Efflux, Uptake] -> Decision Engine -> Final Prediction +
# Biological Explanation. Ported from bbb_digital_twin_v21.ipynb.)
# ══════════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass, field
from typing import Optional, Dict

# ── 10-compound calibration set (Bae et al. 2024 / Yang, Shin, Jeong, Bae et
#    al., Pharmaceutics 2024) used to fit the Bayesian TransportAgent /
#    UptakeAgent posteriors. "dt_pred" is the Digital Twin's own predicted
#    brain:plasma Kp for that compound; "efflux_liability" is the ABC
#    classifier's substrate-probability score at the time of calibration.
_CALIBRATION_DATA = [
    # compound,        actual Kp, dt_pred,  efflux_liability, LogP,    TPSA
    ("Donepezil",      0.4794, 0.134756, 0.106323,  4.3611,  38.77),
    ("Loperamide",     0.0552, 0.156663, 0.915715,  5.0880,  43.78),
    ("Caffeine",       0.4349, 0.533124, 0.669177, -1.0293,  61.82),
    ("Nefazodone",     0.0953, 0.176294, 0.087304,  3.5519,  55.53),
    ("Carbamazepine",  0.2823, 0.204523, 0.433281,  3.3872,  46.33),
    ("Simvastatin",    0.1570, 0.098583, 0.089984,  4.5856,  72.83),
    ("Vincristine",    0.0821, 0.036345, 0.996339,  3.5175, 171.17),
    ("Cetirizine",     0.0771, 0.047726, 0.261203,  3.1482,  53.01),
    ("LP533401",       0.0170, 0.034115, 0.611433,  5.1687, 124.35),
    ("Desipramine",    0.3148, 0.331002, 0.324492,  3.5328,  15.27),
]
_LITERATURE_DISCOUNT = {row[0]: 1.0 for row in _CALIBRATION_DATA}
_LITERATURE_DISCOUNT["Carbamazepine"] = 0.4  # disputed P-gp status (Owen et al. 2001 KO-mouse study)

# Curated literature evidence for brain-uptake transporters -- gate requires a
# human-sourced (evidence_p, transporter, note), never an ML guess.
UPTAKE_EVIDENCE_V2 = {
    "Donepezil":   (1.00, "OCT/OCTN1, choline transport system",
                     "Direct in vivo rat brain kinetics: dose-dependent, saturable apparent "
                     "brain uptake clearance (CLapp,br) measured at the BBB itself "
                     "(Kitamura et al. 2009, J Pharm Sci)."),
    "Vincristine": (0.80, "OATP1B1/1B3/2B1, OATP1A2, MATE1",
                     "Dedicated 2023 mechanistic study of OATP-mediated neuronal uptake -- "
                     "CNS-relevant but characterized at the neuronal level, not the BBB "
                     "endothelium specifically."),
    "Simvastatin": (0.60, "OATP1B1 (SLCO1B1)",
                     "OATP1B1 substrate status is well-confirmed, but the evidence base is "
                     "hepatic (liver uptake), not brain-specific."),
    "Cetirizine":  (0.50, "OAT4",
                     "OAT4 uptake confirmed via levocetirizine study, but explicitly in "
                     "kidney/placenta tissue -- source notes it is not brain-confirmed."),
}
EFFLUX_KD, EFFLUX_N, EFFLUX_THRESHOLD = 0.1, 2.0, 0.3
UPTAKE_EVIDENCE_THRESHOLD = 0.5


@dataclass
class DigitalTwin:
    compound: str
    predicted_kp: float
    experimental_kp: Optional[float] = None
    confidence: Optional[float] = None
    @property
    def log_kp(self):
        return np.log(self.predicted_kp)


@dataclass
class PhysiologicalState:
    name: str
    efflux_kd: Optional[float] = None
    efflux_n: Optional[float] = None
    efflux_threshold: Optional[float] = None
    passive_kp_multiplier: float = 1.0
    uptake_overrides: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


PHYSIOLOGICAL_STATES = {"healthy": PhysiologicalState(name="healthy", notes="Baseline intact-BBB state.")}
def resolve_state(s):
    return s if isinstance(s, PhysiologicalState) else PHYSIOLOGICAL_STATES[s]


def combine_transporter_evidence(pgp=None, bcrp=None, mrp=None):
    probs = [p for p in (pgp, bcrp, mrp) if p is not None]
    if not probs:
        return None
    surv = 1.0
    for p in probs:
        surv *= (1.0 - p)
    return 1.0 - surv


def dominant_mechanism(pgp, bcrp, mrp):
    named = {k: v for k, v in {"P-glycoprotein": pgp, "BCRP": bcrp, "MRP": mrp}.items() if v is not None}
    return max(named, key=named.get) if named else "unspecified"


class TransportAgent:
    """Bayesian grid-posterior agent: log-space multiplicative efflux correction."""
    def __init__(self):
        self.w_grid = np.linspace(0.0, 0.999, 150)
        self.kappa_grid = np.linspace(0.001, 4.0, 150)
        self.sigma_grid = np.linspace(0.05, 2.0, 90)
        self.post = None

    def fit(self, tw, ev):
        log_dt = np.array([t.log_kp for t in tw]); log_a = np.log(np.array([t.experimental_kp for t in tw]))
        p = np.asarray(ev); nt = len(tw)
        W, K, S = np.meshgrid(self.w_grid, self.kappa_grid, self.sigma_grid, indexing="ij")
        log_prior = (-0.5 * (K / 1.5) ** 2) + (-np.log(1 + (S / 0.5) ** 2))
        log_lik = np.zeros_like(W)
        for i in range(nt):
            pred_i = log_dt[i] + K * np.log(np.clip(1 - W * p[i], 1e-12, None))
            log_lik += -0.5 * ((log_a[i] - pred_i) / S) ** 2 - np.log(S)
        lp = log_lik + log_prior; lp -= lp.max(); self.post = np.exp(lp); self.post /= self.post.sum()
        return self

    def evaluate(self, dt, p):
        W = self.w_grid[:, None, None]; K = self.kappa_grid[None, :, None]
        delta = K * np.log(np.clip(1 - W * p, 1e-12, None))
        return np.sum(np.broadcast_to(delta, self.post.shape) * self.post)


class UptakeAgent:
    """Bayesian grid-posterior agent: log-space multiplicative uptake correction (opposite sign of efflux)."""
    def __init__(self):
        self.w_grid = np.linspace(0.0, 0.999, 150)
        self.kappa_grid = np.linspace(0.001, 4.0, 150)
        self.sigma_grid = np.linspace(0.05, 2.0, 90)
        self.post = None

    def fit(self, tw, ev):
        log_dt = np.array([t.log_kp for t in tw]); log_a = np.log(np.array([t.experimental_kp for t in tw]))
        p = np.asarray(ev); nt = len(tw)
        W, K, S = np.meshgrid(self.w_grid, self.kappa_grid, self.sigma_grid, indexing="ij")
        log_prior = (-0.5 * (K / 1.5) ** 2) + (-np.log(1 + (S / 0.5) ** 2))
        log_lik = np.zeros_like(W)
        for i in range(nt):
            pred_i = log_dt[i] + K * np.log(np.clip(1 + W * p[i], 1e-12, None))
            log_lik += -0.5 * ((log_a[i] - pred_i) / S) ** 2 - np.log(S)
        lp = log_lik + log_prior; lp -= lp.max(); self.post = np.exp(lp); self.post /= self.post.sum()
        return self

    def evaluate(self, dt, p):
        W = self.w_grid[:, None, None]; K = self.kappa_grid[None, :, None]
        delta = K * np.log(np.clip(1 + W * p, 1e-12, None))
        return np.sum(np.broadcast_to(delta, self.post.shape) * self.post)


@dataclass
class MechanismDecision:
    mechanism_name: str
    intervention_required: bool
    selected_mechanism: str
    confidence: float
    expected_direction: str
    explanation: str


class DecisionLayerV2:
    """Returns the full 5-field structured decision (per mechanism): was
    intervention required, mechanism selected, confidence, expected direction,
    and a short natural-language explanation -- what a user or reviewer
    actually reads to understand *why*, not just the corrected number."""
    def __init__(self, efflux_kd=EFFLUX_KD, efflux_n=EFFLUX_N, efflux_threshold=EFFLUX_THRESHOLD,
                 uptake_threshold=UPTAKE_EVIDENCE_THRESHOLD):
        self.efflux_kd = efflux_kd
        self.efflux_n = efflux_n
        self.efflux_threshold = efflux_threshold
        self.uptake_threshold = uptake_threshold

    def decide(self, dt, pgp=None, bcrp=None, mrp=None, uptake_evidence_p=0.0,
               uptake_transporter=None, uptake_evidence_note="", physiological_state="healthy"):
        state = resolve_state(physiological_state)
        kd = state.efflux_kd or self.efflux_kd
        nn = state.efflux_n or self.efflux_n
        threshold = state.efflux_threshold or self.efflux_threshold
        effective_kp = dt.predicted_kp * state.passive_kp_multiplier
        rel_passive = effective_kp ** nn / (effective_kp ** nn + kd ** nn)
        efflux_p = combine_transporter_evidence(pgp, bcrp, mrp)

        if rel_passive < threshold:
            efflux_decision = MechanismDecision(
                mechanism_name="efflux", intervention_required=False, selected_mechanism="none",
                confidence=float(np.clip(1 - rel_passive / threshold, 0, 1)), expected_direction="none",
                explanation=(f"Passive permeability (rel_passive={rel_passive:.3f}) is already below "
                             f"the gating threshold ({threshold}); active efflux would have negligible "
                             f"marginal effect regardless of transporter evidence."),
            )
        elif efflux_p is None or efflux_p < threshold:
            efflux_decision = MechanismDecision(
                mechanism_name="efflux", intervention_required=False, selected_mechanism="none",
                confidence=float(np.clip(1 - (efflux_p or 0.0) / threshold, 0, 1)), expected_direction="none",
                explanation=(f"Passive permeability is high enough for an efflux correction to matter "
                             f"(rel_passive={rel_passive:.3f}), but transporter evidence (p={efflux_p}) "
                             f"does not clear the gating threshold ({threshold})."),
            )
        else:
            mech = dominant_mechanism(pgp, bcrp, mrp)
            efflux_decision = MechanismDecision(
                mechanism_name="efflux", intervention_required=True, selected_mechanism=mech,
                confidence=float(np.clip(efflux_p, 0, 1)), expected_direction="decrease",
                explanation=(f"Passive permeability is high (rel_passive={rel_passive:.3f}) and {mech} "
                             f"transporter evidence (p={efflux_p:.3f}) clears the gating threshold "
                             f"({threshold}); active efflux is expected to reduce brain Kp."),
            )

        gate_p = state.uptake_overrides.get(dt.compound, uptake_evidence_p)
        if gate_p is not None and gate_p >= self.uptake_threshold:
            uptake_decision = MechanismDecision(
                mechanism_name="uptake", intervention_required=True,
                selected_mechanism=uptake_transporter or "unspecified transporter",
                confidence=float(np.clip(gate_p, 0, 1)), expected_direction="increase",
                explanation=(uptake_evidence_note or
                             f"Uptake evidence (p={gate_p:.2f}) clears the gating threshold "
                             f"({self.uptake_threshold}); active uptake is expected to increase brain Kp."),
            )
        else:
            gp = gate_p if gate_p is not None else 0.0
            uptake_decision = MechanismDecision(
                mechanism_name="uptake", intervention_required=False, selected_mechanism="none",
                confidence=float(np.clip(1 - gp / self.uptake_threshold, 0, 1)), expected_direction="none",
                explanation=(f"Uptake evidence (p={gp:.2f}) does not clear the gating threshold "
                             f"({self.uptake_threshold}); no dedicated brain uptake transporter evidence "
                             f"supports an intervention for this compound."),
            )

        return {"rel_passive": rel_passive, "efflux": efflux_decision, "uptake": uptake_decision}


@st.cache_resource
def fit_decision_layer():
    """Fits TransportAgent/UptakeAgent once on the 10-compound calibration set
    (cached for the life of the app process, not re-fit per request)."""
    twins = [DigitalTwin(c, dt_pred, actual) for c, actual, dt_pred, *_ in _CALIBRATION_DATA]
    efflux_adj = np.array([
        eff * _LITERATURE_DISCOUNT.get(c, 1.0) for c, _, _, eff, *_ in _CALIBRATION_DATA
    ])
    uptake_p = np.array([UPTAKE_EVIDENCE_V2.get(c, (0.0,))[0] for c, *_ in _CALIBRATION_DATA])
    ta = TransportAgent().fit(twins, efflux_adj)
    ua = UptakeAgent().fit(twins, uptake_p)
    dl = DecisionLayerV2()
    return ta, ua, dl


def lookup_calibration_compound(name: str):
    """Case-insensitive match against the 10 validated calibration compounds."""
    if not name:
        return None
    name_l = name.strip().lower()
    for c, *_ in _CALIBRATION_DATA:
        if c.lower() == name_l:
            return c
    return None


_CALIBRATION_BY_NAME = {row[0]: row for row in _CALIBRATION_DATA}


def run_decision_layer(compound_name, predicted_kp, efflux_liability,
                        uptake_evidence_p=0.0, uptake_transporter=None, uptake_note=""):
    """Runs a compound through the frozen Decision Layer -> Biological Agents ->
    Decision Engine chain. Returns (final_kp, decision_dict, matched_name). If
    the compound matches one of the 10 validated calibration compounds, the
    exact published dt_pred/efflux_liability from the notebook table are used
    (rather than a fresh, slightly-drifted live re-inference) so the validated
    case studies reproduce exactly, and the curated literature discount /
    uptake evidence are applied automatically."""
    ta, ua, dl = fit_decision_layer()
    matched = lookup_calibration_compound(compound_name)

    if matched:
        _, _actual, cal_dt_pred, cal_efflux, *_ = _CALIBRATION_BY_NAME[matched]
        predicted_kp = cal_dt_pred
        efflux_liability = cal_efflux

    discount = _LITERATURE_DISCOUNT.get(matched, 1.0)
    efflux_adj = efflux_liability * discount

    if matched and matched in UPTAKE_EVIDENCE_V2:
        p, transporter, note = UPTAKE_EVIDENCE_V2[matched]
        uptake_evidence_p, uptake_transporter, uptake_note = p, transporter, note

    dt = DigitalTwin(compound=compound_name or "compound", predicted_kp=predicted_kp)
    decision = dl.decide(
        dt, pgp=efflux_adj, uptake_evidence_p=uptake_evidence_p,
        uptake_transporter=uptake_transporter, uptake_evidence_note=uptake_note,
    )
    log_kp = dt.log_kp
    if decision["efflux"].intervention_required:
        log_kp += ta.evaluate(dt, efflux_adj)
    if decision["uptake"].intervention_required:
        log_kp += ua.evaluate(dt, uptake_evidence_p)
    final_kp = float(np.exp(log_kp))
    return final_kp, decision, matched, predicted_kp


def predict_one_compound(models, name_val, smiles_val, pubchem_error=None,
                          uptake_evidence_p=0.0, uptake_transporter=None, uptake_note=""):
    """Runs the full Tier1->Tier2->Tier3->ABC->Decision Layer pipeline for one
    compound and returns a flat dict of results, matching the batch/validation
    table schema. Shared by the Batch Screening and Validation tabs."""
    rec = {"name": name_val or smiles_val[:20], "smiles": smiles_val}

    if not smiles_val:
        note = (f"PubChem lookup failed (not cached, retry-able): {pubchem_error}"
                if pubchem_error else "SMILES not found on PubChem")
        rec.update({
            "bbb_pred": "ERROR", "bbb_conf": None,
            "logBB": None, "fu_plasma": None, "ppb_pct": None,
            "dt_kp": None, "final_kp": None,
            "efflux_intervention": None, "efflux_mechanism": None,
            "efflux_confidence": None, "efflux_direction": None, "efflux_explanation": None,
            "uptake_intervention": None, "uptake_mechanism": None,
            "uptake_confidence": None, "uptake_direction": None, "uptake_explanation": None,
            "matched_calibration_compound": None,
            "note": note,
        })
        return rec

    try:
        tier123_feats, abc_feats, mol = featurize(smiles_val)
        t1_pred, t1_conf = run_tier1(models, tier123_feats)
        t2_logbb         = run_tier2(models, tier123_feats)
        t3_fu            = run_tier3(models, tier123_feats)
        abc_result       = run_abc(models, abc_feats)

        rec.update({
            "bbb_pred":  ("BBB+" if t1_pred == 1 else "BBB−") if t1_pred is not None else None,
            "bbb_conf":  round(t1_conf, 3) if t1_conf is not None else None,
            "logBB":     round(t2_logbb, 3) if t2_logbb is not None else None,
            "fu_plasma": round(t3_fu, 3) if t3_fu is not None else None,
            "ppb_pct":   round((1 - t3_fu) * 100, 1) if t3_fu is not None else None,
            "efflux_liability": round(abc_result["efflux_liability"], 3) if abc_result else None,
            "abc_class": abc_result["label"] if abc_result else None,
        })

        if t2_logbb is not None and abc_result is not None:
            dt_kp = 10 ** t2_logbb
            final_kp, decision, matched, dt_kp = run_decision_layer(
                rec["name"], dt_kp, abc_result["efflux_liability"],
                uptake_evidence_p=uptake_evidence_p,
                uptake_transporter=uptake_transporter,
                uptake_note=uptake_note,
            )
            rec.update({
                "dt_kp":   round(dt_kp, 4),
                "final_kp": round(final_kp, 4),
                "efflux_intervention": decision["efflux"].intervention_required,
                "efflux_mechanism":    decision["efflux"].selected_mechanism,
                "efflux_confidence":   round(decision["efflux"].confidence, 3),
                "efflux_direction":    decision["efflux"].expected_direction,
                "efflux_explanation":  decision["efflux"].explanation,
                "uptake_intervention": decision["uptake"].intervention_required,
                "uptake_mechanism":    decision["uptake"].selected_mechanism,
                "uptake_confidence":   round(decision["uptake"].confidence, 3),
                "uptake_direction":    decision["uptake"].expected_direction,
                "uptake_explanation":  decision["uptake"].explanation,
                "matched_calibration_compound": matched,
            })
        else:
            rec.update({
                "dt_kp": None, "final_kp": None,
                "efflux_intervention": None, "efflux_mechanism": None,
                "efflux_confidence": None, "efflux_direction": None, "efflux_explanation": None,
                "uptake_intervention": None, "uptake_mechanism": None,
                "uptake_confidence": None, "uptake_direction": None, "uptake_explanation": None,
                "matched_calibration_compound": None,
            })
        rec["note"] = ""
    except ValueError as e:
        rec.update({
            "bbb_pred": "ERROR", "bbb_conf": None,
            "logBB": None, "fu_plasma": None, "ppb_pct": None,
            "dt_kp": None, "final_kp": None,
            "efflux_intervention": None, "efflux_mechanism": None,
            "efflux_confidence": None, "efflux_direction": None, "efflux_explanation": None,
            "uptake_intervention": None, "uptake_mechanism": None,
            "uptake_confidence": None, "uptake_direction": None, "uptake_explanation": None,
            "matched_calibration_compound": None,
            "note": str(e),
        })
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# USER-CALIBRATED SURROGATE ("My Chip" digital twin)
#
# Everything above (Tier 1/2/3, ABC, Decision Layer) is the shared BASELINE
# model, trained once on published data (Bae et al. 2024). It is the same for
# every user and never sees anyone's own experimental results.
#
# This section lets an individual lab calibrate a personal correction layer
# on top of that frozen baseline, using a handful of compounds they've
# actually run on their own organ-on-chip. It never modifies the baseline
# model/pkl files -- it only learns a small mapping
# (baseline final_kp) -> (this chip's empirical Kp) and applies it forward to
# new compounds. This is fit fresh per user session (st.session_state), not
# cached at the process level like fit_decision_layer().
# ══════════════════════════════════════════════════════════════════════════════
def normalize_calibration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Case/whitespace/alias-tolerant column matching for the chip
    calibration CSV upload. Renames the first matching column found for
    each target to the exact name the rest of the app expects, so
    'Kp', 'kp_empirical', 'Empirical Kp', etc. all work without the user
    having to rename anything."""
    aliases = {
        "name":         ["name", "compound", "compound_name", "drug", "drug_name"],
        "smiles":       ["smiles", "smile", "canonical_smiles"],
        "empirical_kp": ["empirical_kp", "empirical kp", "kp", "kp_empirical",
                          "measured_kp", "chip_kp", "observed_kp", "kp_measured"],
    }
    lower_map = {c.strip().lower(): c for c in df.columns}
    rename_map = {}
    for target, candidates in aliases.items():
        for cand in candidates:
            if cand in lower_map and lower_map[cand] != target:
                rename_map[lower_map[cand]] = target
                break
            elif cand in lower_map:
                break
    return df.rename(columns=rename_map)


def fit_user_surrogate(cal_df: pd.DataFrame, baseline_col="final_kp", empirical_col="empirical_kp"):
    """Fits a chip-specific calibration from a small table of
    (baseline model prediction, this user's own measured Kp) pairs.

    Works in log-space (Kp values are strictly positive and roughly
    log-normal). Method scales with how much data the user actually has:
      n < 2  -> not enough to fit anything; caller should block.
      n == 2 -> exact 2-point line, no residual/uncertainty estimate.
      3 <= n < 6 -> ordinary least-squares line + residual-based interval.
      n >= 6 and scikit-learn available -> Gaussian Process on the residuals,
        which gives a smoothly shrinking uncertainty band and doesn't force
        a single global slope on a handful of points.
    Falls back one tier down automatically if scikit-learn isn't installed.
    """
    valid = cal_df.dropna(subset=[baseline_col, empirical_col])
    valid = valid[(valid[baseline_col] > 0) & (valid[empirical_col] > 0)]
    n = len(valid)
    if n < 2:
        return {"method": "none", "n": n,
                "error": "Need at least 2 compounds with valid baseline and empirical Kp values."}

    x = np.log(valid[baseline_col].values.astype(float))
    y = np.log(valid[empirical_col].values.astype(float))

    if n >= 6:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 2.0))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
            gp.fit(x.reshape(-1, 1), y)
            y_fit, _ = gp.predict(x.reshape(-1, 1), return_std=True)
            ss_res = float(np.sum((y - y_fit) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
            return {"method": "gp", "model": gp, "n": n, "r2": r2, "x_train": x, "y_train": y}
        except ImportError:
            pass  # fall through to linear

    if n >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        resid = y - y_fit
        resid_std = float(np.std(resid, ddof=2)) if n > 3 else float(np.std(resid, ddof=1))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        return {"method": "linear", "slope": float(slope), "intercept": float(intercept),
                "resid_std": resid_std, "n": n, "r2": r2, "x_train": x, "y_train": y}

    # n == 2: exact fit, no residual/uncertainty available
    slope, intercept = np.polyfit(x, y, 1)
    return {"method": "linear", "slope": float(slope), "intercept": float(intercept),
            "resid_std": None, "n": n, "r2": None, "x_train": x, "y_train": y}


def apply_user_surrogate(calib: dict, baseline_kp: float):
    """Applies a fitted chip-calibration to one baseline Kp prediction.
    Returns (corrected_kp, lower_95, upper_95) -- bounds are None when the
    fit has no usable uncertainty estimate (e.g. an exact 2-point line)."""
    if calib is None or calib.get("method") in (None, "none") or baseline_kp is None or baseline_kp <= 0:
        return None, None, None

    x = np.log(baseline_kp)

    if calib["method"] == "gp":
        mean, std = calib["model"].predict(np.array([[x]]), return_std=True)
        y, s = float(mean[0]), float(std[0])
        return float(np.exp(y)), float(np.exp(y - 1.96 * s)), float(np.exp(y + 1.96 * s))

    if calib["method"] == "linear":
        y = calib["slope"] * x + calib["intercept"]
        if calib.get("resid_std"):
            s = calib["resid_std"]
            return float(np.exp(y)), float(np.exp(y - 1.96 * s)), float(np.exp(y + 1.96 * s))
        return float(np.exp(y)), None, None

    return None, None, None


def surrogate_calibration_summary(calib: dict) -> str:
    """One-line, plain-language description of the fitted chip calibration."""
    if calib is None or calib.get("method") in (None, "none"):
        return "No chip calibration fitted yet."
    n = calib["n"]
    if calib["method"] == "gp":
        r2 = calib.get("r2")
        r2_txt = f", in-sample fit R²={r2:.3f}" if r2 is not None else ""
        return f"Gaussian Process surrogate fit on {n} of your compounds{r2_txt}. Uncertainty bands widen for new compounds far from your calibration set."
    if calib["method"] == "linear":
        slope, intercept = calib["slope"], calib["intercept"]
        r2 = calib.get("r2")
        r2_txt = f", R²={r2:.3f}" if r2 is not None else " (exact fit -- only 2 points, no uncertainty estimate)"
        return f"Log-linear correction fit on {n} of your compounds{r2_txt}: log(your Kp) ≈ {slope:.2f} × log(baseline Kp) + {intercept:.2f}."
    return "Calibration could not be fit."


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE GENERATION (BRICS combinatorial fragment recombination)
# ══════════════════════════════════════════════════════════════════════════════
# NOT a trained deep generative model (no GPU/pretraining corpus available in
# this environment for a REINVENT/VAE-style RL-fine-tuned SMILES generator).
# This is a legitimate, well-established alternative: BRICS decomposes known
# drug-like molecules into chemically sensible fragments at retrosynthetically
# plausible bond positions, then recombines them into new valid structures.
# The pipeline itself (Tier1/2/3 -> ABC -> Decision Layer) is the fitness
# function scoring what comes out. Candidates are hypotheses to filter and
# inspect further -- not validated for synthesizability, novelty vs. patents,
# or purchasability.
GEN_SEED_LIBRARY = {
    "Diazepam":      "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "Haloperidol":   "C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F",
    "Phenytoin":     "C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3",
    "Fluoxetine":    "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
    "Nicotine":      "CN1CCC[C@H]1C2=CN=CC=C2",
    "Caffeine":      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Temozolomide":  "CN1C(=O)N2C=NC(=C2N=N1)C(=O)N",
    "Theophylline":  "CN1C2=NC=NC2C(=O)N(C1=O)C",
    "Donepezil":     "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC",
    "Quinidine":     "O[C@H]([C@]1([H])[C@@H](CC2)CC(C=C)[C@@H]2C1)c3c4cc(OC)ccc4ncc3",
    "Verapamil":     "CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC",
    "Imatinib":      "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Gefitinib":     "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "Riluzole":      "C1=CC2=C(C=C1OC(F)(F)F)SC(=N2)N",
    "Carbamazepine": "C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N",
}


def build_fragment_pool(seed_names, custom_smiles=None):
    """BRICS-decomposes the chosen seed molecules (+ optional custom SMILES)
    into a pool of fragment Mol objects for recombination."""
    frag_smiles = set()
    seeds_used = [GEN_SEED_LIBRARY[n] for n in seed_names if n in GEN_SEED_LIBRARY]
    if custom_smiles:
        seeds_used.append(custom_smiles)
    for smi in seeds_used:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frag_smiles |= BRICS.BRICSDecompose(mol)
    frag_mols = [Chem.MolFromSmiles(f) for f in frag_smiles]
    frag_mols = [m for m in frag_mols if m is not None]
    return frag_mols, seeds_used


def generate_brics_candidates(frag_mols, seed_smiles, n_target=50, mw_min=150, mw_max=550, max_attempts=8000):
    """Runs BRICS.BRICSBuild over the fragment pool, filters to valid, unique,
    drug-like-MW-range candidates that aren't just an unmodified seed."""
    import itertools
    seed_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in seed_smiles if Chem.MolFromSmiles(s)}
    seen = set(seed_canon)
    candidates = []
    builder = BRICS.BRICSBuild(frag_mols)
    for mol in itertools.islice(builder, max_attempts):
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        mw = Descriptors.MolWt(mol)
        if not (mw_min <= mw <= mw_max):
            continue
        candidates.append(canon)
        if len(candidates) >= n_target:
            break
    return candidates


def score_and_rank_candidates(models, candidate_smiles, progress_cb=None):
    """Runs every candidate through the real pipeline (predict_one_compound)
    and computes QED as a drug-likeness sanity check. Returns a DataFrame
    sorted with BBB+ / higher-final_kp candidates first."""
    rows = []
    n = len(candidate_smiles)
    for i, smi in enumerate(candidate_smiles):
        if progress_cb:
            progress_cb(i + 1, n)
        rec = predict_one_compound(models, f"candidate_{i+1}", smi)
        mol = Chem.MolFromSmiles(smi)
        rec["QED"] = round(QED.qed(mol), 3) if mol else None
        rec["MW"] = round(Descriptors.MolWt(mol), 1) if mol else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_bbb_rank"] = (df["bbb_pred"] == "BBB+").astype(int)
    df = df.sort_values(["_bbb_rank", "final_kp"], ascending=[False, False]).drop(columns="_bbb_rank")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def bbb_badge(pred):
    if pred == 1:
        return "🟢 **BBB+** (Penetrates)"
    elif pred == 0:
        return "🔴 **BBB−** (Does not penetrate)"
    return "⚪ Unknown"


def summarize_compound_decision(r):
    """One compact sentence per mechanism, combined into a single readable
    line per compound -- for on-screen display. The full explanation text
    (with rel_passive values, thresholds, citations) stays available in the
    detailed Decision Report CSV for anyone who needs the complete reasoning."""
    bits = []

    eff = r.get("efflux_intervention")
    if eff is True:
        bits.append(
            f"**Efflux** via {r.get('efflux_mechanism')} "
            f"(confidence {r.get('efflux_confidence'):.2f}) is expected to "
            f"**{r.get('efflux_direction')}** brain Kp."
        )
    elif eff is False:
        bits.append("No efflux correction needed.")

    upt = r.get("uptake_intervention")
    if upt is True:
        bits.append(
            f"**Uptake** via {r.get('uptake_mechanism')} "
            f"(confidence {r.get('uptake_confidence'):.2f}) is expected to "
            f"**{r.get('uptake_direction')}** brain Kp."
        )
    elif upt is False:
        bits.append("No uptake correction applied.")

    return " ".join(bits) if bits else "Decision Layer did not run for this compound."


def build_decision_report_df(results_df):
    """Reshapes the wide results table into the long-format Decision Report
    (one row per compound per mechanism) matching bbb_digital_twin_v21.ipynb
    Section 11: Compound | Mechanism | Intervention required? | Mechanism
    selected | Confidence | Expected direction | Explanation."""
    rows = []
    for _, r in results_df.iterrows():
        if r.get("note"):  # skip failed compounds -- nothing to report
            continue
        for mech_label, prefix in [("Efflux", "efflux"), ("Uptake", "uptake")]:
            intervention = r.get(f"{prefix}_intervention")
            if intervention is None:
                continue
            rows.append({
                "Compound": r["name"],
                "Mechanism": mech_label,
                "Intervention required?": "Yes" if intervention else "No",
                "Mechanism selected": r.get(f"{prefix}_mechanism") or "none",
                "Confidence": r.get(f"{prefix}_confidence"),
                "Expected direction": r.get(f"{prefix}_direction") or "none",
                "Explanation": r.get(f"{prefix}_explanation") or "",
            })
    return pd.DataFrame(rows)


def decision_card(mechanism_label, mech):
    """Renders a MechanismDecision's 5 fields (intervention?, mechanism,
    confidence, direction, explanation) as a compact card."""
    fired = mech.intervention_required
    header_icon = "🟠" if fired else "⚪"
    dir_arrow = {"increase": "⬆ increases Kp", "decrease": "⬇ decreases Kp", "none": "no effect"}.get(
        mech.expected_direction, mech.expected_direction
    )
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        c1.markdown(f"**{header_icon} {mechanism_label}**")
        c2.markdown(f"Intervention: **{'Yes' if fired else 'No'}**")
        c3.markdown(f"Confidence: **{mech.confidence:.2f}**")
        if fired:
            st.markdown(f"Mechanism selected: **{mech.selected_mechanism}** · Expected direction: **{dir_arrow}**")
        st.caption(mech.explanation)


def confidence_bar(conf):
    pct = int(conf * 100)
    color = "#2ecc71" if conf >= 0.75 else "#f39c12" if conf >= 0.55 else "#e74c3c"
    st.markdown(
        f"""<div style='background:#e0e0e0;border-radius:6px;height:14px;width:100%;'>
        <div style='background:{color};width:{pct}%;height:14px;border-radius:6px;'></div></div>
        <p style='font-size:0.78rem;margin-top:2px;color:#555;'>{pct}% confidence</p>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 CNS Drug Penetration Predictor")
st.caption("Barrile Lab · University of Cincinnati · GBM Drug Discovery Pipeline")

if not RDKIT_OK:
    st.error("RDKit is not installed. Run `pip install rdkit` and restart.")
    st.stop()

models = load_models()

missing = [k for k, v in models.items() if v is None]
if missing:
    st.warning(
        f"Model file(s) not found in `models/`: **{', '.join(missing)}**. "
        "Those tiers will show as unavailable. Make sure all `.pkl` files are in the `models/` folder."
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Single Compound", "📋 Batch Screening", "🔗 Digital Twin", "🧪 Validation", "🧬 Generate",
    "🧫 My Chip (Personalized Surrogate)"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE COMPOUND
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Compound Prediction")

    # ── PubChem lookup row ────────────────────────────────────────────────────
    col_name, col_btn, col_spacer = st.columns([3, 1, 2])
    with col_name:
        drug_name = st.text_input(
            "Drug name (PubChem lookup)",
            placeholder="e.g. Temozolomide, Erlotinib …",
            key="drug_name_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)   # vertical align
        fetch_clicked = st.button("Fetch SMILES", use_container_width=True)

    # Handle fetch — must happen BEFORE the text_area renders
    if fetch_clicked and drug_name.strip():
        with st.spinner(f"Looking up **{drug_name}** on PubChem…"):
            result, error = pubchem_lookup(drug_name)
        if result:
            st.session_state["smi_box"] = result["smiles"]
            st.success(
                f"✅ **{result['iupac_name']}** · CID {result['cid']} · "
                f"SMILES auto-filled below."
            )
        elif error:
            st.error(f"⚠ PubChem lookup failed (not cached — safe to retry): {error}")
        else:
            st.error(
                f"❌ '{drug_name}' not found on PubChem. "
                "Check spelling or paste SMILES manually."
            )

    # ── SMILES input ──────────────────────────────────────────────────────────
    smiles_input = st.text_area(
        "SMILES string",
        value=st.session_state.get("smi_box", ""),
        height=80,
        key="smi_box",
        placeholder="Paste SMILES here, or use the lookup above.",
    )

    # ── Optional curated uptake evidence (only used if this compound isn't
    #    one of the 10 validated calibration compounds, which auto-fill it) ──
    with st.expander("🧬 Literature uptake evidence (optional)", expanded=False):
        st.caption(
            "The Decision Layer's uptake gate only fires on curated literature evidence, "
            "never an ML guess. Leave at 0 unless you have a specific transporter citation "
            "for this compound. The 10 validated calibration compounds auto-fill this."
        )
        uc1, uc2 = st.columns([1, 2])
        with uc1:
            manual_uptake_p = st.slider("Uptake evidence probability", 0.0, 1.0, 0.0, 0.05)
        with uc2:
            manual_uptake_transporter = st.text_input("Transporter (e.g. OATP1B1, OCT/OCTN1)", value="")
        manual_uptake_note = st.text_area("Evidence note / citation", value="", height=60)

    predict_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=False)

    if predict_btn:
        smiles = smiles_input.strip()
        if not smiles:
            st.warning("Please enter a SMILES string or use the PubChem lookup first.")
        else:
            try:
                tier123_feats, abc_feats, mol = featurize(smiles)
                phys = get_physchem(mol)

                st.markdown("---")

                # ── Physicochemical summary ────────────────────────────────
                with st.expander("📐 Physicochemical Properties", expanded=False):
                    pcols = st.columns(5)
                    labels = {
                        "MolWt": "MW (Da)", "MolLogP": "logP",
                        "NumHDonors": "H-Donors", "NumHAcceptors": "H-Acceptors",
                        "TPSA": "TPSA (Å²)", "NumRotatableBonds": "RotBonds",
                        "RingCount": "Rings", "NumAromaticRings": "Arom. Rings",
                        "HeavyAtomCount": "Heavy Atoms", "FractionCSP3": "Fsp³",
                    }
                    for i, (k, label) in enumerate(labels.items()):
                        with pcols[i % 5]:
                            st.metric(label, f"{phys[k]:.2f}")

                # ── Tier 1 ─────────────────────────────────────────────────
                st.markdown("### Tier 1 — BBB Classification")
                t1_pred, t1_conf = run_tier1(models, tier123_feats)
                if t1_pred is not None:
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.markdown(bbb_badge(t1_pred))
                    with c2:
                        confidence_bar(t1_conf)
                else:
                    st.info("Tier 1 model not loaded.")

                # ── Tier 2 ─────────────────────────────────────────────────
                st.markdown("### Tier 2 — logBB Regression")
                t2_logbb = run_tier2(models, tier123_feats)
                if t2_logbb is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("logBB", f"{t2_logbb:.3f}")
                    c2.metric("BBB Ratio (linear)", f"{10**t2_logbb:.3f}")
                    interp = (
                        "Good penetration" if t2_logbb > 0.3
                        else "Moderate" if t2_logbb > -0.5
                        else "Poor penetration"
                    )
                    c3.metric("Interpretation", interp)
                else:
                    st.info("Tier 2 model not loaded.")

                # ── Tier 3 ─────────────────────────────────────────────────
                st.markdown("### Tier 3 — Plasma Protein Binding (PPB)")
                t3_fu = run_tier3(models, tier123_feats)
                if t3_fu is not None:
                    ppb_pct = (1 - t3_fu) * 100
                    c1, c2, c3 = st.columns(3)
                    c1.metric("fu plasma", f"{t3_fu:.3f}")
                    c2.metric("PPB %", f"{ppb_pct:.1f}%")
                    bound_label = (
                        "Highly bound" if ppb_pct > 90
                        else "Moderately bound" if ppb_pct > 60
                        else "Low binding"
                    )
                    c3.metric("Binding class", bound_label)
                else:
                    st.info("Tier 3 model not loaded.")

                # ── ABC Transporter (efflux liability) ─────────────────────
                st.markdown("### ABC Efflux Transporter")
                abc_result = run_abc(models, abc_feats)
                if abc_result is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted class", abc_result["label"])
                    c2.metric("Efflux liability", f"{abc_result['efflux_liability']:.3f}")
                    c3.metric("P(Substrate) / P(Both)",
                              f"{abc_result['proba'].get('Substrate', 0):.2f} / {abc_result['proba'].get('Both', 0):.2f}")
                    st.caption("Efflux liability = P(Substrate) + P(Both) — the scalar fed into the Decision Layer's efflux gate.")
                else:
                    st.info("ABC transporter model(s) not loaded.")

                # ── Decision Layer / Biological Agents / Decision Engine ───
                if t2_logbb is not None and abc_result is not None:
                    st.markdown("### 🧭 Decision Layer — Structured Decision Report")
                    st.caption(
                        "Compound → Tier 1 BBB predictor → BBB Digital Twin → Decision Layer → "
                        "Biological Agents (Efflux, Uptake) → Decision Engine → Final Prediction + Biological Explanation"
                    )

                    # predicted_kp = 10^logBB (logBB defined as log10(Cbrain/Cplasma));
                    # this is the Digital Twin's own Kp estimate *before* Decision Layer correction.
                    predicted_kp = 10 ** t2_logbb

                    compound_display_name = drug_name.strip() or smiles[:24]
                    final_kp, decision, matched, predicted_kp = run_decision_layer(
                        compound_display_name, predicted_kp, abc_result["efflux_liability"],
                        uptake_evidence_p=manual_uptake_p,
                        uptake_transporter=manual_uptake_transporter or None,
                        uptake_note=manual_uptake_note,
                    )

                    if matched:
                        st.success(
                            f"✅ Matched to validated calibration compound **{matched}** — using the "
                            "published DT prediction / efflux score from the notebook (not a fresh live "
                            "re-inference), plus the curated literature discount/uptake evidence."
                        )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("DT alone (Kp)", f"{predicted_kp:.4f}")
                    c2.metric("Final Kp (Decision Layer corrected)", f"{final_kp:.4f}")
                    c3.metric("rel_passive", f"{decision['rel_passive']:.3f}")

                    decision_card("Efflux", decision["efflux"])
                    decision_card("Uptake", decision["uptake"])

                # ── Digital Twin Papp estimate (legacy Sugano-prior view) ──
                if t2_logbb is not None and t3_fu is not None:
                    st.markdown("### Digital Twin — Papp Estimate (Sugano-prior view)")
                    efflux_ratio = 1.0
                    if abc_result is not None:
                        efflux_ratio = max(1.0, 1.0 + 0.5 * abc_result["efflux_liability"])

                    papp_plasma    = digital_twin_papp(t2_logbb, t3_fu, efflux_ratio)
                    papp_serum_free = digital_twin_papp(t2_logbb, 1.0, efflux_ratio)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Papp (plasma)", f"{papp_plasma:.2f} ×10⁻⁶ cm/s")
                    c2.metric("Papp (serum-free)", f"{papp_serum_free:.2f} ×10⁻⁶ cm/s")
                    c3.metric("Efflux ratio", f"{efflux_ratio:.1f}×")
                    st.caption(
                        "Formula: Papp,free = 10^((logBB − β) / α)  ·  "
                        "Papp,plasma = Papp,free × fu_plasma / efflux_ratio  "
                        "(α=1.99, β=0.357; Sugano 2010 priors — pending on-chip calibration). "
                        "This is a separate, older mechanistic view; the Decision Layer's Final Kp above "
                        "is the current frozen-architecture output."
                    )

            except ValueError as e:
                st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCREENING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Compound Screening")
    st.markdown(
        "Upload a CSV with a **`smiles`** column (and optionally a **`name`** column). "
        "Compounds with only a name will be resolved via PubChem automatically. "
        "Optional columns **`uptake_evidence_p`**, **`uptake_transporter`**, **`uptake_note`** "
        "supply curated literature uptake evidence per row (default: no evidence, gate stays closed). "
        "The 10 validated calibration compounds (matched by name) auto-fill this regardless."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)} rows** · Columns: {list(df.columns)}")

        has_smiles = "smiles" in df.columns
        has_name   = "name"   in df.columns

        if not has_smiles and not has_name:
            st.error("CSV must have a `smiles` and/or `name` column.")
        else:
            if st.button("▶ Run Batch Pipeline", type="primary"):
                results = []
                progress = st.progress(0)
                status   = st.empty()

                for i, row in df.iterrows():
                    pct = int((i + 1) / len(df) * 100)
                    progress.progress(pct)

                    def _clean_cell(val):
                        """pandas reads a blank cell in a present column as
                        NaN (float); str(NaN) == 'nan', which is truthy and
                        would wrongly be treated as a real value. Collapse
                        NaN/empty to a real empty string."""
                        if pd.isna(val):
                            return ""
                        return str(val).strip()

                    name_val   = _clean_cell(row.get("name", ""))   if has_name   else ""
                    smiles_val = _clean_cell(row.get("smiles", "")) if has_smiles else ""

                    # resolve SMILES from name if missing
                    pubchem_error = None
                    if not smiles_val and name_val:
                        status.text(f"PubChem lookup: {name_val} …")
                        res, pubchem_error = pubchem_lookup(name_val)
                        smiles_val = res["smiles"] if res else ""
                        time.sleep(0.2)   # polite rate limit

                    status.text(f"Predicting: {name_val or smiles_val[:20]} …")

                    raw_p = row.get("uptake_evidence_p", 0.0) if "uptake_evidence_p" in df.columns else 0.0
                    row_uptake_p = 0.0 if pd.isna(raw_p) else float(raw_p)
                    raw_t = row.get("uptake_transporter") if "uptake_transporter" in df.columns else None
                    row_uptake_t = None if (raw_t is None or pd.isna(raw_t)) else str(raw_t)
                    raw_n = row.get("uptake_note", "") if "uptake_note" in df.columns else ""
                    row_uptake_n = "" if pd.isna(raw_n) else str(raw_n)

                    rec = predict_one_compound(
                        models, name_val, smiles_val, pubchem_error=pubchem_error,
                        uptake_evidence_p=row_uptake_p,
                        uptake_transporter=row_uptake_t,
                        uptake_note=row_uptake_n,
                    )
                    results.append(rec)

                status.text("Done ✅")
                progress.progress(100)

                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                failed = results_df[results_df["note"].fillna("") != ""]
                if not failed.empty:
                    st.error(f"⚠ {len(failed)} of {len(results_df)} compound(s) failed. Reasons below:")
                    for _, frow in failed.iterrows():
                        st.markdown(f"**{frow['name']}**: {frow['note']}")
                else:
                    st.success(f"✅ All {len(results_df)} compounds processed successfully.")

                report_df = build_decision_report_df(results_df)
                if not report_df.empty:
                    with st.expander("🧭 Decision Report", expanded=True):
                        st.caption("One line per compound, combining both mechanisms.")
                        for _, r in results_df.iterrows():
                            if r.get("note"):
                                continue
                            kp_line = ""
                            if r.get("dt_kp") is not None and r.get("final_kp") is not None:
                                kp_line = f" (DT Kp {r['dt_kp']:.4f} → Final Kp {r['final_kp']:.4f})"
                            st.markdown(f"**{r['name']}**{kp_line}: {summarize_compound_decision(r)}")
                        st.download_button(
                            "⬇ Download full structured Decision Report CSV (with citations/reasoning)",
                            data=report_df.to_csv(index=False).encode("utf-8"),
                            file_name="decision_report_detailed.csv",
                            mime="text/csv",
                        )

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Results CSV",
                    data=csv_out,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Digital Twin — Papp ↔ logBB Calculator")
    st.markdown(
        "Mechanistic formula linking model predictions to on-chip Papp measurements. "
        "Calibrated from Sugano 2010 priors; will be updated with Bae et al. on-chip data."
    )

    direction = st.radio(
        "Direction",
        ["logBB → Papp  (predict chip readout from model output)",
         "Papp → logBB  (back-calculate from chip measurement)"],
        horizontal=True,
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    if direction.startswith("logBB"):
        with c1:
            logbb_in = st.number_input("logBB (from Tier 2)", value=-0.5, step=0.05, format="%.3f")
        with c2:
            fu_in    = st.number_input("fu plasma (from Tier 3)", value=0.1, min_value=0.001, max_value=1.0, step=0.01, format="%.3f")
        with c3:
            er_in    = st.number_input("Efflux ratio (1 = no efflux)", value=1.0, min_value=1.0, step=0.5, format="%.1f")

        alpha = st.sidebar.number_input("α (slope prior)", value=1.99, step=0.01)
        beta  = st.sidebar.number_input("β (intercept prior)", value=0.357, step=0.001)

        if st.button("Calculate Papp"):
            papp_p  = digital_twin_papp(logbb_in, fu_in, er_in, alpha, beta)
            papp_sf = digital_twin_papp(logbb_in, 1.0,   er_in, alpha, beta)
            r1, r2, r3 = st.columns(3)
            r1.metric("Papp (plasma conditions)", f"{papp_p:.3f} ×10⁻⁶ cm/s")
            r2.metric("Papp (serum-free)",        f"{papp_sf:.3f} ×10⁻⁶ cm/s")
            r3.metric("fu correction factor",     f"{papp_p / papp_sf:.3f}×")

            # Round-trip consistency check: feed the just-computed Papp back
            # through the inverse formula and confirm it recovers the input
            # logBB. This only validates the algebra is self-consistent --
            # NOT that alpha/beta are correctly calibrated to real chip data.
            papp_free_check = papp_p * er_in / fu_in * 1e-6  # back to raw cm/s
            recovered_logbb = alpha * np.log10(papp_free_check) + beta
            roundtrip_err = abs(recovered_logbb - logbb_in)
            if roundtrip_err < 1e-6:
                st.success(f"✅ Round-trip check passed: recovered logBB = {recovered_logbb:.6f} (input was {logbb_in:.6f}). Algebra is self-consistent.")
            else:
                st.error(f"⚠ Round-trip check FAILED: recovered logBB = {recovered_logbb:.6f}, input was {logbb_in:.6f} (error = {roundtrip_err:.6f}). This indicates a bug in the formula implementation.")

    else:
        with c1:
            papp_in  = st.number_input("Papp measured on chip (×10⁻⁶ cm/s)", value=10.0, min_value=0.001, step=0.5, format="%.3f")
        with c2:
            fu_in2   = st.number_input("fu plasma", value=0.1, min_value=0.001, max_value=1.0, step=0.01, format="%.3f")
        with c3:
            er_in2   = st.number_input("Efflux ratio", value=1.0, min_value=1.0, step=0.5, format="%.1f")

        alpha2 = st.sidebar.number_input("α (slope prior)", value=1.99, step=0.01, key="alpha2")
        beta2  = st.sidebar.number_input("β (intercept prior)", value=0.357, step=0.001, key="beta2")

        if st.button("Back-calculate logBB"):
            # Papp,plasma = Papp,free * fu / efflux
            # Papp,free   = Papp,plasma * efflux / fu
            # logBB = alpha * log10(Papp,free * 1e-6) + beta
            papp_free_cm = (papp_in * er_in2 / fu_in2) * 1e-6
            if papp_free_cm <= 0:
                st.error("Invalid Papp value.")
            else:
                logBB_calc = alpha2 * np.log10(papp_free_cm) + beta2
                c1o, c2o = st.columns(2)
                c1o.metric("Estimated logBB", f"{logBB_calc:.3f}")
                c2o.metric(
                    "Interpretation",
                    "Good penetration" if logBB_calc > 0.3
                    else "Moderate" if logBB_calc > -0.5
                    else "Poor penetration",
                )

    st.markdown("---")
    with st.expander("ℹ Formula details"):
        st.markdown("""
**logBB → Papp**
```
Papp,free   = 10^((logBB − β) / α)          [cm/s]
Papp,plasma = Papp,free × fu_plasma / ER    [cm/s]
```
**Papp → logBB (back-calculation)**
```
Papp,free  = Papp_measured × ER / fu_plasma
logBB      = α × log₁₀(Papp,free) + β
```
- **α = 1.99**, **β = 0.357** — Sugano (2010) priors, pending calibration against Bae et al. on-chip data
- **ER** = Efflux ratio (ABC transporter model; default 1.0 = no efflux)
- Will be calibrated once Charles completes on-chip Papp measurements (Figure 4, APL Bioengineering submission)
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("External Validation")
    st.markdown(
        "Run the pipeline against a labeled dataset (real experimental values, not "
        "the 10-compound calibration set) and see accuracy metrics directly -- no "
        "manual CSV round-tripping. Upload a CSV with **`name`** and/or **`smiles`**, "
        "plus any column holding ground-truth values (e.g. `true_logBB`, `logBB`, "
        "`Kp`) -- you'll pick which column to compare against below."
    )

    metric_choice = st.radio(
        "Compare predictions against:",
        ["logBB (Tier 2 output)", "Kp (Decision-Layer-corrected final_kp)"],
        horizontal=True,
    )
    pred_col = "logBB" if metric_choice.startswith("logBB") else "final_kp"

    val_uploaded = st.file_uploader("Upload labeled CSV", type=["csv"], key="val_uploader")

    if val_uploaded:
        vdf = pd.read_csv(val_uploaded)
        st.write(f"Loaded **{len(vdf)} rows** · Columns: {list(vdf.columns)}")

        has_smiles_v = "smiles" in vdf.columns
        has_name_v   = "name"   in vdf.columns

        candidate_cols = [c for c in vdf.columns if c not in ("name", "smiles")]

        if not candidate_cols:
            st.error("CSV must have a ground-truth column (any name) besides `name`/`smiles` to compare against.")
        elif not has_smiles_v and not has_name_v:
            st.error("CSV must have a `smiles` and/or `name` column.")
        else:
            # Auto-guess the likely ground-truth column so the common case
            # (a column literally called true_value, true_logBB, logBB, Kp,
            # actual, etc.) needs no renaming -- but always let the user
            # override, since we can't be sure which column they mean.
            aliases = ("true_value", "true_logbb", "true_kp", "logbb", "kp", "actual", "ground_truth")
            guess_idx = 0
            for i, c in enumerate(candidate_cols):
                if c.strip().lower() in aliases:
                    guess_idx = i
                    break
            truth_col = st.selectbox(
                "Ground-truth column in your CSV to compare against:",
                candidate_cols, index=guess_idx,
            )

            if st.button("▶ Run Validation", type="primary"):
                results = []
                progress = st.progress(0)
                status   = st.empty()

                def _clean_cell_v(val):
                    if pd.isna(val):
                        return ""
                    return str(val).strip()

                for i, row in vdf.iterrows():
                    progress.progress(int((i + 1) / len(vdf) * 100))

                    name_val   = _clean_cell_v(row.get("name", ""))   if has_name_v   else ""
                    smiles_val = _clean_cell_v(row.get("smiles", "")) if has_smiles_v else ""

                    pubchem_error = None
                    if not smiles_val and name_val:
                        status.text(f"PubChem lookup: {name_val} …")
                        res, pubchem_error = pubchem_lookup(name_val)
                        smiles_val = res["smiles"] if res else ""
                        time.sleep(0.2)

                    status.text(f"Predicting: {name_val or smiles_val[:20]} …")
                    rec = predict_one_compound(models, name_val, smiles_val, pubchem_error=pubchem_error)
                    rec["true_value"] = row.get(truth_col)
                    results.append(rec)

                status.text("Done ✅")
                progress.progress(100)

                results_df = pd.DataFrame(results)
                valid = results_df.dropna(subset=[pred_col, "true_value"]).copy()
                valid["true_value"] = pd.to_numeric(valid["true_value"], errors="coerce")
                valid = valid.dropna(subset=["true_value"])

                n_total = len(results_df)
                n_valid = len(valid)
                st.write(f"**{n_valid} of {n_total}** compounds produced both a prediction and a usable ground-truth value.")

                if n_valid >= 2:
                    err = valid[pred_col] - valid["true_value"]
                    mae  = err.abs().mean()
                    rmse = (err ** 2).mean() ** 0.5
                    ss_res = (err ** 2).sum()
                    ss_tot = ((valid["true_value"] - valid["true_value"].mean()) ** 2).sum()
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", f"{mae:.3f}")
                    c2.metric("RMSE", f"{rmse:.3f}")
                    c3.metric("R²", f"{r2:.3f}")

                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(valid["true_value"], valid[pred_col], alpha=0.7, edgecolor="k", linewidth=0.5)
                    lims = [
                        min(valid["true_value"].min(), valid[pred_col].min()),
                        max(valid["true_value"].max(), valid[pred_col].max()),
                    ]
                    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
                    ax.set_xlabel(f"True value ({metric_choice.split(' ')[0]})")
                    ax.set_ylabel(f"Predicted ({pred_col})")
                    ax.set_title(f"Predicted vs. True -- n={n_valid}, R²={r2:.3f}")
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Not enough valid predictions to compute metrics (need at least 2).")

                report_df = build_decision_report_df(results_df)
                if not report_df.empty:
                    with st.expander("🧭 Decision Report", expanded=False):
                        st.caption("One line per compound, combining both mechanisms.")
                        for _, r in results_df.iterrows():
                            if r.get("note"):
                                continue
                            kp_line = ""
                            if r.get("dt_kp") is not None and r.get("final_kp") is not None:
                                kp_line = f" (DT Kp {r['dt_kp']:.4f} → Final Kp {r['final_kp']:.4f})"
                            st.markdown(f"**{r['name']}**{kp_line}: {summarize_compound_decision(r)}")
                        st.download_button(
                            "⬇ Download full structured Decision Report CSV (with citations/reasoning)",
                            data=report_df.to_csv(index=False).encode("utf-8"),
                            file_name="decision_report_detailed.csv",
                            mime="text/csv",
                            key="val_report_dl",
                        )

                st.dataframe(results_df, use_container_width=True)

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Validation Results CSV",
                    data=csv_out,
                    file_name="validation_results.csv",
                    mime="text/csv",
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Generate Candidates (BRICS + pipeline scoring)")
    st.warning(
        "This is **not** a trained deep generative model (no GPU/pretraining "
        "corpus available in this environment for a REINVENT/VAE-style RL-"
        "fine-tuned SMILES generator). It's BRICS combinatorial fragment "
        "recombination -- a legitimate, established de novo design method -- "
        "using your actual Digital Twin -> Decision Layer pipeline as the "
        "fitness function. Candidates are hypotheses to filter and inspect "
        "further, not validated for synthesizability, IP novelty, or "
        "purchasability."
    )

    st.markdown("**1. Choose seed molecules** (their BRICS fragments become the building blocks):")
    seed_names = st.multiselect(
        "Seed library", list(GEN_SEED_LIBRARY.keys()),
        default=["Donepezil", "Temozolomide", "Fluoxetine", "Quinidine", "Caffeine"],
    )
    custom_seed = st.text_input("Optional: add your own seed SMILES (biases fragments toward this chemotype)", value="")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_target = st.slider("Candidates to generate & score", 10, 200, 50, step=10)
    with c2:
        mw_min = st.number_input("Min MW", value=150, step=25)
    with c3:
        mw_max = st.number_input("Max MW", value=550, step=25)

    if st.button("🧬 Generate & Score", type="primary"):
        if not seed_names and not custom_seed.strip():
            st.error("Choose at least one seed molecule (or provide a custom SMILES).")
        else:
            with st.spinner("Building fragment pool from seed molecules…"):
                frag_mols, seeds_used = build_fragment_pool(seed_names, custom_seed.strip() or None)
            st.write(f"Fragment pool: **{len(frag_mols)}** unique fragments from **{len(seeds_used)}** seed molecule(s).")

            with st.spinner("Generating candidates via BRICS recombination…"):
                candidates = generate_brics_candidates(frag_mols, seeds_used, n_target=n_target, mw_min=mw_min, mw_max=mw_max)

            if not candidates:
                st.error("No valid candidates generated in this MW range -- try different seeds or widen the MW range.")
            else:
                st.write(f"Generated **{len(candidates)}** unique, valid, MW-filtered candidates. Scoring through the pipeline…")
                progress = st.progress(0)
                status = st.empty()

                def _cb(i, n):
                    progress.progress(int(i / n * 100))
                    status.text(f"Scoring candidate {i}/{n} …")

                results_df = score_and_rank_candidates(models, candidates, progress_cb=_cb)
                status.text("Done ✅")
                progress.progress(100)

                n_bbb_pos = (results_df["bbb_pred"] == "BBB+").sum()
                st.success(f"✅ {n_bbb_pos} of {len(results_df)} candidates predicted BBB+.")

                st.markdown("### Top 10 candidates")
                top10 = results_df.head(10)
                for _, r in top10.iterrows():
                    if r.get("note"):
                        continue
                    kp_line = f" · DT Kp {r['dt_kp']:.4f} → Final Kp {r['final_kp']:.4f}" if r.get("dt_kp") is not None else ""
                    qed_line = f" · QED {r['QED']:.2f} · MW {r['MW']:.0f}" if r.get("QED") is not None else ""
                    with st.container(border=True):
                        st.code(r["smiles"], language=None)
                        st.markdown(f"**{r['bbb_pred']}**{kp_line}{qed_line}")
                        st.caption(summarize_compound_decision(r))

                with st.expander("Full results table (all candidates)"):
                    st.dataframe(results_df, use_container_width=True)

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download all generated candidates CSV",
                    data=csv_out,
                    file_name="generated_candidates.csv",
                    mime="text/csv",
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MY CHIP (PERSONALIZED SURROGATE)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("My Chip — Personalized Digital Twin")
    st.markdown(
        "Every other tab in this app runs the **baseline model**: Tier 1/2/3 + ABC + "
        "Decision Layer, trained once on the published Bae et al. (2024) BBB-on-chip "
        "dataset. It's the same for every user and never sees your own experimental "
        "results.\n\n"
        "This tab lets you calibrate a **personalized surrogate**: give it a handful of "
        "compounds you've actually tested on *your own* organ-on-chip, and it learns a "
        "correction from the baseline model's prediction to what your chip actually "
        "measures. You can then apply that correction to new compounds you haven't "
        "tested yet."
    )
    st.caption(
        "This never modifies the shared baseline model or its `.pkl` files -- the "
        "calibration lives only in your current session, on top of the frozen "
        "baseline pipeline."
    )

    if "user_surrogate" not in st.session_state:
        st.session_state["user_surrogate"] = None
    if "user_surrogate_cal_df" not in st.session_state:
        st.session_state["user_surrogate_cal_df"] = None

    st.markdown("---")
    st.markdown("### Step 1 — Calibrate on your own chip data")
    st.markdown(
        "Upload a CSV with **`name`** and/or **`smiles`**, plus a **`empirical_kp`** "
        "column (your measured brain:plasma Kp for that compound on your chip). "
        "Compounds with only a name are resolved via PubChem, same as Batch Screening. "
        "**You'll want at least 3 compounds** for a real fit with an uncertainty "
        "estimate (2 works but gives an exact, unvalidated line; 6+ unlocks a Gaussian "
        "Process fit if scikit-learn is installed)."
    )

    cal_uploaded = st.file_uploader("Upload your chip calibration CSV", type=["csv"], key="cal_csv")

    if cal_uploaded:
        cal_input_df = pd.read_csv(cal_uploaded)
        cal_input_df = normalize_calibration_columns(cal_input_df)
        has_smiles_c = "smiles" in cal_input_df.columns
        has_name_c   = "name"   in cal_input_df.columns
        has_emp_c    = "empirical_kp" in cal_input_df.columns

        if not has_emp_c or (not has_smiles_c and not has_name_c):
            st.error(
                "CSV must have an `empirical_kp`-type column (accepted: empirical_kp, kp, "
                "kp_empirical, measured_kp, chip_kp, observed_kp), and at least one of "
                "`name` / `smiles`. "
                f"Columns found: {list(cal_input_df.columns)}"
            )
        else:
            st.write(f"Loaded **{len(cal_input_df)} rows** · Columns: {list(cal_input_df.columns)}")
            if st.button("▶ Run baseline model + fit chip calibration", type="primary"):
                cal_results = []
                progress = st.progress(0)
                status = st.empty()

                for i, row in cal_input_df.iterrows():
                    progress.progress(int((i + 1) / len(cal_input_df) * 100))

                    def _clean(val):
                        return "" if pd.isna(val) else str(val).strip()

                    name_val   = _clean(row.get("name", ""))   if has_name_c   else ""
                    smiles_val = _clean(row.get("smiles", "")) if has_smiles_c else ""
                    emp_kp_raw = row.get("empirical_kp")

                    pubchem_error = None
                    if not smiles_val and name_val:
                        status.text(f"PubChem lookup: {name_val} …")
                        res, pubchem_error = pubchem_lookup(name_val)
                        smiles_val = res["smiles"] if res else ""
                        time.sleep(0.2)

                    status.text(f"Running baseline pipeline: {name_val or smiles_val[:20]} …")
                    rec = predict_one_compound(models, name_val, smiles_val, pubchem_error=pubchem_error)
                    rec["empirical_kp"] = float(emp_kp_raw) if pd.notna(emp_kp_raw) else None
                    cal_results.append(rec)

                status.text("Done ✅")
                progress.progress(100)

                cal_results_df = pd.DataFrame(cal_results)
                st.session_state["user_surrogate_cal_df"] = cal_results_df

                failed = cal_results_df[
                    cal_results_df["note"].fillna("") != ""
                ]
                usable = cal_results_df[
                    (cal_results_df["note"].fillna("") == "") &
                    cal_results_df["final_kp"].notna() &
                    cal_results_df["empirical_kp"].notna()
                ]

                if not failed.empty:
                    st.warning(f"⚠ {len(failed)} of {len(cal_results_df)} compound(s) failed the baseline pipeline and were excluded from the fit.")

                calib = fit_user_surrogate(usable)
                if calib.get("method") in (None, "none"):
                    st.error(calib.get("error", "Could not fit a calibration from this data."))
                    st.session_state["user_surrogate"] = None
                else:
                    st.session_state["user_surrogate"] = calib
                    st.success(f"✅ Chip calibration fit on {calib['n']} compound(s).")
                    st.info(surrogate_calibration_summary(calib))

                    display_cols = ["name", "smiles", "dt_kp", "final_kp", "empirical_kp", "note"]
                    display_cols = [c for c in display_cols if c in cal_results_df.columns]
                    st.dataframe(cal_results_df[display_cols], use_container_width=True)

                    # Calibration plot: baseline (final_kp) vs. this chip's empirical Kp
                    fig, ax = plt.subplots(figsize=(5, 4))
                    x_plot = usable["final_kp"].values.astype(float)
                    y_plot = usable["empirical_kp"].values.astype(float)
                    ax.scatter(x_plot, y_plot, color="tab:blue", label="Your calibration compounds", zorder=3)
                    lims = [min(x_plot.min(), y_plot.min()) * 0.7, max(x_plot.max(), y_plot.max()) * 1.3]
                    ax.plot(lims, lims, "k--", linewidth=1, label="y = x (no correction)", alpha=0.5)

                    x_line = np.geomspace(lims[0], lims[1], 60)
                    y_line = [apply_user_surrogate(calib, xv)[0] for xv in x_line]
                    ax.plot(x_line, y_line, color="tab:red", linewidth=2, label="Fitted surrogate")

                    ax.set_xscale("log"); ax.set_yscale("log")
                    ax.set_xlabel("Baseline model Final Kp")
                    ax.set_ylabel("Your empirical Kp")
                    ax.set_title("Chip calibration fit")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig)

    if st.session_state["user_surrogate"]:
        st.markdown("---")
        st.markdown("### Step 2 — Predict new compounds with your personalized surrogate")
        st.caption(surrogate_calibration_summary(st.session_state["user_surrogate"]))

        pred_mode = st.radio("Input mode", ["Single compound", "Batch CSV"], horizontal=True, key="surrogate_pred_mode")

        if pred_mode == "Single compound":
            sc1, sc2 = st.columns([2, 3])
            with sc1:
                surr_name = st.text_input("Drug name (optional, PubChem lookup)", key="surr_name")
                surr_fetch = st.button("Fetch SMILES", key="surr_fetch")
            with sc2:
                surr_smiles_default = st.session_state.get("surr_smi_box", "")
                if surr_fetch and surr_name.strip():
                    res, err = pubchem_lookup(surr_name)
                    if res:
                        st.session_state["surr_smi_box"] = res["smiles"]
                        surr_smiles_default = res["smiles"]
                        st.success(f"✅ SMILES filled from PubChem ({res['iupac_name']}).")
                    elif err:
                        st.error(f"PubChem lookup failed: {err}")
                    else:
                        st.error(f"'{surr_name}' not found on PubChem.")
                surr_smiles = st.text_area("SMILES", value=surr_smiles_default, height=80, key="surr_smi_box")

            if st.button("🚀 Predict (baseline + your chip surrogate)", type="primary", key="surr_predict_btn"):
                if not surr_smiles.strip():
                    st.warning("Enter a SMILES string or use the PubChem lookup.")
                else:
                    rec = predict_one_compound(models, surr_name, surr_smiles.strip())
                    if rec.get("note"):
                        st.error(f"Baseline pipeline failed: {rec['note']}")
                    else:
                        corrected, lo, hi = apply_user_surrogate(st.session_state["user_surrogate"], rec["final_kp"])
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Baseline Final Kp", f"{rec['final_kp']:.4f}")
                        c2.metric("Your chip's predicted Kp", f"{corrected:.4f}" if corrected is not None else "n/a")
                        if lo is not None and hi is not None:
                            c3.metric("95% interval", f"{lo:.4f} – {hi:.4f}")
                        else:
                            c3.metric("95% interval", "n/a (too few calibration points)")
                        st.caption(
                            f"Baseline: {rec['bbb_pred']} · DT Kp {rec['dt_kp']:.4f} · "
                            f"Efflux: {rec['efflux_mechanism']} ({rec['efflux_direction']}) · "
                            f"Uptake: {rec['uptake_mechanism']} ({rec['uptake_direction']})"
                        )

        else:
            surr_batch_file = st.file_uploader("Upload CSV with `name` and/or `smiles`", type=["csv"], key="surr_batch_csv")
            if surr_batch_file and st.button("▶ Run batch through baseline + your surrogate", type="primary", key="surr_batch_btn"):
                batch_df = pd.read_csv(surr_batch_file)
                has_smiles_b = "smiles" in batch_df.columns
                has_name_b   = "name"   in batch_df.columns
                if not has_smiles_b and not has_name_b:
                    st.error("CSV must have a `smiles` and/or `name` column.")
                else:
                    surr_results = []
                    progress = st.progress(0)
                    status = st.empty()
                    for i, row in batch_df.iterrows():
                        progress.progress(int((i + 1) / len(batch_df) * 100))

                        def _clean2(val):
                            return "" if pd.isna(val) else str(val).strip()

                        name_val   = _clean2(row.get("name", ""))   if has_name_b   else ""
                        smiles_val = _clean2(row.get("smiles", "")) if has_smiles_b else ""
                        pubchem_error = None
                        if not smiles_val and name_val:
                            status.text(f"PubChem lookup: {name_val} …")
                            res, pubchem_error = pubchem_lookup(name_val)
                            smiles_val = res["smiles"] if res else ""
                            time.sleep(0.2)

                        status.text(f"Predicting: {name_val or smiles_val[:20]} …")
                        rec = predict_one_compound(models, name_val, smiles_val, pubchem_error=pubchem_error)
                        corrected, lo, hi = apply_user_surrogate(st.session_state["user_surrogate"], rec.get("final_kp"))
                        rec["your_chip_kp"] = round(corrected, 4) if corrected is not None else None
                        rec["your_chip_kp_lo95"] = round(lo, 4) if lo is not None else None
                        rec["your_chip_kp_hi95"] = round(hi, 4) if hi is not None else None
                        surr_results.append(rec)

                    status.text("Done ✅")
                    progress.progress(100)
                    surr_results_df = pd.DataFrame(surr_results)

                    show_cols = ["name", "smiles", "bbb_pred", "dt_kp", "final_kp",
                                 "your_chip_kp", "your_chip_kp_lo95", "your_chip_kp_hi95",
                                 "efflux_mechanism", "uptake_mechanism", "note"]
                    show_cols = [c for c in show_cols if c in surr_results_df.columns]
                    st.dataframe(surr_results_df[show_cols], use_container_width=True)

                    csv_out = surr_results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download personalized predictions CSV",
                        data=csv_out,
                        file_name="my_chip_surrogate_predictions.csv",
                        mime="text/csv",
                    )

        st.markdown("---")
        if st.button("🗑 Clear my chip calibration", key="clear_surrogate"):
            st.session_state["user_surrogate"] = None
            st.session_state["user_surrogate_cal_df"] = None
            st.rerun()
    else:
        st.info("Fit a chip calibration in Step 1 above to unlock personalized predictions here.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Pipeline Info")
    st.markdown("""
**Frozen architecture (v21):**
Compound → Tier 1 BBB predictor → BBB Digital Twin →
**Decision Layer** → Biological Agents (Efflux, Uptake,
future agents) → Decision Engine → Final Prediction +
Biological Explanation

**Tier 1** — BBB Classifier  
RF + XGBoost + LR soft-vote  
Accuracy 92%, MCC 0.853

**Tier 2** — logBB Regressor  
Stacked RF/XGB → Ridge  
CV R² = 0.604

**Tier 3** — PPB Predictor  
Stacked RF/XGB/GB (TDC PPBR_AZ)  
CV R² = 0.604

**ABC Transporter (v4)**  
RF + XGB soft-vote, 3-class  
(Inhibitor / Substrate / Both)  
efflux_liability = P(Substrate)+P(Both)

**Decision Layer** — Bayesian TransportAgent
(efflux) + UptakeAgent (uptake), fit on a
10-compound BBB-on-chip calibration set
(Yang/Bae et al., Pharmaceutics 2024).
Uptake gate requires curated literature
evidence, never an ML guess. Returns a
5-field structured decision per mechanism:
intervention required, mechanism selected,
confidence, expected direction, explanation.

**🧫 My Chip** — optional per-user calibration
layer on top of the frozen baseline above.
Learns baseline Kp → your chip's empirical Kp
from a few compounds you've tested yourself;
session-only, never touches the shared model.

---
**GBM Shortlist**
- Temozolomide
- Erlotinib
- Imatinib
- X147 / X203

---
*APL Bioengineering · Aug 31 2026*  
*Barrile Lab · UC BME*
    """)

    st.markdown("---")
    st.markdown("**Model files expected in `models/`:**")
    for k, v in models.items():
        icon = "✅" if v is not None else "❌"
        st.markdown(f"{icon} `{k}`")

    st.markdown("---")
    with st.expander("⚠ Known caveat — verify before manuscript use"):
        st.caption(
            "Tier 1/2/3 physchem feature order (`TIER123_PHYSCHEM_DESCS`) was "
            "inferred, not recovered from the .pkl files (they carry no "
            "feature_names_in_). Confirm it against the original training "
            "script before trusting these outputs for the APL Bioengineering "
            "submission."
        )
