# CNS Penetration Predictor — Task G
**Barrile Lab · UC Biomedical Engineering**
Sai Madhav Yedupati · 2026

Streamlit web interface for the BBB + ABC transporter prediction pipeline.

---

## Features

| Feature | Description |
|---------|-------------|
| Single compound | Paste any SMILES → full BBB + ABC report with interpretation |
| Batch screening | Upload Excel/CSV or paste up to 100 SMILES → downloadable results table |
| Demo mode | Runs with heuristic estimates when model files are not present |
| Export | CSV download for both single and batch results |

---

## Quick start

```bash
# 1. Install dependencies
pip install streamlit rdkit scikit-learn xgboost pandas numpy openpyxl

# 2. Export trained models from the D1+D2 notebook (run this in Colab)
#    Add the following cell at the end of BBB_TaskD1_D2_Combined.ipynb:
#
#    import pickle, os
#    os.makedirs('models', exist_ok=True)
#    with open('models/bbb_tier1_consensus.pkl', 'wb') as f: pickle.dump(consensus, f)
#    with open('models/bbb_tier2_stacked.pkl',  'wb') as f: pickle.dump(stack, f)
#    # ABC models (from ABC pipeline notebook):
#    # with open('models/abc_pgp_model.pkl', 'wb') as f:  pickle.dump(pgp_model, f)
#    # with open('models/abc_bcrp_model.pkl', 'wb') as f: pickle.dump(bcrp_model, f)
#    # with open('models/abc_mrp1_model.pkl', 'wb') as f: pickle.dump(mrp1_model, f)
#    # with open('models/abc_mrp2_model.pkl', 'wb') as f: pickle.dump(mrp2_model, f)
#
#    Then download the models/ folder and place it next to app.py

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Directory structure

```
bbb_app/
├── app.py              ← main Streamlit application
├── requirements.txt    ← pinned dependencies
├── README.md
└── models/             ← place serialised .pkl files here
    ├── bbb_tier1_consensus.pkl
    ├── bbb_tier2_stacked.pkl
    ├── abc_pgp_model.pkl
    ├── abc_bcrp_model.pkl
    ├── abc_mrp1_model.pkl
    └── abc_mrp2_model.pkl
```

**Without model files:** the app runs in demo mode using physicochemical heuristics.
**With model files:** real predictions from the trained D1+D2 / ABC pipeline models.

---

## Deploy to Streamlit Cloud (free)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and `app.py` as the entry point
4. Add model files to the repo (if <100 MB) or use `st.secrets` + cloud storage

---

## Input format (batch mode)

**CSV or Excel** with a column named `smiles` (case-insensitive):

```csv
smiles
CC(C)NCC(O)COc1cccc2ccccc12
CN(C)C(=N)NC(N)=N
CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21
```

**Or paste directly** into the text box — same format, up to 100 rows.

---

## Output columns (batch CSV)

| Column | Description |
|--------|-------------|
| SMILES | Canonical RDKit SMILES |
| BBB | BBB+ or BBB− |
| Confidence_% | Model confidence (0–100%) |
| logBB | Predicted log brain/plasma ratio |
| P-gp_% | P-gp substrate probability |
| BCRP_% | BCRP substrate probability |
| MRP1_% | MRP1 substrate probability |
| MRP2_% | MRP2 substrate probability |
| Max_efflux_% | Highest efflux probability across 4 transporters |
| Overall_risk | Integrated CNS risk label |

---

## Model export cell for notebook

Add this as the final cell in `BBB_TaskD1_D2_Combined.ipynb`:

```python
# ── Task G: Export models for Streamlit app ─────────────────────────────────
import pickle, os

os.makedirs('models', exist_ok=True)

with open('models/bbb_tier1_consensus.pkl', 'wb') as f:
    pickle.dump(consensus, f)
    print('Saved: bbb_tier1_consensus.pkl')

with open('models/bbb_tier2_stacked.pkl', 'wb') as f:
    pickle.dump(stack, f)
    print('Saved: bbb_tier2_stacked.pkl')

print(f'Model files ready in ./models/')
print(f'Download and place next to app.py, then run: streamlit run app.py')
```

---

## Known limitations

- BBB− logBB prediction is unreliable (R² = −4.55) — flagged in the UI
- MRP1 / MRP2 predictions should be treated as directional (AUC 0.86–0.88)
- Valid for drug-like small molecules (MW ≤ 700 Da) only
- Not valid for peptides, biologics, ionic compounds

---

*Task G · BBB Penetration Platform · Dr. Barrile Lab · UC BME · May 2026*
