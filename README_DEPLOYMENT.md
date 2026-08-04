# Deploying the CNS Drug Penetration Predictor

This app is a single Streamlit script (`app.py`) plus a `models/` folder of
pickled model files. It has no database and no separate backend — everything
needed to run it must live inside the deployed repo.

## Repo layout the deployment expects

```
your-repo/
├── app.py
├── requirements.txt
└── models/
    ├── bbb_tier1_consensus.pkl
    ├── bbb_tier2_stacked.pkl
    ├── tier3_ppb_model.pkl
    ├── rf_final_v4.pkl
    ├── xgb_final_v4.pkl
    └── model_meta_v4.pkl
```

`app.py` resolves `MODEL_DIR` as a `models/` folder next to itself
(`os.path.dirname(__file__)`), so as long as the folder structure above is
preserved in the repo, no code changes are needed for either platform below.

**Before pushing:** check the file sizes of everything in `models/`. GitHub
blocks files over 100 MB by default (and warns above 50 MB) — if any `.pkl`
is close to that, they'll need Git LFS (`git lfs track "models/*.pkl"`)
rather than a plain `git add`.

---

## Option A — Streamlit Community Cloud (recommended first choice)

Free, and the least setup since Streamlit is the native format.

1. Push this repo to GitHub (can be a private repo — Streamlit Cloud reads it
   via GitHub auth either way).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **"New app"**.
3. Point it at the repo, branch, and `app.py` as the entry point.
4. Deploy. First build takes a few minutes (installing RDKit + scikit-learn +
   xgboost from `requirements.txt`).

**Known limits (free tier):** ~1 GB memory shared across the app, apps sleep
after 12 hours with no traffic (wakes back up in under a minute on the next
visit), unlimited public apps but only one private app, no custom domain.
For a manuscript companion demo this is normally fine — the only thing worth
watching is memory, since RDKit + two ensemble classifiers + a regressor
loaded at once can add up on a large batch run.

---

## Option B — Hugging Face Spaces (fallback if memory becomes an issue)

Also free, more headroom (2 CPU cores / 16 GB RAM on the free CPU-basic
tier), but Streamlit is no longer a one-click SDK option here — it now goes
through the **Docker** SDK using Hugging Face's Streamlit template.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick **Docker** as the SDK, then choose the Streamlit template when
   prompted.
3. Push this repo's contents into the Space repo the same way as a normal
   git remote (Spaces are git repos).
4. Space builds and serves automatically from the Dockerfile the template
   provides — `app.py` and `requirements.txt` slot in as-is.

Free-tier Spaces sleep after ~48 hours of inactivity (longer runway than
Streamlit Cloud's 12 hours) and restart on the next visit.

---

## Before either deploy: sanity-check locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Confirm all six model files show ✅ (not ❌) in the sidebar "Model files
expected in `models/`" checklist before pushing — a ❌ locally means the
same tier will silently be unavailable once deployed.
