# RAP Concrete RCPT & DME Predictor (Streamlit)

This app predicts **RCPT** (and **DME** if a DME model file is provided) from inputs:
`C, FA, FA/Binder, A, T CRAP, CA, Age` using an **Optuna-tuned XGBoost regressor**.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to a GitHub repo
2. On https://streamlit.io/cloud create a new app and select the repo
3. Set the main file to `app.py`

## Files

- `app.py` Streamlit app
- `xgb_optuna_rcpt.pkl` trained model for RCPT
- `xgb_optuna_dme.pkl` optional model for DME (add this if you have it)
- `feature_cols.pkl` feature ordering
- `input_feature_ranges.csv` used for UI input bounds
- `xgb_optuna_params.pkl` stored hyperparameters
