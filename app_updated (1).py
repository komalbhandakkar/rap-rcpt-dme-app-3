import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def pth(fname: str) -> str:
    return os.path.join(BASE_DIR, fname)

st.set_page_config(page_title='RCPT and DME Predictor for RAP Concrete', layout='wide')

@st.cache_resource
def load_artifacts():
    feature_cols_local = joblib.load(pth('feature_cols.pkl'))
    ranges_df_local = pd.read_csv(pth('input_feature_ranges.csv'), index_col=0)
    rcpt_model_local = joblib.load(pth('xgb_optuna_rcpt.pkl'))
    params_local = joblib.load(pth('xgb_optuna_params.pkl'))

    dme_model_local = None
    dme_path = pth('xgb_optuna_dme.pkl')
    if os.path.exists(dme_path):
        dme_model_local = joblib.load(dme_path)

    return feature_cols_local, ranges_df_local, rcpt_model_local, dme_model_local, params_local

feature_cols, ranges_df, rcpt_model, dme_model, params = load_artifacts()

st.title('RCPT and DME Predictor for RAP Concrete')

tabs = st.tabs(['Single Prediction', 'Batch Prediction'])

with tabs[0]:
    st.subheader('Single Prediction')

    # Ranges file format in your repo is a bit unusual; attempt to parse robustly
    def get_range(col_name: str):
        try:
            # Expect rows min/max and a list-like string in cell
            if 'min' in ranges_df.index and 'max' in ranges_df.index:
                min_row = ranges_df.loc['min'].values.tolist()
                max_row = ranges_df.loc['max'].values.tolist()
                # If ranges_df is a single column holding list-strings
                if len(ranges_df.columns) == 1:
                    import ast
                    min_list = ast.literal_eval(str(ranges_df.iloc[ranges_df.index.tolist().index('min'), 0]))
                    max_list = ast.literal_eval(str(ranges_df.iloc[ranges_df.index.tolist().index('max'), 0]))
                    # list format: ['C', min, max, mean, std]
                    if len(min_list) >= 3 and str(min_list[0]) == col_name:
                        return float(min_list[1]), float(min_list[2])
                    if len(max_list) >= 3 and str(max_list[0]) == col_name:
                        return float(max_list[1]), float(max_list[2])
                # Fallback defaults
        except Exception:
            pass
        return None, None

    input_vals = {}
    cols_ui = st.columns(3)
    for idx_col, col_name in enumerate(feature_cols):
        mn, mx = get_range(col_name)
        if mn is None or mx is None:
            mn = 0.0
            mx = 1000.0
        default_val = (mn + mx) / 2.0
        with cols_ui[idx_col % 3]:
            input_vals[col_name] = st.number_input(col_name, value=float(default_val), min_value=float(mn), max_value=float(mx))

    single_df = pd.DataFrame([input_vals])

    if st.button('Predict'):
        X_single = single_df[feature_cols].apply(pd.to_numeric, errors='coerce')
        pred_rcpt = float(rcpt_model.predict(X_single)[0])
        pred_dme = None
        if dme_model is not None:
            pred_dme = float(dme_model.predict(X_single)[0])

        st.session_state['last_single_input_df'] = single_df
        st.session_state['last_single_pred_rcpt'] = pred_rcpt
        st.session_state['last_single_pred_dme'] = pred_dme

        st.success('Prediction complete')
        c1, c2 = st.columns(2)
        with c1:
            st.metric('Predicted RCPT (Coulombs)', pred_rcpt)
        with c2:
            if pred_dme is None:
                st.info('DME model not found')
            else:
                st.metric('Predicted DME (GPa)', pred_dme)

with tabs[1]:
    st.subheader('Batch Prediction')
    st.write('Upload a CSV with columns: ' + ', '.join(feature_cols))

    up_file = st.file_uploader('Upload CSV', type=['csv'])
    if up_file is not None:
        batch_df = pd.read_csv(up_file)
        batch_df.columns = [c.strip() for c in batch_df.columns]
        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if len(missing_cols) > 0:
            st.error('Missing columns: ' + ', '.join(missing_cols))
        else:
            X_batch = batch_df[feature_cols].apply(pd.to_numeric, errors='coerce')
            if X_batch.isna().any().any():
                st.warning('Some rows have non-numeric or missing values; those rows may produce NaN predictions.')

            pred_rcpt_b = rcpt_model.predict(X_batch)
            pred_dme_b = None
            if dme_model is not None:
                pred_dme_b = dme_model.predict(X_batch)

            out_df = batch_df.copy()
            out_df['RCPT_pred'] = pred_rcpt_b
            if pred_dme_b is not None:
                out_df['DME_pred'] = pred_dme_b

            st.dataframe(out_df.head(50), use_container_width=True)
            out_csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions (CSV)', data=out_csv, file_name='batch_predictions.csv', mime='text/csv')

# ------------------------
# Results
# ------------------------
st.header('Results')


# --- Clear figures section (auto-inserted) ---
st.subheader('Figures (clear)')
fig_a = pth('image_webclear.png')
fig_b = pth('image_(1)_webclear.png')
if os.path.exists(fig_a):
    st.image(fig_a, use_container_width=True)
if os.path.exists(fig_b):
    st.image(fig_b, use_container_width=True)
# --- end clear figures section ---
st.subheader('a) Predict RCPT and DME')
st.caption('This section echoes your most recent Single Prediction run.')

if 'last_single_input_df' in st.session_state:
    st.write('Last single-input values')
    st.dataframe(st.session_state['last_single_input_df'], use_container_width=True)

col_m1, col_m2 = st.columns(2)
with col_m1:
    if 'last_single_pred_rcpt' in st.session_state:
        st.metric('Predicted RCPT (Coulombs)', float(st.session_state['last_single_pred_rcpt']))
    else:
        st.info('Run a Single Prediction to populate this result.')

with col_m2:
    if 'last_single_pred_dme' in st.session_state and st.session_state['last_single_pred_dme'] is not None:
        st.metric('Predicted DME (GPa)', float(st.session_state['last_single_pred_dme']))
    else:
        st.info('Run a Single Prediction to populate this result (or ensure DME model exists).')

st.subheader('b) Pearson Correlation Coefficients')
if os.path.exists(pth('pearson_corr_table.csv')):
    corr_df_disp = pd.read_csv(pth('pearson_corr_table.csv'), index_col=0)
    st.dataframe(corr_df_disp, use_container_width=True)
    with open(pth('pearson_corr_table.csv'), 'rb') as f_corr:
        st.download_button('Download Pearson correlation table (CSV)', data=f_corr, file_name='pearson_corr_table.csv', mime='text/csv')
else:
    st.info('pearson_corr_table.csv not found in the app directory.')

if os.path.exists(pth('pearson_corr_heatmap_lower_tri.png')):
    st.image(pth('pearson_corr_heatmap_lower_tri.png'), caption='Pearson correlation heatmap (lower-triangular)', use_container_width=True)
    with open(pth('pearson_corr_heatmap_lower_tri.png'), 'rb') as f_img:
        st.download_button('Download Pearson heatmap (PNG)', data=f_img, file_name='pearson_corr_heatmap_lower_tri.png', mime='image/png')
else:
    st.info('pearson_corr_heatmap.png not found in the app directory.')

st.subheader('c) Shap Plot Heatmap')
st.caption('True SHAP summary plot (beeswarm) computed from your Optuna-tuned XGBoost model and your dataset.')

col_sh1, col_sh2 = st.columns(2)
with col_sh1:
    if os.path.exists(pth('SHAP_summary_RCPT_beeswarm.png')):
        st.image(pth('SHAP_summary_RCPT_beeswarm.png'), caption='True SHAP summary (beeswarm) - RCPT', use_container_width=True)
        with open(pth('SHAP_summary_RCPT_beeswarm.png'), 'rb') as f_s1:
            st.download_button('Download SHAP heatmap RCPT (PNG)', data=f_s1, file_name='SHAP_summary_RCPT_beeswarm.png', mime='image/png')

    else:
        # fallback to legacy SHAP-like heatmap if present
        if os.path.exists(pth('SHAP_like_Heatmap_RCPT_600dpi.png')):
            st.image(pth('SHAP_like_Heatmap_RCPT_600dpi.png'), caption='SHAP-like heatmap (fallback) - RCPT', use_container_width=True)
        else:
            st.info('True SHAP heatmap for RCPT not found.')

with col_sh2:
    # Single SHAP figure shown in col_sh1; keep col_sh2 empty for layout consistency
    st.empty()
