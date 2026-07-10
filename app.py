"""
Run:
    streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Telecom Fraud Detection",
    page_icon="📞",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load artifact
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifact(path="model_artifact.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(
        "`model_artifact.pkl` not found. Run `python train_model.py --data "
        "<your_csv_path>` first to train the model and generate this file."
    )
    st.stop()

FEATURE_COLS = artifact["feature_cols"]
MEDIANS = artifact["medians"]
CLIP_BOUNDS = artifact["clip_bounds"]
RAW_MIN_MAX = artifact["raw_min_max"]
BEST_MODEL_NAME = artifact["best_model_name"]
BEST_MODEL = artifact["best_model"]
SCALER = artifact["scaler"]
NEEDS_SCALING = artifact["needs_scaling"]

# Feature metadata: friendly label, help text, whether it's a percentage/ratio (0-1),
# and a sensible step size for the number input.
FEATURE_META = {
    "caller_age_days": {
        "label": "Caller Account Age (days)",
        "help": "How long the caller's number/account has existed. Fraudulent numbers tend to be newer.",
        "step": 1.0, "is_ratio": False,
    },
    "calls_per_day": {
        "label": "Calls per Day",
        "help": "Average number of calls made by this caller per day.",
        "step": 0.1, "is_ratio": False,
    },
    "call_duration_sec": {
        "label": "This Call's Duration (sec)",
        "help": "Duration of this specific call, in seconds.",
        "step": 1.0, "is_ratio": False,
    },
    "avg_call_duration_sec": {
        "label": "Average Call Duration (sec)",
        "help": "This caller's average call duration across all their calls.",
        "step": 1.0, "is_ratio": False,
    },
    "unique_receivers_24h": {
        "label": "Unique Receivers (last 24h)",
        "help": "Number of distinct people this caller contacted in the last 24 hours.",
        "step": 1.0, "is_ratio": False,
    },
    "receiver_block_rate": {
        "label": "Receiver Block Rate",
        "help": "Fraction of receivers who have blocked this caller (0 = none, 1 = all).",
        "step": 0.01, "is_ratio": True,
    },
    "spam_reports_count": {
        "label": "Spam Reports Count",
        "help": "How many times this caller has been reported as spam.",
        "step": 1.0, "is_ratio": False,
    },
    "country_code_risk_score": {
        "label": "Country Code Risk Score",
        "help": "Risk score for the caller's country/region, based on historical fraud rates (0 = low risk, 1 = high risk).",
        "step": 0.01, "is_ratio": True,
    },
    "night_call_ratio": {
        "label": "Night Call Ratio",
        "help": "Proportion of this caller's calls made during night hours.",
        "step": 0.01, "is_ratio": True,
    },
    "sequential_dialing_score": {
        "label": "Sequential Dialing Score",
        "help": "How 'sequential'/automated the calling pattern looks (e.g. dialing numbers in order) — typical of robocalls.",
        "step": 0.01, "is_ratio": True,
    },
    "graph_degree": {
        "label": "Graph Degree",
        "help": "Number of distinct connections this caller has in the call network.",
        "step": 1.0, "is_ratio": False,
    },
    "previous_fraud_associations": {
        "label": "Previous Fraud Associations",
        "help": "Count of past connections to numbers/accounts previously flagged as fraudulent.",
        "step": 1.0, "is_ratio": False,
    },
    "reputation_score": {
        "label": "Reputation Score",
        "help": "Overall trust/reputation score for the caller (higher = more trustworthy, 0-1).",
        "step": 0.01, "is_ratio": True,
    },
}

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("📞 Telecom Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Fraud Checker", "📊 Model Performance", "📈 Data Insights"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Active model: **{BEST_MODEL_NAME}**")
st.sidebar.caption(f"Trained on **{artifact['n_rows']:,}** cleaned rows")

# ========================================================================
# PAGE 1: Fraud Checker (user input form)
# ========================================================================
if page == "🔍 Fraud Checker":
    st.title("🔍 Check a Call for Fraud Risk")
    st.write(
        "Enter the caller/call details below. Fields are constrained to "
        "realistic ranges to prevent invalid input — hover the **?** next "
        "to each field for what it means."
    )

    with st.form("fraud_check_form"):
        col1, col2 = st.columns(2)
        user_values = {}

        for i, feat in enumerate(FEATURE_COLS):
            meta = FEATURE_META.get(feat, {"label": feat, "help": "", "step": 1.0, "is_ratio": False})
            bounds = RAW_MIN_MAX[feat]
            target_col = col1 if i % 2 == 0 else col2

            default_val = float(MEDIANS[feat])

            if meta["is_ratio"]:
                # Ratios/scores are always 0-1, regardless of raw data noise.
                lo, hi = 0.0, 1.0
                default_val = min(max(default_val, lo), hi)
                user_values[feat] = target_col.slider(
                    meta["label"], min_value=lo, max_value=hi,
                    value=round(default_val, 2), step=meta["step"],
                    help=meta["help"], key=feat,
                )
            else:
                lo = max(0.0, float(bounds["min"]))
                hi = float(bounds["max"])
                if hi <= lo:
                    hi = lo + 1.0
                default_val = min(max(default_val, lo), hi)
                user_values[feat] = target_col.number_input(
                    meta["label"], min_value=lo, max_value=hi,
                    value=round(default_val, 2), step=meta["step"],
                    help=meta["help"], key=feat,
                )

        submitted = st.form_submit_button("🚨 Check for Fraud", use_container_width=True)

    if submitted:
        # Build input row, apply the SAME IQR clip bounds used in training
        # so extreme-but-in-range values behave consistently with the model.
        row = {}
        for feat in FEATURE_COLS:
            val = float(user_values[feat])
            b = CLIP_BOUNDS[feat]
            clipped = min(max(val, b["lower"]), b["upper"])
            row[feat] = clipped

        input_df = pd.DataFrame([row])[FEATURE_COLS]

        if NEEDS_SCALING:
            model_input = SCALER.transform(input_df)
        else:
            model_input = input_df

        pred = BEST_MODEL.predict(model_input)[0]
        proba = BEST_MODEL.predict_proba(model_input)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.error("🚨 **FRAUD RISK DETECTED**")
            else:
                st.success("✅ **LIKELY LEGITIMATE**")
            st.metric("Fraud Probability", f"{proba * 100:.1f}%")

        with res_col2:
            st.progress(min(max(proba, 0.0), 1.0))
            if proba >= 0.75:
                st.write("**Risk level: Very High** — recommend blocking/flagging for review.")
            elif proba >= 0.5:
                st.write("**Risk level: High** — recommend manual review.")
            elif proba >= 0.25:
                st.write("**Risk level: Moderate** — monitor this caller.")
            else:
                st.write("**Risk level: Low** — no action needed.")

        # Show which inputs were auto-clipped (helps users understand
        # why a value they typed may have been treated differently)
        clipped_notes = [
            f"- **{FEATURE_META.get(f, {'label': f})['label']}**: {user_values[f]:.2f} → capped to {row[f]:.2f}"
            for f in FEATURE_COLS
            if abs(float(user_values[f]) - row[f]) > 1e-9
        ]
        if clipped_notes:
            with st.expander("ℹ️ Some values were capped to realistic bounds before scoring"):
                st.markdown("\n".join(clipped_notes))

        if hasattr(BEST_MODEL, "feature_importances_") or artifact.get("rf_feature_importance"):
            with st.expander("🔎 What drives this model's predictions?"):
                imp = artifact["rf_feature_importance"]
                imp_series = pd.Series(imp).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.barplot(x=imp_series.values, y=imp_series.index, palette="magma", ax=ax)
                ax.set_xlabel("Importance")
                ax.set_title("Random Forest Feature Importance")
                st.pyplot(fig)

# ========================================================================
# PAGE 2: Model Performance
# ========================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.write(f"Best model selected (highest F1 on the test set): **{BEST_MODEL_NAME}**")

    results_df = artifact["results_df"]
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1": "{:.3f}", "ROC_AUC": "{:.3f}",
        }),
        use_container_width=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    best_row = results_df.iloc[0]
    m1.metric("Accuracy", f"{best_row['Accuracy']*100:.1f}%")
    m2.metric("Precision", f"{best_row['Precision']*100:.1f}%")
    m3.metric("Recall", f"{best_row['Recall']*100:.1f}%")
    m4.metric("F1 Score", f"{best_row['F1']*100:.1f}%")

    st.caption(
        f"5-fold cross-validated F1 (Random Forest): "
        f"{artifact['cv_f1_mean']:.3f} ± {artifact['cv_f1_std']:.3f}"
    )

    st.subheader("Model Comparison Chart")
    fig, ax = plt.subplots(figsize=(9, 5))
    melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"])
    sns.barplot(data=melted, x="Model", y="value", hue="variable", palette="magma", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Confusion Matrix (Best Model)")
    best_result = next(r for r in artifact["results"] if r["Model"] == BEST_MODEL_NAME)
    cm = np.array(best_result["confusion_matrix"])
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
                xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"], ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    with st.expander("Full classification report"):
        st.text(best_result["classification_report"])

    st.subheader("Feature Importance (Random Forest)")
    imp_series = pd.Series(artifact["rf_feature_importance"]).sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=imp_series.values, y=imp_series.index, palette="magma", ax=ax3)
    ax3.set_xlabel("Importance")
    st.pyplot(fig3)

# ========================================================================
# PAGE 3: Data Insights
# ========================================================================
else:
    st.title("📈 Data Insights")

    st.subheader("Class Balance")
    balance = artifact["class_balance"]
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Not Fraud" if k == 0 else "Fraud" for k in balance.keys()]
    ax.pie(balance.values(), labels=labels, autopct="%1.1f%%", colors=["#4c72b0", "#c44e52"])
    st.pyplot(fig)

    st.subheader("Feature Correlation Matrix")
    corr = artifact["correlation_matrix"]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="magma", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Feature Reference")
    ref_rows = []
    for feat in FEATURE_COLS:
        meta = FEATURE_META.get(feat, {"label": feat, "help": ""})
        ref_rows.append({
            "Feature": meta["label"],
            "Description": meta["help"],
            "Median (training data)": round(MEDIANS[feat], 3),
        })
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
