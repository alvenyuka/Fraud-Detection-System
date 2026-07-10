"""
app.py — Step 5 of the model build-up: the live dashboard.

This turns everything the other scripts produced into something a visitor
can actually use in a browser, with four tabs:

    1. Score a Transaction  — type in one transaction, get a fraud score
    2. Model Performance    — the walk-forward results from src/validate.py
    3. Batch Scoring        — upload a CSV, get every row scored
    4. Monitoring           — the drift timeline from src/monitoring.py

Nothing on this page is invented — every number comes from a file that one
of the other scripts already produced, or from running the real model.

Run locally:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import ACTIVE_TYPES, FEATURE_COLS, engineer_features  # noqa: E402
from predict import score_dataframe  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = ROOT / "model" / "xgb_fraud_model.pkl"

st.set_page_config(page_title="Fraud Detection System", page_icon="🕵️", layout="wide")

# ---------------------------------------------------------------------------
# Colors — one small palette, used everywhere, instead of Plotly's defaults.
# ---------------------------------------------------------------------------

COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_BLUE = "#2a78d6"      # the one accent color, used for single-series charts
COLOR_GOOD = "#0ca30c"      # a prediction pushed toward "legitimate"
COLOR_CRITICAL = "#d03b3b"  # a prediction pushed toward "fraud"
# Fixed-order categorical colors for charts with more than one series
# (e.g. one line per feature) — never reassigned when the series list changes.
CATEGORICAL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]


def style_chart(figure: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    """Apply the same light, clean look to every chart on the page."""
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor=COLOR_SURFACE,
        paper_bgcolor=COLOR_SURFACE,
        font=dict(color=COLOR_TEXT),
        margin=dict(t=48, l=10, r=10, b=10),
    )
    figure.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    figure.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    return figure


# ---------------------------------------------------------------------------
# Loading the model and its saved results (done once, then cached by Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained model artifact that train.py produced."""
    artifact = joblib.load(MODEL_PATH)
    return artifact


def get_raw_xgboost_model(artifact):
    """
    Get the plain XGBoost model out of the calibrated model.

    train.py wraps the XGBoost model in a calibration step (see train.py's
    docstring) so its probabilities are trustworthy. SHAP (the tool we use to
    explain individual predictions) needs the plain tree model underneath,
    not the calibration wrapper — so we unwrap it here, once, and reuse it
    just for generating explanations. The calibrated model is still what
    actually produces the fraud score shown to the user.
    """
    calibrated_model = artifact["model"]
    first_fold = calibrated_model.calibrated_classifiers_[0]
    return first_fold.estimator.estimator


@st.cache_resource
def load_shap_explainer(_raw_xgb_model):
    return shap.TreeExplainer(_raw_xgb_model)


@st.cache_data
def load_saved_results():
    """Load every small file the other scripts (validate.py, monitoring.py) produced."""
    results = {}

    metrics_path = DATA_DIR / "metrics_summary.json"
    if metrics_path.exists():
        results["metrics_summary"] = json.loads(metrics_path.read_text())

    confusion_matrix_path = DATA_DIR / "confusion_matrix.json"
    if confusion_matrix_path.exists():
        results["confusion_matrix"] = json.loads(confusion_matrix_path.read_text())

    for csv_name in ["pr_curve", "calibration_curve", "walk_forward_results", "psi_timeline"]:
        csv_path = DATA_DIR / f"{csv_name}.csv"
        if csv_path.exists():
            results[csv_name] = pd.read_csv(csv_path)

    return results


artifact = load_model()
raw_xgb_model = get_raw_xgboost_model(artifact)
shap_explainer = load_shap_explainer(raw_xgb_model)
saved_results = load_saved_results()
OPERATING_THRESHOLD = artifact["operating_threshold"]


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("🕵️ Fraud Detection System")
st.caption(
    "XGBoost fraud classifier trained on PaySim mobile-money data — "
    "[source on GitHub](https://github.com/alvenyuka/Fraud-Detection-System)"
)

with st.sidebar:
    st.header("About this model")
    st.markdown(
        """
Trained on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) simulator
(6.36M transactions). Only `TRANSFER` and `CASH_OUT` types carry fraud in this
dataset — other transaction types are passed through unscored.
        """
    )
    st.info(
        "**Known limitations** (see `MODEL_CARD.md` for the full list):\n\n"
        "- PaySim is a *simulator* — how well this generalises to real transaction "
        "data is unverified.\n"
        "- **A transaction that drains 100% of the sender's balance scores high "
        "fraud probability almost by itself** — even when perfectly consistent. "
        "PaySim's simulated fraud is (almost) always a full drain, and its "
        "legitimate transactions almost never are, so the model picked up that "
        "correlation. A real account closure or full transfer would trigger this "
        "in the demo below — that's a property of this training data, not a bug.\n"
        "- The decision threshold below was chosen for PaySim's fraud rate — a real "
        "deployment would need its own threshold, tuned for its own fraud rate.",
        icon="ℹ️",
    )
    st.metric("Decision threshold", f"{OPERATING_THRESHOLD:.4f}")

tab_score, tab_performance, tab_batch, tab_monitoring = st.tabs(
    ["🔍 Score a Transaction", "📊 Model Performance", "📁 Batch Scoring", "📈 Monitoring"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Score a Transaction
# ---------------------------------------------------------------------------
with tab_score:
    st.subheader("Score a single transaction")
    st.caption("Fill in a transaction's details and see what the model would decide.")

    left_column, right_column = st.columns(2)
    with left_column:
        txn_type = st.selectbox("Transaction type", sorted(ACTIVE_TYPES))
        amount = st.number_input("Amount", min_value=0.0, value=181000.0, step=1000.0)
        old_balance_origin = st.number_input("Sender's balance before", min_value=0.0, value=181000.0, step=1000.0)
        new_balance_origin = st.number_input("Sender's balance after", min_value=0.0, value=0.0, step=1000.0)
    with right_column:
        old_balance_dest = st.number_input("Recipient's balance before", min_value=0.0, value=0.0, step=1000.0)
        new_balance_dest = st.number_input("Recipient's balance after", min_value=0.0, value=0.0, step=1000.0)

    if st.button("Score this transaction", type="primary"):
        # Step 1 — build a one-row table matching what the model expects.
        transaction = pd.DataFrame([{
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": old_balance_origin,
            "newbalanceOrig": new_balance_origin,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
        }])

        # Step 2 — add the same engineered features train.py uses.
        transaction_with_features = engineer_features(transaction)
        model_inputs = transaction_with_features[FEATURE_COLS]

        # Step 3 — ask the model for a fraud probability, then apply the decision threshold.
        fraud_probability = artifact["model"].predict_proba(model_inputs)[:, 1][0]
        is_flagged = fraud_probability >= OPERATING_THRESHOLD

        result_col1, result_col2 = st.columns(2)
        result_col1.metric("Fraud probability", f"{fraud_probability:.4%}")
        with result_col2:
            if is_flagged:
                st.error("🚩 Flagged as fraud", icon="🚩")
            else:
                st.success("✅ Looks legitimate", icon="✅")

        # A full balance drain scores high almost by itself in this model (see
        # the sidebar) — flag that explicitly so the reason is clear rather
        # than looking like a random misfire.
        drain_ratio = transaction_with_features["orig_drain_ratio"].iloc[0]
        if is_flagged and drain_ratio >= 0.999:
            st.warning(
                "This transaction drains ~100% of the sender's balance, which scores high "
                "in this model almost regardless of anything else — see the sidebar's "
                "known limitations.",
                icon="⚠️",
            )

        # Step 4 — explain the decision with SHAP: which features pushed the score up or down.
        st.markdown("**Why this score — feature-by-feature breakdown:**")
        shap_result = shap_explainer(model_inputs)
        contributions = shap_result.values[0]
        waterfall_figure = go.Figure(go.Waterfall(
            orientation="h",
            y=FEATURE_COLS,
            x=contributions,
            base=shap_result.base_values[0],
            increasing=dict(marker=dict(color=COLOR_CRITICAL)),   # pushes toward fraud
            decreasing=dict(marker=dict(color=COLOR_GOOD)),        # pushes toward legitimate
            connector=dict(line=dict(color=COLOR_GRID)),
        ))
        style_chart(waterfall_figure, "Each feature's contribution to the score")
        waterfall_figure.update_layout(height=350)
        st.plotly_chart(waterfall_figure, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2 — Model Performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.subheader("Walk-forward validation results")

    if "metrics_summary" in saved_results:
        summary = saved_results["metrics_summary"]
        st.caption(
            f"Results from {summary['n_folds']} time-based folds spanning the whole dataset — "
            "not just the single train/test split this project started with."
        )

        metric_columns = st.columns(6)
        for column, metric_name in zip(metric_columns, ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier"]):
            metric_stats = summary[metric_name]
            column.metric(metric_name, f"{metric_stats['mean']:.4f}", f"± {metric_stats['std']:.4f}")

        st.markdown("**Results per fold:**")
        st.dataframe(pd.DataFrame(summary["folds"]), width="stretch")
    else:
        st.info("Run `make validate` first to generate these results.")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if "pr_curve" in saved_results:
            pr_curve = saved_results["pr_curve"]
            figure = go.Figure(go.Scatter(
                x=pr_curve["recall"], y=pr_curve["precision"], mode="lines",
                line=dict(color=COLOR_BLUE, width=2),
            ))
            style_chart(figure, "Precision-Recall curve (most recent fold)", "Recall", "Precision")
            st.plotly_chart(figure, width="stretch")
    with chart_col2:
        if "calibration_curve" in saved_results:
            calibration = saved_results["calibration_curve"]
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=calibration["mean_predicted"], y=calibration["fraction_positive"],
                mode="lines+markers", name="Model", line=dict(color=COLOR_BLUE, width=2),
            ))
            figure.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                line=dict(dash="dash", color=COLOR_MUTED),
            ))
            style_chart(figure, "Calibration curve (most recent fold)", "Predicted probability", "Actual fraud rate")
            st.plotly_chart(figure, width="stretch")

    if "confusion_matrix" in saved_results:
        cm = saved_results["confusion_matrix"]
        figure = go.Figure(go.Heatmap(
            z=cm["matrix"], x=cm["labels"], y=cm["labels"],
            text=cm["matrix"], texttemplate="%{text}",
            colorscale=[[0, COLOR_SURFACE], [1, COLOR_BLUE]],
            showscale=False,
        ))
        style_chart(figure, "Confusion matrix (most recent fold)", "Predicted", "Actual")
        st.plotly_chart(figure, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 — Batch Scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a whole CSV of transactions")
    st.caption(
        "Expected columns: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest "
        "— this is the same file format src/predict.py accepts from the command line."
    )
    uploaded_file = st.file_uploader("Upload a transactions CSV", type="csv")
    if uploaded_file is not None:
        transactions = pd.read_csv(uploaded_file)
        scored_transactions = score_dataframe(transactions, artifact)

        n_flagged = int(scored_transactions["fraud_flag"].sum())
        st.success(f"Scored {len(scored_transactions):,} transactions — {n_flagged:,} flagged ({100 * n_flagged / len(scored_transactions):.2f}%)")
        st.dataframe(scored_transactions, width="stretch")
        st.download_button(
            "Download scored CSV",
            scored_transactions.to_csv(index=False).encode("utf-8"),
            file_name="scored.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Tab 4 — Monitoring
# ---------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("Simulated drift monitoring")
    st.caption(
        "How much each feature's distribution has drifted over time, measured against the "
        "earliest time window as the reference. This simulates what production monitoring "
        "would look like — it is not real production traffic."
    )
    if "psi_timeline" in saved_results:
        psi_timeline = saved_results["psi_timeline"]
        figure = go.Figure()
        # Fixed color order per feature, so a feature keeps its color no
        # matter which others are present (see the dataviz "color follows
        # the entity" rule).
        for i, feature_name in enumerate(sorted(psi_timeline["feature"].unique())):
            feature_data = psi_timeline[psi_timeline["feature"] == feature_name]
            color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
            figure.add_trace(go.Scatter(
                x=feature_data["window_start_step"], y=feature_data["psi"],
                mode="lines+markers", name=feature_name,
                line=dict(color=color, width=2), marker=dict(size=6),
            ))
        figure.add_hline(y=0.1, line_dash="dot", annotation_text="moderate shift (0.10)", line_color="#eda100")
        figure.add_hline(y=0.25, line_dash="dot", annotation_text="significant shift (0.25)", line_color=COLOR_CRITICAL)
        style_chart(figure, "Drift (PSI) over time, by feature", "Time step (window start)", "PSI")
        st.plotly_chart(figure, width="stretch")
    else:
        st.info("Run `make monitor` first to generate the drift timeline.")
