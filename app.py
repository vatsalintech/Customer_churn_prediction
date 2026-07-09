from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def _ensure_sklearn_pickle_compat():
    """Allow loading models saved with scikit-learn 1.6.x on newer versions."""
    try:
        import sklearn.compose._column_transformer as column_transformer
    except ImportError:
        return

    if hasattr(column_transformer, "_RemainderColsList"):
        return

    class _RemainderColsList(list):
        pass

    column_transformer._RemainderColsList = _RemainderColsList

BASE_DIR = Path(__file__).parent
ARTIFACTS_PATH = BASE_DIR / "churn_model_artifacts.joblib"

HIDDEN_DEFAULTS = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "PaperlessBilling": "Yes",
}

YES_NO = ["No", "Yes"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
INTERNET_OPTIONS = ["Fiber optic", "DSL", "No"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

LIKELY_CHURN_EXAMPLE = {
    "internet_service": "Fiber optic",
    "tenure": 1,
    "contract": "Month-to-month",
    "monthly_charges": 30.0,
    "payment_method": "Electronic check",
    "multiple_lines": "Yes",
}

LIKELY_STAY_EXAMPLE = {
    "internet_service": "No",
    "tenure": 72,
    "contract": "Two year",
    "monthly_charges": 70.0,
    "payment_method": "Bank transfer (automatic)",
    "multiple_lines": "No",
}

PROJECT_TITLE = "ChurnScope"
PROJECT_LOGO = "📈"
PROJECT_SUBTITLE = "Customer churn intelligence and value forecasting"

DEFAULT_INPUTS = {
    "internet_service": "Fiber optic",
    "tenure": 12,
    "contract": "Month-to-month",
    "monthly_charges": 65.0,
    "payment_method": "Electronic check",
    "multiple_lines": "No",
}


@st.cache_resource
def load_artifacts():
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {ARTIFACTS_PATH}. "
            "Run export_model.py or the training notebook to generate churn_model_artifacts.joblib."
        )

    _ensure_sklearn_pickle_compat()
    artifacts = joblib.load(ARTIFACTS_PATH)
    return artifacts["model"], artifacts["optimal_threshold"]


def inject_compact_css():
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] { display: none; }
        footer { display: none; }
        #MainMenu { visibility: hidden; }
        .block-container {
            padding-top: 0.2rem;
            padding-bottom: 0.35rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 1400px;
        }
        h1 {
            font-size: 1.24rem;
            margin: 0 0 0.1rem 0;
            padding: 0;
        }
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .brand-logo {
            font-size: 1.55rem;
            line-height: 1;
        }
        .brand-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0;
        }
        .brand-subtitle {
            font-size: 0.72rem;
            opacity: 0.85;
            margin: 0.1rem 0 0 0;
        }
        .section-title {
            font-size: 0.88rem;
            font-weight: 700;
            margin: 0.15rem 0 0.08rem 0;
        }
        .section-title-inline {
            font-size: 1rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.4rem;
            white-space: nowrap;
        }
        .section-block {
            margin-bottom: 0;
        }
        .placeholder-card {
            min-height: 10rem;
            border: 1px dashed rgba(255,255,255,0.14);
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1.25rem;
            background: rgba(255,255,255,0.02);
        }
        .results-card {
            border-radius: 18px;
            padding: 0.25rem 0.5rem;
        }
        .placeholder-emoji {
            font-size: 1.9rem;
            margin-bottom: 0.35rem;
        }
        .placeholder-title {
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        [data-testid="stCaptionContainer"] p {
            margin-bottom: 0.2rem;
            font-size: 0.72rem;
        }
        [data-testid="stVerticalBlock"] > div {
            gap: 0;
        }
        [data-testid="element-container"] {
            margin-bottom: 0.1rem;
        }
        [data-testid="stMarkdownContainer"] {
            margin-bottom: 0;
            padding-bottom: 0;
        }
        [data-testid="stHorizontalBlock"] {
            margin-bottom: -0.35rem;
        }
        [data-testid="column"] {
            padding-top: 0;
        }
        div[data-testid="stForm"] {
            margin-top: -0.1rem;
            margin-bottom: -0.45rem;
        }
        div[data-testid="stForm"] form {
            padding-top: 0;
            padding-bottom: 0;
        }
        div[data-testid="stForm"] [data-testid="element-container"] {
            margin-bottom: 0.18rem;
        }
        div[data-testid="stFormSubmitButton"] {
            display: flex;
            justify-content: center;
            margin-top: 0.05rem;
            margin-bottom: 0;
        }
        div[data-testid="stButton"] > button {
            border-radius: 10px;
            font-size: 0.78rem;
            padding-top: 0.32rem;
            padding-bottom: 0.32rem;
            max-width: 10.75rem;
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stSlider"] {
            font-size: 0.84rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
        }
        div[data-testid="stFormSubmitButton"] > button {
            min-width: 11rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("prediction_result", None)


def load_example(example: dict):
    for key, value in example.items():
        st.session_state[key] = value
    st.session_state["prediction_result"] = None


def apply_internet_addons(internet_service: str) -> dict:
    if internet_service == "No":
        value = "No internet service"
        return {
            "OnlineSecurity": value,
            "TechSupport": value,
            "StreamingTV": value,
            "StreamingMovies": value,
        }
    return {
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }


def build_input_frame(inputs: dict) -> pd.DataFrame:
    tenure = int(inputs["tenure"])
    monthly_charges = float(inputs["monthly_charges"])
    internet_service = inputs["internet_service"]
    contract = inputs["contract"]
    total_charges = monthly_charges * max(tenure, 1)
    avg_monthly_spend = total_charges / tenure if tenure > 0 else monthly_charges

    row = {
        **HIDDEN_DEFAULTS,
        **apply_internet_addons(internet_service),
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": inputs["payment_method"],
        "MultipleLines": inputs["multiple_lines"],
        "avg_monthly_spend": avg_monthly_spend,
        "has_internet": int(internet_service != "No"),
        "long_term_contract": int(contract in ["One year", "Two year"]),
    }
    return pd.DataFrame([row])


def predict_customer(model, threshold: float, customer_df: pd.DataFrame) -> tuple[float, int, str]:
    probability = float(model.predict_proba(customer_df)[0, 1])
    will_churn = int(probability >= threshold)
    label = "Likely to churn" if will_churn else "Likely to stay"
    return probability, will_churn, label


def estimate_ltv(inputs: dict, churn_probability: float) -> dict:
    monthly_charges = float(inputs["monthly_charges"])
    tenure = int(inputs["tenure"])
    contract = inputs["contract"]

    contract_horizon = {
        "Month-to-month": 12,
        "One year": 18,
        "Two year": 24,
    }
    expected_future_months = contract_horizon.get(contract, 12) * (1 - churn_probability)
    billed_to_date = monthly_charges * max(tenure, 1)
    projected_future_value = monthly_charges * expected_future_months
    estimated_ltv = billed_to_date + projected_future_value

    return {
        "billed_to_date": billed_to_date,
        "expected_future_months": expected_future_months,
        "projected_future_value": projected_future_value,
        "estimated_ltv": estimated_ltv,
    }


def show_prediction(model, threshold: float, inputs: dict):
    customer_df = build_input_frame(inputs)
    probability, will_churn, label = predict_customer(model, threshold, customer_df)
    ltv = estimate_ltv(inputs, probability)
    st.session_state["prediction_result"] = {
        "probability": probability,
        "will_churn": will_churn,
        "label": label,
        "ltv": ltv,
    }


def render_results():
    result = st.session_state.get("prediction_result")
    if not result:
        st.markdown(
            """
            <div class="placeholder-card">
                <div>
                    <div class="placeholder-emoji">🔎</div>
                    <div class="placeholder-title">Prediction panel</div>
                    <div>Fill the target profile and press <b>Predict</b> to see churn risk, retention outlook, and estimated LTV.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    probability = result["probability"]
    will_churn = result["will_churn"]
    label = result["label"]
    ltv = result["ltv"]

    with st.container(border=True):
        if will_churn:
            st.error(f"**{label}** — {probability:.0%} churn risk")
        else:
            st.success(f"**{label}** — {probability:.0%} churn risk")

        st.progress(min(max(probability, 0.0), 1.0))

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Churn probability", f"{probability:.1%}")
        metric_col2.metric("Estimated LTV", f"${ltv['estimated_ltv']:,.0f}")
        metric_col3.metric("Billed to date", f"${ltv['billed_to_date']:,.0f}")
        metric_col4.metric("Expected future months", f"{ltv['expected_future_months']:.1f}")

        if will_churn:
            st.caption("Consider a retention offer: longer contract, support check-in, or payment review.")
        else:
            st.caption("Customer looks stable. Standard engagement is fine.")

        st.caption("Estimated LTV = billed to date + churn-adjusted future revenue.")


def main():
    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_compact_css()
    init_session_state()

    model, threshold = load_artifacts()

    st.markdown(
        f"""
        <div class="brand-wrap">
            <div class="brand-logo">{PROJECT_LOGO}</div>
            <div>
                <div class="brand-title">{PROJECT_TITLE}</div>
                <div class="brand-subtitle">{PROJECT_SUBTITLE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prefill_title, prefill_btn1, prefill_btn2 = st.columns([0.1, 0.15, 0.8], gap="small")
    with prefill_title:
        st.markdown('<div class="section-title-inline">Prefill the data:</div>', unsafe_allow_html=True)
    with prefill_btn1:
        if st.button("High-risk customer", use_container_width=True):
            load_example(LIKELY_CHURN_EXAMPLE)
            st.rerun()
    with prefill_btn2:
        if st.button("Loyal customer", use_container_width=True):
            load_example(LIKELY_STAY_EXAMPLE)
            st.rerun()

    st.markdown('<div class="section-title">Target profile</div>', unsafe_allow_html=True)
    with st.form("target_profile_form", clear_on_submit=False):
        row1_col1, row1_col2, row1_col3 = st.columns(3, gap="medium")
        with row1_col1:
            st.selectbox("Internet plan", INTERNET_OPTIONS, key="internet_service")
        with row1_col2:
            st.slider("Tenure (months)", 0, 72, key="tenure")
        with row1_col3:
            st.selectbox("Contract", CONTRACT_OPTIONS, key="contract")

        row2_col1, row2_col2, row2_col3 = st.columns(3, gap="medium")
        with row2_col1:
            st.slider("Monthly bill ($)", 18.0, 120.0, step=1.0, key="monthly_charges")
        with row2_col2:
            st.selectbox("Payment method", PAYMENT_OPTIONS, key="payment_method")
        with row2_col3:
            st.selectbox("Multiple phone lines", YES_NO, key="multiple_lines")

        predict_spacer_l, predict_col, predict_spacer_r = st.columns([1.15, 1, 1.15])
        with predict_col:
            predict_clicked = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        inputs = {
            "internet_service": st.session_state.internet_service,
            "tenure": st.session_state.tenure,
            "contract": st.session_state.contract,
            "monthly_charges": st.session_state.monthly_charges,
            "payment_method": st.session_state.payment_method,
            "multiple_lines": st.session_state.multiple_lines,
        }
        show_prediction(model, threshold, inputs)

    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    render_results()


if __name__ == "__main__":
    main()
