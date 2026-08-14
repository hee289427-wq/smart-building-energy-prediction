
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Building Energy Prediction Lab",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at top right, #eef8ff 0%, transparent 25%),
            linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .hero {
        padding: 42px 46px;
        border-radius: 26px;
        background:
            radial-gradient(circle at 90% 10%, rgba(36, 211, 172, .22), transparent 28%),
            linear-gradient(135deg, #071a33 0%, #0d3151 58%, #0b4560 100%);
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 20px 55px rgba(12, 42, 74, .16);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 760;
        line-height: 1.08;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #d8e9f5;
        max-width: 900px;
        line-height: 1.65;
    }

    .eyebrow {
        display: inline-block;
        font-size: 12px;
        letter-spacing: .12em;
        font-weight: 700;
        color: #75e4c6;
        margin-bottom: 12px;
    }

    .info-card {
        background: rgba(255, 255, 255, .95);
        border: 1px solid #e3eaf1;
        border-radius: 18px;
        padding: 21px 22px;
        min-height: 126px;
        box-shadow: 0 8px 28px rgba(28, 50, 74, .07);
    }

    .info-card-title {
        color: #6a7787;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .info-card-value {
        color: #102d49;
        font-size: 27px;
        font-weight: 760;
    }

    .info-card-note {
        color: #718093;
        font-size: 12px;
        margin-top: 6px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #102d49;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #6e7e90;
        margin-bottom: 22px;
    }

    .prediction-box {
        background: white;
        border: 1px solid #dce7f0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 7px 25px rgba(28, 50, 74, .06);
    }

    .model-name {
        color: #667689;
        font-size: 13px;
    }

    .prediction-value {
        color: #102d49;
        font-size: 27px;
        font-weight: 750;
        margin-top: 3px;
    }

    .prediction-error {
        color: #758496;
        font-size: 13px;
        margin-top: 5px;
    }

    .method-chip {
        display: inline-block;
        padding: 6px 11px;
        margin-right: 6px;
        margin-bottom: 8px;
        border-radius: 20px;
        background: #e9f5f4;
        color: #1b665e;
        font-size: 12px;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.96);
        border: 1px solid #e1e8ef;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 5px 20px rgba(28, 50, 74, .05);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071a33 0%, #0b314e 100%);
    }

    div[data-testid="stSidebar"] * {
        color: white;
    }

    div[data-testid="stSidebar"] label {
        color: #e3eef7 !important;
    }

    .footer {
        color: #8a98a7;
        text-align: center;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 22px;
        border-top: 1px solid #dfe7ee;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "meter",
    "primary_use",
    "square_feet",
    "year_built",
    "air_temperature",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "hour",
    "dayofweek",
    "month"
]

TARGET = "meter_reading"

MODEL_NAMES = [
    "Linear Regression",
    "Random Forest",
    "XGBoost"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return {
        "test": pd.read_csv(DATA_DIR / "test_split.csv"),

        "baseline": pd.read_csv(
            DATA_DIR / "baseline_model_comparison.csv"
        ),

        "robustness": pd.read_csv(
            DATA_DIR / "all_models_16_conditions_results.csv"
        ),

        "robustness_summary": pd.read_csv(
            DATA_DIR / "model_robustness_summary.csv"
        ),

        "lr_coefficients": pd.read_csv(
            DATA_DIR / "linear_regression_coefficients.csv"
        ),

        "rf_importance": pd.read_csv(
            DATA_DIR / "random_forest_feature_importance.csv"
        ),

        "xgb_shap": pd.read_csv(
            DATA_DIR / "xgboost_shap_importance.csv"
        ),

        "stability": pd.read_csv(
            DATA_DIR / "explanation_stability_results_display.csv"
        ),

        "stability_summary": pd.read_csv(
            DATA_DIR / "explanation_stability_summary.csv"
        )
    }


@st.cache_resource
def load_models():

    lr = joblib.load(
        MODEL_DIR / "linear_regression_pipeline.joblib"
    )

    rf = joblib.load(
        MODEL_DIR / "random_forest_pipeline.joblib"
    )

    xgb = joblib.load(
        MODEL_DIR / "xgboost_pipeline.joblib"
    )

    return {
        "Linear Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }


data = load_data()
models = load_models()

test_data = data["test"]
baseline_data = data["baseline"]
robustness_data = data["robustness"]
robustness_summary = data["robustness_summary"]
stability_data = data["stability"]
stability_summary = data["stability_summary"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_all(input_df):

    predictions = {}

    for model_name, model in models.items():
        prediction = model.predict(input_df)[0]
        predictions[model_name] = float(prediction)

    return predictions


def info_card(title, value, note=""):

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="info-card-value">{value}</div>
            <div class="info-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def prediction_card(model, prediction, error=None):

    error_html = ""

    if error is not None:
        error_html = (
            f'<div class="prediction-error">'
            f'Absolute Error: {error:,.3f}'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="model-name">{model}</div>
            <div class="prediction-value">{prediction:,.3f}</div>
            {error_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_image(filename, caption=None):

    image_path = ASSET_DIR / filename

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏢 Energy AI Lab")

    st.caption(
        "Robust & Explainable Machine Learning "
        "for Smart Building Energy Prediction"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Prediction Lab",
            "Model Performance",
            "Robustness Analysis",
            "Explainability"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption("RESEARCH DATASET")
    st.write("ASHRAE GEPIII")

    st.caption("TARGET")
    st.write("meter_reading")

    st.caption("MODELS")
    st.write("LR · RF · XGBoost")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
    """
<div class="hero">
<div class="eyebrow">MASTER OF ARTIFICIAL INTELLIGENCE · RESEARCH DEMO</div>
<div class="hero-title">Smart Building Energy<br>Prediction Lab</div>
<div class="hero-subtitle">
An interactive demonstration of robust and explainable machine learning for
energy consumption prediction under imperfect sensor data.
</div>
</div>
    """,
    unsafe_allow_html=True
)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        info_card(
            "Machine Learning Models",
            "3",
            "Linear Regression · Random Forest · XGBoost"
        )

    with c2:
        info_card(
            "Data Conditions",
            "16",
            "1 clean baseline + 15 imperfect conditions"
        )

    with c3:
        info_card(
            "Input Features",
            "13",
            "Building · weather · meter · time features"
        )

    with c4:
        info_card(
            "Test Observations",
            f"{len(test_data):,}",
            "Clean baseline test set"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Research Framework</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'The study evaluates not only prediction performance, '
        'but also robustness and explanation stability.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        info_card(
            "01 · Baseline",
            "Performance",
            "RMSE · MAE · R²"
        )

    with col2:
        info_card(
            "02 · Imperfect Data",
            "Robustness",
            "RMSE degradation under missing values and noise"
        )

    with col3:
        info_card(
            "03 · Baseline",
            "Explainability",
            "Coefficients · Feature Importance · SHAP"
        )

    with col4:
        info_card(
            "04 · Imperfect Data",
            "Stability",
            "SHAP rankings + Spearman correlation"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Experimental Design")

    st.markdown(
        """
        <span class="method-chip">Clean Training Data</span>
        <span class="method-chip">Missing 10% · 20% · 30%</span>
        <span class="method-chip">Gaussian Noise σ=1 · 2 · 3</span>
        <span class="method-chip">9 Missing + Noise Combinations</span>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The three models are trained using the same clean training data. "
        "Controlled missing values and Gaussian noise are introduced only "
        "into test data for robustness and explanation-stability evaluation."
    )


# ============================================================
# PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":

    st.markdown(
        '<div class="section-title">Prediction Lab</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Run the three final trained models on a real test sample '
        'or create a manual sensor input.'
        '</div>',
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Prediction Mode",
        [
            "Test Sample",
            "Manual Input"
        ],
        horizontal=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TEST SAMPLE MODE
    # --------------------------------------------------------

    if mode == "Test Sample":

        left, right = st.columns([1, 2])

        with left:

            sample_id = st.number_input(
                "Test Sample ID",
                min_value=0,
                max_value=len(test_data) - 1,
                value=6,
                step=1
            )

            sample_id = int(sample_id)

            sample = test_data.iloc[[sample_id]]

            actual = float(
                sample[TARGET].iloc[0]
            )

            st.metric(
                "Actual meter_reading",
                f"{actual:,.3f}"
            )

            run_prediction = st.button(
                "Run Prediction",
                type="primary",
                use_container_width=True
            )

        with right:

            st.markdown("#### Selected Sensor Record")

            display_sample = (
                sample[FEATURES]
                .T
                .reset_index()
            )

            display_sample.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                display_sample,
                use_container_width=True,
                hide_index=True,
                height=360
            )

        if run_prediction:

            predictions = predict_all(
                sample[FEATURES]
            )

            errors = {
                name: abs(actual - prediction)
                for name, prediction in predictions.items()
            }

            st.markdown("---")
            st.markdown("### Model Predictions")

            c1, c2, c3 = st.columns(3)

            for col, model_name in zip(
                [c1, c2, c3],
                MODEL_NAMES
            ):

                with col:
                    prediction_card(
                        model_name,
                        predictions[model_name],
                        errors[model_name]
                    )

            best_model = min(
                errors,
                key=errors.get
            )

            st.success(
                f"Closest prediction for this sample: "
                f"**{best_model}** "
                f"(Absolute Error = {errors[best_model]:,.3f})"
            )

            comparison_df = pd.DataFrame({
                "Model": MODEL_NAMES,
                "Prediction": [
                    predictions[m]
                    for m in MODEL_NAMES
                ],
                "Absolute Error": [
                    errors[m]
                    for m in MODEL_NAMES
                ]
            })

            st.markdown("#### Prediction Comparison")

            st.dataframe(
                comparison_df.style.format({
                    "Prediction": "{:,.3f}",
                    "Absolute Error": "{:,.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # MANUAL INPUT MODE
    # --------------------------------------------------------

    else:

        st.info(
            "Manual Input produces model predictions only. "
            "Because no true meter_reading is supplied, "
            "prediction error cannot be calculated."
        )

        st.markdown("### Building & Meter Information")

        c1, c2, c3, c4 = st.columns(4)

        meter_values = sorted(
            test_data["meter"]
            .dropna()
            .unique()
            .tolist()
        )

        primary_use_values = sorted(
            test_data["primary_use"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with c1:

            meter = st.selectbox(
                "Meter Type",
                meter_values
            )

        with c2:

            primary_use = st.selectbox(
                "Primary Use",
                primary_use_values
            )

        with c3:

            square_feet = st.number_input(
                "Square Feet",
                min_value=0.0,
                value=float(
                    test_data["square_feet"].median()
                )
            )

        with c4:

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2030,
                value=int(
                    test_data["year_built"].median()
                ),
                step=1
            )

        st.markdown("### Weather Information")

        c1, c2, c3 = st.columns(3)

        with c1:

            air_temperature = st.number_input(
                "Air Temperature",
                value=float(
                    test_data["air_temperature"].median()
                )
            )

            dew_temperature = st.number_input(
                "Dew Temperature",
                value=float(
                    test_data["dew_temperature"].median()
                )
            )

        with c2:

            precip_depth_1_hr = st.number_input(
                "Precipitation Depth (1 hr)",
                value=float(
                    test_data["precip_depth_1_hr"].median()
                )
            )

            sea_level_pressure = st.number_input(
                "Sea Level Pressure",
                value=float(
                    test_data["sea_level_pressure"].median()
                )
            )

        with c3:

            wind_direction = st.number_input(
                "Wind Direction",
                value=float(
                    test_data["wind_direction"].median()
                )
            )

            wind_speed = st.number_input(
                "Wind Speed",
                value=float(
                    test_data["wind_speed"].median()
                )
            )

        st.markdown("### Time Information")

        c1, c2, c3 = st.columns(3)

        with c1:

            hour = st.slider(
                "Hour",
                0,
                23,
                12
            )

        with c2:

            dayofweek = st.slider(
                "Day of Week",
                0,
                6,
                2
            )

        with c3:

            month = st.slider(
                "Month",
                1,
                12,
                6
            )

        manual_input = pd.DataFrame([{
            "meter": meter,
            "primary_use": primary_use,
            "square_feet": square_feet,
            "year_built": year_built,
            "air_temperature": air_temperature,
            "dew_temperature": dew_temperature,
            "precip_depth_1_hr": precip_depth_1_hr,
            "sea_level_pressure": sea_level_pressure,
            "wind_direction": wind_direction,
            "wind_speed": wind_speed,
            "hour": hour,
            "dayofweek": dayofweek,
            "month": month
        }])

        if st.button(
            "Predict Energy Consumption",
            type="primary",
            use_container_width=True
        ):

            predictions = predict_all(
                manual_input[FEATURES]
            )

            st.markdown("---")
            st.markdown("### Model Predictions")

            c1, c2, c3 = st.columns(3)

            for col, model_name in zip(
                [c1, c2, c3],
                MODEL_NAMES
            ):

                with col:
                    prediction_card(
                        model_name,
                        predictions[model_name]
                    )

            if any(
                value < 0
                for value in predictions.values()
            ):

                st.warning(
                    "One or more raw regression predictions are negative. "
                    "The demo displays the original model outputs without "
                    "post-processing or clipping."
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Baseline Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Performance evaluated using the clean baseline test dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    model_cols = st.columns(3)

    for col, model_name in zip(
        model_cols,
        MODEL_NAMES
    ):

        row = baseline_data[
            baseline_data["Model"] == model_name
        ].iloc[0]

        with col:

            st.markdown(
                f"### {model_name}"
            )

            st.metric(
                "RMSE",
                f"{row['RMSE']:,.3f}"
            )

            st.metric(
                "MAE",
                f"{row['MAE']:,.3f}"
            )

            st.metric(
                "R²",
                f"{row['R²']:.4f}"
            )

    st.markdown("---")

    best_rmse = baseline_data.loc[
        baseline_data["RMSE"].idxmin(),
        "Model"
    ]

    best_mae = baseline_data.loc[
        baseline_data["MAE"].idxmin(),
        "Model"
    ]

    best_r2 = baseline_data.loc[
        baseline_data["R²"].idxmax(),
        "Model"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success(
            f"Lowest RMSE: **{best_rmse}**"
        )

    with c2:
        st.success(
            f"Lowest MAE: **{best_mae}**"
        )

    with c3:
        st.success(
            f"Highest R²: **{best_r2}**"
        )

    st.caption(
        "RMSE is the primary prediction-performance metric "
        "used in this study."
    )

    st.markdown("### Performance Comparison")

    show_image(
        "baseline_grouped_metrics_comparison.png",
        "Baseline model comparison"
    )


# ============================================================
# ROBUSTNESS
# ============================================================

elif page == "Robustness Analysis":

    st.markdown(
        '<div class="section-title">Robustness under Imperfect Sensor Data</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore how prediction error changes when missing values '
        'and Gaussian noise are introduced into test data.'
        '</div>',
        unsafe_allow_html=True
    )

    conditions = (
        robustness_data["Condition"]
        .drop_duplicates()
        .tolist()
    )

    selected_condition = st.selectbox(
        "Select Data Condition",
        conditions
    )

    selected_data = robustness_data[
        robustness_data["Condition"]
        == selected_condition
    ].copy()

    st.markdown("### Selected Condition")

    c1, c2, c3 = st.columns(3)

    for col, model_name in zip(
        [c1, c2, c3],
        MODEL_NAMES
    ):

        row = selected_data[
            selected_data["Model"] == model_name
        ].iloc[0]

        with col:

            st.markdown(
                f"#### {model_name}"
            )

            st.metric(
                "RMSE",
                f"{row['RMSE']:,.3f}"
            )

            st.metric(
                "RMSE Degradation",
                f"{row['RMSE Degradation']:,.3f}"
            )

            st.metric(
                "Degradation (%)",
                f"{row['RMSE Degradation (%)']:.2f}%"
            )

    st.markdown("---")

    st.markdown("### Robustness Summary")

    summary_columns = [
        "Model",
        "Baseline RMSE",
        "Mean RMSE under Imperfect Conditions",
        "Mean RMSE Degradation",
        "Mean Degradation (%)",
        "Worst-Case RMSE",
        "Worst-Case Degradation (%)",
        "Worst Condition"
    ]

    summary_columns = [
        col
        for col in summary_columns
        if col in robustness_summary.columns
    ]

    st.dataframe(
        robustness_summary[summary_columns],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Lower RMSE degradation indicates stronger relative robustness. "
        "Absolute predictive performance should also be considered."
    )

    st.markdown("### Performance & Robustness Across Conditions")

    show_image(
        "model_performance_and_robustness_comparison.png",
        "Model performance and robustness comparison"
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.markdown(
        '<div class="section-title">Model Explainability</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Interpret baseline model behaviour and examine whether '
        'feature rankings remain stable as sensor data quality decreases.'
        '</div>',
        unsafe_allow_html=True
    )

    baseline_tab, stability_tab = st.tabs([
        "Baseline Explainability",
        "Explanation Stability"
    ])

    # --------------------------------------------------------
    # BASELINE EXPLAINABILITY
    # --------------------------------------------------------

    with baseline_tab:

        lr_tab, rf_tab, xgb_tab = st.tabs([
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ])

        with lr_tab:

            st.markdown(
                "### Linear Regression Coefficients"
            )

            st.write(
                "Coefficient signs indicate the direction of the "
                "linear relationship between processed features "
                "and predicted meter reading."
            )

            show_image(
                "linear_regression_top_coefficients.png",
                "Top Linear Regression coefficients"
            )

            st.dataframe(
                data["lr_coefficients"].head(20),
                use_container_width=True,
                hide_index=True
            )

        with rf_tab:

            st.markdown(
                "### Random Forest Feature Importance"
            )

            st.write(
                "Impurity-based feature importance identifies "
                "features that contribute more strongly to the "
                "tree-based prediction process."
            )

            show_image(
                "random_forest_feature_importance.png",
                "Random Forest global feature importance"
            )

            st.dataframe(
                data["rf_importance"],
                use_container_width=True,
                hide_index=True
            )

        with xgb_tab:

            st.markdown(
                "### XGBoost SHAP Analysis"
            )

            st.write(
                "Mean absolute SHAP values quantify the overall "
                "contribution of each original feature."
            )

            show_image(
                "xgboost_global_shap_importance.png",
                "XGBoost global SHAP importance"
            )

            st.dataframe(
                data["xgb_shap"],
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### SHAP Summary Plot"
            )

            show_image(
                "xgboost_shap_summary_plot.png",
                "XGBoost SHAP summary plot"
            )

    # --------------------------------------------------------
    # EXPLANATION STABILITY
    # --------------------------------------------------------

    with stability_tab:

        st.markdown(
            "### Explanation Stability under Imperfect Data"
        )

        st.write(
            "For all three trained models, SHAP-based feature "
            "rankings under each imperfect condition are compared "
            "with the corresponding clean baseline ranking using "
            "Spearman rank correlation."
        )

        st.info(
            "The stability analysis uses the same fixed sample of "
            "300 test observations across all models and data conditions."
        )

        c1, c2, c3 = st.columns(3)

        for col, model_name in zip(
            [c1, c2, c3],
            MODEL_NAMES
        ):

            row = stability_summary[
                stability_summary["Model"]
                == model_name
            ].iloc[0]

            with col:

                st.markdown(
                    f"#### {model_name}"
                )

                st.metric(
                    "Mean Stability",
                    f"{row['Mean Explanation Stability']:.4f}"
                )

                st.metric(
                    "Minimum Stability",
                    f"{row['Minimum Explanation Stability']:.4f}"
                )

                st.caption(
                    f"Worst condition: "
                    f"{row['Worst Condition']}"
                )

        st.markdown("---")

        st.markdown(
            "### Stability Heatmap"
        )

        show_image(
            "explanation_stability_heatmap.png",
            "Spearman feature-ranking stability across imperfect conditions"
        )

        st.markdown(
            "### Stability Trend"
        )

        show_image(
            "explanation_stability_trend.png",
            "Explanation stability trend across imperfect conditions"
        )

        with st.expander(
            "View all 45 explanation-stability scores"
        ):

            st.dataframe(
                stability_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "A Spearman correlation closer to 1 indicates "
            "greater similarity to the clean baseline feature ranking. "
            "Explanation stability measures ranking consistency, "
            "not causal correctness."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Smart Building Energy Prediction Lab ·
        Robust & Explainable Machine Learning under Imperfect Sensor Data
    </div>
    """,
    unsafe_allow_html=True
)
