
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="AI Well Testing Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("rf_well_test_model.joblib")

model = load_model()

# Dashboard heading
st.title("🛢️ AI-Assisted Well Testing Dashboard")

st.markdown("""
**Petroleum Engineering | Pressure Response Classification**

This prototype uses a Random Forest model to classify synthetic
well-test pressure responses using pressure and derivative data.
""")

st.warning(
    "Research prototype: This model was trained on synthetic data. "
    "Its predictions have not been validated against real field well tests."
)

st.divider()

# Upload data
st.header("1. Upload Well-Test Data")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    pressure_cols = [f"dp{i}" for i in range(20)]
    derivative_cols = [f"derp{i}" for i in range(20)]
    time_cols = [f"t{i}" for i in range(20)]

    features = pressure_cols + derivative_cols

    missing = [c for c in features if c not in data.columns]

    if missing:
        st.error(
            "Required pressure or derivative columns are missing. "
            "The CSV must contain dp0–dp19 and derp0–derp19."
        )
        st.stop()

    if data.empty:
        st.error("The uploaded CSV contains no data rows.")
        st.stop()

    st.success(f"Dataset loaded: {len(data):,} rows")

    with st.expander("Preview uploaded data"):
        st.dataframe(data.head())

    # Select response
    st.header("2. Select a Well-Test Response")

    row_number = st.number_input(
        "Select the row to analyse (starting at 0)",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1
    )

    row_number = int(row_number)
    row = data.iloc[row_number]

    X = data.loc[[row_number], features]

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    confidence = max(probabilities) * 100

    # Prediction panel
    st.header("3. AI Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Response Category",
            str(prediction)
        )

    with col2:
        st.metric(
            "Model-Estimated Probability",
            f"{confidence:.2f}%"
        )

    st.caption(
        "The probability is the model's estimate, not a guarantee "
        "of physical correctness."
    )

    # Reservoir properties
    st.header("4. Reservoir and Well Parameters")

    parameter_info = {
        "k": ("Permeability", "mD"),
        "por": ("Porosity", "fraction"),
        "h": ("Thickness", "dataset units"),
        "ct": ("Total compressibility", "dataset units"),
        "Bo": ("Formation volume factor", "dataset units"),
        "visc": ("Viscosity", "dataset units"),
        "q": ("Flow rate", "dataset units"),
        "skin": ("Skin factor", "dimensionless"),
        "stor": ("Storage parameter", "dataset units"),
        "rw": ("Wellbore radius", "dataset units")
    }

    available_parameters = [
        key for key in parameter_info if key in data.columns
    ]

    if available_parameters:
        parameter_table = []

        for key in available_parameters:
            value = row[key]
            label, units = parameter_info[key]

            parameter_table.append({
                "Parameter": label,
                "Symbol": key,
                "Value": value,
                "Units / Description": units
            })

        st.dataframe(
            pd.DataFrame(parameter_table),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No reservoir parameter columns were found.")

    # Pressure plots
    st.header("5. Pressure Response Analysis")

    if all(c in data.columns for c in time_cols):

        time_values = row[time_cols].astype(float).values
        pressure_values = row[pressure_cols].astype(float).values
        derivative_values = row[derivative_cols].astype(float).values

        if (time_values <= 0).any():
            st.warning(
                "Some time values are not positive, so a logarithmic "
                "time plot cannot be drawn safely."
            )
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Pressure Variation vs Time")

                fig, ax = plt.subplots()
                ax.plot(time_values, pressure_values, marker="o")
                ax.set_xscale("log")
                ax.set_xlabel("Time (dataset units)")
                ax.set_ylabel("Pressure variation (dataset units)")
                ax.grid(True, which="both", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.subheader("Pressure Derivative vs Time")

                fig, ax = plt.subplots()
                ax.plot(time_values, derivative_values, marker="o")
                ax.set_xscale("log")
                ax.set_xlabel("Time (dataset units)")
                ax.set_ylabel("Provided derivative (dataset units)")
                ax.grid(True, which="both", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    else:
        st.warning(
            "Time columns t0–t19 are missing, so time-based plots "
            "cannot be generated."
        )

    # Class probabilities
    st.header("6. Classification Probabilities")

    probability_df = pd.DataFrame({
        "Response Category": model.classes_,
        "Estimated Probability (%)": probabilities * 100
    }).sort_values(
        "Estimated Probability (%)",
        ascending=False
    )

    st.bar_chart(
        probability_df.set_index("Response Category")
    )

    # Careful interpretation
    st.header("7. Preliminary Technical Interpretation")

    st.write(
        f"The Random Forest model classified the selected response "
        f"as **{prediction}**, with a model-estimated probability "
        f"of **{confidence:.2f}%**."
    )

    st.write(
        "This result is a data-driven classification of a synthetic "
        "pressure response. It should be compared with pressure "
        "derivative behaviour, expected reservoir conditions and "
        "engineering knowledge before drawing a physical conclusion."
    )

    st.info(
        "The dashboard does not independently calculate permeability "
        "or prove the presence of a physical boundary. Those conclusions "
        "require appropriate engineering analysis and validation."
    )

else:
    st.info(
        "Upload the well-test CSV to view the prediction, reservoir "
        "parameters and pressure-response plots."
    )

st.divider()

st.caption(
    "Academic prototype | AI-assisted well-test response classification"
)
