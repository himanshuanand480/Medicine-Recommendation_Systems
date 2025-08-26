import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import ast  # To safely evaluate string representations of lists
import time  # To simulate progress bars

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Health Advisor 🩺",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads all necessary CSV files into pandas DataFrames."""
    try:
        train_df = pd.read_csv('Training_modified.csv')
        description_df = pd.read_csv('description_modified.csv')
        medications_df = pd.read_csv('medications_modified.csv')
        diets_df = pd.read_csv('diets_modified.csv')
        precautions_df = pd.read_csv('precautions_df_modified.csv')
        socioeconomic_df = pd.read_csv('practical_socioeconomic_modified.csv')

        # Convert string representations of lists to actual lists
        medications_df['Medication'] = medications_df['Medication'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        diets_df['Diet'] = diets_df['Diet'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        return train_df, description_df, medications_df, diets_df, precautions_df, socioeconomic_df
    except FileNotFoundError as e:
        st.error(f"❌ Error loading data file: {e}. Make sure all CSV files are in the same directory as 'app.py'.")
        return None, None, None, None, None, None


@st.cache_resource
def train_model(train_df):
    """Prepares data and trains the RandomForestClassifier model."""
    if train_df is None:
        return None, None, None

    X = train_df.drop('prognosis', axis=1)
    y = train_df['prognosis']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(random_state=42, min_samples_split=20, min_samples_leaf=10)
    model.fit(X, y_encoded)

    all_symptoms = X.columns.tolist()

    return model, le, all_symptoms


# --- Load Data and Train Model ---
(train_df, description_df, medications_df, diets_df,
 precautions_df, socioeconomic_df) = load_data()

model, le, all_symptoms = train_model(train_df)

# --- UI STYLING (with new 3D and interactive elements) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4B0082; /* Indigo */
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #9370DB; /* MediumPurple */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        width: 100%;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #8A2BE2; /* BlueViolet */
        transform: scale(1.02);
    }
    .st-expander {
        border: 2px solid #D8BFD8; /* Thistle */
        border-radius: 10px;
    }
    .result-card {
        background-color: #F8F8FF; /* GhostWhite */
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #DCDCDC; /* Gainsboro */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* --- NEW: Hero Banner for Welcome Page --- */
    .hero-banner {
        width: 100%;
        height: 400px;
        background-image: linear-gradient(rgba(25, 25, 112, 0.6), rgba(75, 0, 130, 0.7)),
                          url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border-radius: 15px;
        padding: 20px;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2), 0 6px 6px rgba(0,0,0,0.23);
        transition: transform 0.3s ease-in-out;
    }
    .hero-banner:hover {
        transform: scale(1.02); /* Interactive hover effect */
    }

    /* --- NEW: 3D Text Styling --- */
    .text-3d {
        color: #FFFFFF; /* White text */
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow:
            1px 1px 0px #1E90FF, /* DodgerBlue */
            2px 2px 0px #1E90FF,
            3px 3px 0px #1E90FF,
            4px 4px 5px rgba(0,0,0,0.5); /* Soft drop shadow for depth */
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a feature",
    ["Welcome", "Symptom Prediction", "Disease Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Disclaimer:** This AI Health Advisor is for informational purposes only and "
    "is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always seek the advice of your physician."
)

# --- Main Application Logic ---
if app_mode == "Welcome":
    st.markdown('<p class="main-header">🩺 Welcome to the AI Health Advisor 🩺</p>', unsafe_allow_html=True)

    # --- NEW ATTRACTIVE HERO BANNER ---
    st.markdown("""
        <div class="hero-banner">
            <p class="text-3d" style="font-size: 2.2rem; font-style: italic; margin-bottom: 0;">Your personal health advisor</p>
            <p class="text-3d" style="font-size: 4rem; font-weight: bold; margin-top: 0;">AI Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    # --- END OF NEW BANNER ---

    st.markdown("### Please select a feature from the sidebar to get started.")


# Only proceed if data and model are loaded successfully
elif all(df is not None for df in
         [train_df, description_df, medications_df, diets_df, precautions_df, socioeconomic_df]) and model is not None:

    if app_mode == "Symptom Prediction":
        st.header("Symptom-Based Disease Prediction 🕵️‍♀️")
        st.write("Select the symptoms you are experiencing from the dropdown below to get a prediction.")

        selected_symptoms = st.multiselect(
            'What symptoms are you feeling?',
            options=sorted(all_symptoms),
            help="You can select multiple symptoms."
        )

        if st.button('Predict Disease', key='predict_btn'):
            if selected_symptoms:
                progress_bar = st.progress(0)
                with st.spinner('Analyzing your symptoms... 🧠'):
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    input_vector = pd.DataFrame([0] * len(all_symptoms), index=all_symptoms).T
                    for symptom in selected_symptoms:
                        input_vector[symptom] = 1

                    prediction_encoded = model.predict(input_vector)[0]
                    predicted_disease = le.inverse_transform([prediction_encoded])[0]

                progress_bar.empty()
                st.success(f"**Predicted Disease:** {predicted_disease}")
                st.balloons()

                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.subheader(f"Information & Recommendations for {predicted_disease}")

                    desc = description_df[description_df['Disease'] == predicted_disease]['Description'].values
                    if desc:
                        st.info(f"**Description:** {desc[0]}")

                    col1, col2 = st.columns(2)
                    with col1:
                        prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
                        precs = precautions_df[precautions_df['Disease'] == predicted_disease][
                            prec_cols].values.flatten().tolist()
                        precs = [p for p in precs if pd.notna(p)]
                        if precs:
                            with st.expander("⚠️ **Recommended Precautions**", expanded=True):
                                for i, p in enumerate(precs):
                                    st.write(f"{i + 1}. {p}")

                        meds = medications_df[medications_df['Disease'] == predicted_disease]['Medication'].values
                        if meds and meds[0]:
                            with st.expander("💊 **Medications**", expanded=True):
                                for med in meds[0]:
                                    st.write(f"- {med}")

                    with col2:
                        diet = diets_df[diets_df['Disease'] == predicted_disease]['Diet'].values
                        if diet and diet[0]:
                            with st.expander("🥗 **Dietary Advice**", expanded=True):
                                for d in diet[0]:
                                    st.write(f"- {d}")

                        socio_info = socioeconomic_df[socioeconomic_df['Disease'] == predicted_disease].to_dict(
                            'records')
                        if socio_info:
                            with st.expander("💼 **Practical & Socio-economic Info**", expanded=True):
                                for key, value in socio_info[0].items():
                                    if key != 'Disease':
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please select at least one symptom.")

    elif app_mode == "Disease Explorer":
        st.header("Disease Information Explorer 📖")
        st.write("Select a disease from the list and click the button to see its details.")

        all_diseases = sorted(description_df['Disease'].unique())
        selected_disease = st.selectbox(
            'Choose a disease',
            options=all_diseases
        )

        if st.button('Show Details', key='details_btn'):
            if selected_disease:
                progress_bar = st.progress(0)
                with st.spinner(f'Fetching details for {selected_disease}...'):
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                progress_bar.empty()
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.subheader(f"Details for: {selected_disease}")

                    desc = description_df[description_df['Disease'] == selected_disease]['Description'].values
                    if desc:
                        st.info(f"**Description:** {desc[0]}")

                    col1, col2 = st.columns(2)
                    with col1:
                        prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
                        precs = precautions_df[precautions_df['Disease'] == selected_disease][
                            prec_cols].values.flatten().tolist()
                        precs = [p for p in precs if pd.notna(p)]
                        if precs:
                            with st.expander("⚠️ **Recommended Precautions**", expanded=True):
                                for i, p in enumerate(precs):
                                    st.write(f"{i + 1}. {p}")

                        meds = medications_df[medications_df['Disease'] == selected_disease]['Medication'].values
                        if meds and meds[0]:
                            with st.expander("💊 **Medications**", expanded=True):
                                for med in meds[0]:
                                    st.write(f"- {med}")

                    with col2:
                        diet = diets_df[diets_df['Disease'] == selected_disease]['Diet'].values
                        if diet and diet[0]:
                            with st.expander("🥗 **Dietary Advice**", expanded=True):
                                for d in diet[0]:
                                    st.write(f"- {d}")

                        socio_info = socioeconomic_df[socioeconomic_df['Disease'] == selected_disease].to_dict(
                            'records')
                        if socio_info:
                            with st.expander("💼 **Practical & Socio-economic Info**", expanded=True):
                                for key, value in socio_info[0].items():
                                    if key != 'Disease':
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please select a disease first.")
else:
    if app_mode != "Welcome":
        st.warning(
            "Application could not start. Please check the error message above and ensure all data files are present.")