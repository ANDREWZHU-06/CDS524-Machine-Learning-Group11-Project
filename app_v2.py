# app.py
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Import functions from models.py
from api_v1 import (
    find_model_files,
    load_models,
    predict_diabetes_risk,
    get_risk_explanation,
    get_base_recommendations,
    get_personalized_recommendations,
    get_treatment_recommendations,
    get_fallback_recommendations,
    parse_treatment_recommendations,
    get_reference_ranges
)

# Set up page configuration
st.set_page_config(
    page_title="Diabetes Detection System",
    page_icon="🏥",
    layout="wide"
)

# Display environment info in debug mode
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False


# Main UI function
def main():
    """Main Streamlit UI function"""
    st.title("🏥 Diabetes Detection System")
    st.subheader("AI-powered diabetes risk assessment tool")

    # Display debug info toggle
    if st.sidebar.checkbox("Enable Debug Mode"):
        st.session_state.debug_mode = True
    else:
        st.session_state.debug_mode = False

    # API Settings in sidebar
    st.sidebar.subheader("API Settings")
    use_api = st.sidebar.checkbox("Use AI Treatment Enhancement", value=False)
    if use_api:
        api_key = st.sidebar.text_input("API Key (Optional)",
                                        value="sk-obprxkhbxdlntmxsuwdchgtxjnuhdpoqhbiysjadslmzpltq",
                                        type="password")
        st.sidebar.info("Using DeepSeek AI to enhance treatment recommendations")

    # Show debug panel if enabled
    if st.session_state.debug_mode:
        st.sidebar.subheader("Debug Information")
        st.sidebar.write(f"Python: {sys.version}")
        st.sidebar.write(f"Working directory: {os.getcwd()}")

        # Find model files
        model_file, preproc_file = find_model_files()

        st.sidebar.write(f"Found model file: {model_file}")
        st.sidebar.write(f"Found preprocessing file: {preproc_file}")

        # Check if files exist
        if model_file:
            st.sidebar.write(f"Model file exists: {os.path.exists(model_file)}")
        if preproc_file:
            st.sidebar.write(f"Preprocessing file exists: {os.path.exists(preproc_file)}")

        # List searched directories
        st.sidebar.subheader("Searched Directories:")
        from models import POTENTIAL_DIRS
        for directory in POTENTIAL_DIRS:
            exists = os.path.exists(directory)
            st.sidebar.write(f"- {directory}: {'Exists' if exists else 'Not found'}")
            if exists:
                files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
                if files:
                    st.sidebar.write(f"  Files: {', '.join(files)}")

    # Load models in session state
    if 'model' not in st.session_state or 'preprocessing_info' not in st.session_state:
        with st.spinner("Loading model..."):
            model, preprocessing_info, error = load_models()
            st.session_state.model = model
            st.session_state.preprocessing_info = preprocessing_info
            st.session_state.model_error = error

        if error:
            st.error(f"Model loading error: {error}")
        elif model is not None and preprocessing_info is not None:
            st.success("Model and preprocessing info loaded successfully!")

    # Check if model is loaded
    if st.session_state.model is None or st.session_state.preprocessing_info is None:
        st.error("Please ensure you've run the main analysis script and generated the necessary model files.")

        # Show detailed error if available
        if hasattr(st.session_state, 'model_error') and st.session_state.model_error:
            st.error(f"Error details: {st.session_state.model_error}")

        # Show instruction on how to fix
        st.subheader("How to fix this issue")
        st.markdown("""
        1. Make sure you've run the main analysis script to generate model files
        2. Check the file paths in the main script
        3. Ensure the model files were created in the expected location
        4. Verify that the model files have valid content (not empty or corrupted)
        """)
        return

    # Get feature list from preprocessing info
    preprocessing_info = st.session_state.preprocessing_info
    top_features = preprocessing_info.get('top_features', [])

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "About the System", "API Documentation"])

    with tab1:
        # Split screen into input and result areas
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Patient Data Input")

            # Create input form
            with st.form("patient_data_form"):
                # Add patient name and ID for the form feeling
                patient_name = st.text_input("Patient Name (Optional)", "")
                patient_id = st.text_input("Patient ID (Optional)", "")
                today_date = datetime.datetime.now().strftime("%Y-%m-%d")

                # Gender input if needed
                if 'Gender' in top_features:
                    gender = st.radio("Gender", ["Male", "Female"])
                    gender_value = 1 if gender == "Male" else 0
                else:
                    gender_value = None

                # Age input if needed
                if 'AGE' in top_features:
                    age = st.slider("Age", 18, 90, 45)
                else:
                    age = None

                # BMI input if needed
                if 'BMI' in top_features:
                    bmi = st.slider("BMI (Body Mass Index)", 15.0, 45.0, 24.5, 0.1)
                else:
                    bmi = None

                # Create columns for remaining inputs
                st.subheader("Clinical Measurements")

                # First row of inputs
                cols = st.columns(2)

                with cols[0]:
                    hba1c = st.number_input("HbA1c (%)" if 'HbA1c' in top_features else "HbA1c (%) - Not used",
                                            3.0, 15.0, 5.0, 0.1,
                                            disabled='HbA1c' not in top_features)

                    urea = st.number_input("Urea (mmol/L)" if 'Urea' in top_features else "Urea (mmol/L) - Not used",
                                           1.0, 30.0, 5.0, 0.1,
                                           disabled='Urea' not in top_features)

                    cr = st.number_input(
                        "Creatinine (µmol/L)" if 'Cr' in top_features else "Creatinine (µmol/L) - Not used",
                        20, 300, 70, 1,
                        disabled='Cr' not in top_features)

                with cols[1]:
                    chol = st.number_input(
                        "Cholesterol (mmol/L)" if 'Chol' in top_features else "Cholesterol (mmol/L) - Not used",
                        2.0, 10.0, 4.5, 0.1,
                        disabled='Chol' not in top_features)

                    tg = st.number_input(
                        "Triglycerides (mmol/L)" if 'TG' in top_features else "Triglycerides (mmol/L) - Not used",
                        0.5, 10.0, 1.5, 0.1,
                        disabled='TG' not in top_features)

                    hdl = st.number_input("HDL (mmol/L)" if 'HDL' in top_features else "HDL (mmol/L) - Not used",
                                          0.5, 3.0, 1.2, 0.1,
                                          disabled='HDL' not in top_features)

                # Second row of inputs
                cols = st.columns(2)

                with cols[0]:
                    ldl = st.number_input("LDL (mmol/L)" if 'LDL' in top_features else "LDL (mmol/L) - Not used",
                                          0.5, 5.0, 2.5, 0.1,
                                          disabled='LDL' not in top_features)

                with cols[1]:
                    vldl = st.number_input("VLDL (mmol/L)" if 'VLDL' in top_features else "VLDL (mmol/L) - Not used",
                                           0.1, 5.0, 0.7, 0.1,
                                           disabled='VLDL' not in top_features)

                # Submit button
                submitted = st.form_submit_button("Analyze Diabetes Risk")

        # Create input data dictionary based on form submission
        if submitted:
            # Prepare input data
            input_data = {}

            if 'Gender' in top_features:
                input_data['Gender'] = gender_value

            if 'AGE' in top_features:
                input_data['AGE'] = age

            if 'BMI' in top_features:
                input_data['BMI'] = bmi

            if 'HbA1c' in top_features:
                input_data['HbA1c'] = hba1c

            if 'Urea' in top_features:
                input_data['Urea'] = urea

            if 'Cr' in top_features:
                input_data['Cr'] = cr

            if 'Chol' in top_features:
                input_data['Chol'] = chol

            if 'TG' in top_features:
                input_data['TG'] = tg

            if 'HDL' in top_features:
                input_data['HDL'] = hdl

            if 'LDL' in top_features:
                input_data['LDL'] = ldl

            if 'VLDL' in top_features:
                input_data['VLDL'] = vldl

            # Process results in column 2
            with col2:
                with st.spinner("Analyzing data..."):
                    # Make prediction
                    predicted_class, class_probabilities, predict_error = predict_diabetes_risk(
                        input_data, st.session_state.model, st.session_state.preprocessing_info
                    )

                    if predict_error:
                        st.error(f"Prediction error: {predict_error}")
                    elif predicted_class is None or class_probabilities is None:
                        st.error("Failed to get prediction results")
                    else:
                        # Display results
                        result_colors = {
                            'N': '#28a745',  # Green
                            'P': '#ffc107',  # Yellow/Orange
                            'Y': '#dc3545'  # Red
                        }

                        result_texts = {
                            'N': 'Normal',
                            'P': 'Pre-diabetic',
                            'Y': 'Diabetic Risk'
                        }

                        # Create a medical report header - using proper formatting without raw HTML showing
                        st.markdown(f"""
                        # Diabetes Risk Assessment Report

                        **Patient:** {patient_name if patient_name else "Anonymous"}  
                        {f"**ID:** {patient_id}" if patient_id else ""}  
                        **Date:** {today_date}
                        """)

                        # Show prediction result with colored box using markdown
                        prediction_text = result_texts.get(predicted_class, 'Unknown')
                        st.markdown(f"## Prediction: {prediction_text}")
                        st.markdown(
                            f"<div style='padding:10px; background-color:{result_colors.get(predicted_class, '#6c757d')}; "
                            f"border-radius:5px; color:white; text-align:center; margin-bottom:15px;'>"
                            f"<h3 style='margin:0;'>{prediction_text}</h3></div>",
                            unsafe_allow_html=True
                        )

                        # Show probability bars
                        st.subheader("Risk Probabilities")

                        # Use progress bar for probabilities
                        for cls, prob in class_probabilities.items():
                            class_label = result_texts.get(cls, cls)
                            st.text(f"{class_label}: {prob * 100:.1f}%")
                            st.progress(float(prob))

                        # Show explanation
                        risk_explanation = get_risk_explanation(predicted_class, input_data)
                        st.subheader("Risk Assessment")
                        st.write(risk_explanation)

                        # --------- UNIFIED TREATMENT PLAN SECTION ---------

                        # Create a unified treatment plan that combines standard recommendations with AI enhancements
                        st.subheader("Treatment Plan")

                        # Add styling for the treatment plan
                        st.markdown("""
                        ## Personalized Treatment Plan
                        *Based on clinical data and risk assessment*
                        """)

                        # Get standard recommendations
                        base_recs = get_base_recommendations(predicted_class)
                        personalized_recs = get_personalized_recommendations(input_data)

                        # Combine recommendations
                        standard_recs = {
                            'lifestyle': base_recs['lifestyle'] + personalized_recs['lifestyle'],
                            'medication': base_recs['medication'] + personalized_recs['medication'],
                            'monitoring': base_recs['monitoring'] + personalized_recs['monitoring']
                        }

                        # Initialize AI recommendations as empty
                        ai_recs = {
                            'lifestyle': [],
                            'medication': [],
                            'monitoring': []
                        }

                        # If API is enabled, get AI recommendations
                        ai_enhanced = False
                        api_error = None

                        if use_api:
                            with st.spinner("Enhancing treatment plan with AI..."):
                                # Get prediction result text
                                result_text = result_texts.get(predicted_class, 'Unknown')

                                # Call the API
                                api_result = get_treatment_recommendations(
                                    input_data,
                                    result_text,
                                    api_key=api_key if 'api_key' in locals() else None
                                )

                                if api_result["success"]:
                                    # Parse AI recommendations
                                    ai_recs = parse_treatment_recommendations(api_result["content"])

                                    # Check if we got any recommendations
                                    if ai_recs['lifestyle'] or ai_recs['medication'] or ai_recs['monitoring']:
                                        ai_enhanced = True
                                else:
                                    # Store error for display
                                    api_error = api_result["error"]

                                    # Use fallback recommendations if API failed
                                    ai_recs = get_fallback_recommendations(predicted_class)
                                    ai_enhanced = True

                        # Display unified treatment plan
                        with st.container():
                            if api_error:
                                st.warning(f"AI enhancement unavailable: {api_error}")
                                st.markdown("Using standard recommendations with fallback enhancements.")

                            cols = st.columns(3)

                            # Lifestyle column
                            with cols[0]:
                                st.markdown("### 🥗 Lifestyle Modifications")
                                if standard_recs['lifestyle']:
                                    for i, item in enumerate(standard_recs['lifestyle']):
                                        st.markdown(f"**{i + 1}.** {item}")

                                    # Add AI recommendations with a divider if they exist
                                    if ai_enhanced and ai_recs['lifestyle']:
                                        st.markdown("<hr style='border: 1px solid #333; margin: 20px 0;'>",
                                                    unsafe_allow_html=True)
                                        st.markdown(
                                            "<div style='margin-bottom:15px;'>"
                                            "<span style='background-color: #f0ad4e; color: black; padding: 4px 8px; "
                                            "border-radius: 4px; margin-right:5px;'>✨</span>"
                                            "<span style='font-weight:bold; font-size:1.1em;'>AI Enhanced Recommendations:</span>"
                                            "</div>",
                                            unsafe_allow_html=True
                                        )

                                        for i, item in enumerate(ai_recs['lifestyle']):
                                            st.markdown(
                                                f"<div style='margin-bottom:10px; border-left:3px solid #4e8df5; padding-left:10px;'>"
                                                f"<strong>{i + 1}.</strong> {item}</div>",
                                                unsafe_allow_html=True
                                            )
                                else:
                                    st.markdown("No specific lifestyle recommendations.")

                            # Medication column
                            with cols[1]:
                                st.markdown("### 💊 Medication")
                                if standard_recs['medication']:
                                    for i, item in enumerate(standard_recs['medication']):
                                        st.markdown(f"**{i + 1}.** {item}")

                                    # Add AI recommendations with a divider if they exist
                                    if ai_enhanced and ai_recs['medication']:
                                        st.markdown("<hr style='border: 1px solid #333; margin: 20px 0;'>",
                                                    unsafe_allow_html=True)
                                        st.markdown(
                                            "<div style='margin-bottom:15px;'>"
                                            "<span style='background-color: #f0ad4e; color: black; padding: 4px 8px; "
                                            "border-radius: 4px; margin-right:5px;'>✨</span>"
                                            "<span style='font-weight:bold; font-size:1.1em;'>AI Enhanced Recommendations:</span>"
                                            "</div>",
                                            unsafe_allow_html=True
                                        )

                                        for i, item in enumerate(ai_recs['medication']):
                                            st.markdown(
                                                f"<div style='margin-bottom:10px; border-left:3px solid #4e8df5; padding-left:10px;'>"
                                                f"<strong>{i + 1}.</strong> {item}</div>",
                                                unsafe_allow_html=True
                                            )
                                else:
                                    st.markdown("No medication recommendations at this time.")

                            # Monitoring column
                            with cols[2]:
                                st.markdown("### 📊 Monitoring Plan")
                                if standard_recs['monitoring']:
                                    for i, item in enumerate(standard_recs['monitoring']):
                                        st.markdown(f"**{i + 1}.** {item}")

                                    # Add AI recommendations with a divider if they exist
                                    if ai_enhanced and ai_recs['monitoring']:
                                        st.markdown("<hr style='border: 1px solid #333; margin: 20px 0;'>",
                                                    unsafe_allow_html=True)
                                        st.markdown(
                                            "<div style='margin-bottom:15px;'>"
                                            "<span style='background-color: #f0ad4e; color: black; padding: 4px 8px; "
                                            "border-radius: 4px; margin-right:5px;'>✨</span>"
                                            "<span style='font-weight:bold; font-size:1.1em;'>AI Enhanced Recommendations:</span>"
                                            "</div>",
                                            unsafe_allow_html=True
                                        )

                                        for i, item in enumerate(ai_recs['monitoring']):
                                            st.markdown(
                                                f"<div style='margin-bottom:10px; border-left:3px solid #4e8df5; padding-left:10px;'>"
                                                f"<strong>{i + 1}.</strong> {item}</div>",
                                                unsafe_allow_html=True
                                            )
                                else:
                                    st.markdown("No specific monitoring recommendations.")

                            # Action buttons
                            action_cols = st.columns(3)
                            with action_cols[0]:
                                if st.button("📄 Print Plan"):
                                    st.info("Printing functionality would be connected here in a production system")
                            with action_cols[1]:
                                if st.button("💾 Save Plan"):
                                    st.info("Plan would be saved to patient records in a production system")
                            with action_cols[2]:
                                if st.button("📧 Share Plan"):
                                    st.info("Plan would be shared with the patient in a production system")

                            # Add disclaimer
                            st.markdown("""
                            *This treatment plan is for informational purposes only and should not replace professional 
                            medical advice. Always consult with a healthcare provider before making any changes to your 
                            healthcare regimen.*
                            """)

                        # Show key metrics visualization
                        st.subheader("Key Health Indicators")

                        # Get reference ranges for metrics
                        ref_ranges = get_reference_ranges()

                        # Create visualizations for key metrics
                        metrics_to_show = [m for m in ['HbA1c', 'BMI', 'Chol', 'TG', 'HDL', 'LDL']
                                           if m in input_data and m in ref_ranges]

                        if metrics_to_show:
                            cols = st.columns(min(3, len(metrics_to_show)))

                            for i, metric in enumerate(metrics_to_show):
                                col_idx = i % len(cols)

                                with cols[col_idx]:
                                    value = input_data[metric]
                                    min_val = ref_ranges[metric]['min']
                                    max_val = ref_ranges[metric]['max']
                                    unit = ref_ranges[metric]['unit']
                                    label = ref_ranges[metric]['label']

                                    # Determine if value is in normal range
                                    if min_val <= value <= max_val:
                                        status = "Normal"
                                        color = "#28a745"  # Green
                                    elif value < min_val:
                                        status = "Low"
                                        color = "#ffc107"  # Yellow
                                    else:
                                        status = "High"
                                        color = "#dc3545"  # Red

                                    # Create card-like display for metrics
                                    st.markdown(f"""
                                    ### {label}
                                    **Value:** <span style="color:{color}; font-size:1.2em;">{value} {unit}</span>  
                                    **Status:** <span style="color:{color};">{status}</span>  
                                    **Normal range:** {min_val}-{max_val} {unit}
                                    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("About the Diabetes Detection System")

        st.markdown("""
        This system uses machine learning to predict diabetes risk based on clinical and demographic data. 
        The prediction model was trained on a dataset of diabetes patients and controls.

        ### Model Information
        The system uses a Random Forest model that was trained with cross-validation to ensure reliability. 
        Key predictive factors include HbA1c (glycated hemoglobin), BMI (body mass index), and age.

        ### How to Use
        1. Enter patient data in the input form
        2. Click "Analyze Diabetes Risk" to get predictions
        3. Review the risk assessment and treatment plan

        ### Important Note
        This system is for informational purposes only and does not substitute for professional medical advice. 
        Please consult a healthcare provider for diagnosis and treatment recommendations.
        """)

    with tab3:
        st.subheader("API Integration Documentation")
        st.markdown("""
        This system integrates with the DeepSeek AI API to enhance treatment recommendations
        based on patient data and prediction results.

        ### How it works

        1. When enabled, the system sends anonymized patient data to the DeepSeek AI API
        2. The AI generates treatment recommendations based on medical literature and guidelines
        3. These recommendations are integrated with the standard recommendations to create a comprehensive treatment plan
        4. If the API is unavailable, the system falls back to enhanced static recommendations

        ### API Configuration

        - You can enable/disable the API integration using the sidebar checkbox
        - The default API key is provided, but you can use your own key if needed
        - No patient data is stored on the API servers

        ### AI Treatment Enhancement Benefits

        - More personalized recommendations based on the specific patient data
        - Up-to-date treatment approaches based on recent medical literature
        - Comprehensive lifestyle, medication, and monitoring recommendations

        ### Data Privacy and Security

        - All API calls are made securely using HTTPS
        - Patient identifying information is never shared with the API
        - Only anonymized clinical measurements are used for generating recommendations
        """)

    # Add footer
    st.markdown("---")
    st.markdown(
        "© 2025 Diabetes Detection System | Disclaimer: This system is for informational purposes only",
    )


if __name__ == "__main__":
    # Display the URL in the terminal instead of automatically opening the browser
    print("\n==============================================")
    print("Diabetes Detection System is running!")
    print("Open your browser and go to: http://localhost:8501")
    print("==============================================\n")
    main()