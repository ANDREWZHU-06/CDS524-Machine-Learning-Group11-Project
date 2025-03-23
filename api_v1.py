# models.py
import pandas as pd
import numpy as np
import pickle
import os
import traceback
import requests
import re

# Configure paths - search in multiple locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POTENTIAL_DIRS = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "results"),
    os.path.join(BASE_DIR, "Diabetes Dataset", "models"),
    os.path.join(BASE_DIR, "Diabetes Dataset", "Diabetes Dataset", "models"),
    '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Group Project/GPv1/models'
]


# Function to find model files
def find_model_files():
    """Search for model and preprocessing files in potential directories"""
    model_file = None
    preproc_file = None

    # Search for model files
    for directory in POTENTIAL_DIRS:
        if os.path.exists(directory):
            # Check for files in this directory
            files = os.listdir(directory)

            # Look for model file
            if model_file is None:
                for file in files:
                    if file.lower().endswith('.pkl') and (
                            'random' in file.lower() or 'rf' in file.lower() or 'model' in file.lower()):
                        model_file = os.path.join(directory, file)
                        break

            # Look for preprocessing file
            if preproc_file is None:
                for file in files:
                    if file.lower().endswith('.pkl') and (
                            'preproc' in file.lower() or 'preprocessing' in file.lower() or 'info' in file.lower()):
                        preproc_file = os.path.join(directory, file)
                        break

    return model_file, preproc_file


# Add API integration function with improved error handling
def get_treatment_recommendations(patient_data, prediction_result,
                                  api_key="sk-obprxkhbxdlntmxsuwdchgtxjnuhdpoqhbiysjadslmzpltq"):
    """Get treatment recommendations from DeepSeek AI API with improved error handling"""

    # Construct a prompt based on patient data and prediction
    prompt = f"""
    Based on the following clinical data and diabetes risk assessment, provide evidence-based treatment recommendations:

    Clinical Data:
    - HbA1c: {patient_data.get('HbA1c', 'N/A')}%
    - BMI: {patient_data.get('BMI', 'N/A')}
    - Age: {patient_data.get('AGE', 'N/A')}
    - Cholesterol: {patient_data.get('Chol', 'N/A')} mmol/L
    - Triglycerides: {patient_data.get('TG', 'N/A')} mmol/L

    Risk Assessment: {prediction_result}

    Please organize comprehensive recommendations into these three categories:

    1. Lifestyle Modifications - Include specific diet, exercise, and weight management advice
    2. Medication Recommendations - Include any medications that should be considered given the clinical profile
    3. Monitoring Plan - Include recommended frequency of tests and assessments

    Format your recommendations as simple bullet points only, without subcategories, section titles, or formatting. Provide only practical, specific recommendations that can be directly implemented. Ensure all text is in English only.
    """

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Increase timeout for more reliable connection
        response = requests.request("POST", url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse response
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return {
                "success": True,
                "content": result['choices'][0]['message']['content']
            }
        else:
            return {
                "success": False,
                "error": "Unable to get treatment recommendations from API response."
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Connection to treatment API timed out. Using standard recommendations only."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to treatment API. Network connection may be unavailable."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Error connecting to treatment API: {str(e)}"
        }


def parse_treatment_recommendations(recommendations_text):
    """Parse API treatment recommendations into structured data with improved cleaning"""
    # Default structure for recommendations
    structured_recommendations = {
        'lifestyle': [],
        'medication': [],
        'monitoring': []
    }

    # Clean the text first - remove unwanted patterns
    # Remove section headers with ###, **, etc.
    cleaned_text = re.sub(
        r'#{1,3}\s*\d*\.?\s*\*{0,2}[A-Za-z\s]+(Modifications|Recommendations|Plan|Target|Monitoring|Additional)\*{0,2}:?',
        '', recommendations_text)

    # Remove any non-ASCII characters (like Chinese characters)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)

    # Remove lines that only have --, **, or other formatting
    cleaned_text = re.sub(r'^\s*[-*]{2,}\s*$', '', cleaned_text, flags=re.MULTILINE)

    # Remove any remaining ** formatting
    cleaned_text = cleaned_text.replace('**', '')

    # Split into lines and process
    lines = cleaned_text.strip().split('\n')

    # Initialize section tracking
    current_section = None

    # First pass: determine sections based on content
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Try to extract section from numbered points (e.g., "1. Lifestyle Modifications:")
        section_match = re.match(r'(?:\d+\.?\s*)?(?:Lifestyle|Diet|Exercise|Physical|Activity|Weight)', line,
                                 re.IGNORECASE)
        if section_match:
            current_section = 'lifestyle'
            lines[i] = None  # Mark for removal
            continue

        section_match = re.match(r'(?:\d+\.?\s*)?(?:Medication|Drug|Metformin|Insulin|Therapy|Prescribe)', line,
                                 re.IGNORECASE)
        if section_match:
            current_section = 'medication'
            lines[i] = None  # Mark for removal
            continue

        section_match = re.match(r'(?:\d+\.?\s*)?(?:Monitor|Test|Check|Screening|Follow-up|Measurement|Level)', line,
                                 re.IGNORECASE)
        if section_match:
            current_section = 'monitoring'
            lines[i] = None  # Mark for removal
            continue

    # Set a default section if none was found
    if current_section is None:
        current_section = 'lifestyle'

    # Second pass: collect actual recommendations
    for line in lines:
        if line is None:  # Skip lines marked for removal
            continue

        line = line.strip()
        if not line:
            continue

        # Clean up the line - remove bullet points and numbering
        for prefix in ['•', '-', '*', '+']:
            if line.startswith(prefix):
                line = line[1:].strip()
                break

        # Remove number prefixes (e.g., "1. ", "2) ", etc.)
        line = re.sub(r'^\d+[.):]\s*', '', line).strip()

        # Skip if line is still empty after cleaning
        if not line:
            continue

        # Skip lines that are just section headers or formatting
        if re.match(r'^[A-Za-z\s]+(Modifications|Recommendations|Plan|Target):?$', line, re.IGNORECASE):
            continue

        # Categorize the line based on content
        if current_section == 'lifestyle' or any(keyword in line.lower() for keyword in
                                                 ['diet', 'exercise', 'physical', 'weight', 'lifestyle', 'activity',
                                                  'eat', 'food']):
            structured_recommendations['lifestyle'].append(line)
        elif current_section == 'medication' or any(keyword in line.lower() for keyword in
                                                    ['medication', 'drug', 'metformin', 'insulin', 'therapy',
                                                     'prescribe', 'pill', 'dosage']):
            structured_recommendations['medication'].append(line)
        elif current_section == 'monitoring' or any(keyword in line.lower() for keyword in
                                                    ['monitor', 'test', 'check', 'screening', 'follow-up',
                                                     'measurement', 'level', 'exam']):
            structured_recommendations['monitoring'].append(line)
        else:
            # If we can't categorize based on keywords, use the current section
            structured_recommendations[current_section].append(line)

    return structured_recommendations


# Provide fallback recommendations when API fails
def get_fallback_recommendations(predicted_class):
    """Provide fallback AI-enhanced recommendations when API fails"""

    # Define some advanced recommendations for each class
    fallback_recommendations = {
        'N': {
            'lifestyle': [
                "Follow a Mediterranean-style diet rich in fruits, vegetables, whole grains, and olive oil",
                "Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity weekly",
                "Maintain adequate hydration with water rather than sugary beverages"
            ],
            'medication': [],
            'monitoring': [
                "Check HbA1c levels annually as a preventive measure",
                "Consider regular blood pressure and cholesterol screening"
            ]
        },
        'P': {
            'lifestyle': [
                "Follow a low glycemic index diet to prevent blood sugar spikes",
                "Incorporate resistance training twice weekly in addition to aerobic exercise",
                "Aim for gradual weight loss of 5-7% of body weight over 6 months",
                "Limit alcohol consumption and avoid smoking"
            ],
            'medication': [
                "Consider metformin if HbA1c levels exceed 6.0% along with other risk factors"
            ],
            'monitoring': [
                "Monitor HbA1c every 3-6 months",
                "Consider continuous glucose monitoring to identify patterns",
                "Regular screening for diabetes-related complications"
            ]
        },
        'Y': {
            'lifestyle': [
                "Limit carbohydrate intake to 45-60g per meal, focusing on complex carbohydrates",
                "Practice portion control using the plate method (½ vegetables, ¼ protein, ¼ whole grains)",
                "Incorporate both aerobic and resistance exercises for optimal glycemic control",
                "Consider working with a registered dietitian for personalized meal planning"
            ],
            'medication': [
                "Consider diabetes medication options based on individual factors",
                "Review potential medication options such as metformin, SGLT2 inhibitors, or GLP-1 receptor agonists",
                "Discuss insulin therapy if HbA1c targets are not achieved with other medications"
            ],
            'monitoring': [
                "Monitor blood glucose levels daily",
                "Check HbA1c every 3 months until target is reached",
                "Schedule annual eye and foot examinations",
                "Regular kidney function tests and cardiovascular assessments"
            ]
        }
    }

    # Return appropriate fallback recommendations
    return fallback_recommendations.get(predicted_class, {
        'lifestyle': [],
        'medication': [],
        'monitoring': []
    })


# Load models and preprocessing info
def load_models():
    """Load model and preprocessing info"""
    try:
        # Find model files
        model_file, preproc_file = find_model_files()

        if model_file is None or preproc_file is None:
            # If not found, try specific paths
            model_file = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Group Project/GPv1/models/Random_Forest_model.pkl'
            preproc_file = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Group Project/GPv1/models/preprocessing_info.pkl'

        # Check if files exist
        if not os.path.exists(model_file):
            return None, None, f"Model file not found: {model_file}"

        if not os.path.exists(preproc_file):
            return None, None, f"Preprocessing file not found: {preproc_file}"

        # Load the files
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        with open(preproc_file, 'rb') as f:
            preprocessing_info = pickle.load(f)

        return model, preprocessing_info, None

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        return None, None, error_msg


# Function to predict diabetes risk
def predict_diabetes_risk(input_data, model, preprocessing_info):
    """Make prediction using the model"""
    try:
        if model is None or preprocessing_info is None:
            return None, None, "Model or preprocessing info not loaded"

        # Extract preprocessing components
        scaler = preprocessing_info['scaler']
        imputer = preprocessing_info['imputer']
        top_features = preprocessing_info['top_features']
        class_mapping = preprocessing_info['class_mapping']

        # Prepare input data
        input_df = pd.DataFrame([input_data])

        # Ensure all required features are present
        for feature in top_features:
            if feature not in input_df.columns:
                input_df[feature] = np.nan

        # Select only needed features in correct order
        input_features = input_df[top_features]

        # Apply preprocessing
        input_imputed = imputer.transform(input_features)
        input_scaled = scaler.transform(input_imputed)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # Map prediction to class label
        inv_class_mapping = {v: k for k, v in class_mapping.items()}

        # Handle the case where prediction is not in mapping
        if prediction in inv_class_mapping:
            predicted_class = inv_class_mapping[prediction]
        else:
            # Fallback to the class with highest probability
            max_prob_idx = np.argmax(probabilities)
            if max_prob_idx in inv_class_mapping:
                predicted_class = inv_class_mapping[max_prob_idx]
            else:
                predicted_class = "Unknown"

        # Map probabilities to class labels
        class_probabilities = {}
        for i, prob in enumerate(probabilities):
            if i in inv_class_mapping:
                class_probabilities[inv_class_mapping[i]] = prob

        return predicted_class, class_probabilities, None

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        return None, None, error_msg


# Risk explanation function
def get_risk_explanation(predicted_class, feature_values):
    """Provide explanation based on prediction and features"""
    explanations = {
        'N': "Your data shows results within normal range. It's important to maintain a healthy lifestyle to prevent diabetes.",
        'P': "Your data suggests you may be in a pre-diabetic state. It's recommended to consult a doctor for further advice.",
        'Y': "Your data indicates a risk of diabetes. We strongly recommend consulting a doctor for further examination."
    }

    additional_info = ""

    # Add explanations for specific metrics
    if 'HbA1c' in feature_values:
        hba1c = feature_values['HbA1c']
        if hba1c > 6.5:
            additional_info += f"• Your HbA1c value of {hba1c}% exceeds the 6.5% diabetes diagnostic threshold.\n"
        elif hba1c > 5.7:
            additional_info += f"• Your HbA1c value of {hba1c}% falls within the 5.7%-6.4% pre-diabetic range.\n"

    if 'BMI' in feature_values:
        bmi = feature_values['BMI']
        if bmi >= 30:
            additional_info += f"• Your BMI of {bmi} is in the obese range, which is a risk factor for diabetes.\n"
        elif bmi >= 25:
            additional_info += f"• Your BMI of {bmi} is in the overweight range, which may increase diabetes risk.\n"

    base_explanation = explanations.get(predicted_class, "Unable to provide risk explanation.")

    if additional_info:
        return f"{base_explanation}\n\nAdditional Analysis:\n{additional_info}"
    else:
        return base_explanation


# Get base recommendations function
def get_base_recommendations(predicted_class):
    """Get base recommendations based on predicted class"""
    general_recommendations = {
        'N': {
            'lifestyle': [
                "Maintain a healthy diet with plenty of vegetables and fruits",
                "Exercise regularly, at least 150 minutes of moderate aerobic activity weekly",
                "Maintain a healthy weight"
            ],
            'medication': [],
            'monitoring': [
                "Get regular check-ups (every 1-2 years)"
            ]
        },
        'P': {
            'lifestyle': [
                "Develop a low-sugar, low-fat diet plan",
                "Increase physical activity to 150-300 minutes of moderate aerobic exercise weekly",
                "If overweight or obese, try to lose 5-10% of your body weight"
            ],
            'medication': [],
            'monitoring': [
                "Monitor blood sugar levels regularly (every 6-12 months)",
                "Consider consulting a doctor or nutritionist for a personalized prevention plan"
            ]
        },
        'Y': {
            'lifestyle': [
                "Follow a strict diet, limiting carbohydrate and sugar intake",
                "Exercise regularly, at least 150 minutes of moderate aerobic activity weekly"
            ],
            'medication': [
                "Consult a doctor immediately for a comprehensive assessment",
                "Consider medication options"
            ],
            'monitoring': [
                "Monitor blood sugar levels regularly",
                "Control other risk factors (such as high blood pressure, high cholesterol)"
            ]
        }
    }

    # Return default recommendations if class not found
    return general_recommendations.get(predicted_class, {
        'lifestyle': ["Maintain a healthy lifestyle"],
        'medication': [],
        'monitoring': ["Consult with a healthcare provider"]
    })


# Get personalized recommendations based on patient data
def get_personalized_recommendations(feature_values):
    """Get personalized recommendations based on patient data"""
    personalized_recs = {
        'lifestyle': [],
        'medication': [],
        'monitoring': []
    }

    # Add BMI-specific recommendations
    if 'BMI' in feature_values and feature_values['BMI'] >= 25:
        personalized_recs['lifestyle'].append(
            "Consider developing a weight loss plan and consult a nutritionist for personalized dietary advice")

    # Add age-specific recommendations
    if 'AGE' in feature_values and feature_values['AGE'] > 45:
        personalized_recs['monitoring'].append(
            "Age is a risk factor for diabetes; consider more frequent blood sugar monitoring")

    # Add triglyceride-specific recommendations
    if 'TG' in feature_values and feature_values['TG'] > 1.7:
        personalized_recs['lifestyle'].append(
            "Your triglyceride levels are high; consider reducing refined carbohydrates and alcohol consumption")

    return personalized_recs


# Reference ranges for health metrics
def get_reference_ranges():
    return {
        'HbA1c': {'min': 4.0, 'max': 5.6, 'unit': '%', 'label': 'HbA1c'},
        'BMI': {'min': 18.5, 'max': 24.9, 'unit': 'kg/m²', 'label': 'BMI'},
        'Chol': {'min': 3.1, 'max': 5.2, 'unit': 'mmol/L', 'label': 'Cholesterol'},
        'TG': {'min': 0.5, 'max': 1.7, 'unit': 'mmol/L', 'label': 'Triglycerides'},
        'HDL': {'min': 1.0, 'max': 1.6, 'unit': 'mmol/L', 'label': 'HDL'},
        'LDL': {'min': 2.0, 'max': 3.0, 'unit': 'mmol/L', 'label': 'LDL'},
        'Urea': {'min': 2.8, 'max': 7.1, 'unit': 'mmol/L', 'label': 'Urea'},
        'Cr': {'min': 40, 'max': 106, 'unit': 'µmol/L', 'label': 'Creatinine'},
        'VLDL': {'min': 0.2, 'max': 1.0, 'unit': 'mmol/L', 'label': 'VLDL'}
    }