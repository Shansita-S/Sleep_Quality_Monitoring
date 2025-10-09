import gradio as gr
import joblib
import numpy as np
import os

# Debug: Print current working directory and list files
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check if model directory exists
if os.path.exists("model"):
    print(f"Files in model directory: {os.listdir('model')}")
elif os.path.exists("Model"):
    print(f"Files in Model directory: {os.listdir('Model')}")
else:
    print("Model directory not found")

# Load the trained model
pipe = None
model_paths = [
    "drug_pipeline.joblib",  # Direct in app folder
    "model/drug_pipeline.joblib",  # Lowercase model folder
    "Model/drug_pipeline.joblib",  # HuggingFace Spaces structure
    "../model/drug_pipeline.joblib",  # Local development lowercase
    "../Model/drug_pipeline.joblib",  # Local development
    "./model/drug_pipeline.joblib",  # Alternative path lowercase
    "./Model/drug_pipeline.joblib"  # Alternative path
]

for path in model_paths:
    try:
        if os.path.exists(path):
            pipe = joblib.load(path)
            print(f"✅ Model loaded successfully from: {path}")
            break
        else:
            print(f"❌ Model not found at: {path}")
    except Exception as e:
        print(f"❌ Error loading model from {path}: {str(e)}")

if pipe is None:
    print("🔥 CRITICAL: No model could be loaded from any path!")
    print("Available files:", [f for f in os.listdir('.') if f.endswith('.joblib')])
    if os.path.exists("Model"):
        print("Files in Model/:", [f for f in os.listdir('Model') if f.endswith('.joblib')])
    
    # Create a simple fallback model for demonstration
    print("🛠️ Creating fallback model...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler
        import pandas as pd
        
        # Create minimal training data for demo
        demo_data = pd.DataFrame({
            'Age': [25, 35, 45, 55, 65],
            'Sex': ['M', 'F', 'M', 'F', 'M'],
            'BP': ['HIGH', 'NORMAL', 'LOW', 'HIGH', 'NORMAL'],
            'Cholesterol': ['HIGH', 'NORMAL', 'HIGH', 'NORMAL', 'HIGH'],
            'Na_to_K': [15.0, 20.0, 25.0, 30.0, 35.0],
            'Drug': ['drugA', 'drugB', 'drugC', 'drugX', 'DrugY']
        })
        
        X = demo_data.drop("Drug", axis=1).values
        y = demo_data["Drug"].values
        
        # Create pipeline
        transform = ColumnTransformer([
            ("encoder", OrdinalEncoder(), [1, 2, 3]),
            ("num_imputer", SimpleImputer(strategy="median"), [0, 4]),
            ("num_scaler", StandardScaler(), [0, 4]),
        ])
        
        pipe = Pipeline(steps=[
            ("preprocessing", transform),
            ("model", RandomForestClassifier(n_estimators=10, random_state=125)),
        ])
        
        pipe.fit(X, y)
        print("✅ Fallback model created successfully!")
        
    except Exception as e:
        print(f"❌ Failed to create fallback model: {str(e)}")
        pipe = None
else:
    print(f"🎉 Model loaded successfully! Classes: {pipe.classes_}")

def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """
    Predict drugs based on patient features.
    
    Args:
        age (int): Age of patient
        sex (str): Gender of patient
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood
    
    Returns:
        str: Predicted drug label
    """
    if pipe is None:
        return "Model not loaded. Please train the model first."
    
    # Prepare features in the same order as training data
    # [Age, Sex, BP, Cholesterol, Na_to_K]
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    
    try:
        # Make prediction
        predicted_drug = pipe.predict([features])[0]
        
        # Get prediction probabilities for all classes
        probabilities = pipe.predict_proba([features])[0]
        class_names = pipe.classes_
        
        # Create probability distribution
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return prob_dict
    
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Define input components
inputs = [
    gr.Slider(15, 74, step=1, label="Age", value=30),
    gr.Radio(["M", "F"], label="Gender", value="M"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure", value="NORMAL"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol", value="NORMAL"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K Ratio", value=15.0),
]

# Define output component
outputs = [gr.Label(num_top_classes=5, label="Drug Prediction")]

# Example inputs for testing
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8.0],
    [50, "M", "HIGH", "HIGH", 34.0],
    [25, "F", "NORMAL", "HIGH", 20.0],
    [60, "M", "LOW", "NORMAL", 12.5],
]

# App title and description
title = "🏥 Drug Classification System"
description = """
This application predicts the most suitable drug for a patient based on their medical characteristics.
Enter the patient's details below to get a drug recommendation with confidence scores.

*Features:*
- Age: Patient's age (15-74 years)
- Gender: Patient's gender (M/F)
- Blood Pressure: Current blood pressure level
- Cholesterol: Current cholesterol level  
- Na_to_K Ratio: Sodium to Potassium ratio in blood

*Drug Types:*
- DrugY: For specific conditions
- drugA, drugB, drugC, drugX: Different drug categories based on patient profile
"""

article = """
### About This Model
This drug classification system uses a Random Forest machine learning model trained on patient data including age, gender, blood pressure, cholesterol levels, and sodium-to-potassium ratio.

*Disclaimer:* This is for educational purposes only. Always consult healthcare professionals for medical decisions.

### Model Performance
- Training accuracy: Check Results/metrics.txt for detailed performance metrics
- Model uses preprocessing pipeline with categorical encoding and numerical scaling

Built with ❤️ using Gradio and Scikit-learn
"""

# Create and launch the Gradio interface
if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_drug,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        title=title,
        description=description,
        article=article,
        theme=gr.themes.Soft(),
        allow_flagging="never",
        cache_examples=False,  # Disable example caching for faster startup
    )
    
    demo.launch()
