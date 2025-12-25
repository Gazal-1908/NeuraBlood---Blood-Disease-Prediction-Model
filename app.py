from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__, static_folder="static")

# Load the trained Random Forest model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Blood_diseases.pkl")
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


# Define feature names (must match model training order)
FEATURES = [
    "Glucose", "Cholesterol", "Hemoglobin", "Platelets", "White Blood Cells",
    "Red Blood Cells", "Hematocrit", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
    "Mean Corpuscular Hemoglobin Concentration", "Insulin", "BMI", "Systolic Blood Pressure",
    "Diastolic Blood Pressure", "Triglycerides", "HbA1c", "LDL Cholesterol", "HDL Cholesterol",
    "ALT", "AST", "Heart Rate", "Creatinine", "Troponin", "C-reactive Protein"
]

# Disease recommendations
disease_recommendations = {
    "Healthy": "Eat a balanced diet, exercise regularly, manage stress, and monitor health metrics.",
    "Diabetes": "Monitor blood sugar, maintain a low-carb diet, exercise regularly, and get regular checkups.",
    "Anemia": "Consume iron-rich foods, vitamin C for better absorption, and consult a doctor for supplements.",
    "Thrombocytosis": "Avoid smoking, manage stress, exercise moderately, and monitor platelet counts.",
    "Thalassemia": "Limit iron intake, consider folic acid supplements, and consult a hematologist."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate input data
        data = request.json
        if not data:
            return jsonify({"error": "No input received!"}), 400

        # Ensure all parameters are provided
        input_data = []
        for param in FEATURES:
            if param not in data or data[param] == "":
                return jsonify({"error": f"Missing value for {param}!"}), 400
            input_data.append(float(data[param]))  # Convert to float
        
        # Convert input into numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Get model prediction
        prediction = model.predict(input_array)

        # Handling different prediction formats
        disease_labels = ["Diabetes", "Anemia", "Thalassemia", "Thrombocytosis", "Healthy"]
        predicted_diseases = []

        if isinstance(prediction, (list, np.ndarray)):
            predicted_diseases = [disease_labels[i] for i in range(len(prediction[0])) if prediction[0][i] == 1]
        elif isinstance(prediction, str):  
            predicted_diseases.append(prediction) if prediction in disease_labels else ["Unknown"]

        # Convert list to readable text
        predicted_disease_text = ", ".join(predicted_diseases) if predicted_diseases else "No disease detected"

        # Get recommendations for all predicted diseases
        recommendations = [disease_recommendations.get(disease, "Consult a doctor for a detailed diagnosis.") for disease in predicted_diseases]
        recommendations_text = " ".join(recommendations) if recommendations else "Stay healthy and keep monitoring your health!"

        return jsonify({
            "predicted_disease": predicted_disease_text,
            "recommendation": recommendations_text
        })

    except Exception as e:
        print("Error in prediction:", str(e))  # Debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
