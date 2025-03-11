from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # Pour charger le modèle ML
import numpy as np

app = Flask(__name__)
CORS(app)  # Permet à React d’accéder à l’API

# Charger le modèle ML et le vectorizer
model_classifier = joblib.load("model_grid_search_classifier.pkl")
model_regressor = joblib.load("dream_model_regressor.pkl")  # Si ton modèle utilise du texte
vectorizer = joblib.load("preprocessor_pipeline.pkl")



@app.route("/classificationDream/request", methods=["POST"])
def classify_dream():
    try:
        data = request.get_json()
        dream_text = data.get("dream")  # Récupération de la description du rêve

        if not dream_text:
            return jsonify({"error": "No dream provided"}), 400

        # Transformer le texte en features exploitables par le modèle
        dream_vector = vectorizer.transform([dream_text])

        # Prédire la classe
        prediction_classifier = model_classifier.predict(dream_vector)[0]
        confidence_classifier = model_classifier.predict_proba(dream_vector).max()

        # Prédire la classe
        prediction_regressor = model_regressor.predict(dream_vector)[0]
        confidence_regressor = model_regressor.predict_proba(dream_vector).max()

        return jsonify({"prediction_classifier": prediction_classifier, "confidence_classifier": confidence_classifier, "prediction_regressor": confidence_regressor, "confidence": confidence_regressor})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
