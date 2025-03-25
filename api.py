from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from dotenv import load_dotenv
import os
import pandas as pd
from pipeline import clean_code_column, MultiLabelBinarizerTransformer, filter_empty_text

load_dotenv()

app = Flask(__name__)
CORS(app)  # Permet à React d’accéder à l’API


model_classifier = joblib.load("test_dream_model_characters_code.pkl")
preprocessor = joblib.load("preprocessor_pipeline_fit.pkl")
print(preprocessor)

@app.route("/classificationDream/request", methods=["POST"])
def classify_dream():
    try:
        data = request.get_json()
        dream_text = data.get("descriptionDream")  # Récupération de la description du rêve
        
        if not dream_text:
            return jsonify({"error": "No dream provided"}), 400
        dream_df = pd.DataFrame({ 
            "characters_code": [""],
            "emotions_code": [""],
            "aggression_code": [""],
            "friendliness_code": [""],
            "sexuality_code": [""],
            "text_dream": [dream_text],
        })

        data_transformed = preprocessor.transform(dream_df)
        vectorizer_feature_names = preprocessor.transformers_[1][1].named_steps['vectorizer'].get_feature_names_out()       
        vectorized_data = data_transformed[:, :len(vectorizer_feature_names)]
        dream_vectorize_df = pd.DataFrame(vectorized_data, columns=vectorizer_feature_names)

        # prédiction du rêve
        prediction_classifier = model_classifier.predict(dream_vectorize_df)
        vectorizer_labels_names = preprocessor.transformers_[0][1].named_steps['mlb'].get_feature_names_out()    

        # Identifier les indices des colonnes où la prédiction est à 1
        predicted_columns = [vectorizer_labels_names[i] for i in range(len(prediction_classifier[0])) if prediction_classifier[0][i] == 1]

        result_dict = {}

        for col in predicted_columns:
            # On sépare le nom de la catégorie (avant le '_') et la valeur (après le '_')
            category, value = col.rsplit('_', 1)
    
            # On crée une clé dans le dictionnaire si celle-ci n'existe pas
            if category not in result_dict:
                result_dict[category] = []
    
            # On ajoute la valeur à la liste associée à la catégorie
            result_dict[category].append(value)

        response_data = result_dict
        print("response_data", response_data)
        # Retourner les résultats en JSON
        return jsonify(response_data)


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = os.getenv("PORT", 5000)
    app.run(debug=True, host="127.0.0.1", port=int(port))
