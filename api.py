from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # Pour charger le modèle ML
from dotenv import load_dotenv
import os
import pandas as pd
from pipeline import clean_code_column, MultiLabelBinarizerTransformer, filter_empty_text

load_dotenv()

app = Flask(__name__)
CORS(app)  # Permet à React d’accéder à l’API

model_classifier = joblib.load("test_dream_model_characters_code.pkl")
preprocessor = joblib.load("preprocessor_pipeline.pkl")
print(preprocessor)

@app.route("/classificationDream/request", methods=["POST"])
def classify_dream():
    try:
        data = request.get_json()
        dream_text = data.get("descriptionDream")  # Récupération de la description du rêve
        
        if not dream_text:
            return jsonify({"error": "No dream provided"}), 400
        dream_df = pd.DataFrame({ 
            "characters_code": [""],  # Remplissez avec des valeurs appropriées
            "emotions_code": [""],
            "aggression_code": [""],
            "friendliness_code": [""],
            "sexuality_code": [""],
            "text_dream": [dream_text],
        })
        data_transformed = preprocessor.transform(dream_df)
        print("data_transformed", data_transformed)
        vectorizer_feature_names = preprocessor.transformers_[1][1].named_steps['vectorizer'].get_feature_names_out()
        
        
        vectorized_data = data_transformed[:, :len(vectorizer_feature_names)]  # Garder seulement les colonnes du vectoriseur
        dream_vectorize_df = pd.DataFrame(vectorized_data, columns=vectorizer_feature_names)
        print(dream_vectorize_df[dream_vectorize_df == 1])

        prediction_classifier = model_classifier.predict(dream_vectorize_df)
        print("prediction_classifier", prediction_classifier)
        
        # Identifier les indices des colonnes où la prédiction est 1
        # Par exemple, si prediction_classifier[i] == 1, on garde cette colonne
        predicted_columns = [vectorizer_feature_names[i] for i in range(len(prediction_classifier[0])) if prediction_classifier[0][i] == 1]
        
        # Si vous souhaitez retourner un tableau des codes (par exemple: 'characters_code_1FSA', 'characters_code_2ISA', etc.)
        # Vous pouvez ajuster cette partie selon les noms des colonnes de vos données
        response_data = {
            "predicted_codes": predicted_columns  # Les colonnes où la prédiction est 1
        }
        
        # Retourner les résultats en JSON
        return jsonify(response_data)


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = os.getenv("PORT", 5000)
    app.run(debug=True, host="127.0.0.1", port=int(port))
