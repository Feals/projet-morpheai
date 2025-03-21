import joblib
import pandas as pd
from pipeline import clean_code_column, MultiLabelBinarizerTransformer, filter_empty_text, TextProcessor, VectorizerProcessor

# Charger le pipeline de prétraitement
preprocessor = joblib.load("preprocessor_pipeline.pkl")

# Charger le modèle pré-entraîné
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")

df = pd.read_csv("dream_data_dryad.tsv", sep='\t')

# Colonnes à supprimer
columns_to_drop = ['A/CIndex', 'F/CIndex', 'S/CIndex', "dream_id", "dreamer", "description", 
                   "dream_date", "dream_language", 'Male', 'Animal', 'Friends', 'Family', 
                   'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']

df=df.drop(columns_to_drop, axis=1)

# Colonnes categorielles
categorical_cols = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]


# Colonne textes
text_cols = ["text_dream"]

# Entraînement du pipeline de prétraitement...
data_transformed = preprocessor.fit_transform(df)

joblib.dump(preprocessor, "preprocessor_pipeline_fit.pkl")
# Vérification que le préprocesseur est bien ajusté
try:
    # Récupération des noms de colonnes transformées
    cat_feature_names = preprocessor.transformers_[0][1].named_steps['mlb'].get_feature_names_out(categorical_cols)
    vectorizer_feature_names = preprocessor.transformers_[1][1].named_steps['vectorizer'].get_feature_names_out()
    all_feature_names = list(cat_feature_names) + list(vectorizer_feature_names)
    
    print("Le préprocesseur a été correctement ajusté.")
except Exception as e:
    print(f"Erreur lors de la vérification du préprocesseur : {e}")
    print("Le préprocesseur n'a peut-être pas été correctement ajusté.")

# Conversion en DataFrame
df_transformed = pd.DataFrame(data_transformed, columns=all_feature_names)
filtered_columns = [col for col in df_transformed.columns if not any(col.startswith(prefix) for prefix in categorical_cols)]

# Afficher les colonnes restantes
df_transformed.to_csv("test_df_after_preprocessing.csv", index=False)