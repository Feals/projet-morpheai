import joblib
import pandas as pd
from pipeline_test import clean_code_column, MultiLabelBinarizerTransformer, filter_empty_text




# Charger le pipeline de prétraitement
preprocessor = joblib.load("test_preprocessor_pipeline.pkl")

# Charger le modèle pré-entraîné
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")

df = pd.read_csv("dream_data_dryad.tsv", sep='\t').head(1000)

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

joblib.dump(preprocessor, "test_preprocessor_pipeline_fit.pkl")

# Récupération des noms de colonnes transformées
cat_feature_names = preprocessor.transformers_[0][1].named_steps['mlb'].get_feature_names_out(categorical_cols)
vectorizer_feature_names = preprocessor.transformers_[1][1].named_steps['vectorizer'].get_feature_names_out()
all_feature_names = cat_feature_names + vectorizer_feature_names

# Conversion en DataFrame
df_transformed = pd.DataFrame(data_transformed, columns=all_feature_names)
df_transformed.to_csv("test_df_after_preprocessing.csv", index=False)