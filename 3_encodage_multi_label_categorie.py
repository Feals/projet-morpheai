import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Charger le dataset
df = pd.read_csv("dataset_dream_dryad_col_num_strandard.csv").head(5)

# Fonction pour enlever les espaces superflus dans chaque cellule
def clean_code_column(column):
    # Enlève les espaces en début et fin de chaîne
    column = column.apply(lambda x: str(x).strip())   
    return column

# Colonnes contenant les catégories multiples
code_columns = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]

# Remplir les NaN avec des chaînes vides et nettoyer les espaces
df[code_columns] = df[code_columns].fillna("").apply(clean_code_column)

# Dictionnaire pour stocker les encodeurs MultiLabelBinarizer pour chaque colonne
mlb_encoders = {}  
encoded_dfs = []   # Liste pour stocker les DataFrames encodés

# Transformation de chaque colonne multi-label
for col in code_columns:
    mlb = MultiLabelBinarizer()
    # Transformer la colonne en liste de catégories (séparées par ",")
    df[col] = df[col].apply(lambda x: x.split(",") if x else [])  
    print("df[col]", df[col])
    # Appliquer l'encodage binaire
    encoded_array = mlb.fit_transform(df[col])
    # Créer un DataFrame avec les colonnes encodées
    encoded_df = pd.DataFrame(encoded_array, columns=[f"{col}_{cls}" for cls in mlb.classes_])
    encoded_dfs.append(encoded_df)
    mlb_encoders[col] = mlb

# Concaténer les nouvelles colonnes encodées avec le DataFrame original
df_final = pd.concat([df] + encoded_dfs, axis=1)

# Supprimer les anciennes colonnes de catégories
df_final = df_final.drop(columns=code_columns)

df_final.to_csv("dataset_dream_dryad_col_categories_encoded.csv", index=False)
