from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np
import spacy

# Charger le dataset prétraité
df = pd.read_csv("dataset_dream_dryad_clean.csv", sep='\t')

df = df.drop(columns=['A/CIndex', 'F/CIndex', 'S/CIndex', "dream_id", "dreamer", "description", "dream_date", "dream_language"], errors='ignore')

# 1. Standardisation des colonnes numériques
scaler = StandardScaler()
df[['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']] = scaler.fit_transform(df[['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']])


# 2. Transformation des colonnes de codes en One-Hot Encoding multi-label
def clean_code_column(column):
    # Enlève les espaces superflus
    column = column.apply(lambda x: str(x).strip())   
    return column



# Colonnes contenant des codes à transformerpp
code_columns = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]

# Remplir les NaN dans les colonnes cibles avec des listes vides ou nettoye la colonne avant la transformation
df[code_columns] = df[code_columns].fillna("").apply(clean_code_column)

# Appliquer le MultiLabelBinarizer sur chaque colonne multi-label
mlb_encoders = {}  # Dictionnaire pour stocker les encodeurs
encoded_dfs = []   # Liste pour stocker les nouveaux DataFrames encodés

# Appliquer le split sur toutes les colonnes concernées
for col in code_columns:
    mlb = MultiLabelBinarizer()
    # Transformer la colonne en liste de valeurs (split par ",")
    df[col] = df[col].apply(lambda x: x.split(",") if x else [])  
    # Encoder la colonne
    encoded_array = mlb.fit_transform(df[col])
    # Créer un DataFrame avec les nouvelles colonnes encodées
    encoded_df = pd.DataFrame(encoded_array, columns=[f"{col}_{cls}" for cls in mlb.classes_])
    encoded_dfs.append(encoded_df)
    mlb_encoders[col] = mlb

# Concaténer toutes les nouvelles colonnes encodées avec le DataFrame original
df_final = pd.concat([df] + encoded_dfs, axis=1)

# Supprimer les anciennes colonnes de codes
df_final = df_final.drop(columns=code_columns)
print("df_final", df_final)

# 3. NLP : Vectorisation des colonnes `entities`, `dependencies`, `unique_lemmas`
nlp = spacy.load("en_core_web_lg")

# Fonction pour vectoriser chaque liste de mots
def mean_word_vector(words):
    vectors = [nlp(word).vector for word in words if word in nlp.vocab]
    return np.mean(vectors, axis=0) if vectors else np.zeros(nlp.vocab.vectors.shape[1])


# Vectorisation des colonnes `entities`, `dependencies`, `unique_lemmas`
df_final["entities_vec"] = df_final["entities"].apply(mean_word_vector)
df_final["dependencies_vec"] = df_final["dependencies"].apply(mean_word_vector)
df_final["unique_lemmas_vec"] = df_final["unique_lemmas"].apply(mean_word_vector)

# Supprimer les anciennes colonnes `entities`, `dependencies`, et `unique_lemmas`
df_final = df_final.drop(columns=["entities", "dependencies", "unique_lemmas"])

# 4. Résumé des données finales
print(list(df_final.columns))
print(df_final.head())

# Sauvegarde du dataset nettoyé
df_final.to_csv("dataset_dream_dryad_clean_v4.csv", index=False)
print("Dataset nettoyé et mis à jour.")