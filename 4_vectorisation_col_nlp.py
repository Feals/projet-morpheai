import pandas as pd
import numpy as np
import spacy

# Charger le dataset
df = pd.read_csv("dataset_dream_dryad_col_categories_encoded.csv", sep='\t')

nlp = spacy.load("en_core_web_lg")

def mean_word_vector(words):
    vectors = [nlp(word).vector for word in words if word in nlp.vocab]
    return np.mean(vectors, axis=0) if vectors else np.zeros(nlp.vocab.vectors.shape[1])

# Vectorisation des colonnes `entities`, `dependencies`, `unique_lemmas`
df["entities_vec"] = df["entities"].apply(mean_word_vector)
df["dependencies_vec"] = df["dependencies"].apply(mean_word_vector)
df["unique_lemmas_vec"] = df["unique_lemmas"].apply(mean_word_vector)



# Supprimer les anciennes colonnes `entities`, `dependencies`, et `unique_lemmas`
df = df.drop(columns=["entities", "dependencies", "unique_lemmas"])

# 4. Résumé des données finales
print(list(df.columns))
print(df.head())

# Sauvegarde du dataset nettoyé
df.to_csv("dataset_dream_dryad_clean.csv", index=False)
print("Dataset nettoyé et mis à jour.")