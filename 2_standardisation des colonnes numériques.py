from sklearn.preprocessing import StandardScaler
import pandas as pd


# Charger le dataset prétraité
df = pd.read_csv("dataset_dream_dryad_nlp_text_dream.csv", sep='\t')

# 1. Standardisation des colonnes numériques
scaler = StandardScaler()
df[['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']] = scaler.fit_transform(df[['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']])

df.to_csv("dataset_dream_dryad_col_num_strandard.csv", sep='\t', index=False)