import spacy
import pandas as pd
import unidecode
import nltk
from nltk.corpus import wordnet as wn
import numpy as np

# Téléchargement des ressources NLTK
nltk.download('wordnet')
nltk.download('stopwords')

# Importation de la base de données
df = pd.read_csv("dream_data_dryad.tsv", sep='\t').head(100)

# Supression des colonnes inutiles
df = df.drop(columns=['A/CIndex', 'F/CIndex', 'S/CIndex', "dream_id", "dreamer", "description", "dream_date", "dream_language"], errors='ignore')


# Suppression des valeurs NaN ou vide
df_clean = df.loc[df["text_dream"].notna() & (df["text_dream"].str.strip() != "")].reset_index(drop=True)

# Transformation de la colonne text_dream en une liste python manipulable
text_dreams = df_clean["text_dream"].tolist()

# Chargement du modèle spaCy
parser = spacy.load("en_core_web_lg")

# Générer une liste de mots-clés liés aux rêves via WordNet
"""important_concepts = ["dream", "sleep", "nightmare", "vision", "fantasy", "hallucination", "subconscious", 
                      "cat", "dog", "bird", "fish", "snake", "horse", "tree", "house", "car", "forest", "mountain"]

important_words = set()
for concept in important_concepts:
    for syn in wn.synsets(concept):
        for lemma in syn.lemmas():
            important_words.add(lemma.name().lower())

# Extraction des mots importants du dataset
all_words = []
for dream in text_dreams:
    doc = parser(dream.lower())
    all_words.extend([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Mise à jour des mots fréquents
freq_word = Counter(all_words)
frequent_words = {word for word, freq in freq_word.items() if freq > 10}
important_words.update(frequent_words)"""

# Traitement des rêves
results_texts_dreams_spacy = []
for dream in text_dreams:
    # Transformation des caractères accentués ou spéciaux en caractères ASCII + mettre le texte en minuscule
    dream = unidecode.unidecode(dream).lower()


    doc = parser(dream)
    # On récupère les mots lemmatisé sans ponctuation, etc
    text_dream_clean = [segment.lemma_ for segment in doc 
                        if segment.is_alpha and
                        segment.pos_ not in ['PUNCT', 'CCONJ', 'DET'] 
                        # and (segment.lemma_ in important_words or not segment.is_stop)
                        ]
    # Utilisation directe de set() pour obtenir les lemmas uniques
    dico_ref = list(set(text_dream_clean))

    # Filtrer les entités pour ne garder que celles contenant des mots présents dans text_dream_clean
    entities = [(entity.text, entity.label_) 
                for entity in doc.ents 
                if any(word in text_dream_clean for word in entity.text.split())]

    # Filtrer les dépendances pour ne garder que celles où le token est dans text_dream_clean
    dependencies = [(token.text, token.dep_, token.head.text) 
                for token in doc 
                if token.text in text_dream_clean]

    results_texts_dreams_spacy.append({
        "entities": entities,
        "dependencies": dependencies,
        "unique_lemmas": dico_ref
    })

    
# Création du DataFrame final

df_dreams_spacy = pd.DataFrame(results_texts_dreams_spacy)
df_dreams_spacy = df_dreams_spacy.reset_index(drop=True)
df_clean = df_clean.reset_index(drop=True)
# fusion du dataset d'origine et du nouveau
df_final_with_text_dreams = pd.concat([df_clean, df_dreams_spacy], axis=1)
# suppression de la colonne text_dream qui est devenu 3 colonnes (entities, dependencies, unique_lemmas)
df_final = df_final_with_text_dreams.drop(columns=['text_dream'])

# Sauvegarde des fichiers
df_final.to_csv("dataset_dream_dryad_nlp_text_dream.csv", sep='\t', index=False)
