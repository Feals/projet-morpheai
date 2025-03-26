import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import spacy
import unidecode
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from scipy.sparse import hstack, csr_matrix
import numpy as np
from joblib import Memory
from xgboost import XGBClassifier 
import re



# Preprocessing for categorical data


categorical_cols = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]

# on remplace les cellules vides par une chaine de charactères vide.
def clean_code_column(column):
    column = column.fillna("")
    return column



# Transformation avec MultiLabelBinarizer
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb_dict = {}

    def fit(self, X, y=None):
        for column in X.columns:
            mlb = MultiLabelBinarizer()
            # transforme le contenu de la cellule en une liste d'élément, les éléments sont découpées par les ","
            #  et ont retire pour chaques éléments de la liste les potentiels espaces avant et après la chaine de charactères
            X[column] = X[column].apply(lambda x: [item.strip() for item in x.split(",")] if x else [])
            # Filtrer les labels commençant par un nombre
            if column == "characters_code" :
                filtered_labels = [
                [label for label in row if re.match(r'^\d', label)]
                for row in X[column]
                ]
            else : filtered_labels = X[column]

            
            # Apprendre uniquement sur les labels filtrés
            mlb.fit(filtered_labels)
            self.mlb_dict[column] = mlb
        return self

    def transform(self, X):      
        transformed_data = []
    
        for column in X.columns:
            if column == "characters_code" :
                filtered_column = [
                [label for label in row if re.match(r'^\d', label)]
                for row in X[column]
                ]
            else : filtered_column = X[column]
            transformed_column = self.mlb_dict[column].transform(filtered_column)
            transformed_data.append(csr_matrix(transformed_column))
    
        transformed_data = hstack(transformed_data)

        # on nomme les colonnes générées   
        column_names = self.get_feature_names_out(X.columns)    
        transformed_data = pd.DataFrame.sparse.from_spmatrix(transformed_data, columns=column_names)
        return transformed_data

    # permet de donner un nom compréhensible à chaque colonnes binarizé
    def get_feature_names_out(self, input_features=None):
        column_names = []
        for column, mlb in self.mlb_dict.items():
            column_names.extend([f"{column}_{cls}" for cls in mlb.classes_])
        return column_names

# Stockage temporaire en RAM (ou sur disque avec location="/tmp")
memory_categorical_transformer = Memory(location=None, verbose=0)

# Création du pipeline de transformations pour les labels
categorical_transformer = Pipeline(steps=[
    ('clean_columns', FunctionTransformer(clean_code_column)),
    ("mlb", MultiLabelBinarizerTransformer())
], memory=memory_categorical_transformer)



# Preprocessing pour le text
text_cols = ['text_dream']


# Charger le modèle spaCy
nlp = spacy.load("en_core_web_lg")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

lemmatizer = WordNetLemmatizer()
# Fonction de filtrage des lignes vides
def filter_empty_text(dreams):
    return dreams[dreams['text_dream'].str.strip() != ""]

# Classe personnalisée pour le traitement des textes
class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []
        
        for dream in X['text_dream']:
            # Normalisation des accents et mise en minuscule
            dream_normalized = unidecode.unidecode(dream).lower()
            
            # crée un objet doc avec spacy qui contient toutes les informations du texte qu'on lui passe afin de pouvoir lui appliquer des traitements
            doc = nlp(dream_normalized)
            
            # Extraction des lemmes (mots de base)
            lemmatized_words = [
                token.lemma_
                for token in doc
                if token.is_alpha and token.text not in stop_words and token.pos_ not in ['PUNCT', 'CCONJ', 'DET']
            ]
            dico_ref = list(set(lemmatized_words))

            # Ajout des entités extraites
            entities = [(entity.text, entity.label_) for entity in doc.ents]
            
            # Extraction des dépendances syntaxiques
            dependencies = [(token.text, token.dep_, token.head.text) for token in doc if token.text in lemmatized_words]
            
            # Ajout des résultats dans la liste
            results.append({
                "text_lemmatized": dico_ref,  # Texte lemmatisé
                "entities": entities,  # Entités extraites
                "dependencies": dependencies  # Dépendances syntaxiques
            })
        
        # Convertir le résultat en DataFrame
        result_df = pd.DataFrame(results)
        
        # Gestion des entités et des dépendances : les transformer en colonnes supplémentaires
        # Les entités et dépendances peuvent être séparées dans des colonnes spécifiques
        result_df['text_lemmatized'] = result_df['text_lemmatized'].apply(lambda x: " ".join(x))
        result_df['entities'] = result_df['entities'].apply(lambda x: ", ".join([f"{text}:{label}" for text, label in x]))
        result_df['dependencies'] = result_df['dependencies'].apply(lambda x: ", ".join([f"{word}:{dep}:{head}" for word, dep, head in x]))
        # Afficher la forme du DataFrame avant la vectorisation
        return result_df[['text_lemmatized', 'entities', 'dependencies']]
    
    
    def get_feature_names_out(self, input_features=None):
        column_names = ['text_lemmatized', 'entities', 'dependencies']
        return column_names

vectorizer_cols = ['text_lemmatized', 'entities', 'dependencies']

class VectorizerProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = {}

    def fit(self, X, y=None):
        # Apprendre le vocabulaire de chaque colonne
        for column in X.columns:
            encodage = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, dtype=np.float32)
            encodage.fit(X[column])  # Apprentissage du vocabulaire
            self.vectorizer[column] = encodage  # Stockage du vectorizer pour chaque colonne
        return self

    def transform(self, X):
        transformed_data = []

        # Transforme chaque colonne
        for column in X.columns:
            transformed_column = self.vectorizer[column].transform(X[column])  # Transformation avec le bon vectorizer
            transformed_data.append(csr_matrix(transformed_column))

        # Empiler horizontalement les données transformées
        transformed_data = hstack(transformed_data)
        # Générer un DataFrame avec les bonnes colonnes
        column_names = self.get_feature_names_out(X.columns)    
        transformed_data = pd.DataFrame.sparse.from_spmatrix(transformed_data, columns=column_names)
        return transformed_data

    def get_feature_names_out(self, input_features=None):
        column_names = []
        for column, encodage in self.vectorizer.items():
            # Récupérer les noms des caractéristiques du vectorizer pour chaque colonne
            feature_names = encodage.get_feature_names_out()
            column_names.extend([f"{column}_{name}" for name in feature_names])  # Renomme les caractéristiques
        return column_names

   
# Stockage temporaire en RAM (ou sur disque avec location="/tmp")
memory_text_transformer = Memory(location=None, verbose=0)

text_transformer = Pipeline(steps=[
    ('filter_empty_text', FunctionTransformer(filter_empty_text, validate=False)),
    ('nlp', TextProcessor()),  # Traitement NLP
    ('vectorizer', VectorizerProcessor()),  # encodage
], memory = memory_text_transformer)

# Création du pipeline de transformations pour les features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('text', text_transformer, text_cols),
    ])

# pipeline modele
joblib.dump(preprocessor, "preprocessor_pipeline.pkl")


# Définition de la grille de recherche pour optimiser les hyperparamètres

model_classifier = Pipeline(steps=[
    ('model_classifier', XGBClassifier(
        random_state=42, 
        eval_metric='logloss', 
        base_score=0.5, 
        tree_method='hist', 
        device = "cuda",
        subsample=0.8,
    )),
])

multi_target_classifier = MultiOutputClassifier(model_classifier)

# GridSearchCV pour la classification
param_grid_classifier = {
    'estimator__model_classifier__n_estimators': [350], 
    'estimator__model_classifier__learning_rate': [0.08],
    'estimator__model_classifier__max_depth': [5]
}

grid_search_classifier = GridSearchCV(
    multi_target_classifier, 
    param_grid_classifier, 
    cv=3, 
    scoring='accuracy',
    n_jobs=-2,
    pre_dispatch='2*n_jobs',     # Éviter la surcharge  
    error_score="raise"
)

# Sauvegarde du pipeline complet (prétraitement + modèle)
joblib.dump(grid_search_classifier, "model_grid_search_classifier_pipeline.pkl")