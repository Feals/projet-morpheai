from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import spacy
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

# Preprocessing for numerical data
numerical_cols = ['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']

numerical_transformer = Pipeline(steps=[
    # imputation des données vide, on remplace ces cellules par 0
    ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
    # Normalisation, on s'assure que les données soient bien compris entre 0 et 1
    ('minMax', MinMaxScaler())
])


# Preprocessing for categorical data
categorical_cols = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]

def clean_code_column(column):
    # on remplace les cellules vides par une chaine de charactères vide.
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
            # apprend toutes les classes uniques de X[column])
            mlb.fit(X[column])
            # crée un dictionnaire qui contient l'ensembles des classes
            self.mlb_dict[column] = mlb
        return self

    def transform(self, X):      
        transformed_data = []
        
        for column in X.columns:
            # crée les valeurs binaires pour chaques colonnes dans une list
            transformed_column = self.mlb_dict[column].transform(X[column])
            # on sauvegarde ces valeurs dans la variable transformed
            transformed_data.append(transformed_column)
        
        # Empile horizontalement les données transformées
        transformed_data = np.hstack(transformed_data)
        # Générer un DataFrame avec les bonnes colonnes
        column_names = self.get_feature_names_out(X.columns)        
        transformed_data = pd.DataFrame(transformed_data, columns=column_names)
        return transformed_data

    def get_feature_names_out(self, input_features=None):
        column_names = []
        # permet de donner un nom compréhensible à chaque colonnes binarizé
        for column, mlb in self.mlb_dict.items():
            column_names.extend([f"{column}_{cls}" for cls in mlb.classes_])
        return column_names

# Transformer catégoriel pour les colonnes où les codes sont séparés par des virgules
categorical_transformer = Pipeline(steps=[
    ('clean_columns', FunctionTransformer(clean_code_column)),
    ("mlb", MultiLabelBinarizerTransformer())
])

# Preprocessing for text
text_cols = ['text_dream']

# Téléchargement des ressources NLTK
nltk.download('wordnet')
nltk.download('stopwords')

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_lg")

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
        stop_words = set(stopwords.words('english'))
        
        for dream in X['text_dream']:
            # Normalisation des accents et mise en minuscule
            dream_normalized = unidecode.unidecode(dream).lower()
            
            # crée un objet doc avec spacy qui contient toutes les informations du texte qu'on lui passe afin de pouvoir lui appliquer des traitements
            doc = nlp(dream_normalized)
            
            # Extraction des lemmes (mots de base)
            lemmatized_words = [
                lemmatizer.lemmatize(token.text, pos='v') if token.pos_ == 'VERB' else
                lemmatizer.lemmatize(token.text, pos='n') if token.pos_ == 'NOUN' else
                lemmatizer.lemmatize(token.text, pos='a') if token.pos_ == 'ADJ' else
                lemmatizer.lemmatize(token.text)
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
            encodage = TfidfVectorizer()
            encodage.fit(X[column])  # Apprentissage du vocabulaire
            self.vectorizer[column] = encodage  # Stockage du vectorizer pour chaque colonne
        return self

    def transform(self, X):
        transformed_data = []

        # Transformer chaque colonne
        for column in X.columns:
            transformed_column = self.vectorizer[column].transform(X[column])  # Transformation avec le bon vectorizer
            transformed_data.append(transformed_column.toarray())  # Conversion en array dense pour empilement

        # Empiler horizontalement les données transformées
        transformed_data = np.hstack(transformed_data)

        # Générer un DataFrame avec les bonnes colonnes
        column_names = self.get_feature_names_out(X.columns)

        transformed_data = pd.DataFrame(transformed_data, columns=column_names)
        return transformed_data

    def get_feature_names_out(self, input_features=None):
        column_names = []
        for column, encodage in self.vectorizer.items():
            # Récupérer les noms des caractéristiques du vectorizer pour chaque colonne
            feature_names = encodage.get_feature_names_out()
            column_names.extend([f"{column}_{name}" for name in feature_names])  # Renommer les caractéristiques
        return column_names

   

text_transformer = Pipeline(steps=[
    ('filter_empty_text', FunctionTransformer(filter_empty_text, validate=False)),  # Fonction de filtrage des textes vides
    ('nlp', TextProcessor()),  # Traitement NLP
    ('vectorizer', VectorizerProcessor()),  # encodage
])






# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('text', text_transformer, text_cols),
    ])






# pipeline modele

model_classifier = Pipeline(steps=[
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),  # Entraînement du modèle RandomForest
])

model_regressor = Pipeline(steps=[
    ('classifier', RandomForestRegressor(n_estimators=100, random_state=42)),  # Entraînement du modèle RandomForest
])



# Sauvegarde du pipeline de prétraitement
joblib.dump(preprocessor, "preprocessor_pipeline.pkl")

# Sauvegarde du pipeline complet (prétraitement + modèle)
joblib.dump(model_classifier, "model_classifier_pipeline.pkl")
joblib.dump(model_regressor, "model_regressor_pipeline.pkl")


# Définition de la grille de recherche pour optimiser les hyperparamètres
param_grid = {
    'estimator__model_classifier__n_estimators': [50, 100, 200], 
    'estimator__model_classifier__learning_rate': [0.01, 0.1, 0.2],
    'estimator__model_classifier__max_depth': [3, 5, 7]
}

model_classifier = Pipeline(steps=[
    ('model_classifier', GradientBoostingClassifier(random_state=42)),
])

model_regressor = Pipeline(steps=[
    ('model_regressor', GradientBoostingRegressor(random_state=42)),
])
multi_target_classifier = MultiOutputClassifier(model_classifier)
# GridSearchCV pour la classification
grid_search_classifier = GridSearchCV(multi_target_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score="raise")

# GridSearchCV pour la régression
grid_search_regressor = GridSearchCV(model_regressor, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Sauvegarde du pipeline complet (prétraitement + modèle)
joblib.dump(grid_search_classifier, "model_grid_search_classifier_pipeline.pkl")
joblib.dump(grid_search_regressor, "model_grid_search_regressor_pipeline.pkl")