import time
import joblib
import pandas as pd
from pipeline import clean_code_column
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)
import os


# Charger le pipeline de prétraitement
preprocessor = joblib.load("preprocessor_pipeline.pkl")

# Charger le modèle pré-entraîné
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")

# Charger le dataset par chunks
chunk_size = 200
df = pd.read_csv("dream_data_dryad.tsv", sep='\t', chunksize=chunk_size)

# Colonnes à supprimer
columns_to_drop = ['A/CIndex', 'F/CIndex', 'S/CIndex', "dream_id", "dreamer", "description", 
                   "dream_date", "dream_language", 'Male', 'Animal', 'Friends', 'Family', 
                   'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']

categorical_cols = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]


categorical_cols_for_trainning_model = ["characters_code"]


text_cols = ["text_dream"]
df_train_columns = categorical_cols + text_cols

iteration_count = 0

for chunk in df:
    iteration_count += 1
    print("Iteration:", iteration_count)
    start_time = time.time()

    # Nettoyage et transformation
    chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
    chunk_train = chunk[df_train_columns]
    data_transformed = preprocessor.fit_transform(chunk_train)
    end_time_preprocessing = time.time()
    print(f"Temps pour l'itération {iteration_count}: {end_time_preprocessing - start_time:.6f} secondes")
    # Récupération des noms de colonnes après transformation
    cat_feature_names = preprocessor.transformers_[0][1].named_steps['mlb'].get_feature_names_out(categorical_cols)
    vectorizer_feature_names = preprocessor.transformers_[1][1].named_steps['vectorizer'].get_feature_names_out()
    all_feature_names = list(cat_feature_names) + list(vectorizer_feature_names)

    df_transformed = pd.DataFrame(data_transformed, columns=all_feature_names)

    # Séparation des features et labels
    X = df_transformed.drop(columns=cat_feature_names, axis=1).astype('float32')

    # Groupement des labels par famille (ex: characters_code_XXXX)
    models = {}
    for category in categorical_cols_for_trainning_model:
        category_labels = [col for col in cat_feature_names if col.startswith(category)]
        
        if not category_labels:
            print(f"Aucune colonne trouvée pour {category}, on passe.")
            continue
        
        y = df_transformed[category_labels].astype('float32')

        # Vérifier qu'il y a suffisamment d'échantillons pour entraîner un modèle
        class_counts = y.sum(axis=0)
        classes_to_drop = class_counts.loc[class_counts < 10].index
        y = y.drop(classes_to_drop, axis=1)
        print("y", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("y_test", y_test.shape)
        print("y.columns", y.columns.shape)
        # On utilise le modèle Grid Search déjà chargé
        if os.path.exists("dream_model_characters_code.pkl"):
            model_grid_search_classifier = joblib.load("dream_model_characters_code.pkl")
        model = model_grid_search_classifier.fit(X_train, y_train)

        y_preds = model.predict(X_test)
        print("y_preds.shape", y_preds.shape)
        print("y_preds", y_preds)
        accuracy = accuracy_score(y_test, y_preds) # mesure la proportion de bonnes prédictions.
        print(f"Accuracy for {category}: {accuracy}")
        precision = precision_score(y_test, y_preds, average="weighted") # Mesure la fiabilité des prédictions positives.
        print(f"precision for {category}: {precision}")
        recall = recall_score(y_test, y_preds, average="weighted")# Mesure la capacité du modèle à détecter les vrais positifs.
        print(f"recall for {category}: {recall}")
        f1 = f1_score(y_test, y_preds, average="weighted") # Moyenne harmonique de la précision et du rappel
        print(f"f1 for {category}: {f1}")

        # affichage des résultats attendus vs résultats prédits
        y_test_df = pd.DataFrame(y_test, columns=y.columns).reset_index(drop=True)
        y_preds_df = pd.DataFrame(y_preds, columns=y.columns).reset_index(drop=True)

        # Création d’un DataFrame pour comparer prédictions et réalité
        comparison_df = pd.DataFrame()

        for class_name in y.columns:
            comparison_df[f"Réel - {class_name}"] = y_test_df[class_name]
            comparison_df[f"Prédit - {class_name}"] = y_preds_df[class_name]

        # Affichage des 5 premières lignes
        print("\n📊 Comparaison Réel vs Prédit (5 premières lignes) :")
        print(comparison_df.head(5))

        print("\nClassification Report:")
        print(classification_report(y_test, y_preds, target_names=y.columns, zero_division=0))

        models[category] = model

        # Sauvegarde du modèle spécifique à cette famille de labels
        joblib.dump(model, f'dream_model_{category}.pkl')

    end_time = time.time()
    print(f"Temps pour l'itération {iteration_count}: {end_time - start_time:.6f} secondes")
