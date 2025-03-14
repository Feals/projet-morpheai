import time
import joblib
import pandas as pd
from pipeline import clean_code_column
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score


# Charger le pipeline de prétraitement (pré-trainé)
preprocessor = joblib.load("preprocessor_pipeline.pkl")
model_classifier = joblib.load("model_classifier_pipeline.pkl")
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")


# 1000 trop long
chunk_size = 50

# Charger le dataset
df = pd.read_csv("dream_data_dryad.tsv", sep='\t',  chunksize=chunk_size)

# Supprimer les colonnes inutiles
columns_to_drop = ['A/CIndex', 'F/CIndex', 'S/CIndex', "dream_id", "dreamer", "description", "dream_date", "dream_language", 'Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']

categorical_cols = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]
text_cols=["text_dream"]

# Sélectionner les colonnes présentes dans le dataset
df_train_columns  = categorical_cols + text_cols

iteration_count = 0
for chunk in df:
    iteration_count += 1
    print("iteration_count", iteration_count)
    start_time = time.time()

    # Supprimer les colonnes inutiles
    chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
    chunk_train = chunk[df_train_columns]
    # Utiliser fit_transform pour entraîner et transformer directement les données
    data_transformed = preprocessor.fit_transform(chunk_train)
    # Pour les colonnes catégoriques, récupérer les noms générés par MultiLabelBinarizer
    cat_feature_names = preprocessor.transformers_[0][1].named_steps['mlb'].get_feature_names_out(categorical_cols)
    vectorizer_feature_names = preprocessor.transformers_[1][1].named_steps['vectorizer'].get_feature_names_out()


    # Combiner les noms de colonnes numériques et catégoriques
    all_feature_names = cat_feature_names + vectorizer_feature_names

    # Convertir en DataFrame pour affichage
    df_transformed = pd.DataFrame(data_transformed, columns=all_feature_names)
    #df_transformed.to_csv("df_transformed.csv", index=False)
    






# machine learning
    # Séparer les variables X et y
    X = df_transformed  # Toutes les features transformées
    X = X.drop(cat_feature_names, axis=1)
    X= X.astype('float32')
    y_classification = df_transformed[list(cat_feature_names)]  # Labels de classification
    y_classification = y_classification.astype('float32')

    class_counts = y_classification.sum(axis=0)
    classes_to_drop = class_counts.loc[class_counts < 10].index
    y_class_filtered = y_classification.drop(classes_to_drop, axis=1)
    # Séparer les jeux de données
    X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class_filtered,  test_size=0.2, random_state=42
    )

    # Gradiant Boost

    # --- Classification ---
    model_grid_search_classifier.fit(X_train, y_class_train)

    y_class_preds = model_grid_search_classifier.predict(X_test)

    # Calculer la précision
    accuracy = accuracy_score(y_class_test, y_class_preds)
    print('Accuracy Classification:', accuracy)

    # Classification : Rapport de classification (précision, rappel, F-mesure)
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_class_preds, zero_division=0))

    comparison_class = pd.DataFrame({"Réel": y_class_test.values.flatten(), "Prédit": y_class_preds.flatten()})
    print("\n📊 Comparaison Classification:")
    print(comparison_class.head(10))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps pour l'itération {iteration_count}: {elapsed_time:.6f} secondes")


# 12. **Sauvegarde du modèle optimisé**
joblib.dump(model_grid_search_classifier, 'dream_model_classifier.pkl')