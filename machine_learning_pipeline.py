import joblib
import pandas as pd
from pipeline import clean_code_column
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score


# Charger le pipeline de prétraitement (pré-trainé)
preprocessor = joblib.load("preprocessor_pipeline.pkl")
model_classifier = joblib.load("model_classifier_pipeline.pkl")
model_regressor = joblib.load("model_regressor_pipeline.pkl")
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")
model_grid_search_regressor = joblib.load("model_grid_search_regressor_pipeline.pkl")



chunk_size = 50

# Charger le dataset
df = pd.read_csv("dream_data_dryad.tsv", sep='\t',  chunksize=chunk_size)

# Supprimer les colonnes inutiles
columns_to_drop = ['A/CIndex', 'F/CIndex', 'S/CIndex', "dream_id", "dreamer", 
                      "description", "dream_date", "dream_language"]

# Définir les colonnes numériques et catégoriques
numerical_cols = ['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'Aggression/Friendliness', 'NegativeEmotions']
categorical_cols = ["characters_code", "emotions_code", "aggression_code", "friendliness_code", "sexuality_code"]
text_cols=["text_dream"]

# Sélectionner les colonnes présentes dans le dataset
df_train_columns  = numerical_cols + categorical_cols + text_cols

iteration_count = 0
for chunk in df:
    iteration_count += 1
    print("iteration_count", iteration_count)
    # Supprimer les colonnes inutiles
    chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
    chunk_train = chunk[df_train_columns]
    # Utiliser fit_transform pour entraîner et transformer directement les données
    data_transformed = preprocessor.fit_transform(chunk_train)

    # Récupérer les noms des colonnes transformées (en fonction du type de prétraitement)
    num_feature_names = numerical_cols
    # Pour les colonnes catégoriques, récupérer les noms générés par MultiLabelBinarizer
    cat_feature_names = preprocessor.transformers_[1][1].named_steps['mlb'].get_feature_names_out(categorical_cols)
    vectorizer_feature_names = preprocessor.transformers_[2][1].named_steps['vectorizer'].get_feature_names_out()


    # Combiner les noms de colonnes numériques et catégoriques
    all_feature_names = num_feature_names + cat_feature_names + vectorizer_feature_names

    # Convertir en DataFrame pour affichage
    df_transformed = pd.DataFrame(data_transformed, columns=all_feature_names)
    df_transformed.to_csv("df_transformed.csv", index=False)
    print("df_transformed", df_transformed)





# machine learning
    # Séparer les variables X et y

    X = df_transformed  # Toutes les features transformées
    X = X.drop(num_feature_names + cat_feature_names, axis=1)
    X = X.astype('float32')
    y_classification = df_transformed[list(cat_feature_names)].astype('float32')  # Labels de classification
    y_regression = df_transformed[numerical_cols].astype('float32')  # Labels de régression
    class_counts = y_classification.sum(axis=0)
    classes_to_drop = class_counts.loc[class_counts < 10].index
    y_class_filtered = y_classification.drop(classes_to_drop, axis=1)
    # Séparer les jeux de données
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class_filtered, y_regression, test_size=0.2, random_state=42
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
    print(classification_report(y_class_test, y_class_preds))

    comparison_class = pd.DataFrame({"Réel": y_class_test.values.flatten(), "Prédit": y_class_preds.flatten()})
    print("\n📊 Comparaison Classification:")
    print(comparison_class.head(10))

    # --- Régression ---
    model_grid_search_regressor.fit(X_train, y_reg_train)
    y_reg_preds = model_grid_search_regressor.predict(X_test)

    # Calculer l'erreur absolue moyenne (MAE)
    mae = mean_absolute_error(y_reg_test, y_reg_preds)
    print('MAE Régression:', mae)

    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_reg_test, y_reg_preds)
    print('MSE Régression:', mse)

    # Calculer R2 score (coefficient de détermination)
    r2 = r2_score(y_reg_test, y_reg_preds)
    print('R2 Régression:', r2)

    comparison_reg = pd.DataFrame({"Réel": y_reg_test.values.flatten(), "Prédit": y_reg_preds.flatten()})
    print("\n📊 Comparaison Régression:")
    print(comparison_reg.head(10))


# 12. **Sauvegarde du modèle optimisé**
joblib.dump(model_grid_search_classifier, 'dream_model_classifier.pkl')
joblib.dump(model_grid_search_regressor, 'dream_model_regressor.pkl')


'''
# Random Forest Classique

# --- Classification ---
model_classifier.fit(X_train, y_class_train)
y_class_preds = model_classifier.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_class_test, y_class_preds)
print('Accuracy Classification:', accuracy)

# Classification : Rapport de classification (précision, rappel, F-mesure)
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_preds))

# --- Régression ---
model_regressor.fit(X_train, y_reg_train)
y_reg_preds = model_regressor.predict(X_test)

# Calculer l'erreur absolue moyenne (MAE)
mae = mean_absolute_error(y_reg_test, y_reg_preds)
print('MAE Régression:', mae)

# Calculer l'erreur quadratique moyenne (MSE)
mse = mean_squared_error(y_reg_test, y_reg_preds)
print('MSE Régression:', mse)

# Calculer R2 score (coefficient de détermination)
r2 = r2_score(y_reg_test, y_reg_preds)
print('R2 Régression:', r2)
'''