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

# Charger le modèle pré-entraîné
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")

df = pd.read_csv("test_df_after_preprocessing.csv")

# récupérer le nom des colonnes pour les features
columns_to_X = ['text_lemmatized', 'entities', 'dependencies']
categorical_cols_for_trainning_model = ["characters_code"]

nlp_types_labels = [col for col in df.columns if any(col.startswith(prefix) for prefix in columns_to_X)]

# initialisation du nombre d'itération
iteration_count = 0

# réinitialisation du dataframe qui a été consommé précédement
df = pd.read_csv("test_df_after_preprocessing.csv")
iteration_count += 1
print("Iteration:", iteration_count)
# permet de lancer un chronométre afin de connaître le temps d'une itération 
start_time = time.time()

# Séparation des features au format float32
X = df[nlp_types_labels].astype('float32')

# Groupement des labels par famille (ex: characters_code_XXXX)
models = {}
for category in categorical_cols_for_trainning_model:
    category_labels = [col for col in df if col.startswith(category)]
        
    if not category_labels:
            print(f"Aucune colonne trouvée pour {category}, on passe.")
            continue

    # Séparation des labels au format float32
    y = df[category_labels].astype('float32')

    # Vérifier qu'il y a suffisamment d'échantillons pour entraîner un modèle
    class_counts = y.sum(axis=0)
    classes_to_drop = class_counts.loc[class_counts < 10].index
    # on supprime les classes qui n'ont pas assez d'échantillons
    y = y.drop(classes_to_drop, axis=1)

    # séparation des jeux de données pour l'entrainement et pour les tests
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
joblib.dump(model, f'test_dream_model_{category}.pkl')

end_time = time.time()
print(f"Temps pour l'itération {iteration_count}: {end_time - start_time:.6f} secondes")
