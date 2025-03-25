import time
import joblib
import pandas as pd
from pipeline import clean_code_column
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# Paramètre clé : seuil minimal d'échantillons par classe
MIN_SAMPLES_PER_CLASS = 300
# Charger le modèle pré-entraîné
model_grid_search_classifier = joblib.load("model_grid_search_classifier_pipeline.pkl")

df = pd.read_csv("test_df_after_preprocessing.csv")

# récupérer le nom des colonnes pour les features
columns_to_X = ['text_lemmatized', 'entities', 'dependencies']
categorical_cols_for_trainning_model = ["characters_code"]

nlp_types_labels = [col for col in df.columns if any(col.startswith(prefix) for prefix in columns_to_X)]

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
    classes_to_drop = class_counts.loc[class_counts <= MIN_SAMPLES_PER_CLASS].index
    # on supprime les classes qui n'ont pas assez d'échantillons
    y = y.drop(classes_to_drop, axis=1)

    # séparation des jeux de données pour l'entrainement et pour les tests
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = model_grid_search_classifier.fit(X_train, y_train)

    
    # Prédictions sur le jeu de test et le jeu d'entraînement
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    # Évaluation sur les données d'entraînement
    print(f"\nÉvaluation sur les données d'entraînement pour {category}:")
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
        "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
        "F1-score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")
    }

    for metric_name, metric_func in metrics.items():
        score = metric_func(y_train, y_train_preds)
        print(f"Entraînement - {metric_name}: {score:.4f}")

    # Évaluation sur les données de test
    print(f"\nÉvaluation sur les données de test pour {category}:")
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_test, y_test_preds)
        print(f"Test - {metric_name}: {score:.4f}")

    # Classification report pour le test
    print("\nClassification Report pour le test :")
    print(classification_report(y_test, y_test_preds, target_names=y.columns, zero_division=0))

    # Comparaison détaillée
    y_test_df = pd.DataFrame(y_test, columns=y.columns).reset_index(drop=True)
    y_test_pred_df = pd.DataFrame(y_test_preds, columns=y.columns).reset_index(drop=True)

    comparison_df = pd.DataFrame()
    for class_name in y.columns:
        comparison_df[f"Réel - {class_name}"] = y_test_df[class_name]
        comparison_df[f"Prédit - {class_name}"] = y_test_pred_df[class_name]

    print("\n📊 Comparaison Réel vs Prédit (5 premières lignes) :")
    print(comparison_df.head())

    # Sauvegarde de la comparaison complète
    comparison_df.to_csv(f'comparison_{category}.csv', index=False)
    print(f"Comparaison complète sauvegardée dans 'comparison_{category}.csv'")

    # Matrice de confusion pour le test
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test.values.argmax(axis=1), y_test_preds.argmax(axis=1))

    # On remplace les indices par les noms des classes dans les axes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    xticklabels=y.columns, yticklabels=y.columns)

    # Titre et labels des axes
    plt.title(f'Matrice de confusion pour {category} (Test)')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')

    # Afficher la matrice de confusion
    plt.show()

    models[category] = model

    # Sauvegarde du modèle spécifique à cette famille de labels
    joblib.dump(model, f'test_dream_model_{category}.pkl')

end_time = time.time()
print(f"Temps de l'apprentissage: {end_time - start_time:.6f} secondes")