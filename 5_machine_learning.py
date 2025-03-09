import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, f1_score
from sklearn.metrics import confusion_matrix
import joblib




# Charger le dataset prétraité
df = pd.read_csv("dataset_dream_dryad_clean.csv")

# Fonction de conversion des chaînes en vecteurs
def convert_to_vector(x):
    return np.fromstring(x.strip('[]'), sep=' ')

# Colonnes contenant des vecteurs sous forme de texte
vector_columns = ['entities_vec', 'dependencies_vec', 'unique_lemmas_vec']

# Appliquer la conversion pour chaque colonne de vecteur
for col in vector_columns:
    df[col] = df[col].apply(convert_to_vector)

# Colonnes contenant des vecteurs sous forme de texte
vector_columns = ['entities_vec', 'dependencies_vec', 'unique_lemmas_vec']

# **Préparation des features (X)**
# On empile les colonnes de vecteurs sous forme de matrices
X = np.hstack([np.vstack(df[col].values) for col in vector_columns])

# **Préparation des labels (y)**
label_columns = [col for col in df.columns if col not in vector_columns]
y = df[label_columns]

# **Appliquer MultiLabelBinarizer aux labels**


mlb = MultiLabelBinarizer()
y_encoded = pd.DataFrame(mlb.fit_transform(y.apply(lambda row: [str(val) for val in row if pd.notna(val)], axis=1)),
                         columns=mlb.classes_)

# **Diviser les données en ensembles d'entraînement et de test**


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# **Entraînement du modèle RandomForest**


model = RandomForestClassifier(n_estimators=100, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

model.fit(X_train, y_train)

# **Prédiction sur les données de test**
y_pred = model.predict(X_test)

# **Évaluation du modèle**


print(classification_report(y_test, y_pred))

# **Calcul des métriques**
accuracy = accuracy_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Hamming Loss: {hamming:.4f}")
print(f"F1 Score (micro): {f1:.4f}")

# **Afficher la matrice de confusion pour chaque label**


# **Sauvegarder le modèle**


joblib.dump(model, 'dream_model.pkl')


