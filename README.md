# 🌙 Classification de Rêves selon Hall Van De Castle

Ce projet vise à classer des rêves en fonction de la méthode **Hall Van De Castle** en utilisant **du Machine Learning traditionnel** (pas de Deep Learning).  
Nous utilisons un dataset issu de **Dryade** contenant **21 000 rêves catégorisés**.

## 📂 Structure du Projet

Le code est organisé en **quatre fichiers principaux** :

- **`pipeline.py`** : Définit les modèles de préprocessing et d'entraînement.
- **`preprocessing.py`** : Transforme le dataset pour obtenir des données compréhensibles par l’IA.
- **`machinelearning.py`** : Gère l'apprentissage du modèle et l'évaluation des métriques.
- **`api.py`** : Permet de recevoir un rêve envoyé par un utilisateur et retourne la prédiction des catégories.

## 🎯 Problématiques du Projet

### 🚀 **1. Temps de computation trop long**

- L’apprentissage est **très lent** sur CPU.
- Tentative d’utiliser le **GPU avec XGBoost**, mais cela ne fonctionne pas.
- Problème d’optimisation des hyperparamètres à cause du **temps de calcul et de la mémoire limitée**.
  - **GridSearchCV** : Impossible de tester plusieurs valeurs sans faire planter le PC.

### 🎭 **2. Classification Multi-Label difficile**

- Un rêve peut appartenir à plusieurs catégories.
- Le modèle peine à **prédire correctement plusieurs classes simultanément**.

### 🔍 **3. Données déséquilibrées**

- Certaines catégories sont **très peu représentées**.
- Problème : **Comment rééquilibrer les classes avec des données textuelles ?**

## 💡 Pistes d’Amélioration

✅ **Accélération des calculs**

- Trouver un moyen d’utiliser le **GPU plutôt que le CPU**.
- Utiliser une **machine externe** (ex : Google Colab, serveur distant).

✅ **Amélioration du NLP**

- Vérifier si le **prétraitement des textes** est optimal.
- Essayer des techniques d'**augmentation de données textuelles**.

✅ **Rééquilibrage des classes**

- Chercher des **méthodes adaptées aux données textuelles**.
- Explorer des approches comme **SMOTE pour texte**, pondération des classes, ou autres.

✅ **Optimisation des Hyperparamètres**

- Trouver une **méthode pour trouver les hyperparamètre optimum**

## ⚙️ Installation des dépendances

pip install -r requirements.txt
