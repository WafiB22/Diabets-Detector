# ============================================================
# PROJET MACHINE LEARNING - CLASSIFICATION
# Dataset : Pima Indians Diabetes
# Cible : Outcome (0 = pas diabétique, 1 = diabétique)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CHARGEMENT DU DATASET
# ============================================================
# Dataset Pima Indians Diabetes (disponible via URL publique)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
colonnes = ["Grossesses", "Glucose", "Pression", "Epaisseur_peau",
            "Insuline", "IMC", "Diabete_heritage", "Age", "Diabetique"]

df = pd.read_csv(url, names=colonnes)
print("Shape du dataset :", df.shape)
print("\nAperçu :")
print(df.head())

# ============================================================
# 2. NETTOYAGE
# ============================================================
# Certaines colonnes ont des 0 qui sont en réalité des valeurs manquantes
# (ex : une pression artérielle de 0 est impossible)
cols_zeroes = ["Glucose", "Pression", "Epaisseur_peau", "Insuline", "IMC"]
for col in cols_zeroes:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

print("\nAprès nettoyage - valeurs manquantes :")
print(df.isnull().sum())

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
features = ["Grossesses", "Glucose", "Pression", "Epaisseur_peau",
            "Insuline", "IMC", "Diabete_heritage", "Age"]

X = df[features]
y = df["Diabetique"]

print(f"\nFeatures : {features}")
print(f"Cible    : Diabetique (0 = Non, 1 = Oui)")
print(f"Taux de diabétiques : {y.mean()*100:.1f}%")

# Normalisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ============================================================
# 4. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ============================================================
def evaluer_modele(nom, model):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    cv = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    print(f"\n{'='*50}")
    print(f"Modèle : {nom}")
    print(f"  Accuracy Train  : {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"  Accuracy Test   : {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Recall          : {recall_score(y_test, y_pred_test):.4f}")
    print(f"  Precision       : {precision_score(y_test, y_pred_test):.4f}")
    print(f"  F1-Score        : {f1_score(y_test, y_pred_test):.4f}")
    print(f"  CV (5-fold) moy : {cv.mean():.4f} ± {cv.std():.4f}")

    return {
        "Modèle": nom,
        "Acc Train": round(accuracy_score(y_train, y_pred_train), 4),
        "Acc Test":  round(accuracy_score(y_test, y_pred_test), 4),
        "Recall":    round(recall_score(y_test, y_pred_test), 4),
        "Precision": round(precision_score(y_test, y_pred_test), 4),
        "F1-Score":  round(f1_score(y_test, y_pred_test), 4),
        "CV Mean":   round(cv.mean(), 4),
    }, model

resultats = []

res1, m1 = evaluer_modele("Régression Logistique", LogisticRegression(max_iter=1000, random_state=42))
res2, m2 = evaluer_modele("Arbre de Décision",     DecisionTreeClassifier(random_state=42))
res3, m3 = evaluer_modele("SVC",                   SVC(random_state=42))
res4, m4 = evaluer_modele("Random Forest",         RandomForestClassifier(n_estimators=100, random_state=42))

resultats = [res1, res2, res3, res4]

# ============================================================
# 5. GRIDSEARCH SUR RANDOM FOREST
# ============================================================
print("\n\nGRIDSEARCH - Random Forest")
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Meilleurs paramètres : {grid.best_params_}")
print(f"Meilleur score CV    : {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# ============================================================
# 6. TABLEAU COMPARATIF
# ============================================================
df_res = pd.DataFrame(resultats)
print("\n\nTABLEAU COMPARATIF :")
print(df_res.to_string(index=False))

# ============================================================
# 7. FEATURE IMPORTANCES
# ============================================================
importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n\nFEATURE IMPORTANCES :")
print(importances)

plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="steelblue")
plt.title("Feature Importances - Random Forest (Diabète)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/feature_importances_diabete.png")
print("\nGraphique sauvegardé.")
