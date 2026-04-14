# ============================================================
# DASHBOARD STREAMLIT - PRÉDICTION DIABÈTE
# Fichier : dashboard_diabete.py
# Lancement : streamlit run dashboard_diabete.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="Prédiction Diabète",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Prédiction du Diabète")
st.markdown("Dataset : **Pima Indians Diabetes** | Cible : **Diabétique (0 = Non / 1 = Oui)**")
st.divider()

# ============================================================
# CHARGEMENT & NETTOYAGE
# ============================================================
@st.cache_data
def charger_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    colonnes = ["Grossesses", "Glucose", "Pression", "Epaisseur_peau",
                "Insuline", "IMC", "Diabete_heritage", "Age", "Diabetique"]
    df = pd.read_csv(url, names=colonnes)

    # Remplacer les 0 impossibles par la médiane
    cols_zeroes = ["Glucose", "Pression", "Epaisseur_peau", "Insuline", "IMC"]
    for col in cols_zeroes:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df

df = charger_data()

features = ["Grossesses", "Glucose", "Pression", "Epaisseur_peau",
            "Insuline", "IMC", "Diabete_heritage", "Age"]

X = df[features]
y = df["Diabetique"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Paramètres")
modeles_choisis = st.sidebar.multiselect(
    "Modèles à entraîner",
    ["Régression Logistique", "Arbre de Décision", "SVC", "Random Forest"],
    default=["Régression Logistique", "Random Forest"]
)
test_size = st.sidebar.slider("Taille du jeu de test (%)", 10, 40, 20)
cv_folds  = st.sidebar.slider("Folds (Cross-Validation)", 3, 10, 5)

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🧠 Modèles", "📈 Évaluation", "🔍 Feature Importances"])

# ---------- TAB 1 ----------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", len(df))
    col2.metric("Features", len(features))
    col3.metric("Taux de diabétiques", f"{y.mean()*100:.1f}%")

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Distribution diabétique / non-diabétique")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = y.value_counts()
    ax.bar(["Non diabétique (0)", "Diabétique (1)"], counts.values,
           color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Nombre de patients")
    st.pyplot(fig)

    st.subheader("Corrélation entre les features")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------- TAB 2 ----------
with tab2:
    st.subheader("Entraînement des modèles")

    catalogue = {
        "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
        "Arbre de Décision":     DecisionTreeClassifier(random_state=42),
        "SVC":                   SVC(random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
    }

    resultats = []
    modeles_entraines = {}

    if not modeles_choisis:
        st.warning("Sélectionne au moins un modèle dans la barre latérale.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=test_size/100, random_state=42, stratify=y
        )

        for nom in modeles_choisis:
            model = catalogue[nom]
            model.fit(X_tr, y_tr)
            modeles_entraines[nom] = model

            y_pred_tr = model.predict(X_tr)
            y_pred_te = model.predict(X_te)
            cv = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")

            resultats.append({
                "Modèle":    nom,
                "Acc Train": round(accuracy_score(y_tr, y_pred_tr), 4),
                "Acc Test":  round(accuracy_score(y_te, y_pred_te), 4),
                "Recall":    round(recall_score(y_te, y_pred_te), 4),
                "Precision": round(precision_score(y_te, y_pred_te), 4),
                "F1-Score":  round(f1_score(y_te, y_pred_te), 4),
                "CV Mean":   round(cv.mean(), 4),
            })

        df_res = pd.DataFrame(resultats)
        st.dataframe(df_res.set_index("Modèle"), use_container_width=True)
        st.session_state["resultats"] = df_res
        st.session_state["modeles"]   = modeles_entraines
        st.session_state["X_te"]      = X_te
        st.session_state["y_te"]      = y_te

# ---------- TAB 3 ----------
with tab3:
    st.subheader("Comparaison des modèles")

    if "resultats" not in st.session_state:
        st.info("Lance l'entraînement dans l'onglet 🧠 Modèles d'abord.")
    else:
        df_res = st.session_state["resultats"]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(df_res))
        w = 0.35
        ax.bar(x - w/2, df_res["Acc Train"], w, label="Train", color="#3498db")
        ax.bar(x + w/2, df_res["Acc Test"],  w, label="Test",  color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(df_res["Modèle"], rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Train vs Test")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Matrice de confusion")
        modele_sel = st.selectbox("Choisir un modèle", df_res["Modèle"].tolist())
        if modele_sel in st.session_state["modeles"]:
            model = st.session_state["modeles"][modele_sel]
            y_pred = model.predict(st.session_state["X_te"])
            cm = confusion_matrix(st.session_state["y_te"], y_pred)
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay(cm, display_labels=["Non diabétique", "Diabétique"]).plot(
                ax=ax2, colorbar=False, cmap="Blues")
            ax2.set_title(f"Matrice - {modele_sel}")
            st.pyplot(fig2)

# ---------- TAB 4 ----------
with tab4:
    st.subheader("Feature Importances — Random Forest")

    if "modeles" not in st.session_state or "Random Forest" not in st.session_state["modeles"]:
        st.info("Entraîne le modèle Random Forest dans l'onglet 🧠 Modèles d'abord.")
    else:
        rf = st.session_state["modeles"]["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        importances.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Importance des features (Random Forest)")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

        st.dataframe(
            importances.sort_values(ascending=False)
                       .rename("Importance")
                       .reset_index()
                       .rename(columns={"index": "Feature"}),
            use_container_width=True
        )

# ============================================================
# PRÉDICTION MANUELLE
# ============================================================
st.divider()
st.subheader("🔮 Faire une prédiction sur un nouveau patient")

if "modeles" not in st.session_state:
    st.info("Entraîne au moins un modèle d'abord.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        grossesses = st.number_input("Nombre de grossesses", 0, 20, 2)
        glucose    = st.number_input("Glucose (mg/dL)", 50, 250, 120)
        pression   = st.number_input("Pression artérielle (mmHg)", 30, 130, 70)
    with col2:
        epaisseur  = st.number_input("Épaisseur peau (mm)", 0, 100, 20)
        insuline   = st.number_input("Insuline (µU/mL)", 0, 900, 80)
        imc        = st.number_input("IMC (kg/m²)", 10.0, 70.0, 28.0)
    with col3:
        heritage   = st.number_input("Héritage diabète", 0.0, 3.0, 0.5)
        age        = st.slider("Âge", 18, 80, 35)
        modele_sel = st.selectbox("Modèle", list(st.session_state["modeles"].keys()))

    patient = pd.DataFrame([[grossesses, glucose, pression, epaisseur,
                              insuline, imc, heritage, age]], columns=features)
    patient_scaled = pd.DataFrame(scaler.transform(patient), columns=features)

    if st.button("🩺 Prédire"):
        pred = st.session_state["modeles"][modele_sel].predict(patient_scaled)[0]
        if pred == 1:
            st.error("⚠️ Ce patient est **à risque de diabète** selon le modèle.")
        else:
            st.success("✅ Ce patient **n'est pas à risque** selon le modèle.")

        st.caption("⚠️ Ce résultat est uniquement à titre éducatif, pas un diagnostic médical.")
