📌 Contexte
Le diabète est une maladie chronique touchant des centaines de millions de personnes dans le monde. Un diagnostic précoce améliore considérablement le pronostic et permet d'initier rapidement un traitement adapté.
Ce projet vise à développer un modèle de machine learning capable de prédire si un patient est diabétique à partir de mesures médicales non invasives (glycémie, IMC, âge…).
Il s'agit d'un problème de classification supervisée binaire :

0 → patient non diabétique
1 → patient diabétique


⚕️ Priorité médicale : le Recall est la métrique prioritaire dans ce contexte. Il vaut mieux détecter tous les malades (avec quelques faux positifs) que de manquer un patient diabétique.


📊 Dataset — Pima Indians Diabetes
CaractéristiqueValeurSourceUCI / KagglePatients768Features8Prévalence diabétique34.9%
Features médicales
FeatureUnitéGrossessesnombreGlucosemg/dLPression artériellemm HgÉpaisseur peaummInsulineµU/mLIMCkg/m²Hérédité diabètefacteurÂgeannées

⚙️ Pipeline ML
Données brutes
    │
    ▼
Nettoyage ── Remplacement des valeurs 0 impossibles par la médiane
    │         (Glucose, Pression artérielle, Épaisseur peau, Insuline, IMC)
    ▼
Préparation ── Séparation X / y
    │        ── StandardScaler (moyenne=0, écart-type=1)
    │        ── Split 80% train / 20% test (stratifié)
    ▼
Modélisation ── 4 algorithmes comparés
    │
    ▼
Optimisation ── GridSearchCV sur Random Forest
    │
    ▼
Évaluation ── Accuracy, Recall, Precision, F1-Score, Courbes ROC
    │
    ▼
Dashboard Streamlit (prédiction en temps réel)

🤖 Modèles testés
ModèleDescriptionRégression LogistiqueBaseline linéaire, simple et interprétableArbre de DécisionIntuitif mais susceptible d'overfittingSVCMaximise la marge entre les classes, nécessite normalisationRandom ForestEnsemble de 100 arbres, le plus robuste ✅

📈 Résultats
Comparaison des modèles
ModèleAccuracy (test)Recall (test)NotesRégression Logistique~77%~65%Bonne baselineArbre de Décision~72%~63%⚠️ Overfitting (100% en train)SVC~77%~65%Performances similaires à LRRandom Forest79.2%68.9%✅ Meilleur modèle
Validation croisée (5-fold) — Random Forest
Score moyen : 0.766 ± 0.028
Le faible écart-type confirme la robustesse du modèle.
🔑 Feature Importance — Random Forest
RangFeatureImportance1Glucose~27%2IMC~15%3Âge~13%4Hérédité~12%………

Le glucose sanguin est la variable la plus prédictive, ce qui est cohérent avec la littérature médicale.


🖥️ Dashboard Streamlit
L'interface interactive comporte 5 sections :
SectionContenu📁 DonnéesAperçu du dataset, distribution, carte de corrélation🤖 ModèlesChoix des algorithmes, entraînement, tableau comparatif📊 ÉvaluationAccuracy Train/Test, matrice de confusion, courbes ROC📌 Feature ImportanceImportance des variables (graphique interactif)🔮 PrédictionEntrer les mesures d'un patient → prédiction + niveau de confiance

🚀 Installation & Lancement
1. Cloner le repo
bashgit clone https://github.com/votre-username/diabetes-prediction.git
cd diabetes-prediction
2. Créer un environnement virtuel (recommandé)
bashpython -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
3. Installer les dépendances
bashpip install -r requirements.txt
4. Lancer l'application Streamlit
bashstreamlit run app/streamlit_app.py

📁 Structure du projet
diabetes-prediction/
│
├── data/
│   └── diabetes.csv                  # Dataset Pima Indians
│
├── notebooks/
│   └── exploration.ipynb             # EDA et expérimentations
│
├── src/
│   ├── preprocessing.py              # Nettoyage, normalisation, split
│   ├── train.py                      # Entraînement des 4 modèles
│   └── evaluate.py                   # Métriques, ROC, feature importance
│
├── app/
│   └── streamlit_app.py              # Dashboard Streamlit
│
├── models/
│   └── random_forest_best.pkl        # Modèle sauvegardé
│
├── assets/
│   └── presentation.pptx             # Présentation du projet
│
├── requirements.txt
├── .gitignore
└── README.md

📦 Dépendances principales
scikit-learn
pandas
numpy
streamlit
matplotlib
seaborn
joblib

Générer le fichier complet : pip freeze > requirements.txt


📚 Références

Pima Indians Diabetes Dataset — Kaggle
Scikit-learn Documentation
Streamlit Documentation

👥 Auteur
Wafi BOUIMEDJ github.com/WafiB22
