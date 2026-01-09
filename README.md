# 🎬 Analyse de Sentiments sur les Avis IMDB

Comparaison entre le Machine Learning classique (SVM) et le Deep Learning (LSTM)

Ce projet explore différentes approches pour analyser le sentiment d’avis de films issus du **IMDB Dataset of 50K Movie Reviews**.
J’ai implémenté plusieurs pipelines de prétraitement et testé différentes techniques de vectorisation et de modélisation afin de comparer leurs performances.

---

## 🚀 Objectif

L’objectif principal est de construire un système capable de classifier un avis de film comme **positif** ou **négatif**, et de comparer :

* Les différentes étapes de prétraitement
* Deux librairies de lemmatisation : **NLTK** vs **SpaCy**
* Un modèle de Machine Learning classique : **SVM (avec Bag-of-Words)**
* Un modèle de Deep Learning : **LSTM (avec Keras Embedding et Word2Vec)**

---

## 📊 Présentation des Pipelines

### **Pipeline 1 — SVM + Bag-of-Words + NLTK**

* Nettoyage (suppression HTML, mise en minuscule, suppression ponctuation)
* Tokenisation
* Suppression des stopwords
* Lemmatisation avec **NLTK**
* Vectorisation en **Bag-of-Words**
* Classification via **SVM linéaire**

### **Pipeline 2 — SVM + Bag-of-Words + SpaCy**

Même pipeline que ci-dessus, mais avec **SpaCy** pour la lemmatisation afin de comparer la précision et le temps d’exécution.

### **Pipeline 3 — LSTM + Embedding Keras**

* Prétraitement adapté aux modèles séquentiels (ponctuation et stopwords conservés)
* Tokenizer Keras + padding des séquences
* Entraînement d’un **LSTM utilisant un Embedding interne (Keras)**

### **Pipeline 4 — LSTM + Word2Vec**

* Entraînement de **Word2Vec** sur le dataset
* Construction d’une matrice d’embedding
* Utilisation d’un Embedding initialisé avec Word2Vec
* Même architecture LSTM que le pipeline précédent

---

## 🧠 Modèles & Résultats (Résumé)

| Modèle                     | Accuracy | Notes                                            |
| -------------------------- | -------- | ------------------------------------------------ |
| **SVM + BoW + NLTK**       | 0.857    | Rapide, baseline simple                          |
| SVM + BoW + SpaCy          | 0.855    | Précision similaire mais beaucoup plus lent      |
| **LSTM + Embedding Keras** | 0.87     | Meilleure compréhension du contexte              |
| **LSTM + Word2Vec**        | **0.89** | Meilleur modèle : plus rapide et plus sémantique |

👉 Le modèle **LSTM + Word2Vec** est celui qui offre la meilleure performance globale.

🎯 **Ce modèle a été déployé dans une application Streamlit pour une démonstration interactive.**

---

## 🛠️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/aboodAJ/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

### 2. Créer une virtual environment (linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Utilisation

**Entraîner les modèles :**

Simplement depuis les notebooks.

**Lancer l’application Streamlit :**

```bash
streamlit run app.py
```

---

⭐ Si le projet vous a aidé, n’hésitez pas à mettre une étoile au repository ⭐


