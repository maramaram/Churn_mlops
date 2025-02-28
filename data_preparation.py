import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Prétraitement des données
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from scipy.stats import zscore

# Modèles de classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Sélection des fonctionnalités
from sklearn.feature_selection import SelectKBest, chi2

# Modèles non supervisés
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Échantillonnage et gestion des données déséquilibrées
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Validation croisée et optimisation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV

# Évaluation des modèles
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score)

# Interprétation des modèles
import shap
from lime.lime_tabular import LimeTabularExplainer

from sklearn.tree import plot_tree
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importer les bibliothèques nécessaires
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
import time

def load_and_preprocess_data(filepath):
 # Encoder les colonnes catégoriques
 categorical_columns = data_transformed.select_dtypes(include=['object', 'category']).columns
 label_encoder = LabelEncoder()

 # Vérifier si la colonne 'state' existe parmi les colonnes catégoriques
 if 'State' in categorical_columns:
    # Encoder la colonne 'state'
    data_transformed['State'] = label_encoder.fit_transform(data_transformed['State'])
    # Obtenir le mapping pour 'state'
    state_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Mapping pour la colonne 'state' :")
    print(state_mapping)
 else:
    print("La colonne 'state' n'existe pas dans les colonnes catégoriques.")
# Convertir la colonne 'Churn' en binaire (0 ou 1)
 if data_transformed['Churn'].dtype == 'bool' or data_transformed['Churn'].nunique() == 2:
    data_transformed['Churn'] = data_transformed['Churn'].astype(int)
 else:
    # Si Churn contient des valeurs textuelles comme "Yes"/"No"
    data_transformed['Churn'] = label_encoder.fit_transform(data_transformed['Churn'])
# Transformer les colonnes en valeurs quantitatives : 1 pour yes, 0 pour no
    data_transformed['International plan'] = data_transformed['International plan'].map({'Yes': 1, 'No': 0})
    data_transformed['Voice mail plan'] = data_transformed['Voice mail plan'].map({'Yes': 1, 'No': 0})
# Afficher les premières lignes du DataFrame transformé
    data_transformed.head()

    correlation_matrix = data_transformed.corr()
# 2. Identifier et éliminer les variables fortement corrélées
    threshold = 0.8  # Seuil de corrélation
    columns_to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            columns_to_drop.add(colname)
# 4. Afficher les colonnes à éliminer et celles qui sont corrélées
print("Colonnes fortement corrélées (à éliminer):", columns_to_drop)
# 5. Créer la variable data_reduced pour stocker les données restantes (sans les colonnes supprimées)
data_reduced = data_transformed.drop(columns=columns_to_drop)
# 6. Créer la variable data_reduced_sans_churn avec les données sans la colonne 'Churn'
data_reduced_sans_churn = data_reduced.drop(columns=['Churn'])
# 7. Normaliser les données restantes
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reduced_sans_churn)
# 8. Appliquer PCA
pca = PCA()
pca_result = pca.fit_transform(data_scaled)
# 9. Créer un nouveau DataFrame contenant les composantes principales et la colonne cible
explained_variance_ratio = pca.explained_variance_ratio_
df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(len(explained_variance_ratio))])
df_pca['Churn'] = data['Churn'].values  # Ajouter la colonne cible 'Churn' après PCA
# 10. Visualiser la variance expliquée par chaque composante
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
plt.title('Variance expliquée cumulée')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée')
plt.grid()
plt.show()
# 11. Afficher les résultats
print("Variance expliquée par chaque composante principale:")
for i, var in enumerate(explained_variance_ratio):
    print(f"Composante {i+1}: {var:.2%}")
print("\nLes premières lignes des données après PCA:")
print(df_pca.head())
# 12. Sauvegarder le DataFrame transformé pour une utilisation future
data_pca = df_pca.copy()
