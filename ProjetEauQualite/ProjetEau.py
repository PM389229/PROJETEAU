import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Charger les bases de données pour les différentes années
data_2018 = pd.read_csv('data_2018.csv')
data_2019 = pd.read_csv('data_2019.csv')
data_2020 = pd.read_csv('data_2020.csv')

# Dictionnaire de correspondance entre les valeurs actuelles et les nouvelles valeurs
mapping = {
    'C2S1': 'Moderate',
    'C3S1': 'Poor',
    'C4S2': 'Bad',
    'C4S1': 'Bad',
    'C3S2': 'Poor',
    'C4S4': 'Bad',
    'C4S3': 'Bad',
    'C1S1': 'Good',
    'C3S4': 'Bad',
    'C3S3': 'Poor',
    'C2S2': 'Moderate',
}

# Modifier les valeurs de la colonne "Classification" dans chaque base de données
data_2018['Classification'] = data_2018['Classification'].replace(mapping)
data_2019['Classification'] = data_2019['Classification'].replace(mapping)
data_2020['Classification'] = data_2020['Classification'].replace(mapping)

# Créer une instance de LabelEncoder
label_encoder = LabelEncoder()

def encode_labels(data):
    # Encoder les étiquettes de classe en valeurs numériques
    data['Classification'] = label_encoder.fit_transform(data['Classification'])

def select_top_correlated_features(data, year):
    # Matrice de corrélation
    correlation_matrix = data.corr()

    # Tracé de la matrice de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Matrice de Corrélation - Année {year}")
    st.pyplot()

    # Sélection des 10 fonctionnalités les plus corrélées avec la cible
    target_column = 'Classification'
    top_correlated_features = correlation_matrix[target_column].abs().nlargest(11).index
    top_correlated_features = top_correlated_features.drop(target_column)

    st.write(f"Les 10 fonctionnalités les plus corrélées avec la cible - Année {year}:")
    st.write(top_correlated_features)

# Encodage des étiquettes et sélection des fonctionnalités pour chaque année
for year, data in zip(['2018', '2019', '2020'], [data_2018, data_2019, data_2020]):
    encode_labels(data)
    select_top_correlated_features(data, year)