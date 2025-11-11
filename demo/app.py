import streamlit as st
import pandas as pd
import joblib
import os

st.title(" IDS-ML : Détection d'intrusions réseau")

# Vérification des fichiers
if not os.path.exists('models/ids_model.pkl') or not os.path.exists('models/scaler.joblib'):
    st.error("❌ Modèle ou scaler manquant. Lance d'abord train.py")
    st.stop()

# Chargement du modèle et scaler
clf = joblib.load('models/ids_model.pkl')
scaler = joblib.load('models/scaler.joblib')

uploaded = st.file_uploader("Uploader un fichier CSV KDD Cup 99", type="csv")
if uploaded:
    # Lecture du CSV sans en-tête
    df = pd.read_csv(uploaded, header=None)

    # Renommer les colonnes exactement comme dans train.py (avec "label")
    columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
    ]
    df.columns = columns

    st.write("Aperçu des données uploadées :", df.head())

    # Séparer features et target
    X = df.drop("label", axis=1)
    y = df["label"].apply(lambda x: 0 if x == "normal." else 1)

    # Encodage des colonnes texte
    for col in ["protocol_type", "service", "flag"]:
        X[col] = X[col].astype('category').cat.codes

    # Application du scaler
    X_scaled = scaler.transform(X)

    # Prédiction
    preds = clf.predict(X_scaled)
    df['Prediction'] = preds
    df['Prediction'] = df['Prediction'].map({0:'normal', 1:'attaque'})

    # Affichage et téléchargement
    st.write("Aperçu avec prédictions :", df.head())
    st.download_button(
        "Télécharger les prédictions",
        df.to_csv(index=False).encode(),
        "predictions.csv"
    )
