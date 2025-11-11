import pandas as pd
import joblib
import os

print("📦 Chargement du modèle...")

# Vérifie que les fichiers existent
if not os.path.exists('models/ids_model.pkl') or not os.path.exists('models/scaler.joblib'):
    print("❌ Modèle ou scaler manquant. Lance d'abord train.py")
    exit()

clf = joblib.load('models/ids_model.pkl')
scaler = joblib.load('models/scaler.joblib')

print("📥 Chargement des nouvelles données pour prédiction...")
df = pd.read_csv('data/kddcup.csv', header=None)

# Encodage des colonnes texte comme dans train.py
for col_idx in [1, 2, 3]:  # protocol_type, service, flag
    df[col_idx] = df[col_idx].astype('category').cat.codes

X = df.iloc[:, :-1]
X_scaled = scaler.transform(X)

preds = clf.predict(X_scaled)
df['Prediction'] = preds
df['Prediction'] = df['Prediction'].map({0:'normal', 1:'attaque'})

df.to_csv('demo/predictions.csv', index=False)
print("✅ Prédictions terminées. Résultat dans demo/predictions.csv")
