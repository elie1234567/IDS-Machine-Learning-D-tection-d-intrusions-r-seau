import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("📥 Chargement du dataset...")

# Vérifie que le fichier existe
if not os.path.exists('data/kddcup.csv'):
    print("❌ Erreur : Le fichier data/kddcup.csv est introuvable.")
    print("➡️ Télécharge-le avec : wget https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz")
    exit()

# Chargement du dataset
df = pd.read_csv('data/kddcup.csv', header=None)

# Attribution des noms de colonnes (selon KDD)
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

print("🔧 Encodage des données texte...")
# Encodage automatique des colonnes non numériques
for col in ["protocol_type", "service", "flag"]:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# Transformation des labels en binaire : normal (0) / attaque (1)
df["label"] = df["label"].apply(lambda x: 0 if x == "normal." else 1)

print("⚙️ Normalisation des données...")
X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✂️ Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("🧠 Entraînement du modèle RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("📊 Évaluation du modèle...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("✅ Accuracy :", accuracy_score(y_test, y_pred))

# Sauvegarde du modèle
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ids_model.pkl")
print("💾 Modèle enregistré dans models/ids_model.pkl")

# Sauvegarde du modèle et du scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ids_model.pkl")
joblib.dump(scaler, "models/scaler.joblib")
print("💾 Modèle et scaler enregistrés dans le dossier models/")

