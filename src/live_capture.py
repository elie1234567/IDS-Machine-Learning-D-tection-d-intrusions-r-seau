# src/live_capture.py
import os
import joblib
import pandas as pd
from scapy.all import sniff
import csv
from datetime import datetime

print("IDS-ML en mode surveillance...")

# Vérification des fichiers modèle/scaler
if not os.path.exists('models/ids_model.pkl') or not os.path.exists('models/scaler.joblib'):
    print("❌ Modèle ou scaler manquant. Lance d'abord train.py")
    exit()

# Chargement du modèle et du scaler
clf = joblib.load('models/ids_model.pkl')
scaler = joblib.load('models/scaler.joblib')

# Colonnes du dataset KDD (sans la colonne label)
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# ----- ALERT LOGGING SETUP -----
ALERT_CSV = "demo/alerts.csv"
os.makedirs("demo", exist_ok=True)
# créer l'en-tête si le fichier n'existe pas
if not os.path.exists(ALERT_CSV):
    with open(ALERT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "prediction", "packet_len", "protocol", "note"])

# compteur d'attaques (optionnel, utile pour la démo)
attack_count = 0

# Fonction d'analyse d'un paquet
def analyze_packet(packet):
    global attack_count

    # Crée un DataFrame avec des valeurs par défaut pour toutes les colonnes
    data = {col: [0] for col in columns}

    # Exemple : remplir certaines colonnes avec des valeurs issues du paquet
    data["duration"] = [0]
    # protocole simple (peut être 'IP'/'TCP' selon paquet)
    proto_name = packet.payload.name if hasattr(packet.payload, "name") else "tcp"
    data["protocol_type"] = [proto_name]
    data["service"] = ["http"]
    data["flag"] = ["SF"]
    data["src_bytes"] = [len(packet)]
    data["dst_bytes"] = [0]

    df = pd.DataFrame(data)

    # Encodage des colonnes texte
    for col in ["protocol_type", "service", "flag"]:
        df[col] = df[col].astype('category').cat.codes

    # Normalisation
    X_scaled = scaler.transform(df)

    # Prédiction
    pred = clf.predict(X_scaled)  # 0 = normal, 1 = attaque

    # ------ Logging / affichage ------
    label = "attaque" if pred[0] == 1 else "normal"

    # Pour démo propre : n'affiche que les attaques (décommente la ligne suivante pour tout afficher)
    if pred[0] == 1:
        attack_count += 1
        print(f"🚨 ALERT #{attack_count}: attaque détectée — packet_len={len(packet)} proto={proto_name}")

    # Toujours enregistrer les alertes dans demo/alerts.csv (on peut aussi enregistrer tout)
    if label == "attaque":
        with open(ALERT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                label,
                len(packet),
                proto_name,
                "live"
            ])

# Sniffer le réseau (tu peux ajouter iface="wlan0" ou "eth0" si besoin)
# Utilise sudo -E si nécessaire ou donne les capabilities au python binaire
sniff(prn=analyze_packet, store=False)
