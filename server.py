# server.py — Version Railway (cloud)
# Le thread série est supprimé
# Les données arrivent via POST /data depuis le PC/Raspberry Pi local
#
# Architecture :
# Arduino → USB → PC local (serial_reader.py) → POST /data → Railway → App Flutter

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import json
import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─── Chargement du modèle ML ───────────────────────────────────────────────────
print("Chargement du modèle ML...")
with open('lora_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print(f"Modèle : {model_info['best_model']}")
print(f"MAE    : {model_info['mae']} m")
print(f"R²     : {model_info['r2']}")

# ─── Stockage dernière mesure ──────────────────────────────────────────────────
latest_data = {
    "rssi":        0.0,
    "snr":         0.0,
    "temperature": 0.0,
    "humidity":    0.0,
    "obstacle":    0,
    "distance":    0.0,
    "status":      "normal",
    "timestamp":   datetime.now().isoformat(),
    "connected":   False,
}

ALERT_THRESHOLD = 30.0  # seuil en mètres

# ─── Helpers ───────────────────────────────────────────────────────────────────
def predict_distance(rssi, snr, temperature, humidity, obstacle):
    features = pd.DataFrame(
        [[rssi, snr, temperature, humidity, obstacle]],
        columns=['rssi', 'snr', 'temperature', 'humidity', 'obstacle']
    )
    log_distance = model.predict(features)[0]
    distance = np.exp(log_distance)

    # Clamp physique basé sur RSSI
    if rssi >= -65:
        distance = min(distance, 20.0)
    elif rssi >= -75:
        distance = min(distance, 50.0)
    elif rssi >= -85:
        distance = min(distance, 100.0)

    return round(float(distance), 4)

def compute_status(distance):
    return "danger" if distance > ALERT_THRESHOLD else "normal"

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "app":     "LoRa Track API",
        "status":  "running",
        "version": "1.0.0"
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "server":    "LORA TRACK API",
        "status":    "running",
        "model":     model_info['best_model'],
        "mae":       model_info['mae'],
        "r2":        model_info['r2'],
        "threshold": ALERT_THRESHOLD,
        "connected": latest_data["connected"],
        "timestamp": datetime.now().isoformat(),
    })

@app.route('/latest', methods=['GET'])
def get_latest():
    return jsonify(latest_data)

@app.route('/data', methods=['POST'])
def receive_data():
    """
    Reçoit les données depuis le PC/Raspberry Pi local
    Body JSON : { rssi, snr, temperature, humidity, obstacle }
    """
    global latest_data
    body = request.get_json()
    if not body:
        return jsonify({"error": "Body JSON manquant"}), 400
    try:
        rssi        = float(body['rssi'])
        snr         = float(body['snr'])
        temperature = float(body['temperature'])
        humidity    = float(body['humidity'])
        obstacle    = int(body.get('obstacle', 0))

        distance = predict_distance(rssi, snr, temperature, humidity, obstacle)
        status   = compute_status(distance)

        latest_data = {
            "rssi":        rssi,
            "snr":         snr,
            "temperature": temperature,
            "humidity":    humidity,
            "obstacle":    obstacle,
            "distance":    distance,
            "status":      status,
            "timestamp":   datetime.now().isoformat(),
            "connected":   True,
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"RSSI:{rssi} T:{temperature}°C H:{humidity}% "
              f"→ {distance}m ({status.upper()})")

        return jsonify({"success": True, "distance": distance, "status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    if not body:
        return jsonify({"error": "Body JSON manquant"}), 400
    try:
        distance = predict_distance(
            float(body.get('rssi', 0)),
            float(body.get('snr', 0)),
            float(body.get('temperature', 0)),
            float(body.get('humidity', 0)),
            int(body.get('obstacle', 0)),
        )
        return jsonify({
            "distance":  distance,
            "status":    compute_status(distance),
            "threshold": ALERT_THRESHOLD,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/threshold', methods=['GET'])
def get_threshold():
    return jsonify({"threshold": ALERT_THRESHOLD})

@app.route('/threshold', methods=['PUT'])
def set_threshold():
    global ALERT_THRESHOLD
    body = request.get_json()
    if not body or 'threshold' not in body:
        return jsonify({"error": "Champ 'threshold' manquant"}), 400
    ALERT_THRESHOLD = float(body['threshold'])
    print(f"[SETTINGS] Seuil mis à jour : {ALERT_THRESHOLD} m")
    return jsonify({"threshold": ALERT_THRESHOLD, "success": True})

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print("   LORA TRACK — Serveur Flask (Railway)")
    print(f"{'='*50}")
    print(f"   Port         : {port}")
    print(f"   Seuil alerte : {ALERT_THRESHOLD} m")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=False)