import json
import base64
import logging
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery
from flask import Flask, request
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Vertex AI y BigQuery
PROJECT_ID = "cobalt-entropy-473700-i0"
VERTEX_REGION = "us-central1"  # Región para Vertex AI (Gemini disponible)
MODEL_ID = "gemini-1.5-flash"
BQ_DATASET = "minedash_data"
BQ_TABLE = "sensor_analysis"

# Inicializa Vertex AI
vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)

# Cliente BigQuery
bq_client = bigquery.Client()

@app.route('/', methods=['POST'])
def analyze_sensor_data():
    try:
        # Extraer datos de Pub/Sub
        pubsub_message = request.get_json()
        if 'message' in pubsub_message:
            data = base64.b64decode(pubsub_message['message']['data']).decode('utf-8')
        else:
            data = pubsub_message.get('data', '{}')
        sensor_data = json.loads(data)
        logger.info(f"Datos recibidos: {sensor_data}")

        # Prompt para análisis
        prompt = f"""
        Analiza los siguientes datos de un sensor minero en formato JSON. Si la temperatura (campo "value") supera los 50°C, genera una alerta de riesgo de fallo. Si no, indica que está en rango seguro. Datos: {data}

        Formato de salida:
        - Si hay anomalía: "Alerta: Temperatura alta ({{value}}°C), riesgo de fallo en sensor {{sensor_id}}."
        - Si no hay anomalía: "Temperatura en rango seguro ({{value}}°C) para sensor {{sensor_id}}."
        """

        # Llamada a Vertex AI con Gemini 1.5 Flash
        model = GenerativeModel(MODEL_ID)
        response = model.generate_content(prompt)
        prediction = response.text
        logger.info(f"Respuesta de Vertex AI: {prediction}")

        # Insertar en BigQuery
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        row = {
            "sensor_id": sensor_data.get("sensor_id", ""),
            "type": sensor_data.get("type", ""),
            "value": float(sensor_data.get("value", 0.0)),
            "timestamp": sensor_data.get("timestamp", datetime.now().isoformat()),
            "analysis_result": prediction
        }
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            logger.error(f"Error al insertar en BigQuery: {errors}")
        else:
            logger.info(f"Datos insertados en BigQuery: {row}")

        return prediction, 200
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)