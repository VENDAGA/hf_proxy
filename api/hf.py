from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = "https://api.router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route("/hf", methods=["POST"])
def hf_proxy():
    data = request.json
    user_message = data.get("message", "")

    payload = {
        "inputs": user_message,
        "parameters": {"max_new_tokens": 250, "temperature": 0.7, "return_full_text": False}
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return jsonify({"response": result[0]["generated_text"]})
        return jsonify({"response": "HF returned unexpected format."})
    except Exception as e:
        return jsonify({"response": f"HF error: {e}"})
