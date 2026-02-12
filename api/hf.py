import requests
import os
import json

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method Not Allowed"})
        }

    try:
        body = json.loads(request.body)
        user_message = body.get("message", "")

        HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
        HF_API_URL = "https://api.router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": user_message,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            reply = result[0]["generated_text"]
        else:
            reply = "Unexpected response from Hugging Face."

        return {
            "statusCode": 200,
            "body": json.dumps({"response": reply})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
