import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json().get("data", [])
    print("Available Models on Groq:")
    print("-" * 30)
    for model in sorted(models, key=lambda x: x['id']):
        # Filter for vision models if preferred, but listing all for clarity
        print(f"ID: {model['id']}")
else:
    print(f"Error fetching models: {response.status_code}")
    print(response.text)
