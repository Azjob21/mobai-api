# Add this debug script to see the actual error
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"product_ids": [31554], "date": "2026-02-14"}
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")