"""
Debug script to test API endpoints and see actual errors
"""
from fastapi.testclient import TestClient
from api import api_app

client = TestClient(api_app)

print("Testing POST /research endpoint...")
try:
    response = client.post(
        "/research",
        json={"ticker": "AAPL", "instructions": "Quick analysis"}
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error Response: {response.text}")
        print(f"Headers: {response.headers}")
except Exception as e:
    print(f"Exception occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
