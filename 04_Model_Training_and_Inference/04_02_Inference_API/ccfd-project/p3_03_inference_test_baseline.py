"""
Credit Card Fraud Detection - P3_01 Baseline API Test Script (Basics)

Tests the P3_01 baseline inference API with sample transactions.

Usage:
    1. Start the API: python p3_02_inference_api_baseline.py
    2. In another terminal: python p3_03_inference_test_baseline.py
"""

import requests
import json

API_URL = "http://localhost:8000"

# Test cases
test_transactions = [
    {
        "name": "Normal Transaction",
        "data": {
            "transaction_amount": 50.00,
            "transaction_hour": 14,
            "days_since_last_txn": 1,
            "avg_transaction_amount": 75.00,
            "merchant_category": "grocery",
            "card_present": "yes",
            "international": "no",
            "transaction_count_24h": 2
        }
    },
    {
        "name": "Suspicious Transaction (High Amount)",
        "data": {
            "transaction_amount": 2500.00,
            "transaction_hour": 3,
            "days_since_last_txn": 0,
            "avg_transaction_amount": 100.00,
            "merchant_category": "electronics",
            "card_present": "no",
            "international": "yes",
            "transaction_count_24h": 7
        }
    },
    {
        "name": "Edge Case (High Value Jewelry)",
        "data": {
            "transaction_amount": 1500.00,
            "transaction_hour": 2,
            "days_since_last_txn": 0,
            "avg_transaction_amount": 120.00,
            "merchant_category": "jewelry",
            "card_present": "no",
            "international": "yes",
            "transaction_count_24h": 5
        }
    }
]


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Model Loaded: {data['model_loaded']}")
        print(f"Model Name: {data['model_name']}")
        print(f"Model Type: {data['model_type']}")
        return data['model_loaded']
    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Is it running?")
        print(f"   Start it with: python p3_02_inference_api_baseline.py")
        return False


def test_inference(transaction):
    """Test inference endpoint"""
    print(f"\n{transaction['name']}")
    print("-" * 40)

    try:
        response = requests.post(
            f"{API_URL}/inference",
            json=transaction['data']
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Probability: {result['probability']:.2%}")
            print(f"Risk Score: {result['risk_score']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Message: {result['message']}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.ConnectionError:
        print("Connection error")
        return False


def main():
    print("=" * 60)
    print("P3_01 BASELINE API TEST SUITE (Pipeline)")
    print("=" * 60)

    # Test health
    if not test_health():
        print("\nAPI not available. Exiting.")
        return

    # Test inference
    print("\n" + "=" * 60)
    print("Testing Inference Endpoint")
    print("=" * 60)

    success_count = 0
    for transaction in test_transactions:
        if test_inference(transaction):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Tests passed: {success_count}/{len(test_transactions)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
