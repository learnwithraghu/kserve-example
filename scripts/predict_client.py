"""
KServe Churn Prediction Client
This script sends prediction requests to the deployed KServe inference service.
"""

import requests
import json
import argparse
import sys
from typing import Dict, List, Any

# Sample customer data for testing
SAMPLE_CUSTOMERS = [
    {
        "customer_id": "CUST001",
        "credit_score": 650,
        "geography": "France",
        "gender": "Male",
        "age": 42,
        "tenure": 8,
        "balance": 125000.50,
        "num_products": 2,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 75000.00
    },
    {
        "customer_id": "CUST002",
        "credit_score": 450,
        "geography": "Germany",
        "gender": "Female",
        "age": 28,
        "tenure": 1,
        "balance": 5000.00,
        "num_products": 1,
        "has_credit_card": 0,
        "is_active_member": 0,
        "estimated_salary": 35000.00
    },
    {
        "customer_id": "CUST003",
        "credit_score": 800,
        "geography": "Spain",
        "gender": "Male",
        "age": 55,
        "tenure": 10,
        "balance": 200000.00,
        "num_products": 4,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 150000.00
    }
]

def prepare_inference_request(customers: List[Dict[str, Any]]) -> Dict:
    """
    Prepare the inference request payload in KServe v1 format.
    
    Args:
        customers: List of customer dictionaries
        
    Returns:
        Dictionary in KServe v1 inference protocol format
    """
    # Extract features in the correct order (excluding customer_id)
    feature_order = [
        'credit_score', 'geography', 'gender', 'age', 'tenure',
        'balance', 'num_products', 'has_credit_card', 
        'is_active_member', 'estimated_salary'
    ]
    
    instances = []
    for customer in customers:
        instance = [customer[feature] for feature in feature_order]
        instances.append(instance)
    
    return {"instances": instances}

def send_prediction_request(url: str, customers: List[Dict[str, Any]], verbose: bool = False) -> Dict:
    """
    Send prediction request to KServe endpoint.
    
    Args:
        url: KServe inference endpoint URL
        customers: List of customer dictionaries
        verbose: Whether to print detailed request/response info
        
    Returns:
        Prediction response dictionary
    """
    # Prepare request payload
    payload = prepare_inference_request(customers)
    
    if verbose:
        print("\n" + "="*60)
        print("REQUEST PAYLOAD")
        print("="*60)
        print(json.dumps(payload, indent=2))
    
    # Set headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Send POST request
        if verbose:
            print("\n" + "="*60)
            print(f"SENDING REQUEST TO: {url}")
            print("="*60)
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if verbose:
            print("\n" + "="*60)
            print("RAW RESPONSE")
            print("="*60)
            print(json.dumps(result, indent=2))
        
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to {url}")
        print("Please ensure:")
        print("  1. The KServe InferenceService is deployed and ready")
        print("  2. The URL is correct")
        print("  3. You have network access to the cluster")
        sys.exit(1)
        
    except requests.exceptions.Timeout:
        print(f"\n❌ Error: Request timed out after 30 seconds")
        print("The model server might be overloaded or not responding")
        sys.exit(1)
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"Response: {response.text}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

def display_predictions(customers: List[Dict[str, Any]], predictions: Dict):
    """
    Display predictions in a user-friendly format.
    
    Args:
        customers: List of customer dictionaries
        predictions: Prediction response from KServe
    """
    print("\n" + "="*60)
    print("CHURN PREDICTIONS")
    print("="*60)
    
    # Extract predictions from response
    pred_values = predictions.get('predictions', [])
    
    for i, (customer, pred) in enumerate(zip(customers, pred_values)):
        print(f"\nCustomer {i+1}: {customer['customer_id']}")
        print("-" * 40)
        print(f"  Credit Score:      {customer['credit_score']}")
        print(f"  Geography:         {customer['geography']}")
        print(f"  Gender:            {customer['gender']}")
        print(f"  Age:               {customer['age']}")
        print(f"  Tenure:            {customer['tenure']} years")
        print(f"  Balance:           ${customer['balance']:,.2f}")
        print(f"  Products:          {customer['num_products']}")
        print(f"  Has Credit Card:   {'Yes' if customer['has_credit_card'] else 'No'}")
        print(f"  Active Member:     {'Yes' if customer['is_active_member'] else 'No'}")
        print(f"  Estimated Salary:  ${customer['estimated_salary']:,.2f}")
        print()
        
        # Interpret prediction
        churn_prediction = pred
        if churn_prediction == 0:
            print(f"  ✓ PREDICTION: Customer will STAY (No Churn)")
            print(f"    Risk Level: LOW")
        else:
            print(f"  ⚠ PREDICTION: Customer will CHURN")
            print(f"    Risk Level: HIGH")
            print(f"    Recommended Action: Engage retention strategy")

def main():
    """Main function to run the prediction client."""
    parser = argparse.ArgumentParser(
        description='Send prediction requests to KServe churn prediction service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict using sample data
  python scripts/predict_client.py --url http://churn-predictor.default.example.com/v1/models/churn-predictor:predict
  
  # Predict with verbose output
  python scripts/predict_client.py --url http://churn-predictor.default.example.com/v1/models/churn-predictor:predict --verbose
  
  # Predict using custom JSON file
  python scripts/predict_client.py --url http://churn-predictor.default.example.com/v1/models/churn-predictor:predict --input customers.json
        """
    )
    
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='KServe inference endpoint URL (e.g., http://churn-predictor.default.example.com/v1/models/churn-predictor:predict)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to JSON file containing customer data (optional, uses sample data if not provided)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed request and response information'
    )
    
    args = parser.parse_args()
    
    # Load customer data
    if args.input:
        try:
            with open(args.input, 'r') as f:
                customers = json.load(f)
            print(f"✓ Loaded {len(customers)} customers from {args.input}")
        except FileNotFoundError:
            print(f"❌ Error: File '{args.input}' not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in file '{args.input}'")
            sys.exit(1)
    else:
        customers = SAMPLE_CUSTOMERS
        print(f"Using {len(customers)} sample customers for prediction")
    
    # Send prediction request
    predictions = send_prediction_request(args.url, customers, args.verbose)
    
    # Display results
    display_predictions(customers, predictions)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
