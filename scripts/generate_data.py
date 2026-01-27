"""
Banking Churn Data Generator
This script generates synthetic banking customer data for churn prediction modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_banking_churn_data(n_samples=10000):
    """
    Generate synthetic banking customer churn data.
    
    Features:
    - customer_id: Unique identifier
    - credit_score: Credit score (300-850)
    - age: Customer age (18-80)
    - tenure: Number of years with the bank (0-10)
    - balance: Account balance (0-250000)
    - num_products: Number of bank products (1-4)
    - has_credit_card: Whether customer has credit card (0/1)
    - is_active_member: Whether customer is active (0/1)
    - estimated_salary: Annual salary (10000-200000)
    - geography: Country (France, Germany, Spain)
    - gender: Gender (Male, Female)
    - churn: Target variable (0=stayed, 1=churned)
    """
    
    print(f"Generating {n_samples} synthetic banking customer records...")
    
    # Generate customer IDs
    customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    
    # Generate features
    credit_scores = np.random.randint(300, 851, n_samples)
    ages = np.random.randint(18, 81, n_samples)
    tenures = np.random.randint(0, 11, n_samples)
    balances = np.random.uniform(0, 250000, n_samples).round(2)
    num_products = np.random.randint(1, 5, n_samples)
    has_credit_card = np.random.binomial(1, 0.7, n_samples)
    is_active_member = np.random.binomial(1, 0.5, n_samples)
    estimated_salaries = np.random.uniform(10000, 200000, n_samples).round(2)
    geographies = np.random.choice(['France', 'Germany', 'Spain'], n_samples, p=[0.5, 0.25, 0.25])
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    
    # Generate churn with realistic patterns
    # Higher churn probability for:
    # - Lower credit scores
    # - Fewer products
    # - Inactive members
    # - Very low or very high balance
    # - Shorter tenure
    
    churn_prob = np.zeros(n_samples)
    
    # Base probability
    churn_prob += 0.2
    
    # Credit score impact
    churn_prob += (850 - credit_scores) / 5500
    
    # Age impact (younger and older customers more likely to churn)
    age_factor = np.abs(ages - 45) / 100
    churn_prob += age_factor
    
    # Tenure impact (newer customers more likely to churn)
    churn_prob += (10 - tenures) / 50
    
    # Balance impact (very low or very high balance increases churn)
    balance_normalized = balances / 250000
    balance_factor = np.abs(balance_normalized - 0.5) / 2
    churn_prob += balance_factor
    
    # Number of products (fewer products = higher churn)
    churn_prob += (5 - num_products) / 20
    
    # Active member impact (inactive = higher churn)
    churn_prob += (1 - is_active_member) * 0.3
    
    # Credit card impact (no credit card = slightly higher churn)
    churn_prob += (1 - has_credit_card) * 0.1
    
    # Geography impact
    geography_factor = np.where(geographies == 'Germany', 0.15, 
                                np.where(geographies == 'France', 0.05, 0.1))
    churn_prob += geography_factor
    
    # Clip probabilities to valid range
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate actual churn based on probabilities
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'credit_score': credit_scores,
        'geography': geographies,
        'gender': genders,
        'age': ages,
        'tenure': tenures,
        'balance': balances,
        'num_products': num_products,
        'has_credit_card': has_credit_card,
        'is_active_member': is_active_member,
        'estimated_salary': estimated_salaries,
        'churn': churn
    })
    
    return df

def main():
    """Generate and save training and test datasets."""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate training data (10,000 samples)
    print("\n" + "="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    train_data = generate_banking_churn_data(n_samples=10000)
    
    # Save training data
    train_path = 'data/banking_churn_train.csv'
    train_data.to_csv(train_path, index=False)
    print(f"\n✓ Training data saved to: {train_path}")
    print(f"  Shape: {train_data.shape}")
    print(f"  Churn rate: {train_data['churn'].mean():.2%}")
    
    # Generate test data (2,000 samples)
    print("\n" + "="*60)
    print("GENERATING TEST DATA")
    print("="*60)
    np.random.seed(123)  # Different seed for test data
    test_data = generate_banking_churn_data(n_samples=2000)
    
    # Save test data
    test_path = 'data/banking_churn_test.csv'
    test_data.to_csv(test_path, index=False)
    print(f"\n✓ Test data saved to: {test_path}")
    print(f"  Shape: {test_data.shape}")
    print(f"  Churn rate: {test_data['churn'].mean():.2%}")
    
    # Display sample records
    print("\n" + "="*60)
    print("SAMPLE RECORDS (First 5 from training data)")
    print("="*60)
    print(train_data.head())
    
    # Display feature statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(train_data.describe())
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated data in the 'data/' directory")
    print("2. Run 'python scripts/train_model.py' to train the churn prediction model")

if __name__ == "__main__":
    main()
