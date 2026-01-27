# Banking Churn Prediction with KServe

This project demonstrates how to deploy a machine learning model for banking customer churn prediction on a Kubernetes cluster using KServe. The model predicts whether a banking customer is likely to churn (leave the bank) based on various customer attributes.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Making Predictions](#making-predictions)
- [Troubleshooting](#troubleshooting)
- [Understanding the Components](#understanding-the-components)

---

## 🎯 Overview

**What is Churn Prediction?**
Churn prediction identifies customers who are likely to stop using a service. For banks, this helps in:
- Proactive customer retention
- Targeted marketing campaigns
- Reducing customer acquisition costs

**Technology Stack:**
- **ML Framework**: scikit-learn (Random Forest Classifier)
- **Model Serving**: KServe (Kubernetes-native model serving)
- **Storage**: Kubernetes Persistent Volume Claims (PVC)
- **Runtime**: KServe SKLearn runtime (built-in)

**Model Features:**
The model uses 10 banking customer features:
1. Credit Score (300-850)
2. Geography (France, Germany, Spain)
3. Gender (Male, Female)
4. Age (18-80)
5. Tenure (years with bank: 0-10)
6. Account Balance ($0-$250,000)
7. Number of Products (1-4)
8. Has Credit Card (Yes/No)
9. Is Active Member (Yes/No)
10. Estimated Salary ($10,000-$200,000)

---

## 📁 Project Structure

```
kserve-training/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── scripts/                          # Python scripts for data and model
│   ├── generate_data.py             # Script to generate synthetic training data
│   ├── train_model.py               # Script to train the churn prediction model
│   └── predict_client.py            # Client script to call KServe endpoint
│
├── kubernetes/                       # Kubernetes manifests
│   ├── pvc.yaml                     # Persistent Volume Claim for model storage
│   ├── model-upload-job.yaml        # Job to verify PVC (helper)
│   └── inferenceservice.yaml        # KServe InferenceService definition
│
├── data/                            # Generated data (created by generate_data.py)
│   ├── banking_churn_train.csv     # Training dataset
│   └── banking_churn_test.csv      # Test dataset
│
└── model/                           # Trained model artifacts (created by train_model.py)
    ├── model.joblib                # Trained Random Forest model
    ├── scaler.joblib               # Feature scaler
    ├── label_encoders.joblib       # Categorical encoders
    ├── feature_names.json          # Feature names list
    ├── metrics.json                # Model performance metrics
    └── metadata.json               # Model metadata for KServe
```

---

## ✅ Prerequisites

### Local Machine Requirements

1. **Python 3.8+** installed
   ```bash
   python --version
   ```

2. **pip** (Python package manager)
   ```bash
   pip --version
   ```

### Kubernetes Cluster Requirements

1. **Kubernetes cluster** (v1.22+) with kubectl access
   ```bash
   kubectl version --client
   kubectl cluster-info
   ```

2. **KServe installed** on the cluster (v0.10+)
   - KServe requires Istio or Knative for networking
   - Check if KServe is installed:
   ```bash
   kubectl get crd inferenceservices.serving.kserve.io
   ```
   - If not installed, follow: https://kserve.github.io/website/latest/admin/serverless/serverless/

3. **kubectl configured** to access your cluster
   ```bash
   kubectl get nodes
   ```

4. **Sufficient cluster resources**
   - At least 2 CPU cores available
   - At least 4GB RAM available
   - Storage provisioner for PVC

---

## 💻 Local Development

Follow these steps on your **local machine** to generate data and train the model.

### Step 1: Clone the Repository (on K8s Cluster)

```bash
# SSH into your Kubernetes cluster node or bastion host
ssh <user>@<k8s-host>

# Clone your repository directly on the cluster
git clone <your-repo-url>
cd kserve-training
```

**Why?** Since you'll be deploying to Kubernetes, cloning directly on the cluster eliminates the need to transfer files later. You can train the model locally on the cluster and deploy it immediately.

### Step 2: Create Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

**Why?** Virtual environments isolate project dependencies, preventing conflicts with other Python projects on your system.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Why?** This installs all required Python libraries:
- `pandas` & `numpy`: Data manipulation
- `scikit-learn`: Machine learning framework
- `joblib`: Model serialization
- `requests`: HTTP client for API calls

**Expected output:** You'll see packages being downloaded and installed.

### Step 4: Generate Training Data

```bash
python scripts/generate_data.py
```

**What this does:**
- Creates a `data/` directory
- Generates 10,000 synthetic banking customer records for training
- Generates 2,000 records for testing
- Saves as CSV files with realistic churn patterns

**Why?** We need data to train the model. This script creates synthetic but realistic banking customer data with features that influence churn.

**Expected output:**
```
============================================================
GENERATING TRAINING DATA
============================================================

Generating 10000 synthetic banking customer records...

✓ Training data saved to: data/banking_churn_train.csv
  Shape: (10000, 12)
  Churn rate: 20.45%
...
```

**Verify:** Check that `data/banking_churn_train.csv` and `data/banking_churn_test.csv` exist.

### Step 5: Train the Model

```bash
python scripts/train_model.py
```

**What this does:**
- Loads the training data
- Preprocesses features (encoding, scaling)
- Trains a Random Forest classifier
- Evaluates model performance
- Saves model artifacts to `model/` directory

**Why?** This creates the trained machine learning model that will be deployed to Kubernetes. The model learns patterns from historical data to predict future churn.

**Expected output:**
```
============================================================
BANKING CHURN PREDICTION MODEL TRAINING
============================================================

Loading data from: data/banking_churn_train.csv
✓ Data loaded successfully. Shape: (10000, 12)

============================================================
PREPROCESSING DATA
============================================================
...
Training set size: 8000 samples
Validation set size: 2000 samples

============================================================
TRAINING MODEL
============================================================
...
✓ Model training complete!

============================================================
MODEL EVALUATION
============================================================

Performance Metrics:
  Accuracy:  0.8542
  Precision: 0.7823
  Recall:    0.7156
  F1-Score:  0.7475
  ROC-AUC:   0.9123
...
```

**Verify:** Check that the `model/` directory contains:
- `model.joblib` (the trained model)
- `scaler.joblib` (feature scaler)
- `label_encoders.joblib` (categorical encoders)
- `feature_names.json` (feature list)
- `metrics.json` (performance metrics)
- `metadata.json` (model metadata)

**Important:** The `model/` directory will be uploaded to Kubernetes in the next section.

---

## ☸️ Kubernetes Deployment

Now we'll deploy the trained model to your Kubernetes cluster using KServe. Since you've already cloned the repo on the cluster and trained the model there, you can proceed directly with deployment.

### Step 6: Create Persistent Volume Claim (PVC)

```bash
kubectl apply -f kubernetes/pvc.yaml
```

**What this does:** Creates a 1GB persistent volume claim named `model-storage-pvc` in the default namespace.

**Why?** KServe needs to load the model from somewhere. A PVC provides persistent storage in Kubernetes where we'll store our model files. This storage persists even if pods are deleted or restarted.

**Verify:**
```bash
kubectl get pvc model-storage-pvc
```

**Expected output:**
```
NAME                 STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
model-storage-pvc    Bound    pvc-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx     1Gi        RWO            standard       10s
```

**Status should be "Bound"** - this means storage has been successfully allocated.

### Step 7: Upload Model to PVC

We need to copy the trained model files into the PVC. We'll create a temporary pod to access the PVC storage.

```bash
# Create a temporary pod with the PVC mounted
kubectl run model-uploader --image=busybox --restart=Never --overrides='
{
  "spec": {
    "containers": [{
      "name": "model-uploader",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{
        "name": "model-storage",
        "mountPath": "/mnt/models"
      }]
    }],
    "volumes": [{
      "name": "model-storage",
      "persistentVolumeClaim": {
        "claimName": "model-storage-pvc"
      }
    }]
  }
}'
```

**What this does:** Creates a temporary pod with the PVC mounted at `/mnt/models`.

**Why?** This pod acts as a bridge to copy files from your local filesystem into the Kubernetes PVC storage.

**Wait for pod to be ready:**
```bash
kubectl wait --for=condition=Ready pod/model-uploader --timeout=60s
```

**Copy model files to the pod:**
```bash
kubectl cp model/ model-uploader:/mnt/models/
```

**What this does:** Copies the entire `model/` directory (with all model artifacts) into the PVC at `/mnt/models/`.

**Why?** KServe will look for the model files in this location when the InferenceService starts.

**Verify the files were copied:**
```bash
kubectl exec model-uploader -- ls -la /mnt/models/model/
```

**Expected output:**
```
total XXX
drwxr-xr-x    2 root     root          4096 Jan 27 05:00 .
drwxr-xr-x    3 root     root          4096 Jan 27 05:00 ..
-rw-r--r--    1 root     root        XXXXX Jan 27 05:00 feature_names.json
-rw-r--r--    1 root     root        XXXXX Jan 27 05:00 label_encoders.joblib
-rw-r--r--    1 root     root        XXXXX Jan 27 05:00 metadata.json
-rw-r--r--    1 root     root        XXXXX Jan 27 05:00 metrics.json
-rw-r--r--    1 root     root        XXXXX Jan 27 05:00 model.joblib
-rw-r--r--    1 root     root        XXXXX Jan 27 05:00 scaler.joblib
```

**Clean up the temporary pod:**
```bash
kubectl delete pod model-uploader
```

**Why?** We no longer need this pod; the files are now in the PVC.

### Step 8: Deploy KServe InferenceService

```bash
kubectl apply -f kubernetes/inferenceservice.yaml
```

**What this does:** Creates a KServe InferenceService named `churn-predictor` that:
- Uses the SKLearn runtime (built-in support for scikit-learn models)
- Loads the model from `pvc://model-storage-pvc/model/`
- Allocates resources (100m CPU, 256Mi memory)
- Exposes a prediction endpoint

**Why?** The InferenceService is KServe's main resource. It automatically:
- Creates a deployment for serving the model
- Sets up autoscaling (scales to zero when idle)
- Configures networking and load balancing
- Provides a standardized prediction API

**Monitor deployment:**
```bash
kubectl get inferenceservice churn-predictor
```

**Expected output (initially):**
```
NAME              URL                                                 READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION                    AGE
churn-predictor   http://churn-predictor.default.example.com          False          100                              churn-predictor-predictor-default-00001   10s
```

**Wait for READY to become True:**
```bash
kubectl wait --for=condition=Ready inferenceservice/churn-predictor --timeout=300s
```

**Why we wait:** KServe needs time to:
1. Pull the SKLearn runtime container image
2. Mount the PVC and load the model
3. Start the prediction server
4. Run health checks

**Check detailed status:**
```bash
kubectl get inferenceservice churn-predictor -o yaml
```

**View the predictor pod:**
```bash
kubectl get pods -l serving.kserve.io/inferenceservice=churn-predictor
```

**Expected output:**
```
NAME                                                           READY   STATUS    RESTARTS   AGE
churn-predictor-predictor-default-00001-deployment-xxxxx       2/2     Running   0          2m
```

**Check logs (if needed for debugging):**
```bash
kubectl logs -l serving.kserve.io/inferenceservice=churn-predictor -c kserve-container
```

### Step 9: Get the Inference Endpoint URL

```bash
kubectl get inferenceservice churn-predictor -o jsonpath='{.status.url}'
```

**What this does:** Retrieves the external URL for the inference service.

**Expected output:**
```
http://churn-predictor.default.example.com
```

**Why?** This URL is where you'll send prediction requests. The actual URL depends on your cluster's ingress configuration.

**For local/development clusters (minikube, kind, etc.):**

If you're using a local cluster without external DNS, you may need to use port-forwarding:

```bash
# Port-forward to the inference service
kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80
```

Then use `http://localhost:8080` with the appropriate Host header.

**Test connectivity:**
```bash
# Get the service URL
SERVICE_URL=$(kubectl get inferenceservice churn-predictor -o jsonpath='{.status.url}')
echo "Service URL: $SERVICE_URL"

# Test the endpoint (should return model metadata or 404 for root path)
curl -v $SERVICE_URL
```

---

## 🔮 Making Predictions

Now that the model is deployed, let's send prediction requests!

### Step 10: Run Prediction Client

The prediction client sends sample customer data to the KServe endpoint and displays churn predictions.

**Basic usage with sample data:**
```bash
python scripts/predict_client.py --url http://churn-predictor.default.example.com/v1/models/churn-predictor:predict
```

**What this does:**
- Sends 3 sample customers to the model
- Receives churn predictions (0 = no churn, 1 = churn)
- Displays results in a readable format

**Why?** This demonstrates how to integrate the model into applications. The client shows the request format and how to interpret responses.

**Expected output:**
```
Using 3 sample customers for prediction

============================================================
CHURN PREDICTIONS
============================================================

Customer 1: CUST001
----------------------------------------
  Credit Score:      650
  Geography:         France
  Gender:            Male
  Age:               42
  Tenure:            8 years
  Balance:           $125,000.50
  Products:          2
  Has Credit Card:   Yes
  Active Member:     Yes
  Estimated Salary:  $75,000.00

  ✓ PREDICTION: Customer will STAY (No Churn)
    Risk Level: LOW

Customer 2: CUST002
----------------------------------------
  Credit Score:      450
  Geography:         Germany
  Gender:            Female
  Age:               28
  Tenure:            1 years
  Balance:           $5,000.00
  Products:          1
  Has Credit Card:   No
  Active Member:     No
  Estimated Salary:  $35,000.00

  ⚠ PREDICTION: Customer will CHURN
    Risk Level: HIGH
    Recommended Action: Engage retention strategy
...
```

**Verbose mode (see raw request/response):**
```bash
python scripts/predict_client.py --url http://churn-predictor.default.example.com/v1/models/churn-predictor:predict --verbose
```

**Why use verbose?** Helps debug issues and understand the exact API format.

### Step 11: Test with Custom Data

Create a JSON file with your own customer data:

```bash
cat > custom_customers.json << 'EOF'
[
  {
    "customer_id": "CUST999",
    "credit_score": 720,
    "geography": "France",
    "gender": "Female",
    "age": 35,
    "tenure": 5,
    "balance": 85000.00,
    "num_products": 3,
    "has_credit_card": 1,
    "is_active_member": 1,
    "estimated_salary": 95000.00
  }
]
EOF
```

**Run prediction with custom data:**
```bash
python scripts/predict_client.py --url http://churn-predictor.default.example.com/v1/models/churn-predictor:predict --input custom_customers.json
```

**Why?** This shows how to integrate the model with real customer data from your systems.

### Step 12: Direct API Testing with curl

You can also test the endpoint directly with curl:

```bash
curl -X POST \
  http://churn-predictor.default.example.com/v1/models/churn-predictor:predict \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [
      [650, "France", "Male", 42, 8, 125000.50, 2, 1, 1, 75000.00]
    ]
  }'
```

**What this does:** Sends a raw prediction request in KServe v1 format.

**Why?** Useful for testing from any environment or integrating with non-Python applications.

**Expected response:**
```json
{
  "predictions": [0]
}
```

**Note:** The instance array must have features in this exact order:
1. credit_score
2. geography
3. gender
4. age
5. tenure
6. balance
7. num_products
8. has_credit_card
9. is_active_member
10. estimated_salary

---

## 🔧 Troubleshooting

### Issue: InferenceService not becoming Ready

**Check the status:**
```bash
kubectl describe inferenceservice churn-predictor
```

**Common causes:**
1. **Model files not found in PVC**
   - Verify: `kubectl exec model-uploader -- ls /mnt/models/model/`
   - Solution: Re-upload model files (Step 8)

2. **Insufficient resources**
   - Check: `kubectl describe pod -l serving.kserve.io/inferenceservice=churn-predictor`
   - Solution: Reduce resource requests in `inferenceservice.yaml`

3. **Image pull errors**
   - Check: `kubectl get pods -l serving.kserve.io/inferenceservice=churn-predictor`
   - Solution: Ensure cluster has internet access to pull KServe images

### Issue: PVC stuck in Pending state

**Check PVC status:**
```bash
kubectl describe pvc model-storage-pvc
```

**Common causes:**
1. **No storage provisioner**
   - Solution: Install a storage provisioner or use a specific storageClassName
   - For minikube: `minikube addons enable storage-provisioner`

2. **Insufficient storage**
   - Solution: Free up space or reduce PVC size

### Issue: Cannot connect to inference endpoint

**Check if service is ready:**
```bash
kubectl get inferenceservice churn-predictor
```

**For local clusters:**
```bash
# Use port-forwarding
kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80

# Then use localhost
python scripts/predict_client.py --url http://localhost:8080/v1/models/churn-predictor:predict \
  -H "Host: churn-predictor.default.example.com"
```

### Issue: Prediction errors or wrong results

**Check model logs:**
```bash
kubectl logs -l serving.kserve.io/inferenceservice=churn-predictor -c kserve-container --tail=100
```

**Verify model files:**
```bash
# Create temporary pod again
kubectl run model-checker --image=busybox --restart=Never --overrides='...'  # (same as Step 8)

# Check files
kubectl exec model-checker -- ls -lh /mnt/models/model/

# Clean up
kubectl delete pod model-checker
```

### Issue: Model serving is slow

**Check resource usage:**
```bash
kubectl top pod -l serving.kserve.io/inferenceservice=churn-predictor
```

**Solution:** Increase resource limits in [`inferenceservice.yaml`](kubernetes/inferenceservice.yaml):
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "2"
    memory: "2Gi"
```

Then reapply:
```bash
kubectl apply -f kubernetes/inferenceservice.yaml
```

---

## 📚 Understanding the Components

### What is KServe?

KServe is a Kubernetes-native platform for model serving that provides:
- **Standardized APIs**: Consistent prediction interface across frameworks
- **Autoscaling**: Scales to zero when idle, scales up under load
- **Multi-framework support**: SKLearn, TensorFlow, PyTorch, XGBoost, etc.
- **Production features**: Canary rollouts, explainability, monitoring

### Why use the SKLearn Runtime?

The SKLearn runtime is a pre-built container that:
- Automatically loads scikit-learn models from storage
- Handles preprocessing (if saved with the model)
- Provides a standardized prediction API
- Requires no custom code

**Alternative:** You could write a custom predictor for more control over preprocessing/postprocessing.

### Understanding the Storage Path

In [`inferenceservice.yaml`](kubernetes/inferenceservice.yaml):
```yaml
storageUri: pvc://model-storage-pvc/model/
```

This tells KServe:
- Use a PVC named `model-storage-pvc`
- Look for model files in the `model/` subdirectory
- The SKLearn runtime expects `model.joblib` or `model.pkl`

### KServe Prediction Protocol (v1)

**Request format:**
```json
{
  "instances": [
    [feature1, feature2, ..., featureN],
    [feature1, feature2, ..., featureN]
  ]
}
```

**Response format:**
```json
{
  "predictions": [prediction1, prediction2, ...]
}
```

**Why this format?** It's standardized across all KServe runtimes, making it easy to swap models.

### Model Artifacts Explained

- **`model.joblib`**: The trained Random Forest classifier
- **`scaler.joblib`**: StandardScaler for numerical features
- **`label_encoders.joblib`**: LabelEncoders for categorical features
- **`feature_names.json`**: Ordered list of feature names
- **`metrics.json`**: Model performance metrics
- **`metadata.json`**: Model metadata (framework, version, etc.)

**Note:** The SKLearn runtime only requires `model.joblib`. The other files are for reference and potential custom preprocessing.

### Resource Management

**Why set resource requests/limits?**
- **Requests**: Guaranteed resources for the pod
- **Limits**: Maximum resources the pod can use
- Helps Kubernetes schedule pods efficiently
- Prevents resource starvation

**Autoscaling:**
KServe automatically scales based on:
- Request rate (requests per second)
- Concurrency (concurrent requests)
- Can scale to zero when idle (saves resources)

---

## 🎓 Next Steps

1. **Monitor the deployment:**
   ```bash
   kubectl get inferenceservice churn-predictor -w
   ```

2. **View metrics (if Prometheus is installed):**
   ```bash
   kubectl port-forward -n istio-system service/prometheus 9090:9090
   # Open http://localhost:9090
   ```

3. **Integrate with applications:**
   - Use the prediction client as a reference
   - Build REST APIs that call the KServe endpoint
   - Integrate with batch processing pipelines

4. **Improve the model:**
   - Retrain with more data
   - Try different algorithms (XGBoost, Neural Networks)
   - Add feature engineering
   - Upload new model version to PVC

5. **Production considerations:**
   - Set up monitoring and alerting
   - Implement A/B testing with multiple model versions
   - Add authentication/authorization
   - Use S3/GCS instead of PVC for model storage
   - Configure custom domains and TLS

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review KServe logs: `kubectl logs -l serving.kserve.io/inferenceservice=churn-predictor`
3. Check KServe documentation: https://kserve.github.io/website/
4. Verify your cluster meets prerequisites

---

## 📄 License

This project is for educational purposes.

---

**Happy Model Serving! 🚀**
