# Keystroke Dynamics Authentication System

## 📋 Overview

This project implements **user authentication via keystroke dynamics** — a biometric authentication method that analyzes typing patterns to distinguish genuine users from imposters. The system uses machine learning classifiers to authenticate users based on their unique keystroke timing patterns when typing a fixed password.

## 🎯 Problem Statement

**Binary Classification Task**: Given a typing sample (31 keystroke timing features), classify it as:
- **Genuine** = the typing sample matches the authenticated user's keystroke profile
- **Imposter** = the typing sample does NOT match the user's profile

## 📁 Project Structure

```
Behavioural Authentication/
├── Data/
│   └── KeystrokeData.csv          # Main training dataset (~1,423 samples)
├── Jupyter Notebooks/
│   └── Keystroke.ipynb            # Primary analysis & model training
├── test/
│   ├── test.ipynb                 # Testing & prediction script
│   └── keystroke_predictions.csv  # Model predictions on test data
├── KeystrokeLoggingApplication.jar # Keystroke data collection tool
├── Keystrokes.csv                 # Sample keystroke data (generated during testing)
├── KeystrokesInNano.csv           # Same data in nanoseconds
└── PROJECT_ARCHITECTURE.md        # Detailed technical documentation
```

## 📊 Dataset

### [`Data/KeystrokeData.csv`](Data/KeystrokeData.csv)
- **Samples**: ~1,423 typing sessions
- **Target User**: "Atharwa"
- **Genuine Samples**: Multiple sessions from the authenticated user
- **Imposter Samples**: Sessions from other users attempting to impersonate the target user
- **Class Distribution**: Roughly balanced

### Feature Columns (31 total)
Each row contains timing metrics for typing a fixed password sequence (`.tieRianl`):

| Feature Type | Description | Example |
|---|---|---|
| **H.X** | Hold time for key X (ms) | H.period = 0.119 |
| **DD.X.Y** | Keydown-to-Keydown time (ms) | DD.period.t = 0.272 |
| **UD.X.Y** | Keyup-to-Down time (ms) | UD.period.t = 0.153 |

## 🤖 Machine Learning Models

### Implemented Algorithms

1. **K-Nearest Neighbors (KNN)**
   - `n_neighbors = 5`
   - Distance-based classification
   - Accuracy: ~99.7%

2. **Logistic Regression**
   - Linear probabilistic classifier
   - Accuracy: ~89.1%

3. **Random Forest**
   - Ensemble of decision trees
   - Accuracy: ~91.9%

4. **Extra Trees Classifier**
   - Optimized tree ensemble
   - Accuracy: ~100%

5. **Gradient Boosting**
   - Sequential ensemble boosting
   - Accuracy: 100%

6. **Neural Networks (MLP)**
   - Hidden layers: (100, 200, 330, 10)
   - Accuracy: ~98.3%

7. **Voting Classifier**
   - Ensemble of multiple models
   - Accuracy: ~98.3%

### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Reduced to 5 components
- **SVD (Singular Value Decomposition)**: Used with feature selection

## 📈 Performance Metrics

All models are evaluated using:
- **Accuracy**: Percentage of correct predictions
- **Precision/Recall**: Class-specific performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, TN, FP, FN breakdown
- **ROC Curve**: Standard for biometric systems

### Best Model Performance
- **Model**: Extra Trees Classifier / Gradient Boosting
- **Accuracy**: 100%
- **F1-Score**: 1.0

---

## 🚀 Quick Start Guide

### Step 1: Create Python Environment

Create a virtual environment to isolate project dependencies:

**On Windows (PowerShell/Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Requirements Summary:**
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **matplotlib & seaborn**: Data visualization
- **jupyter**: Interactive notebook environment

### Step 3: Run the Keystroke Logging Application

Execute the JAR file to collect keystroke data:

```bash
java -jar KeystrokeLoggingApplication.jar
```

This application will:
- Launch an interactive window for keystroke collection
- Record timing patterns when you type

### Step 4: Enter the Password

When prompted, **type the password**: `.tieRianl`

⚠️ **Important Notes:**
- Type this password exactly as specified
- Maintain your natural typing rhythm
- If you want to use a different password, you'll need to update the feature column names in the dataset later
- Multiple typing sessions from the same user will create "Genuine" samples

### Step 5: Create Your Dataset

Create your custom dataset using the JAR file:

#### A. Generate Genuine Samples
1. Run `KeystrokeLoggingApplication.jar`
2. Type the password `.tieRianl` multiple times (200-400 times recommended)
3. Save each session as "Genuine" in the dataset in Data/KeystrokeData.csv

#### B. Generate Imposter Samples
1. Have other users run the same JAR file
2. Have them type the same password `.tieRianl`
3. Save their sessions as "Imposter" in the dataset

#### C. Combine Into Training Dataset
Merge all collected data into `Data/KeystrokeData.csv` with the following structure:

```csv
User,H.period,DD.period.t,UD.period.t,H.t,DD.t.i,UD.t.i,...,H.Return,Target
YourName,0.119,0.272,0.153,0.103,0.208,0.105,...,0.112,Genuine
YourName,0.119,0.272,0.153,0.103,0.216,0.113,...,0.151,Genuine
OtherUser,0.234,0.301,0.120,0.095,0.250,0.130,...,0.098,Imposter
OtherUser,0.245,0.310,0.115,0.100,0.255,0.125,...,0.105,Imposter
```

**Columns Required:**
- `User`: Name of the person typing
- Features: H.*, DD.*, UD.* (31 timing columns from the JAR output)
- `Target`: Either "Genuine" or "Imposter"

### Step 6: Train the Model

Train the machine learning models on your dataset:

1. Navigate to `Jupyter Notebooks/Keystroke.ipynb`
2. Open the notebook
3. Run all cells sequentially (use **Shift + Enter** to execute)
4. The notebook will:
   - Load your data from `Data/KeystrokeData.csv`
   - Preprocess and scale features
   - Train multiple classification models
   - Display accuracy and performance metrics
   - Save the trained model

**Expected Output:**
- Model accuracies for each algorithm
- Confusion matrices
- Feature importance plots
- Classification reports with precision, recall, F1-scores

### Step 7: Test with New Data

Collect new keystroke samples for testing:

1. Run `KeystrokeLoggingApplication.jar` again
2. Type the password `.tieRianl` several times
3. A new `Keystrokes.csv` file will be generated in the project root directory

**Example Output:**
```
Keystrokes.csv (created automatically by the JAR)
```

### Step 8: Run the Test File

Test your trained model on the new keystroke data:

1. Navigate to `test/test.ipynb`
2. Open the notebook
3. Run all cells sequentially
4. The notebook will:
   - Load your trained model
   - Read new keystroke data from `Keystrokes.csv`
   - Make predictions (Genuine or Imposter)
   - Save results to `test/keystroke_predictions.csv`

**Output Files Generated:**

The test results are saved in **two locations**:

1. **Inside `test.ipynb` notebook**: 
   - Display predictions directly in the notebook cells
   - View confidence scores and detailed analysis

2. **In `test/keystroke_predictions.csv`** (CSV file):
   - Contains prediction results for each keystroke sample
   - Includes weightage scores in 0 to 1 range for both classes:
     - **Genuine**: Probability score (0.0 to 1.0) - likelihood of genuine user
     - **Imposter**: Probability score (0.0 to 1.0) - likelihood of imposter
   - Allows for further analysis and integration with other systems


**Interpretation:**
- `Genuine_Probability = 0.95`: 95% confidence the sample is from the genuine user
- `Imposter_Probability = 0.05`: 5% confidence the sample is from an imposter
- `Predicted_Class`: Final classification (determined by which probability is higher)

---

## 📝 Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│              KEYSTROKE DYNAMICS WORKFLOW                     │
└─────────────────────────────────────────────────────────────┘

1. CREATE ENVIRONMENT
   └─> python -m venv .venv
       .venv\Scripts\Activate.ps1

2. INSTALL DEPENDENCIES
   └─> pip install pandas numpy scikit-learn jupyter

3. COLLECT TRAINING DATA
   └─> java -jar KeystrokeLoggingApplication.jar
       Type password: .tieRianl (genuine samples)
       └─> Repeat with other users (imposter samples)

4. PREPARE DATASET
   └─> Combine all samples into Data/KeystrokeData.csv
       Add columns: User, Features (H.*, DD.*, UD.*), Target

5. TRAIN MODEL
   └─> jupyter notebook
       Run: Jupyter Notebooks/Keystroke.ipynb
       └─> Trains KNN, Logistic Regression, Random Forest, etc.
           Saves trained models

6. COLLECT TEST DATA
   └─> java -jar KeystrokeLoggingApplication.jar
       Type password: .tieRianl (new samples)
       └─> Generates Keystrokes.csv

7. TEST & PREDICT
   └─> jupyter notebook
       Run: test/test.ipynb
       └─> Makes predictions: Genuine or Imposter
           Saves to test/keystroke_predictions.csv

8. ANALYZE RESULTS
   └─> Review accuracy, confusion matrix, and predictions
```

---

## 🔧 Key Configuration

### Password
- **Default**: `.tieRianl`
- **To change**: Update feature column names in both notebooks and JAR configuration
- **Recommendation**: Use consistent password for all data collection

### Feature Extraction
- **Total Features**: 31 (Hold times + Digraph times)
- **Timing Unit**: Milliseconds (ms)
- **Collection Method**: Automatic via `KeystrokeLoggingApplication.jar`

### Train-Test Split
```python
# Used in Keystroke.ipynb
train_test_split(data, test_size=0.2, random_state=42)
# 80% for training, 20% for testing
```

### Feature Scaling
```python
# Standardizes features to mean=0, std=1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

---

## 📊 Understanding the Data

### Hold Time (H.X)
Time (in milliseconds) that key X is held down.
- Example: `H.period = 0.119` means holding "." for 0.119 ms
- Varies between users based on typing style

### Keydown-Keydown (DD.X.Y)
Time from pressing key X to pressing key Y.
- Example: `DD.period.t = 0.272` means 0.272 ms between pressing "." and "t"
- Captures inter-keystroke rhythm

### Keyup-Down (UD.X.Y)
Time from releasing key X to pressing key Y.
- Example: `UD.period.t = 0.153` means 0.153 ms between releasing "." and pressing "t"
- Fine-grained timing detail

---

## 🎯 Expected Results

### Genuine User Authentication
- **Accuracy**: 95-100%
- **False Rejection Rate (FRR)**: < 5%
- **True Positive Rate**: > 95%

### Imposter Detection
- **False Acceptance Rate (FAR)**: < 5%
- **True Negative Rate**: > 95%

### Model Performance Ranking
1. **Best**: Gradient Boosting / Extra Trees → ~100% accuracy
2. **Very Good**: KNN (k=5) → ~99.7% accuracy
3. **Good**: Neural Networks → ~98.3% accuracy
4. **Fair**: Logistic Regression → ~89.1% accuracy

---

## 🔐 Security Considerations

1. **Fixed-Text Password**: Current implementation uses a fixed password
2. **Liveness Detection**: Ensure typing speed is natural (not artificially modified)
3. **Multi-Session Training**: Collect 10+ genuine sessions for robust profile
4. **Imposter Diversity**: Include imposters with different typing speeds
5. **Feature Normalization**: Always use StandardScaler for consistency

---

## 📁 Output Files Generated

| File | Description |
|---|---|
| `Keystrokes.csv` | New keystroke data generated by JAR during testing |
| `test/keystroke_predictions.csv` | **Model predictions on test data with probability scores** (see Step 8 for details) |
| `Data/KeystrokeData.csv` | Your custom training dataset |
| Trained models (pkl/joblib) | Saved in `Jupyter Notebooks/` after training |
---

## 🛠️ Troubleshooting

### Issue: JAR file won't run
**Solution**: 
- Ensure Java is installed: `java -version`
- Check file permissions
- Try: `java -jar KeystrokeLoggingApplication.jar` (with full path if needed)

### Issue: Feature column mismatch
**Solution**:
- Ensure dataset columns match the password used
- If using different password: update feature names in notebooks
- Verify no missing or extra columns

### Issue: Low model accuracy
**Solutions**:
- Collect more genuine samples (200-300+ recommended)
- Include diverse imposters (different typing speeds)
- Check for data quality issues
- Verify correct password was used during collection
- Consider feature scaling issues

### Issue: Jupyter notebook won't start
**Solution**:
- Ensure virtual environment is activated
- Reinstall jupyter: `pip install --upgrade jupyter`
- Try: `python -m jupyter notebook`

---

## 📚 Files Reference

| File | Purpose |
|---|---|
| `Keystroke.ipynb` | Main training notebook - model development & training |
| `test.ipynb` | Testing notebook - evaluation on new keystroke samples |
| `KeystrokeData.csv` | Training dataset (modify with your collected data) |
| `keystroke_predictions.csv` | Test predictions output |
| `KeystrokeLoggingApplication.jar` | Keystroke collection tool |
| `PROJECT_ARCHITECTURE.md` | Detailed technical documentation |

---

## 🎓 Key Concepts

### Keystroke Dynamics
Biometric authentication based on unique typing patterns. Unlike passwords, keystroke dynamics are:
- **Hard to copy**: Even if someone knows the password, they type differently
- **Passive**: Verified during normal typing (no extra equipment needed)
- **Complementary**: Works alongside traditional passwords for multi-factor authentication

### Binary Classification
- **Input**: 31 keystroke timing features
- **Output**: "Genuine" or "Imposter"
- **Use Case**: Authenticate if a typing sample belongs to the claimed user

### Machine Learning Pipeline
1. **Data Collection** → JAR application captures keystroke timings
2. **Preprocessing** → Clean, validate, and format data
3. **Feature Scaling** → Standardize to mean=0, std=1
4. **Training** → Fit multiple classification models
5. **Evaluation** → Measure accuracy, precision, recall, F1-score
6. **Prediction** → Classify new keystroke samples

---

## 📧 Project Details

- **Authentication Type**: Keystroke Dynamics (Behavioral Biometric)
- **Classification**: Binary (Genuine vs. Imposter)
- **Features**: 31 keystroke timing metrics
- **Target Accuracy**: > 95%

---
