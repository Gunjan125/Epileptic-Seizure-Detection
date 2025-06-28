# Epileptic-Seizure-Detection

This project aims to detect epileptic seizures  using various Machine Learning (ML) and Deep Learning (DL) techniques. It includes data preprocessing, model training, evaluation, and visualization.

### ✅ Dataset Used:
- **Source**:= Kaggle
- **Type**: EEG time-series data in CSV
- **Target**: Binary classification (Seizure = 1, Non-seizure = 0)

### 🔧 Preprocessing:
- Removed `Unnamed` column
- Converted multi-class labels to binary (1 = seizure, others = non-seizure)
- Standardized using `StandardScaler`
- Split data into train/test using `train_test_split`

### 📈 Models Built:
| Model              | Accuracy | Key Insights |
|--------------------|----------|--------------|
| Logistic Regression | ~95%     | Basic model, decent start |
| Random Forest       | ~98%     | Strong results, improved recall |
| XGBoost             | ~98%     | Similar to RF, more tunable |
| Deep Learning (MLP) | ~98%     | Performed equally well; potential for image inputs |

### 🔬 Evaluation Metrics Used:
- **Accuracy**
- **Precision / Recall / F1-Score**
- **Confusion Matrix**
- **Loss & Accuracy Plots (for DL)**

## 🔄 Phase 2 - Hyperparameter Tuning (Done for RF and DL)
- Grid Search CV used for Random Forest
- Manual tuning (epochs, layers, dropout) done for DL
- Improved recall and class balance

## 🧠 Deep Learning Implementation

The model was trained on preprocessed EEG data, where each sample consists of 178 extracted features per time window.

- **Model Type:** Feedforward Artificial Neural Network (ANN)
- **Architecture:**
  - Input Layer: 178 neurons
  - Hidden Layer 1: Dense (128), Activation:eLU
  - Hidden Layer 2: Dense (64), Activation:eLU
  - Output Layer: Dense (1), Activation: Sigmoid
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Training Split:** Train/Test split on preprocessed dataset
- **Epochs:** 20 (approx.)
- **Final Accuracy:** High accuracy on validation set (subject-dependent)

The model was saved as `seizure_detector1.keras` and is loaded directly into the Streamlit app for real-time prediction.

---

## 💻 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow streamlit

**## Output Screenshot**
![image](https://github.com/user-attachments/assets/cb235400-970a-41dd-b753-2246bb15e965)



