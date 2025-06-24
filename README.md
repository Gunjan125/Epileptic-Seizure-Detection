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

## 🧭 Upcoming Phases:

### Phase 3 - CNN with EEG Images
- Use real EEG spectrogram/image data
- Train CNN to detect seizures from images

### Phase 4 - Streamlit App
- Upload EEG data (CSV or image)
- Predict seizure activity
- Show results with charts

---

## 💻 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow streamlit

