# 🫀 Indian Liver Patient Disease Prediction

A machine learning project to classify liver patients using the **Indian Liver Patient Dataset (ILPD)**. This notebook trains and compares 10 classifiers, handles class imbalance with SMOTE and class weighting, and includes full visualizations and sample predictions.

---

## 📁 Project Structure

```
liver-shiva.ipynb       # Main notebook
visualizations/         # Auto-generated plots (created at runtime)
models/                 # Saved model .pkl files (created at runtime)
```

---

## 📊 Dataset

**Indian Liver Patient Dataset (ILPD)**  
Source: [Kaggle – sharadhiv/indian-liver-patient-dataset-ilpd](https://www.kaggle.com/datasets/sharadhiv/indian-liver-patient-dataset-ilpd)

| Feature | Description |
|---|---|
| `age` | Patient age |
| `gender` | Gender (Female=0, Male=1 after encoding) |
| `tot_bilirubin` | Total bilirubin |
| `direct_bilirubin` | Direct bilirubin |
| `tot_proteins` | Total proteins |
| `albumin` | Albumin |
| `ag_ratio` | Albumin/Globulin ratio |
| `sgpt` | SGPT (liver enzyme) |
| `sgot` | SGOT (liver enzyme) |
| `alkphos` | Alkaline phosphotase |
| `is_patient` | **Target** — 1 = Liver Patient, 0 = Healthy |

---

## ⚙️ Pipeline

1. **Load & Clean** — handle duplicates, impute 4 missing `alkphos` values with median
2. **Encode & Preprocess** — label-encode `gender`, remap target to binary (1/0)
3. **Scale** — MinMaxScaler (saved as `scaler.pkl`)
4. **Split** — 80/20 stratified train/test split
5. **SMOTE** — applied to training data only for tree/kernel models
6. **Train 10 models** — see below
7. **Evaluate** — accuracy, classification report, ROC-AUC, confusion matrix
8. **Visualize** — 15 saved plots
9. **Sample Predictions** — 5 test cases with majority vote and probability heatmap

---

## 🤖 Models Trained

| Model | Imbalance Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| SVM (Linear) | `class_weight='balanced'` |
| SVM (RBF) | SMOTE |
| SVM (Poly) | SMOTE |
| KNN | SMOTE |
| Decision Tree | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| AdaBoost | SMOTE |
| Gradient Boosting | SMOTE |
| XGBoost | `scale_pos_weight` |

---

## 📈 Visualizations Generated

| File | Description |
|---|---|
| `01_class_distribution.png` | Bar chart of target class counts |
| `02_feature_distributions.png` | Histograms of all numeric features |
| `03_boxplots_by_class.png` | Boxplots: Healthy vs Patient per feature |
| `04_correlation_heatmap.png` | Feature correlation matrix |
| `05_gender_by_class.png` | Gender split by class |
| `06_age_distribution_by_class.png` | Age distribution: Healthy vs Patient |
| `07_*_confusion_matrix.png` | Confusion matrix per model |
| `08_roc_curves.png` | ROC curves for all models |
| `09_model_comparison.png` | Train vs Test accuracy comparison bar chart |
| `10_feature_importances_rf.png` | Random Forest feature importances |
| `11_pca_explained_variance.png` | PCA explained variance per component |
| `12_kfold_accuracy.png` | KFold cross-validation accuracy per fold (SVM) |
| `13_bootstrap_accuracy.png` | Bootstrap accuracy distribution (SVM) |
| `15_sample_prediction_heatmap.png` | Probability heatmap across 5 samples × 8 models |

---

## 🧪 Sample Test Cases

5 hand-crafted patients (varying age, gender, enzyme levels) run through all models:

| Sample | Description |
|---|---|
| S1 | Older male, elevated bilirubin & enzymes — likely Patient |
| S2 | Young female, normal values — likely Healthy |
| S3 | Middle-aged male, borderline values — uncertain |
| S4 | Young female, low-normal values — likely Healthy |
| S5 | Elderly male, severely elevated values — likely Patient |

Majority vote and per-model probability scores are reported for each.

---

## 🚀 How to Run

### On Kaggle
1. Upload the notebook to Kaggle
2. Add the dataset: `sharadhiv/indian-liver-patient-dataset-ilpd`
3. Enable GPU/CPU accelerator and run all cells

### Locally
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
jupyter notebook liver-shiva.ipynb
```

Update the dataset path in the **Load Dataset** cell:
```python
df = pd.read_csv('path/to/Indian Liver Patient Dataset (ILPD).csv')
```

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
joblib
```

---

## 📝 Notes

- Models are saved as `.pkl` files under `/kaggle/working/models/`
- The scaler is also saved (`scaler.pkl`) for use in inference
- SMOTE is applied **only to training data** — test set remains untouched
- XGBoost uses `scale_pos_weight = count(negative) / count(positive)` instead of SMOTE
