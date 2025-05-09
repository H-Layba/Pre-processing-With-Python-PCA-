# HR Attrition Prediction Project (With and Without PCA)

This project involves building and comparing two machine learning pipelines to predict employee attrition. One pipeline applies **standard preprocessing only**, while the other includes **Principal Component Analysis (PCA)** for dimensionality reduction. The goal is to evaluate how PCA affects model performance, data handling, and feature interpretability.

---

## 📁 Dataset

* **File:** `HR-Employee-Attrition_with_missing_values.csv`
* **Target Variable:** `Attrition` (`Yes` = 1, `No` = 0)
* **Features:** Employee demographics, job role, travel frequency, department, etc.
* **Missing Values:** Present in multiple columns
* **Outliers:** Present in numerical columns

---

## 🧪 Objectives

1. Handle missing values using **random imputation**.
2. Detect and cap outliers using **Interquartile Range (IQR)**.
3. Apply encoding for nominal and ordinal features.
4. Train a **Random Forest Classifier**.
5. Evaluate model **accuracy**.
6. Compare performance with and without **PCA (2 components)**.

---

## ⚙️ Preprocessing Techniques

### 1. **Random Imputer**

A custom transformer to impute missing values by randomly sampling from non-missing values in the same column.

### 2. **Outlier Capper**

A custom transformer that caps outliers beyond 1.5×IQR range.

### 3. **Column-wise Pipelines**

* **Numerical:** Random imputation → Outlier capping → Standard scaling
* **Nominal:** Constant fill → One-hot encoding
* **Ordinal:** Constant fill → Ordinal encoding

---

## 🔄 Pipelines

### 🛠️ Without PCA

* Full pipeline: `Preprocessing → RandomForestClassifier`
* Outputs:

  * Final feature matrix
  * Accuracy score
  * Count of outliers before/after
  * Missing values before/after

### 📉 With PCA

* Full pipeline: `Preprocessing → PCA (2 components) → RandomForestClassifier`
* Outputs:

  * Transformed data in 2D PCA space
  * Accuracy score
  * Count of outliers before/after
  * Visualization-ready principal components

---

## 📊 Evaluation Metrics

* **Accuracy Score**
* **Outlier Count** (Before and After preprocessing)
* **Missing Value Check**

---

## ✅ Results Snapshot

| Model Type  | PCA Used | Accuracy | Notes                               |
| ----------- | -------- | -------- | ----------------------------------- |
| Without PCA | ❌        | \~0.8x   | Preserves original features         |
| With PCA    | ✅        | \~0.7x   | Dimension reduction to 2 components |

> Exact accuracy may vary depending on random seed and data shuffling.

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🗃️ File Structure

```
project/
├── HR-Employee-Attrition_with_missing_values.csv
├── pipeline_without_pca.py
├── pipeline_with_pca.py
├── README.md
```

---

## 📈 Sample Output (Without PCA)

```
Missing Values Before Preprocessing:
...
Outliers Before Preprocessing:
Age: 10, MonthlyIncome: 22, ...

Missing Values After Preprocessing:
...
Outliers After Preprocessing:
All capped to bounds.

Accuracy: 0.85
```

---

## 💡 Conclusion

* PCA may help in reducing dimensionality but might slightly compromise interpretability and performance for some models.
* Custom transformers for imputation and outlier handling improve robustness.
* Modular pipelines enable easy experimentation and reproducibility.

---
