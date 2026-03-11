# ML-Insurance-Cost-Predictor

# 🏥 ML Insurance Cost Predictor

> Predicting medical insurance charges using machine learning — from raw data to a clean, feature-engineered model pipeline.

---

## 📌 Overview

This project builds a machine learning pipeline to predict **individual medical insurance costs** based on personal attributes such as age, BMI, smoking status, and region. The notebook walks through every stage of a real-world ML workflow — from exploratory data analysis all the way to a statistically validated feature set ready for modeling.

---

## 📂 Dataset

The project uses the classic **Medical Cost Personal Dataset** (`insurance.csv`), which contains the following features:

| Feature | Description |
|---|---|
| `age` | Age of the primary beneficiary |
| `sex` | Gender of the insured (male / female) |
| `bmi` | Body Mass Index |
| `children` | Number of dependents covered |
| `smoker` | Whether the person smokes (yes / no) |
| `region` | Residential region in the US (NE, NW, SE, SW) |
| `charges` | 🎯 **Target** — Individual medical costs billed by insurance |

---

## 🔍 Project Pipeline

### 1. 📊 Exploratory Data Analysis (EDA)
- Distribution plots for all numeric features (`age`, `bmi`, `children`, `charges`)
- Count plots for categorical features (`sex`, `smoker`, `children`)
- Box plots to identify outliers
- Correlation heatmap to understand feature relationships

### 2. 🧹 Data Cleaning & Preprocessing
- Removed duplicate records
- Encoded binary categorical features:
  - `sex` → `is_female` (0 / 1)
  - `smoker` → `is_smoker` (0 / 1)
- One-hot encoded the `region` column (with `drop_first=True` to avoid multicollinearity)

### 3. ⚙️ Feature Engineering
- Created a `bmi_category` column using WHO BMI thresholds:
  - Underweight (`< 18.5`), Normal (`18.5–24.9`), Overweight (`25–29.9`), Obese (`≥ 30`)
- One-hot encoded BMI categories
- Applied **StandardScaler** to normalize continuous features: `age`, `bmi`, `children`

### 4. 📐 Feature Selection
- **Pearson Correlation** — measured linear relationships between numeric features and `charges`
- **Chi-Square Test** — evaluated statistical significance of categorical features against binned charge quartiles
- Final selected features: `age`, `bmi`, `children`, `is_female`, `is_smoker`, `region_southeast`, `bmi_category_Obese`

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn seaborn matplotlib scipy
```

### Run the Notebook
```bash
git clone https://github.com/your-username/ML_Insurance_Cost_Predictor.git
cd ML_Insurance_Cost_Predictor
jupyter notebook ML_Insurance_Cost_Predictor.ipynb
```

> 📝 **Note:** Update the dataset path in the second cell to point to your local `insurance.csv` file.

---

## 📁 Project Structure

```
ML_Insurance_Cost_Predictor/
│
├── ML_Insurance_Cost_Predictor.ipynb   # Main notebook
├── insurance.csv                        # Dataset
└── README.md                            # Project documentation
```

---

## 📈 Key Insights

- 🚬 **Smoking** is the strongest predictor of high insurance charges
- 🍔 **Obesity (BMI ≥ 30)** significantly increases predicted costs
- 📅 **Age** shows a strong positive correlation with charges
- 🌍 **Region** has a relatively minor but statistically significant effect

---

## 🙋‍♂️ Author

**Bijoy** — Built as part of the AIM-2025S Machine Learning curriculum.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE) .
