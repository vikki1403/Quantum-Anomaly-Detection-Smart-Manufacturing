# Quantum Anomaly Detection for Smart Manufacturing

A production-grade, 3-level predictive maintenance system (**Normal / Warning / Critical**) built on the **AI4I 2020 Predictive Maintenance Dataset**. The project benchmarks Classical Machine Learning (**Logistic Regression**, **Random Forest**) against Quantum Machine Learning (**Variational Quantum Classifier / VQC via Qiskit Aer**), computes a 0–100 **Machine Risk Score**, generates rule-based maintenance recommendations, and provides an interactive **Streamlit dashboard**.

---

## Key Features

1. **AI4I 2020 Dataset Ingestion & Target Engineering**:
   - Automated download and local caching of the official UCI AI4I 2020 predictive maintenance dataset (10,000 samples, 0 missing values).
   - Physics-informed 3-level target (`Fault_Level`):
     - **Normal (0)**: Machine failure = 0 and all telemetry within nominal tolerances (89.60%).
     - **Warning (1)**: Machine failure = 0 with borderline sensor conditions or early wear signals (7.01%).
     - **Critical (2)**: Machine failure = 1 (3.39%).
2. **Domain-Specific Feature Engineering**:
   - $\text{Temperature\_Difference} = \text{Process\_temperature} - \text{Air\_temperature}$ (Thermal dissipation failure indicator)
   - $\text{Power} = \text{Torque} \times \text{Rotational\_speed}$ (Drive motor load proxy)
   - $\text{Tool\_Wear\_Rate} = \frac{\text{Tool\_wear}}{\text{Torque} + 10^{-5}}$ (Cutting edge friction gradient)
   - $\text{Strain\_Index} = \text{Torque} \times \text{Tool\_wear}$ (Overstrain structural fatigue metric)
3. **Quantum-Classical Dimensionality Pipeline**:
   - Stratified 80/20 train/test split.
   - SMOTE oversampling applied strictly to training data to handle severe class imbalance.
   - PCA dimensionality reduction to 4 components (**retains 88.52% cumulative variance**), mapped directly to 4 qubits.
   - Quantum phase scaling to $[0, \pi]$ radians for angle encoding.
4. **Variational Quantum Classifier (VQC)**:
   - 4-qubit parameterized quantum circuit using `ZZFeatureMap` (2nd-order non-linear entanglement) and `RealAmplitudes` ansatz ($R_y$ rotations + linear entanglers).
   - Executed **locally on Qiskit Aer simulator** (`SamplerV2`) with classical `COBYLA` optimizer—no IBM Quantum API keys or paid services required.
5. **Machine Risk Score & Recommendations**:
   - Formula: $\text{Risk} = P(\text{Warning}) \times 50 + P(\text{Critical}) \times 100$.
   - Operational bands: **Normal (0–30)**, **Warning (31–70)**, **Critical (71–100)**.
   - Prescriptive maintenance actions tailored to active failure physics modes (TWF, HDF, PWF, OSF).
6. **Interactive Streamlit Web Dashboard**:
   - Real-time diagnostic sliders with preset scenarios (Nominal, HDF Risk, TWF Risk, PWF Risk, OSF Risk).
   - Visual risk gauge, model class probability distributions, and immediate action cards.
   - In-app EDA visualization gallery and classical vs. quantum benchmarking charts.
   - Batch CSV file uploader with one-click diagnostic scoring and downloadable CSV export.

---

## Model Benchmark Results

| Model | Architecture / Optimizer | Accuracy | F1-Score (Weighted) | F1-Score (Macro) | Test Samples |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Multinomial L-BFGS ($C=10.0$) | 84.95% | 0.8794 | 0.6467 | 2,000 |
| **Random Forest** | 100 Trees, Gini Impurity | **99.00%** | **0.9900** | **0.9378** | 2,000 |
| **Quantum VQC** | 4 Qubits, ZZFeatureMap + RealAmplitudes, COBYLA | 40.67% | 0.4086 | 0.4086 | 150 |

---

## Project Structure

```
├── requirements.txt                  # Python dependencies (Qiskit, Aer, Scikit-learn, Streamlit)
├── README.md                         # Project documentation and run guide
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv              # Downloaded raw UCI dataset
│   └── processed/
│       ├── ai4i2020_processed.csv    # Cleaned dataset with Fault_Level and engineered features
│       ├── X_train_classical.csv     # Standardized features for classical training
│       ├── X_test_classical.csv      # Standardized features for classical testing
│       ├── X_train_quantum.csv       # [0, pi] PCA features for quantum training
│       ├── X_test_quantum.csv        # [0, pi] PCA features for quantum testing
│       ├── y_train.csv               # SMOTE-balanced training labels
│       └── y_test.csv                # Holdout test labels
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis notebook
│   └── feature_engineering.ipynb     # Feature engineering & PCA angle mapping notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Step 1: Ingestion, cleaning, target engineering
│   ├── eda.py                        # Step 2: Generates high-res figures to reports/figures/
│   ├── feature_scaling_pca.py        # Step 2: SMOTE, StandardScaler, PCA, and Quantum scaler
│   ├── train_classical_models.py     # Step 3: Logistic Regression & Random Forest (GridSearchCV)
│   ├── train_vqc.py                  # Step 4: Quantum VQC via local Qiskit Aer simulator
│   ├── risk_score.py                 # Step 5: 0-100 Machine Risk Score and operational bands
│   ├── recommendations.py            # Step 5: Rule-based prescriptive maintenance engine
│   └── evaluate_compare.py           # Step 6: Multi-model benchmarking and comparison export
├── app/
│   └── dashboard.py                  # Step 7: Full interactive Streamlit web dashboard
├── models/
│   ├── scaler_standard.joblib        # Fitted StandardScaler
│   ├── pca_transformer.joblib        # Fitted 4-component PCA
│   ├── scaler_quantum.joblib         # Fitted [0, pi] MinMaxScaler
│   ├── logistic_regression.joblib    # Tuned Logistic Regression model
│   ├── random_forest.joblib          # Tuned Random Forest model
│   ├── vqc_model.joblib              # Serialized Quantum VQC model
│   └── vqc_weights.npy               # Trained variational parameter vector
└── reports/
    ├── metrics.json                  # Quantitative metrics for all models
    ├── comparison_report.csv         # Side-by-side benchmark table
    ├── pca_variance.json             # PCA cumulative variance breakdown
    └── figures/
        ├── fault_level_distribution.png
        ├── correlation_matrix.png
        ├── sensor_distributions_by_fault.png
        ├── engineered_features_analysis.png
        ├── confusion_matrices_comparison.png
        └── model_comparison_chart.png
```

---

## Installation & Setup

### 1. Clone & Install Dependencies
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline Step-by-Step

Execute each modular stage independently:

### Step 1: Data Ingestion & Target Engineering
Downloads the AI4I 2020 dataset from UCI, cleans headers, computes engineered features, and creates the 3-level `Fault_Level` target:
```bash
python src/data_preprocessing.py
```

### Step 2: Exploratory Data Analysis & Scaling
Generates publication-quality figures saved to `reports/figures/`:
```bash
python src/eda.py
```
Performs stratified 80/20 splitting, SMOTE balancing, standard scaling, and 4-component PCA quantum angle scaling:
```bash
python src/feature_scaling_pca.py
```

### Step 3: Train Classical Models
Tunes Logistic Regression and Random Forest using `GridSearchCV`:
```bash
python src/train_classical_models.py
```

### Step 4: Train Quantum Classifier (VQC)
Simulates a 4-qubit `ZZFeatureMap` + `RealAmplitudes` variational circuit on the local Qiskit Aer simulator:
```bash
python src/train_vqc.py
```

### Step 5: Evaluate & Benchmark Models
Generates comparative confusion matrices, grouped performance bar charts, and exports `reports/metrics.json` and `reports/comparison_report.csv`:
```bash
python src/evaluate_compare.py
```

---

## Launching the Streamlit Web Application

To launch the interactive dashboard:
```bash
streamlit run app/dashboard.py
```

The application will open in your browser at `http://localhost:8501`.
