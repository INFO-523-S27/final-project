# ✅ INFO 511 — Final Project  
## From Data to Decisions: A Reproducible Data Science Investigation with TidyTuesday  
**⚠️ This is an INDIVIDUAL project (not a team project).**

---

## 🎯 Project Purpose

This final project assesses your ability to:

- Work with **real, messy, open datasets**
- Implement the **entire data science lifecycle**
- Apply:
  - Regression
  - Classification
  - Clustering
- Build **reproducible machine learning pipelines**
- Write and validate **testable, modular Python code**
- Interpret results in **real-world terms**
- Reflect on **ethical and societal consequences**
- Communicate findings **professionally**

You will **select your own dataset from the TidyTuesday archive**, justify your methodological approach, and build a **fully reproducible, end-to-end data science investigation**.

---

## 📁 Repository Structure

```

.
├── data/                       # Your TidyTuesday CSV(s)
│   └── your_dataset.csv
│
├── src/
│   ├── ds.py                   # ✅ Main reproducible ML pipeline (YOU WRITE)
│   └── test_ds.py              # ✅ Unit tests for your pipeline (YOU WRITE)
│
├── proposal.qmd                # ✅ Project proposal & justification
├── index.qmd                   # ✅ Final scientific & ethical report
│
├── final_project.qmd           # ✅ Analysis, EDA, visualization & interpretation
│
├── requirements.txt            # ✅ Reproducible environment
├── .gitignore                  # ✅ Prevents junk files from being tracked
│
└── README.md                   # ✅ This file

````

⚠️ **Only modify files explicitly marked as editable.**

---

## ✅ What You Must Deliver

| Component | File | Points |
|----------|------|--------|
| Dataset Justification | `proposal.qmd` | 10 |
| Reproducible ML Pipeline | `ds.py` + `test_ds.py`  | 30 |
| Analysis & Visualization | `final_project.qmd` | 20 |
| Final Written Report | `index.qmd` | 15 |
| Final Presentation | Google Slide + Panopto | 15 |
| Peer-review + reflection + Repo Quality | GitHub Issues + repo wide | 5 |
| **TOTAL** |  | **100** |

---

## 📦 Dataset Requirements (TidyTuesday Only)

You must choose **one dataset** from:

🔗 https://github.com/rfordatascience/tidytuesday  

Your dataset must:

- Come from **one specific week**
- Include **at least one numeric outcome**
- Support **at least TWO of the following**:
  - Regression
  - Classification
  - Clustering

You may combine multiple tables **from the same TidyTuesday week only**.

---

## ✅ Required Files & Their Roles

### ✅ `proposal.qmd` — Project Design & Justification (10 pts)

Must include:

- Dataset name + TidyTuesday link
- Why you chose this dataset
- Your real-world research question(s)
- Which modeling tasks you will apply:
  - Regression (target variable)
  - Classification (class definition)
  - Clustering (expected structure)
- Potential **bias, harm, and ethical risks**

⛔ You may not begin modeling until this is approved.

---

### ✅ `src/ds.py` — Reproducible ML Pipeline (30 pts)

This is the **engineering backbone of your project**.

Your pipeline **must include all of the following categories**:

#### Data Handling
- `load_data`
- `initial_summary`
- `clean_data`
- `feature_engineering`

#### Modeling (At Least 2 Required)
- `run_regression`
- `run_classification`
- `run_clustering`

#### Evaluation
- Regression: R², RMSE  
- Classification: Accuracy, F1, ROC-AUC  
- Clustering: Silhouette, Calinski–Harabasz  

✅ Must run end-to-end from **raw CSV → metrics**  
✅ Must be **pure functions only** (no prints, no plotting)  
✅ Must be **deterministic** (fixed random seeds)  
✅ Must be **fully testable**

---

### ✅ `src/test_ds.py` — Automated Testing (5 pts)

You must implement **at least 8 real unit tests** that validate:

- Data loading
- Cleaning behavior
- Feature engineering outputs
- Train/test splits
- Model training & prediction
- Metric correctness
- Handling of edge cases:
  - NaNs
  - Small data
  - Class imbalance

✅ These tests enforce professional engineering discipline.

---

### ✅ `final_project.qmd` — Analysis & Visualization (20 pts)

Your notebook must:

1. Load data using `ds.py`
2. Perform EDA:
   - Distributions
   - Correlations
   - Missingness
3. Run:
   - At least **1 regression**
   - At least **1 classification or clustering**
4. Visualize:
   - Predictions vs truth
   - Cluster structure
   - Feature relationships
5. Interpret results in Markdown

⛔ **No model logic is allowed inside the notebook.**  
✅ All models must come from `ds.py`.

---

### ✅ `index.qmd` — Final Scientific & Ethical Report (15 pts)

Must include:

- Research questions
- Dataset background
- Modeling choices & justification
- Results & limitations
- Ethical risks & societal implications
- What you would do next with more time or data

✅ This is where your **thinking is graded**, not just your code.

---

## 🧪 Running the Project Locally

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
````

---

### 2️⃣ Run Unit Tests

```bash
pytest
```

✅ All tests must pass before submission.

---

### 3️⃣ Run Code Quality Check

```bash
pylint ds.py
```

✅ Minimum acceptable score: **9.0**

---

### 4️⃣ Render Final Analysis

```bash
quarto render final_project.qmd
```

✅ Notebook must render without error.

---

## ✅ Rules & Constraints

✅ You may use:

* pandas
* numpy
* scikit-learn
* matplotlib / seaborn
* statsmodels (optional)

❌ You may NOT:

* Use AutoML tools
* Use LLMs to generate conclusions without verification
* Copy existing TidyTuesday solutions
* Fit models directly inside the notebook
* Hardcode answers to pass tests
* Ignore ethics and interpretability

---

## ✅ Peer-review + Reflection + Professionalism (5 pts)

You will lose points for:

* Poor peer-review and obvious GenAI usage
* Poor reflection (e.g., I did everything great)
* Tracking large datasets in Git
* Poor Git history
* Missing documentation
* Non-reproducible results

---

## ✅ Submission Checklist

Before your final push:

* ✅ Proposal approved
* ✅ `ds.py` complete & clean
* ✅ `test_ds.py` complete
* ✅ `pytest` passes
* ✅ `pylint ds.py` ≥ 9.0
* ✅ `final_project.qmd` renders
* ✅ `report.md` submitted
* ✅ Dataset placed in `/data`
* ✅ Clean Git history

---

## ✅ Why This Project Is Structured This Way

This mirrors real-world data science in:

* Healthcare analytics
* Climate modeling
* Economic forecasting
* Policy research
* Responsible AI development

You are being evaluated on your ability to:

✅ Think like a scientist
✅ Build like an engineer
✅ Reflect like an ethicist
✅ Communicate like a professional analyst

---

## ✅ Academic Integrity

This is an **individual project**.

You may:

* Discuss ideas at a high level
* Debug conceptually with classmates

You may NOT:

* Share code
* Share tests
* Copy solutions
* Reuse previous TidyTuesday analyses

All work must be your own.

---

✅ **If you can successfully complete this project, you are demonstrating real industry-ready applied data science skills.**
