# Risk-assessment
This project uses ML to predict the likelihood of credit default within two years using financial and demographic features. It includes data preprocessing, LightGBM model training, risk probability prediction, and exploratory data analysis to profile borrowers into low, medium, and high risk categories.

---

## Dataset

The dataset contains anonymized financial and demographic information of individuals, with the following key features:
- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfOpenCreditLinesAndLoans`
- and more.

Target variable: `SeriousDlqin2yrs` (1 = default, 0 = no default)

---

## Project Workflow

### 1. Data Preprocessing
- Missing values handled using row-wise deletion (`MonthlyIncome`, `NumberOfDependents`)
- Unnecessary columns dropped (`Unnamed: 0`)

### 2. Model Training
- Train-test split on `df_train` (80/20, stratified)
- Models trained:
  -  LightGBM Classifier
  -  Random Forest Classifier
- Performance metric: **ROC AUC Score**

### 3. Risk Categorization
Based on model prediction probabilities, individuals are classified into:
- **Low Risk**: < 5%
- **Medium Risk**: 5%–10%
- **High Risk**: > 10%

### 4. EDA
- Distribution of predicted risk levels
- Risk level trends by:
  - **Age**
  - **Monthly Income**
  - **Age Group Bins** (e.g., 18–24, 25–34, etc.)

---

## Sample Visualizations

- ROC Curves comparing LightGBM vs Random Forest
- Boxplots of Age risk groups
- Countplot of Risk Level by Age Group

---

## Results

| Model              | Validation ROC AUC |
|-------------------|--------------------|
| LightGBM          | **0.85**           |
| Random Forest     | 0.83               |

LightGBM showed better performance and was used for final predictions.

---

## Output

Final predictions are saved to `submission.csv` with:
