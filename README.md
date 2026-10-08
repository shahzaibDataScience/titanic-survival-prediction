# 🚢 Titanic Survival Prediction

Predict whether a Titanic passenger survived, using the famous 891-passenger dataset. Full pipeline: EDA → feature engineering → two ML models compared head-to-head.

## What you'll learn

- Exploratory data analysis: survival rate by sex, class, and age
- Feature engineering: extracting **Title** from names with regex, building **FamilySize**
- Handling missing values (median for Age, mode for Embarked)
- One-hot encoding categorical features with `pd.get_dummies()`
- **Train/test split** — what it is and why we test on unseen data
- Training **Logistic Regression** vs **Random Forest**
- Reading a **confusion matrix** and classification report
- Comparing models by accuracy

## Dataset

`data/titanic.csv` — the real Titanic dataset (891 passengers, 12 columns):

| Column | Meaning |
|---|---|
| Survived | Target: 1 = survived, 0 = did not |
| Pclass | Ticket class (1 = richest, 3 = cheapest) |
| Name | Passenger name (used to extract Title) |
| Sex | Male / female |
| Age | Age in years (177 missing) |
| SibSp / Parch | Siblings-spouses / parents-children aboard |
| Ticket | Ticket number (dropped) |
| Fare | Ticket price |
| Cabin | Cabin number (dropped — too messy) |
| Embarked | Port: C = Cherbourg, Q = Queenstown, S = Southampton |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook titanic_survival.ipynb
```

Then run all cells top to bottom (Cell → Run All).

## Key findings

- Overall survival rate: **38.4%**
- Women survived far more: **74.2%** vs men **18.9%**
- 1st class: **63.0%** survived; 3rd class: only **24.2%**
- **Logistic Regression: 79.33%** accuracy
- **Random Forest: 83.24%** accuracy — the winner 🌲

## 🎓 Explain it yourself

1. Why do we split data into train and test sets? What goes wrong if we test on the training data?
2. Explain the four boxes of a confusion matrix in your own words.
3. Why did we extract "Title" from the Name column? What information does it add?
4. Why fill missing Age with the **median** instead of the **mean**?
5. Why does Random Forest usually beat a single simple model like Logistic Regression?
