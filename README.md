# Titanic - Machine Learning Project 🚢
This project aims to predict passenger survival on the Titanic using **machine learning models** and **feature engineering techniques**. The dataset is sourced from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic). Through exploratory data analysis (EDA), feature engineering, and model optimization, achieved a **Kaggle leaderboard score of 0.78947**.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Workflow](#workflow)
3. [Key Insights](#key-insights)
4. [Models and Performance](#models-and-performance)
5. [How to Run the Code](#how-to-run-the-code) 
---

## **Project Overview**
The Titanic dataset includes passenger information like age, class, fare, family relationships, and survival status. The task involves building a predictive model to classify passengers into survivors (`1`) and non-survivors (`0`).

- **Dataset**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- **Target Variable**: `Survived` (1 = survived, 0 = did not survive)
- **Final Kaggle Score**: **0.78947**
![image](https://github.com/user-attachments/assets/4851b291-948b-439e-b1b4-ffac0484c58b)
---

## **Workflow**
This project follows a structured machine learning pipeline:

### **1. Exploratory Data Analysis (EDA)**

#### **Understanding the Dataset**
The Titanic dataset includes passenger details such as demographic, socio-economic, and ticketing information. The target variable, `Survived`, indicates whether a passenger survived (`1`) or not (`0`). Key insights from the dataset are as follows:

#### **Key EDA Observations**
- **Survival Distribution**:
  - Approximately 38.4% of passengers survived, while 61.6% did not.
  - This highlights an imbalanced dataset where survival is the minority class.

- **Pclass (Passenger Class)**:
  - First-class passengers had the highest survival rate (~63%), followed by second-class (~47%), and third-class (~24%).

- **Sex**:
  - Females had a significantly higher survival rate (~74%) compared to males (~19%), showing a strong correlation with survival.

- **Age**:
  - Younger passengers (children) had higher survival rates compared to adults and seniors. Missing values in `Age` were imputed to avoid data loss.

- **Embarked (Port of Embarkation)**:
  - Passengers who embarked at Cherbourg (`C`) had the highest survival rate (~55%), followed by Queenstown (`Q`, ~39%) and Southampton (`S`, ~34%).

- **Fare**:
  - Higher fares were associated with higher survival rates, likely due to their correlation with passenger class.

- **Family Features (SibSp and Parch)**:
  - Passengers traveling with small family groups (2–4 members) had better survival rates than solo travelers or those with large families.

- **Cabin and Deck**:
  - Passengers with known cabin information (e.g., Decks B, C, D) had higher survival rates than those without cabin information (`Z`).

- **Title**:
  - Titles extracted from names (e.g., `Mr.`, `Mrs.`, `Miss.`) revealed distinct survival trends:
    - Females with titles like `Miss.` and `Mrs.` had significantly higher survival rates.
    - Males with titles like `Mr.` had the lowest survival rates.
    - Rare titles (e.g., `Countess`, `Rev`) varied but often correlated with class or social status.

These observations were used to guide feature engineering and model development.


---

### **2. Data Preprocessing**
- **Handling Missing Values**:
  - Used `IterativeImputer` to fill missing `Age` values based on correlated features (`Pclass`, `SibSp`, `Parch`).
  - Replaced missing `Embarked` with the mode.
  - Filled missing `Fare` with the median and replaced missing `Cabin` with a placeholder (`Z`).

- **Feature Engineering**:
  1. **Family Features**:
     - Created `FamilySize = SibSp + Parch + 1` to represent total family members onboard.
     - Categorized passengers into family groups:
       - `Solo`: Alone travelers
       - `Small`: Families of 2–4
       - `Large`: Families of 5 or more
  2. **Deck Extraction**:
     - Extracted deck information from the `Cabin` feature.
     - Decks `D`, `E`, and `B` had higher survival rates, while passengers without cabin info (`Z`) had the lowest.
  3. **Titles from Names**:
     - Extracted titles (`Mr.`, `Mrs.`, `Miss.`, etc.) from passenger names.
     - Grouped rare titles (`Lady`, `Countess`, `Jonkheer`) into a `Rare` category.
  4. **Binned Features**:
     - Created categories for `Age` and `Fare`:
       - **AgeBands**: `Child`, `Teenager`, `Adult`, `Senior`
       - **FareBands**: `Low`, `Medium`, `High`

- **One-Hot Encoding**:
  - Encoded categorical features (`Pclass`, `Sex`, `Embarked`, etc.) into numerical values.
  - Ensured no multicollinearity by dropping one category from each encoded feature.

- **Scaling**:
  - Applied `StandardScaler` to scale numerical features (`Age`, `Fare`) for consistency.

---

### **3. Model Training**
- **Split Dataset**:
  - Split data into training and validation sets (80%-20%) using `train_test_split` with stratification on the target variable.

- **Selected Models**:
  - Trained the following models with hyperparameter tuning:
    1. Random Forest
    2. Gradient Boosting
    3. K-Nearest Neighbors (KNN)
    4. XGBoost
    5. CatBoost

- **Hyperparameter Optimization**:
  - Used **GridSearchCV** with 5-fold cross-validation to tune parameters like `n_estimators`, `max_depth`, and `learning_rate`.
  - Ensured models were robust to overfitting by evaluating cross-validation scores.

- **Results**:
  - Best models: **Random Forest** and **XGBoost**, achieving cross-validation scores of **83.85%**.

---

### **4. Model Evaluation**
- **Validation Accuracy**:
  - Random Forest and XGBoost outperformed other models with validation accuracy of **81.01%**.
- **Feature Importance**:
  - `Pclass`, `Sex`, and `Fare` were among the most important predictors.

- **Submission**:
  - Generated predictions using all models and prepared `.csv` files for Kaggle submission.

---

## **Key Insights**
1. **Survival Factors**:
   - Gender: Females were prioritized during evacuation.
   - Class: Higher-class passengers had better access to lifeboats.
   - Family: Passengers with small families had higher survival chances compared to those traveling alone or in large families.

2. **Feature Engineering Impact**:
   - Adding `FamilySize`, `AgeBand`, and `Title` significantly improved model performance.

3. **Model Performance**:
   - Ensemble models like Random Forest and XGBoost performed better than simpler models like KNN.

---

## **Models and Performance**
| **Model**           | **Training Accuracy** | **Validation Accuracy** | **Cross-Validation** |
|---------------------|-----------------------|-------------------------|----------------------|
| Random Forest       | 91.01%               | 81.01%                  | 83.85%              |
| XGBoost             | 91.85%               | 81.01%                  | 83.85%              |
| Gradient Boosting   | 87.08%               | 78.21%                  | 83.85%              |
| K-Nearest Neighbors | 85.81%               | 78.77%                  | 81.18%              |
| CatBoost            | 83.85%               | 80.45%                  | 83.85%              |

![image](https://github.com/user-attachments/assets/4afc0f2a-ebde-463a-b219-d7ffd2c148e9)

![image](https://github.com/user-attachments/assets/36b6c1a8-6730-4417-aa13-47d89b0feb45)


## **How to Run the Code**

Follow these steps to run the project on your local machine:

### **1. Clone the Repository**

Clone the repository to your local system using the following command:
```bash
git clone https://github.com/Igoras6534/Titanic-ML-Project.git
cd Titanic-ML-Project
```
Ensure you have Python 3.8+ installed. To install the required libraries, run:
```bash
pip install -r requirements.txt
```
Launch Jupyter Notebook or Jupyter Lab and open the notebook:
```bash
jupyter notebook Titanic-Project-EDA-ML.ipynb
```
Now you're ready to run the notebook.
