# Ames Housing Price Prediction: Decision Tree Optimization

An end-to-end Machine Learning project demonstrating data preprocessing, advanced feature engineering, and the mitigation of overfitting in Decision Tree Regressors using the Ames Housing dataset.

##  Project Objective
The goal of this project is to accurately predict house prices (`SalePrice`) based on various property attributes. The primary focus is on handling high-dimensional categorical data, engineering high-signal features, and tuning a baseline Decision Tree Regressor to prevent pure memorization (overfitting) and improve generalization on unseen data.

##  Workflow & Methodology

1. **Exploratory Data Analysis (EDA):**
   * Identified strong positive correlations (`Overall Qual`, `Gr Liv Area`, `Total Bsmt SF`) and negative correlations (`Overall Cond`, `Enclosed Porch`).
   * Detected and visualized extreme outliers skewing the price distribution.

2. **Feature Engineering:**
   * Consolidated redundant spatial metrics into `Total_SqFt`.
   * Created chronological features such as `House_Age` from construction and sale years.
   * Aggregated bathroom counts into a single `Total_Baths` feature.
   * Dropped legacy features (`Gr Liv Area`, `Year Built`, etc.) to prevent multicollinearity and reduce model complexity.

3. **Data Preprocessing:**
   * **Categorical Encoding:** Converted 39 categorical columns into a machine-readable format using One-Hot Encoding (`pd.get_dummies` with `drop_first=True`), expanding the feature space to 255 dimensions.
   * **Scaling:** Intentionally bypassed scaling, as Decision Trees are rule-based algorithms (splitting on thresholds) and do not rely on distance metrics.

4. **Model Training & Optimization:**
   * **Baseline Model:** Trained a raw `DecisionTreeRegressor`. Demonstrated severe overfitting (Training R²: 100%, Testing R²: 86.79%).
   * **Outlier Removal:** Filtered out massive, undervalued properties (Total_SqFt > 6000 & SalePrice < $300,000) that degraded model logic.
   * **Hyperparameter Tuning:** Utilized `GridSearchCV` to constrain tree complexity (`max_depth`, `min_samples_split`, `min_samples_leaf`), ensuring robust, generalized rule creation.

##  Final Evaluation (Model V2)

The optimization pipeline successfully closed the overfitting gap from ~13% down to a stable **5.21%**, while simultaneously reducing error margins.

* **Training R² Score:** 92.40%
* **Testing R² Score:** 87.19%
* **Mean Absolute Error (MAE):** $20,035.64
* **Root Mean Squared Error (RMSE):** $30,234.64

An R² of 87.19% confirms that the engineered features successfully explain the vast majority of the variance in Ames housing prices, while the reduced RMSE proves the model's resilience against outlier-driven penalties.

##  Technologies Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

##  How to Run
1. Clone the repository: `git clone <https://github.com/yusuferdemgungorr/Ames_Housing_Price_Prediction>`
2. Install dependencies: `pip install -r requirements.txt`
3. Place the Ames Housing dataset in the `data/` directory.
4. Run the Jupyter Notebook step-by-step.
