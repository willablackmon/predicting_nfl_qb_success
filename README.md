# Predicting the Success of NFL Quarterbacks

### I. Summary

Predicting the success of NFL Quarterbacks, with emphasis on college performance, pre-draft rankings, and team investment in money and draft capital.

**X Feature Data** from college football will be used as predictive variables (inputs), while the **y Target Data** serves as the outcome to predict NFL success (as defined by the various success targets grouped and combined into aggregate measures.

**Grouped Success** targets (binary (1, 0):

* **hof_success** : Hall of Fame induction.
* **wins_success** : Winning success (e.g., win % > 50%, 1 or more Super Bowl wins).
* **award_success** : Awards success (e.g., more than 1 Pro Bowl, more than 1 All-Pro selection).
* **stat_success** : Statistical success (e.g., passing_yards > 15,000, touchdown_passes > 50).
* **earn_success** : Earnings success (career earnings > $5M).
* **long_success** : Longevity success (years_played > 5 years).

**Aggregate Overall 'NFL Success'** target:

* **nfl_success** : Aggregate NFL success, based on a combination of the above success targets.  Eg. `nfl_agg_succ = 1` for any player with 1 in at least 2 or more of the other columns (e.g., both `stat_succ =1` and `wins_succ = 1`).

---

### II. Data Sources

* [drafthistory.com](https://www.drafthistory.com)
* [Pro Football Reference](https://www.pro-football-reference.com)
* [Sports-Reference.com]()

##### A.  y Target Data (NFL Career Statistics, Achievements)

Obtain raw data starting from a historical list of all QBs drafted into the NFL quarterbacks and use this base list to obtain more granular statistical data that can be used to determine the player's success in the NFL.

* `draft_history.ipynb`, scrapes quarterback draft history data from [drafthistory.com](https://www.drafthistory.com). This file provides a comprehensive list of all NFL quarterbacks drafted since the league's inception.
* `pfb_ref_sourcing.ipynb` takes `draft_history` data and dynamically builds URLs that point to quarterback's profiles on Pro Football Reference (PFR).
* granular statistics for each quarterback are obtained, passing yards, completion rates, touchdowns, win percentage, and other relevant performance metrics.
* sourced data (both successful retrievals and failures) into CSV and PKL files for further analysis and modeling.
* `hof_monitor.ipynb` further enhances data from [Pro Football Reference](https://www.pro-football-reference.com) by pulling Pro Football Hall of Fame monitor metrics, adding additional layers of insight by providing milestone achievements (e.g., Pro Bowls, All-Pro selections, AV, etc.).

 **Libraries Used**:

* `requests` for sending HTTP requests to the target website.
* `BeautifulSoup` (from `bs4`) for parsing and navigating the HTML content.
* `pandas` for organizing and manipulating the scraped data.

**Output, Artifacts:**  The ultimate goal will be to set definitions of 'success' in the NFL.  Different Target/y Data fields will be grouped to help determine which are most indicative of overall quarterback success.

The different **Grouped Success Targets** will further be combined into a single, **Aggregate Overall 'NFL Success'** target.  The aggregate overall success metric, along with the grouped success metrics can be investigated using the feature/X data from college football data.

##### B.  X Feature Data (College Football Statistics)

For all quarterbacks where detailed NFL metrics were successfully obtained, the dataset was enhanced by scraping college football statistics from Pro-Football-Reference's sister site, [Sports-Reference.com]().  The college statistics provide the features we will investigate to help predict a player's success in the NFL.

* `cfb_ref_sourcing.ipynb` takes the list of quarterbacks for whom NFL data was sourced, and extracts college football data for each quarterback found, including:
  * Passing statistics: Completion percentage, passing yards, touchdowns, interceptions, passer rating
  * Game participation: Years played, games played, games started, wins, loses.
  * Season-by-season breakdown: Individual performance data for each season at the college level.

 **Libraries Used**:

* `pandas` for reading and organizing college football data.
* `requests` for sending HTTP requests to the Sports-Reference site.
* `BeautifulSoup` (from `bs4`) for parsing the HTML content to find relevant data.

**Output, Artifacts:**  The results are exported as CSV and PKL files, to create a comprehensive dataset for further analysis and modeling.


---

### III. Data Cleaning and Organization

This section provides a comprehensive view of how data was cleaned and organized, ensuring that all processes involved in preparing the data for machine learning models are covered.  Goals were to prepare and clean the combined dataset by addressing missing data, handling duplicates, merging college and NFL datasets, and performing necessary transformations to ensure the data is ready for modeling.

* `pfb_ref_cleaning.ipynb` Clean and organize NFL statistics data from Pro-Football-Reference.com to generate Y target data for use in determining Group Success metrics.

  * **Columns** : `['name', 'hall_of_fame', 'college', 'weighted_career_av', 'approximate_value', 'pass_rating', 'draft_year', 'retire_year', 'years', 'games_played', 'games_started', 'qb_rec', 'pass_yds', 'pass_yds_per_g', 'pass_td', 'pass_td_pct', 'pass_int', 'pass_int_pct', 'pass_att', 'pass_cmp', 'pass_cmp_pct', 'pass_yds_per_att', 'pass_adj_yds_per_att', 'pass_net_yds_per_att', 'pass_adj_net_yds_per_att', 'pass_yds_per_cmp', 'draft_round', 'draft_pick', 'all_star', 'superbowls', 'comebacks', 'gwd', 'height', 'weight', 'wins', 'losses', 'ties', 'height_in']`
* `cfb_ref_cleaning.ipynb` Clean and organize college football data sourced from Sports-Reference.com, for use in generating X feature data, including all columns available for modeling.

  * **Columns** : `['player', 'G', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'TD%', 'Int', 'Int%', 'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'blank', 'awards', 'school', 'draft', 'pro_stats', 'draft_rd', 'draft_overall', 'draft_yr', 'draft_team']`

 **Libraries Used** :

* `pandas` for data manipulation, cleaning, and merging.
* `pickle` for loading and saving intermediate data files.
* Display settings adjusted for visibility of large datasets during the cleaning process.

 **Data Transformation Highlights** :

* **Handling Missing Data** : NaN/null values were addressed either by filling in reasonable defaults or removing rows with excessive missing information.
* **De-duplication** : Duplicate rows were identified and removed where necessary to ensure the accuracy of the merged dataset.
* **Feature Engineering** : Additional features like player earnings and career longevity were calculated and added to the dataset, creating a richer set of attributes for modeling.
* **Exporting Data** : Cleaned and merged datasets were exported as both CSV and PKL files, ensuring compatibility for further analysis and modeling tasks.


---

### IV. Merging Data and Success Metrics Determination

`merge_cfb_ref_pfb_ref.ipynb`:

* Applies specific rules to calculate the various success metrics with NFL data (designed to quantify a player's performance and career success in the NFL)
* merges the professional football success metrics and cleaned college statistic datasets to create a comprehensive dataset for investigation and modeling the determination of the success of NFL quarterbacks.

**Group Success Metrics** determination:

* **`win_success `:** measure of players' wins, career win percentage. Quarterback assigned `win_success = 1` with more than **50 career wins** or win percentage greater than **50%.**
* **`stats_success` :** players who achieved significant statistical milestones in their NFL careers.  Quarterback assigned `stats_success = 1`  with **passing yards > 15,000,**, **touchdown passes > 50** , and **completion percentage >= 60%** .
* **`metrics_success`** : quantify success based on advanced metrics, including weighted Approximate Value (AV) and passer rating.  Quarterback assigned `metrics_success = 1` with **weighted Approximate Value (wAV)** greater than 50 or **career passer rating** exceeding **65.**
* **`longevity_success`** :  assess whether a quarterback had a long-lasting career.  Quarterback assigned `longevity_success = 1` if they played in the NFL for at least **4 years** and started at least **32 games** .
* **`superbowl_success`** identify quarterback won at least one Super Bowl  Quarterback assigned `superbowl_success = 1` with **at least 1 Super Bowl** win.

**Aggregate Success determination:**

* **`nfl_success`** overall measure of a quarterback's success by combining the individual success metrics.  Quarterback is assigned `nfl_success = 1` if they meet the criteria (value = 1) for at least **2 or more of the individual success metrics** (`win_success`, `stats_success`, `metrics_success`, `longevity_success`, `superbowl_success`).


---

### V. Modeling

##### A.  Methodology

Given the nature of the task - predicting success metrics - the focus will be on models that handle classification problems.

First/baseline models were **Logistic Regression** and **Random Forest Classifier.**

Also included was **Basic Neural Network (DNN)** Tensorflow/Keras model as there may be temporal dependencies in the data (value or state of one variable at a given time point influences the value or state of another variable at a future time point).

Practices utilized:

* **SimpleImputer()** in replacing missing or null/NA data.
* **Correlation analysis:** compute correlation matrix to identify highly correlated features
* **Feature Importance** (with Random Forest Model); calculate the feature importances and visualize with seaborn bar graph.
* **Feature Scaling** with StandardScaler to standardize/normalize numerical features (esp. important in the case of Logistic Regression and KNeighborsClassifier)
* **Visualizations** with seaborn, matplotlib
* **Hyperparameter Tuning** with GridSearchCV
* **K-Fold Cross-Validation** to assess performance of model across multiple subsets of the data; gives insights into how well model model generalizes.  Utilizes SciKeras **KerasClassifier** and Scikit-Learn **cross_val_score.**

Please See Appendix A, "Model Consideration and Selection" for a full list of the models reviewed, along with details of the characteristics.

##### B.  Modeling Process

###### 1. Data Inspection, Cleaning

* Inspect Data
* Divide into X feature and y target DataFrames
* Handle N/A, NaN Values: eg. fillna() or **SimpleImputer()** to fill or remove NA values.

###### 2. Exploratory Data Analysis (EDA)

The **target variable,** a 'success' value determined by applying numerical thresholds to statistics and metrics, as well as aggregating awards and honors (Pro-Bowl elections, Hall of Fame Indicutions), and career success (winning percentage, SuperBowl wins) were all of the nature of categorical classification.

* **Check **feature distributions**: determine if transformations are needed (log transformation for skewed data, etc.)**
* **Correlation analysis: compute correlation matrix** to identify highly correlated features, which can impact certain models; visualize with heat map.

![1729297617264](image/1729297617264.png)

* **Feature Importance** (with Random Forest Model); calculate the feature importances and visualize with seaborn bar graph.

![1729297671032](image/1729297671032.png)

* **Visualization** with seaborn, matlibplot.

  ![1729297802604](image/1729297802604.png)

  ![1729297822652](image/1729297822652.png)

###### 3. Data Preparation, Split and Scale

* Train-Test Split (prepare data for model training and evaluation).
* **Standard Scaling** (ensure your features are on the same scale): StandardScaler()

###### 4. Modeling

* Model (create model)
* Fit (train model)
* Evaluate Training Model fit (accuracy_score/score on training data)
* Predict (accuracy_score/score)
* Evaluate Model (accuracy_score/score on y test data (actual data)  vs. predictions)
* K-Fold Cross-Validation to assess performance of model across multiple subsets of the data (utlizes KerasClassifier and Scikit-Learn cross_val_score)

![1729297728611](image/1729297728611.png)

![1729298177810](image/1729298177810.png)

###### 5. Optimization / Hyperparameter Tuning

* **Grid Search**

###### 6. Re-Model / Evaluate with Optimized Params/Methods/Hyperparameters

* **Cross (K-Folds) Validation**


---

### VII. Future Updates to this Model:

##### A.  Additional Success Metrics

`pfb_ref_cleaning_kitchensink.ipynb` is currently being used to pursue additional success metrics for inclusion in the modeling, including:

* **`hof_success`** : Hall of Fame induction.
* **`draft_success`** : Draft success, based on whether a player was picked in the 1st round, mid rounds, or later rounds.
* **`award_success`** : Award success, defined as having **3 or more total awards** (e.g., Pro Bowls, All-Pros, MVPs).
* **`earn_success`** : Earnings success, defined as having  **career earnings > $5M.**

##### B.  Additional Models, Data Techniques

For expansion on these studies, the following is recommended as a starting point for future models:

* **Support Vector Machine**
* **LSTM (Long Short-Term Memory)**
* **K-Nearest Neighbors (KNN)**

Additional practices for future modeling:

* **Encoding Categorical Variables:**  explore categorical variables, requiring encoding (one-hot encoding, ordinal encoding).


---

### Appendix A: Model Consideration and Selection

The following models were considered for the modeling and prediction of this dataset:

**Logistic Regression:**

* Good starting point for binary classification problems: simple and interpretable, making it a good baseline model.
* **Pros:** Simple and interpretable. Works well for binary classification. **Cons:** Limited to linear decision boundaries. **Best for** : Problems with linear relationships.

**Random Forest Classifier:**

* Robust ensemble method that works well for many classification tasks. It handles overfitting better than some other models and can manage a mix of numerical and categorical data
* **Pros:** Handles non-linear relationships.  Robust to outliers and overfitting. **Cons:** Less interpretable.  Can be slow for large datasets. **Best for:** Complex datasets with many features and interactions.

**K-Nearest Neighbors (KNN, KNeighborsClassifier):**

* Model can be effective if you have enough data. It makes predictions based on the nearest neighbors in the feature space. However, it can be sensitive to the choice of `k` and might struggle with high-dimensional data
* **Pros:** Simple and intuitive.  Non-parametric, flexible decision boundaries. **Cons:** Computationally intensive for large datasets. **Best for** : Small datasets with well-defined boundaries.

**Support Vector Machine (SVM) / Support Vector Classifier (SVC):**

* Can be powerful, especially with a non-linear kernel. It's effective for high-dimensional spaces but might require more tuning.
* **Pros:** Effective in high-dimensional spaces. Works well for clear margin of separation. **Cons:** Computationally intensive.  Requires careful tuning of hyperparameters. **Best for** : Medium-sized datasets with distinct class separability.

**Deep Neural Network (DNN):**

* A simple feedforward DNN can work well for structured data.  Ensure there is enough data to avoid overfitting, and tuning of hyperparameters is crucial.
* **Pros:** Flexible and can model complex relationships.  Scales well with large datasets. **Cons:** Requires more data. Can overfit without proper regularization. **Best for:** General use with non-linear relationships.

**Convolutional Neural Network (CNN)**:

* Generally used for image data, CNNs might not be the best fit for structured tabular data.
* **Pros:** Captures spatial hierarchies. Highly effective for image data. **Cons:** Computationally heavy. **Best for:** Image data, spatial data.

**LSTM (Long Short-Term Memory):**

* Designed for sequential data, such as time series or natural language; likely not appropriate for player statistics.
* **Pros:** Overcomes vanishing gradient issues. Captures long-term dependencies effectively.  **Cons:** Computationally heavy **Best for:** Long-term sequential data, time-series.

**Recurrent Neural Network (RNN)**

* Designed for sequential data, such as time series or natural language; likely not appropriate for player statistics.
* **Pros:** Captures temporal dependencies. Effective for time-series prediction. **Cons:** Vanishing gradient problems. **Best for:** Time-series data, sequential data.

---

### Appendix B: Other Similar Studies

1. Excellent culling of NFL prospect data by Jack Lich, with highly clean datasets and feature descriptions available on both Kaggle [https://www.kaggle.com/datasets/jacklichtenstein/espn-nfl-draft-prospect-data](https://www.kaggle.com/datasets/jacklichtenstein/espn-nfl-draft-prospect-data) and github [https://github.com/jacklich10/nfl-draft-data](https://github.com/jacklich10/nfl-draft-data)
2. "Does Your NFL Team Draft to Win? New Research Reveals Rounds 3, 4, and 5 are the Key to Future On-Field Performance," Chandler Smith, [https://www.samford.edu/sports-analytics/fans/2024/Does-Your-NFL-Team-Draft-to-Win-New-Research-Reveals-Rounds-3-4-and-5-are-the-Key-to-Future-On-Field-Performance](https://www.samford.edu/sports-analytics/fans/2024/Does-Your-NFL-Team-Draft-to-Win-New-Research-Reveals-Rounds-3-4-and-5-are-the-Key-to-Future-On-Field-Performance)
3. "Using Machine Learning and College Profiles to Predict NFL Success," [Northwestern Sports Analytics Group](https://sites.northwestern.edu/nusportsanalytics/ "Northwestern Sports Analytics Group") [https://sites.northwestern.edu/nusportsanalytics/2024/03/29/using-machine-learning-and-college-profiles-to-predict-nfl-success/]()
4. "Predicting QB Success in the NFL," [Adam McCann](https://www.linkedin.com/in/adam-mccann-bb94774/), Chief Data Officer at KeyMe, [https://duelingdata.blogspot.com/2017/04/predicting-qb-success-in-nfl.html]()
5. "NFL Draft Day Dreams: Analyzing the Success of Drafted Players vs. Undrafted Free Agents," Breanna Wright and Major Bottoms Jr. [https://www.endava.com/insights/articles/nfl-draft-day-analyzing-the-success-of-drafted-players-vs-undrafted-free-agents]()
6. "Can the NFL Combine Predict Future Success?"  [https://nfldraftcombineanalysis.wordpress.com/2016/05/13/can-the-nfl-combine-predict-future-success/]()  Includes information about Combine Results and NFL Success, Combine Results and Draft Pick Position, Draft Pick Position and NFL Success, draft pick valuation (uses Jimmy Johnson's chart for pick valuation),
7. "Can NFL career success be predicted based on Combine results?" Caroline Malin-Mayor, Monica-Ann Mendoza, Victor Li, Tyler Devlin https://nfldraftcombineanalysis.wordpress.com/2016/05/03/52/  https://nfldraftcombineanalysis.wordpress.com/2016/04/27/11/
8. "Valuing the NFL Draft", Caroline Malin-Mayor, Monica-Ann Mendoza, Victor Li, Tyler Devlin https://nfldraftcombineanalysis.wordpress.com/2016/04/20/2/  Uses Weighted Career Approximate Value to label success metric.
