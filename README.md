### Study

Predicting the success of NFL Quarterbacks, with emphasis on college performance, pre-draft rankings, and team investment in money and draft capital.

### Data Sources:

* Historical Draft Data: all QBs ever drafted.
* Take that list of QBs and fetch the QB data for all of them from pro-football-reference (ran that last night).  The search for the proper QB name is done much more thoroughly, double checking the name of the Quarterback fully before pulling data.
* Take that DataFrame, pull the list of QBs with valid data (over 600) and lookup their college stats at [sports-reference.com/cfb](http://sports-reference.com/cfb)  .

https://www.sports-reference.com/bot-traffic.html

* FBref and Stathead sites more often than ten requests in a minute.
* our other sites more often than twenty requests in a minute.

##### Data File Organization

* sourcing_notebooks are the ipynb files that are used to source data
* sourcing_artifacts are the results of the ipynb files in sourcing_notebooks
* data_cleaning_notebooks are the ipynb files used to take any files and clean them.. resulting cleaned files will be placed in data_cleaning_artifacts

##### Target/y Data

The ultimate goal will be to determine which of those target/y are indicative of overall quarterback success (or can be investigated on their own) using the feature/X data from college football data.

Target/y columns:

* **hof:** Hall of Fame induction
* **wins_succ:** win % > XX, 1 or more Superbowl wins),
* **award_succ:** Awards (Pro Bowl > 1, All Pro > 1)
* **stat_succ:** Statistical Success (passing_yards > 15,000, touchdown_passes > 50....)
* **earn_succ:** Earnings Success (career_earnings > $5M
* **long_succ:** Longevity success (years_played > 5 years)
* **nfl_agg_succ:** aggregate NFL success, based on combination of the aforementioned success targets.
  * eg.: nfl_agg_succ = 1 for any player with 1 in >= 2 or more of the other columns (stat_succ =1 and wins_succ = 1).

All columns will be binary (1, 0), ready-to-use as a target/y.  We will begin with NFL Aggregate Sucess target as well as perform branched/multi-prediction Neural Network models for multiple target/y columns.

##### Feature/X Data

To gather X features, we will gather the following data for college Quarterback prospects entering the NFL:

**Source**:


### Modeling Process

#### I. Data Preparation

1. Investigate Data
2. Handle NA Values: SimpleImputer() to fill or remove NA values.
3. Compute Correlation Matrix ()  (before scaling to keep original values intact for more accurate correlation matrix).
4. Standard Scaling (ensure your features are on the same scale): StandardScaler()

#### II. Train-Test Split (prepare data for model training and evaluation).

#### III. Modeling

1. Model (create model)
2. Fit (train modell)
3. Evaluate Training Model fit (accuracy_score/score on training data)
4. Predict (accuracy_score/score)
5. Evaluate Model (accuracy_score/score on y test data (actual data)  vs. predictions)

#### IV. Optimization / Hyperparameter Tuning

1. Grid Search

#### V. Re-Model / Evaluate with Optimized Params/Methods/Hyperparameters



### Other Studies

1. Excellent culling of NFL prospect data by Jack Lich, with highly clean datasets and feature descriptions available on both Kaggle [https://www.kaggle.com/datasets/jacklichtenstein/espn-nfl-draft-prospect-data](https://www.kaggle.com/datasets/jacklichtenstein/espn-nfl-draft-prospect-data) and github [https://github.com/jacklich10/nfl-draft-data](https://github.com/jacklich10/nfl-draft-data)
2. "Does Your NFL Team Draft to Win? New Research Reveals Rounds 3, 4, and 5 are the Key to Future On-Field Performance," Chandler Smith, [https://www.samford.edu/sports-analytics/fans/2024/Does-Your-NFL-Team-Draft-to-Win-New-Research-Reveals-Rounds-3-4-and-5-are-the-Key-to-Future-On-Field-Performance](https://www.samford.edu/sports-analytics/fans/2024/Does-Your-NFL-Team-Draft-to-Win-New-Research-Reveals-Rounds-3-4-and-5-are-the-Key-to-Future-On-Field-Performance)
3. "Using Machine Learning and College Profiles to Predict NFL Success," [Northwestern Sports Analytics Group](https://sites.northwestern.edu/nusportsanalytics/ "Northwestern Sports Analytics Group") [https://sites.northwestern.edu/nusportsanalytics/2024/03/29/using-machine-learning-and-college-profiles-to-predict-nfl-success/]()
4. "Predicting QB Success in the NFL," [Adam McCann](https://www.linkedin.com/in/adam-mccann-bb94774/), Chief Data Officer at KeyMe, [https://duelingdata.blogspot.com/2017/04/predicting-qb-success-in-nfl.html]()
5. "NFL Draft Day Dreams: Analyzing the Success of Drafted Players vs. Undrafted Free Agents," Breanna Wright and Major Bottoms Jr. [https://www.endava.com/insights/articles/nfl-draft-day-analyzing-the-success-of-drafted-players-vs-undrafted-free-agents]()
6. "Can the NFL Combine Predict Future Success?"  [https://nfldraftcombineanalysis.wordpress.com/2016/05/13/can-the-nfl-combine-predict-future-success/]()  Includes information about Combine Results and NFL Success, Combine Results and Draft Pick Position, Draft Pick Position and NFL Success, draft pick valuation (uses Jimmy Johnson's chart for pick valuation),
7. "Can NFL career success be predicted based on Combine results?" Caroline Malin-Mayor, Monica-Ann Mendoza, Victor Li, Tyler Devlin https://nfldraftcombineanalysis.wordpress.com/2016/05/03/52/  https://nfldraftcombineanalysis.wordpress.com/2016/04/27/11/
8. "Valuing the NFL Draft", Caroline Malin-Mayor, Monica-Ann Mendoza, Victor Li, Tyler Devlin https://nfldraftcombineanalysis.wordpress.com/2016/04/20/2/  Uses Weighted Career Approximate Value to label success metric.
