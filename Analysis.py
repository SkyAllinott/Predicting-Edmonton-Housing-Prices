import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV
import pickle
import os
import time

os.chdir("G:/My Drive/Python Projects/Edmonton Housing/")

# Loading in data. Downloaded from (https://github.com/SkyAllinott/Edmonton-Housing-Dataset)
train = pd.read_csv('G:/My Drive/Python Projects/Edmonton Housing/Edmonton_Housing_train.csv')
test = pd.read_csv('G:/My Drive/Python Projects/Edmonton Housing/Edmonton_Housing_test.csv')
seed = 9
train_labels = train['realprice']
test_labels = test['realprice']
train_features = train.drop('realprice', axis=1)
test_features = test.drop('realprice', axis=1)
X, y = train_features, train_labels

# Parameter space:
# This was adjusted over several iterations. For a first coarse pass, I ran HalvingRandomSearch on wide parameter
# ranges for its speed to understand what the data liked. Once I understand where the model performed best,
# I restricted it to this HalvingGridSearch.
params = {'max_depth': [24, 26, 28, 30, 32, 35],
          'learning_rate': [0.01],
          'subsample': np.arange(0.5, 0.7, 0.1),
          'colsample_bytree': np.arange(0.5, 0.8, 0.1),
          'n_estimators': np.arange(600, 1000, 100),
          'min_child_weight': [1, 2],
          'reg_alpha': [0, 1]}

# Tuning hyper parameters:
boost = xgb.XGBRegressor(seed=seed)
clf = HalvingGridSearchCV(estimator=boost,
                          param_grid=params,
                          scoring='neg_mean_absolute_error',
                          n_jobs=-1,
                          verbose=1)

# Testing 768 candidate models (parameter grid above) takes about 15 minutes on an intel i7-8700k at 100%
# (there was no performance improvement to GPU usage on such a small dataset). Without halving, testing 200 models
# took approximately 130 minutes. As expected, there are significant performance gains to successive halving.
start = time.time()
clf.fit(X, y)
print("Best parameters: ", clf.best_params_)
print("Lowest MAE: ", (-clf.best_score_))
end = time.time()
print((end-start)/60)

# Exploring the results of the search:
# (used to update the next grid and get an idea of what works)
cv_results = pd.DataFrame(clf.cv_results_)
cv_results = cv_results.sort_values(by='rank_test_score')

# Fitting the optimal model from the search:
best_parameters = clf.best_params_
boost_best = xgb.XGBRegressor(seed=seed, **prev_best_params)
boost_best.fit(train_features, train_labels)

xgb.plot_importance(boost_best, max_num_features=10)
plt.subplots_adjust(left=.35)

# Predicting on the test set:
predictions = boost_best.predict(test_features)
errors_boost = abs(predictions - test_labels)
MAE = round(np.mean(errors_boost), 2)

# Plotting fitted vs actual values:
xvalues = [0, 2000000]
yvalues = [0, 2000000]
fig, ax = plt.subplots()
plt.scatter(test_labels, predictions)
plt.plot(xvalues, yvalues, color='red')
ax.ticklabel_format(style='plain')
plt.title("Edmonton Housing: Fitted vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Fitted Values")
plt.subplots_adjust(left=.17)
plt.show()

# Saving the best parameters to a .txt file due to the random nature of tuning, and it's time to run:
# Previous best MAE:
with open('MAE.txt', 'rb') as file:
    MAE_prev_best = pickle.load(file)

# Determines whether to update the MAE and model parameters from previous runs:
if MAE_prev_best['MAE'] < MAE:
    print('Previous MAE was better, do not update the model')
else:
    print('New MAE is better, updating the parameters and MAE text files')
    MAE_dict = {'MAE': MAE}
    with open('MAE.txt', 'wb') as file:
        pickle.dump(MAE_dict, file)
    with open('params.txt', 'wb') as file:
        pickle.dump(best_parameters, file)


# Load in the best parameters if needed, to compare:
with open('params.txt', 'rb') as file:
    prev_best_params = pickle.load(file)


