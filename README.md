# Predicting Edmonton Housing Prices
I fit and tune the hyperparameters of a XGBoost decision tree model. The data is 10,000 observations total, and was constructed as part of my Master's thesis. 

The data can be accessed here: https://github.com/SkyAllinott/Edmonton-Housing-Dataset

## Results:
The optimal model had a mean absolute error (MAE) of $16,810.36. The mean price of a home in the dataset was $238,310.21. Below is a graph of fitted and actual values, along with the 45 degree line. The model performs exceptionally.

![fittedvsactual](https://user-images.githubusercontent.com/52394699/180627806-63538a0f-ffc3-4d66-82c9-debee200c7f1.png)

### Feature Importance:
![feature_importance](https://user-images.githubusercontent.com/52394699/181067188-df11fd5c-45a8-4802-877b-302c479449b5.png)

Lot size and structure size were the most important featuers, followed by several distance measures (to downtown and LRT stations). Interestingly, the Valley Line West was more important than Valley Line SE, despite only being in the annoucement phase during this year (compared to SE's construction phase). Both neighbourhood crime measures were relatively unimportant, but since we controlled for neighbourhoods through dummies, and included neighbourhood crime, then this crime measure is more or less the true effect of neighbourhood crime, everything else constant. 


## Hyperparameter Tuning:
### What is a hyperparameter?
Hyperparameters are a set of variables in machine learning models, that cannot be inferred. That is, they cannot be determined from the data, and instead are defined by the user. Hyperparameters are critical to model accuracy, and therefore different hyperparameters can lead to drastically better or worse models.

This quickly becomes a computationally taxing problem. Evaluating a model with two hyperparameters, each with 3 possible values, yields 9 combinations of models. If we test these models 5 times, we're estimating 45 models. With a typical XGBoost model having 2 to 10 tunable parameters, the number of candidate models can quickly get into the thousands. Since I like to keep my electricity bill low, I focus on how to evaluate these models as efficiently as possible. 

### General Approach:
When tuning hyperparameters, I began with a coarse grid that I searched utilising randomized search. Random search takes x models and evaluates a subset of them. I then analysed the results to refine the parameter grid, and continued by using grid search. Grid search evaluates all x models. 

### Successive halving:
Successive halving begins by estimating all models in the parameter grid on a tiny subsample (about 10 observations of 8,000). After this, the top third of the models carry on to the next iterations, and the process repeats, with each iteration getting access to more data.

Initially, I did not utilise successive halving, and evaluating 200 candidate models (with 5 cross validation folds) from the first randomized search took approximately 2 hours and 10 minutes on an Intel i7-8700k at 100% usage. 

When implementing successive halving, I was able to evaluate 768 candidate models (again with 5 fold cv) in 15 minutes at 100% usage. That means that successive halving is nearly 36 times faster. With this significant reduction in computation time, you do not have to trade accuracy for speed as much. This ensures you are getting the best (or very close!) model. 

### GPU Usage:
My GPU (GTX 1080) offered no performance enhacements over the CPU, and was actually much slower. I think there are two reasons for this:
1. The data is relatively small. At a best case of 10,000 observations, this is not enough to saturate the GPU and offers no performance enhancement over a CPU
2. Some parameters are known to tune poorly/inefficiently on a GPU, and may have slowed down computation.
