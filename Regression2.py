#Project made by Nuno Baptista ist196295 & Francisco Matias ist199936
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, RidgeCV, Lasso
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV


warnings.filterwarnings("ignore")

# Given a model, we want to estimate the MSE and R2
def Estimate_metrics(x_train, y_train, model, kfold, cv):
    # Cross-validation on the test data
    cv_results = cross_validate(model, x_train, y_train, cv=kfold, scoring=('neg_mean_squared_error', 'r2'))

    # The formula to calculate MSE and R2
    MSE = abs(cv_results['test_neg_mean_squared_error'].mean())
    R2 = cv_results['test_r2'].mean()
    SSE = MSE * len(x_train)

    return MSE, SSE, R2

# Functions that will fit both models (Inlier and oulier).
def fitting(X, y):
    
    #We already tested for the other two models and this one was the best.
    Chosen_model = LinearRegression().fit(X, y)
   
    #Estimate some important parameters 
    MSE, SSE, R2 = Estimate_metrics(X, y, Chosen_model, 5, 5)
    
    return Chosen_model, MSE, SSE, R2

# Function to identify and remove the outlier with the largest residual
def remove_outlier(X, y, model):
   
    y_predict = model.predict(X)
    residuals = np.abs(y - y_predict)
    index_outlier = np.argmax(residuals)
    
    #Deleting outliers
    X_inlier = np.delete(X, index_outlier, axis=0)
    y_inlier = np.delete(y, index_outlier)
    
    #Saving here the outliers
    X_outlier = X[index_outlier].reshape(1, -1)  # Reshape to maintain consistency
    y_outlier = y[index_outlier]
    
    return X_inlier, y_inlier, X_outlier, y_outlier

# Function to perform the iterative outlier removal process
def Recursive_Outlier_remove(X, y):
    
    best_models = None
    best_datasets = None  
    lower_sse_sum = float('inf')
    
    removed_outliers_X = []
    removed_outliers_y = []
    #best_mean_r2 = 0
    #counter = 0
    
    #Running all the samples, one by one in the training set.
    for i in range(y.shape[0]):  
        
        model, mse, sse, r_squared = fitting(X, y)
        
        X_inlier, y_inlier, X_outlier, y_outlier = remove_outlier(X, y, model)
        
        #Adding here the removed outliers.
        removed_outliers_X.append(X_outlier)
        removed_outliers_y.append(y_outlier)

        if len(removed_outliers_X) < 5:
            X, y = X_inlier, y_inlier 
            continue
        
        if X_inlier.shape[0] < 5:
            break 
        
        #Fitting the models
        inliers_model, mse_inlier, sse_inlier, r2_inlier = fitting(X_inlier, y_inlier) 
        outliers_model, mse_outliers, sse_outliers, r2_outliers = fitting(np.vstack(removed_outliers_X).reshape(-1, X.shape[1]), np.hstack(removed_outliers_y).reshape(-1, 1))

        #We tested both ways ( With r2 and mse )
        #mean_r2 = (r2_inlier + r2_outliers) / 2
        mse_sum = mse_inlier + mse_outliers
        
        #if mean_r2 > best_mean_r2:
        if mse_sum < lower_sse_sum:
          best_models = (inliers_model, outliers_model)
          best_datasets = (X_inlier, y_inlier, X_outlier, y_outlier)
          #best_mean_r2 = mean_r2
          lower_sse_sum = mse_sum
        
        #Setting the new inliers.
        X, y = X_inlier, y_inlier
    
    #If something went wrong
    if best_models is None:
        print("No best models found.")
    
    return  best_datasets, best_models

# Loading data from the given files
X_train = np.load('X_train_regression2.npy')
y_train = np.load('y_train_regression2.npy')
X_test = np.load('X_test_regression2.npy')

# Perform iterative outlier removal
adjusting = Recursive_Outlier_remove(X_train, y_train)

inliers_model = adjusting[1][0]
outliers_model = adjusting[1][1]

#adjusting[2][0] is the first model (Inlier model) from the returns of the Recursive_Outlier_remove(X_train, y_train)
predict_inliers_model = inliers_model.predict(X_test)
predict_outliers_model = outliers_model.predict(X_test)

previsao_final = np.hstack((predict_inliers_model.reshape(len(predict_inliers_model), 1), predict_outliers_model.reshape(len(predict_outliers_model), 1)))
np.save('Y_test_regression2', previsao_final)
