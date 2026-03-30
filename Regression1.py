#Project made by Nuno Batista ist196295 & Francisco Matias ist199936
import numpy as np
from sklearn.metrics import mean_squared_error
from statistics import mean
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, RidgeCV, Lasso
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

#Given a module, we want to estimate the MSE and R2
def Estimate_metrics(x_train, y_train, model, kfold, cv):
    
    #Croos validation in the test data
    cv_results = cross_validate(model, x_train, y_train.ravel(), cv = kfold, scoring = ('neg_mean_squared_error', 'r2'))
    
    #Print the cv_results dictionary
    #print("cv_results dictionary:")
    #print(cv_results)
    
    #The formula to calculate MSE and R2
    MSE = abs(cv_results['test_neg_mean_squared_error'].mean()) 
    r2 = cv_results['test_r2'].mean()
    
    return MSE, r2
    
def Find_Best_Alpha(alphas_to_train, k, x, y, model, cv):
    
    #The list of alphas set to be trained.
    grid_params = {'alpha': alphas_to_train}
    #Use GridSearchCV to perform a grid search over the specified alpha values, using k-fold cross-validation.
    grid_search = GridSearchCV(model, grid_params, cv = k, scoring = 'neg_mean_squared_error').fit(x, y)
    #The best alpha from the results of the grid search.
    best_alpha = grid_search.best_params_['alpha']

    return best_alpha

def Regression1():
    
    #Loading the training data
    x_train = np.load('X_train_regression1.npy')
    y_train = np.load('y_train_regression1.npy')
    
    #MSE and R2 will be saved here.
    MSE = []
    R2 = []
    
    #Number of samples
    N = x_train.shape[0]
    # We tried with other values as k = 5, 10. 
    kFolds = 5  
    #Dont know the best alpha yet
    Best_alpha = None 
    
                            # *************** Ridge regression model *************** #
               
    #Training the model with a range of alphas.                                                                                                                    
    reg_alphas = np.arange(0, 3, 0.01) #regularization with a step size of 0.01
    model = Ridge()
    Best_alpha = Find_Best_Alpha(reg_alphas, kFolds, x_train, y_train, model, N)
    
                                    #We trained the model, so now we are ready to advance#
    #Loading the test data
    x_test = np.load('X_test_regression1.npy')
    
    trained_model = Ridge(alpha = Best_alpha).fit(x_train, y_train)
    
    #Verifying the params.
    mse, r2 = Estimate_metrics(x_train, y_train, trained_model, kFolds, N)
    MSE.append(mse)
    R2.append(r2)
    
    #Print our values
    #print(f"Best_alpha: {Best_alpha:.2f}")
    #print(f"MSE: {mse:.2f}")
    #print(f"R2: {r2:.2f}")
        
    #Predicition
    y_pred = trained_model.predict(x_test)
    y_pred_reshaped = y_pred.reshape((len(y_pred), 1))
    np.save('Y_test_regression1', y_pred_reshaped)
    
if __name__ == "__main__":
    Regression1()