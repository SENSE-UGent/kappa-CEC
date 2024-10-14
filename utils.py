import numpy as np

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def RMSE(predictions, targets):
    """
    Compute the Root Mean Square Error (RMSE) between predicted and actual target values.

    Parameters:
    - predictions: array-like, predicted values.
    - targets: array-like, true target values.

    Returns:
    - float: The RMSE value which indicates the standard deviation of the prediction errors.
    """
    differences = np.array(predictions) - np.array(targets)
    return np.sqrt(np.mean(differences**2))


# Function to perform linear regression and print results
def perform_regression_and_select_best(feature_sets, target, df):
    """
    Perform linear regression for multiple feature sets and select the best performing model based on R² score.

    Parameters:
    - feature_sets: list of feature sets to be evaluated (each feature set is a list of column names).
    - target: str, the name of the target variable in the dataframe.
    - df: DataFrame, the input dataset containing feature columns and the target variable.

    Returns:
    - tuple: The best performing linear regression model and the corresponding feature set.
    """
    print('perform_regression_and_select_best:', feature_sets, target)
    best_score = float('-inf')
    best_regressor = None
    best_features = None
    
    for features in feature_sets:
        print(len(df[list(features)].to_numpy()))
        print(len(df[target]))

        reg = LinearRegression().fit(df[list(features)].to_numpy(), df[target])
        predictions = reg.predict(df[list(features)].to_numpy())
        score = reg.score(df[list(features)].to_numpy(), df[target])
        rmse = np.sqrt(np.median((predictions - df[target])**2))

        # Print results for each feature set
        print(f'Feature(s) {features}')
        print(f'RMSE: {rmse:.8f}')
        print(f'Score: {score:.3f}')
        print(f"Coefficients: {reg.coef_}", f"Intercept: {reg.intercept_}\n")
                
        # Update best regressor based on R² score
        if score > best_score:
            best_score = score
            best_regressor = reg
            best_features = features
    
    # Return the best regressor and its features
    return best_regressor, best_features


def stochastic_poly(df, feature_columns, Y, n=3, iters=100, round_n=3):
    """
    Perform polynomial regression using stochastic sampling for multiple degrees and iterations to find the best model.

    Parameters:
    - df: DataFrame, the input dataset containing feature columns.
    - feature_columns: list of str, the feature columns to be used for regression.
    - Y: array-like, the target variable for regression.
    - n: int, the maximum polynomial degree to evaluate (default is 3).
    - iters: int, the number of iterations for stochastic sampling (default is 100).
    - round_n: int, the number of decimal places to round the final results (default is 3).

    Returns:
    - tuple: 
        - best_n: int, the best degree of the polynomial model based on test R² scores.
        - median R² score for the best polynomial degree on the test data.
        - median R² score for the best polynomial degree on the train data.
        - median RMSE for the best polynomial degree on the test data.
        - median RMSE for the best polynomial degree on the train data.
    """
    ypred_train_best, ypred_test_best, R2_train_t_best, R2_test_t_best, RMSE_train_t_best, RMSE_test_t_best = [], [], [], [], [], []
    X = df[feature_columns]

    for i in range(iters):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
        LinReg = LinearRegression()
        ypred_train_, ypred_test_, R2_train_t_, R2_test_t_, RMSE_train_t_, RMSE_test_t_ = [], [], [], [], [], []

        for k in range(n):
            poly = PolynomialFeatures(degree=k)
            poly.fit(X_train)
            Xt_train = poly.transform(X_train)
            Xt_test = poly.transform(X_test)

            LinReg.fit(Xt_train, y_train)
            ypred_train = LinReg.predict(Xt_train)
            ypred_test = LinReg.predict(Xt_test)

            R2_train_t = r2_score(y_train, ypred_train)
            R2_test_t = r2_score(y_test, ypred_test)
            RMSE_train_t = RMSE(y_train, ypred_train)
            RMSE_test_t = RMSE(y_test, ypred_test)

            ypred_train_.append(ypred_train)
            ypred_test_.append(ypred_test)
            R2_train_t_.append(R2_train_t)
            R2_test_t_.append(R2_test_t)
            RMSE_train_t_.append(RMSE_train_t)
            RMSE_test_t_.append(RMSE_test_t)

        ypred_train_best.append(ypred_train_)
        ypred_test_best.append(ypred_test_)
        R2_train_t_best.append(R2_train_t_)
        R2_test_t_best.append(R2_test_t_)
        RMSE_train_t_best.append(RMSE_train_t_)
        RMSE_test_t_best.append(RMSE_test_t_)

    r2_test_n1 = [inner_list[0] for inner_list in R2_test_t_best]
    r2_test_n2 = [inner_list[1] for inner_list in R2_test_t_best]
    r2_test_n3 = [inner_list[2] for inner_list in R2_test_t_best]
#    r2_test_n4 = [inner_list[3] for inner_list in R2_test_t_best]
    
#    r2_medians = [np.median(r2_test_n1), np.median(r2_test_n2), np.median(r2_test_n3), np.median(r2_test_n4)]
    r2_medians = [np.median(r2_test_n1), np.median(r2_test_n2), np.median(r2_test_n3)]

    best_n = r2_medians.index(np.max(r2_medians))
    return best_n, round(np.median([inner_list[best_n] for inner_list in R2_test_t_best]), round_n), round(np.median([inner_list[best_n] for inner_list in R2_train_t_best]), round_n), round(np.median([inner_list[best_n] for inner_list in RMSE_test_t_best]), round_n), round(np.median([inner_list[best_n] for inner_list in RMSE_train_t_best]), round_n)                                             