import numpy as np

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def RMSE(predictions, targets):
    """
    Compute the Root Mean Square Error.

    Parameters:
    - predictions: array-like, predicted values
    - targets: array-like, true values

    Returns:
    - RMSE value
    """
    differences = np.array(predictions) - np.array(targets)
    return np.sqrt(np.mean(differences**2))


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df


# Function to perform linear regression and print results
def perform_regression_and_select_best(feature_sets, target, df):
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


# Generalize the meshgrid and Z values generation
def generate_meshgrid_and_Z(reg, df, features, n_points=50):
    x_range = np.linspace(df[features[0]].min(), df[features[0]].max(), n_points)
    y_range = np.linspace(df[features[1]].min(), df[features[1]].max(), n_points)
    X, Y = np.meshgrid(x_range, y_range)
    Z = reg.coef_[0] * X + reg.coef_[1] * Y + reg.intercept_
    return X, Y, Z


def stochastic_poly(df, feature_columns, Y, n=3, iters=100, round_n=3):
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