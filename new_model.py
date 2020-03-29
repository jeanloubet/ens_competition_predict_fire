import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import Counter
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


def load_file():
    # Data reading
    x_train = pd.read_csv('x_train.csv', sep=',')
    y_train = pd.read_csv('y_train.csv', sep=',')
    x_test = pd.read_csv('x_test.csv', sep=',')
    return x_train, y_train, x_test




def traitement_x(df):
    # ajust types
    df = df[["floor", "longitude intervention", "latitude intervention", "longitude before departure",
             "latitude before departure", "OSRM estimated distance", "OSRM estimated duration"]]
    df["floor"] = df["floor"].astype(float)
    df["longitude intervention"] = df["longitude intervention"].astype(float)
    df["latitude intervention"] = df["latitude intervention"].astype(float)
    df["longitude before departure"] = df["longitude before departure"].astype(float)
    df["latitude before departure"] = df["latitude before departure"].astype(float)
    df["OSRM estimated distance"] = df["OSRM estimated distance"].astype(float)
    df["OSRM estimated duration"] = df["OSRM estimated duration"].astype(float)

    return df


def traitement_y(df):
    reference_for_code = pd.factorize(df["emergency vehicle selection"])
    df["emergency vehicle selection"] = pd.factorize(df["emergency vehicle selection"])[0]
    df["delta selection-departure"] = df["delta selection-departure"].astype(float)
    df["delta departure-presentation"] = df["delta departure-presentation"].astype(float)
    df["delta selection-presentation"] = df["delta selection-presentation"].astype(float)
    return df, reference_for_code


def create_model_regression(x_train, y_train, model_name: str, objective_model: str, extra=""):
    # selection des meilleurs parametres du model
    """
    parameters = {
    'objective':['reg:linear'],
     'max_depth':[10,12,14],
     'min_child_weight':[2,3],
     'learning_rate':[0.1,0],
     'n_estimators':range(200,400,50)
    }
    xgb1 = XGBRegressor()
    xgb_grid = GridSearchCV(xgb1,parameters,scoring='r2', cv = 5,n_jobs = 5,verbose=True)

    xgb_grid.fit(x_train,y_train)
    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)


    param_final =xgb_grid.best_params_
    """
    data_dmatrix = xgb.DMatrix(data=x_train, label=y_train)
    params = {
        'objective': objective_model,
        'max_depth': 12,
        'min_child_weight': 2,
        'learning_rate': 0.1,
        'n_estimators': 300,
    }

    params.update(extra)

    xg_reg2 = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=200)  # params=param_final,
    ##sauvegarde du model
    joblib.dump(xg_reg2, 'model/' + model_name + '.pkl')

    return xg_reg2


if __name__ == '__main__':
    ##lecture fichiers
    x_train, y_train, x_test = load_file()
    # traitement
    x_train_treated = traitement_x(x_train)
    x_test_treated = traitement_x(x_test)
    y_train_treated, reference = traitement_y(y_train)

    nb_classes = len(Counter(reference[0])) - 1

    model0 = create_model_regression(x_train_treated, y_train_treated["emergency vehicle selection"],
                                     'emergency vehicle selection', 'multi:softmax', extra={'num_class': nb_classes})
    model1 = create_model_regression(x_train_treated, y_train_treated["delta selection-departure"],
                                     "delta selection-departure", 'reg:linear')
    model2 = create_model_regression(x_train_treated, y_train_treated["delta departure-presentation"],
                                     "delta departure-presentation", 'reg:linear')
    model3 = create_model_regression(x_train_treated, y_train_treated["delta selection-presentation"],
                                     "delta selection-presentation", 'reg:linear')

    ## Submission
    submission = pd.concat([pd.DataFrame(x_test[['emergency vehicle selection']].values), \
                            pd.DataFrame(np.full((len(x_test), 1), y_train['delta selection-departure'].median())), \
                            pd.DataFrame(model.predict(x_test_transit_poly)), \
                            pd.DataFrame(y_selection_presentation_predicted)], \
                           axis=1)

    submission.columns = list(y_train.columns.values)

    submission.set_index('emergency vehicle selection', inplace=True)

    submission.to_csv('./submission.csv', sep=",")
