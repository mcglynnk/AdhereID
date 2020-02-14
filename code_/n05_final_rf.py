# Setup
import pandas as pd
pd.options.display.max_columns = 14
pd.options.display.width = 200

import numpy as np

import pickle
from collections import defaultdict, OrderedDict
import statistics
from termcolor import colored

# ML setup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN

# Model scoring
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold # Validation

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier # Random chance classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# Feature selection
from sklearn.linear_model import Ridge, RidgeClassifier, Lasso
from sklearn.feature_selection import RFE

# Tuning
from sklearn.model_selection import GridSearchCV


# SWITCHES
impute_feats=True
drop_cols=True
get_dummies=True
drop_baddummies=True
drop_bad_feats = True
scale=True


# -------------------------------------------------------------------------------------------------------------------
# Import data
from code_.required_files import ncpa_stopped_or_didnt_fill_file
ncpa_data = pd.read_csv(ncpa_stopped_or_didnt_fill_file)

# Separate target and features
X = ncpa_data.iloc[:,1:] # Features
y = ncpa_data.iloc[:,0] # Target (adherence)

le = LabelEncoder()
y = le.fit_transform(y) # ********* 0 = adherence, 1 = non-adherence

# -------------------------------------------------------------------------------------------------------------------
# Impute some missing values; some columns have 'no reponse' to the survey (not a lot)
label_dict = defaultdict(LabelEncoder)
print__ = False # code switch for testing
def impute_missing_feat_vals(X, col_to_impute, value_of_missing_data, print_=print__):
    '''
    Choose a dataframe, a column, and a missing value.  Imputes missing values across the column.
    If print_ = True, the function prints out, in this order:
    The original dataframe, numeric-encoded dataframe, imputed numeric dataframe, and inverse encoded dataframe
    so that we can see what the function did.

    **Note: all of the columns will be imputed in the printed results.  Can ignore this.  Only the selected column
    (col_to_impute) is returned and can be replaced into the desired dataframe.
    '''
    if print_ == True:
        print('Original dataframe:')
    rowstolookat = X.index[X[col_to_impute] == value_of_missing_data].tolist()[0]
    colstolookat = X.columns.get_loc(col_to_impute)

    if print_ == True:
        print(X.iloc[rowstolookat - 2:rowstolookat + 3, colstolookat - 1:colstolookat + 3], '\n')

    # Optional:
    # To always use the least abundant value from value_counts() as the missing val:
    # missing_val = list(X[col_to_impute].value_counts().to_dict().keys())[len(X[col_to_impute].value_counts()) - 1]
    # print('Least abundant class is {}. Imputing...'.format(missing_val))
    # # Encoding
    # X = X.apply(lambda x: label_dict[x.name].fit_transform(x))
    #
    # # Get a dict (and dataframe) of what the classes in this column were encoded into
    # encoded_labels = list(
    #     zip(label_dict[col_to_impute].classes_,label_dict[col_to_impute].transform(label_dict[col_to_impute].classes_))
    # )
    # encoded_labels_df = pd.DataFrame(encoded_labels, columns=['name','encoded'])
    # # Get the encoding label for the least abundant class
    # encoding_chosen = encoded_labels_df[encoded_labels_df['name']==missing_val].iloc[:,1]
    # print('{} was encoded into {}'.format(missing_val, encoding_chosen))

    # To specify the missing value:
    # Encoding
    X = X.apply(lambda x: label_dict[x.name].fit_transform(x))

    encoded_labels = list(
        zip(label_dict[col_to_impute].classes_, label_dict[col_to_impute].transform(label_dict[col_to_impute].classes_))
    )
    encoded_labels_df = pd.DataFrame(encoded_labels, columns=['name', 'encoded'])
    # Get the encoding label for the chosen missing value
    encoding_chosen = encoded_labels_df[encoded_labels_df['name'] == value_of_missing_data].iloc[:, 1]
    if print_ == True:
        print(colored('{} was encoded into {}'.format(value_of_missing_data, encoding_chosen), 'red'), '\n')

        print('Encoded:')
        print(X.iloc[rowstolookat - 2:rowstolookat + 3, colstolookat - 1:colstolookat + 3], '\n')

    # KNN Impute
    imputer = KNNImputer(n_neighbors=4, missing_values=float(encoding_chosen))
    X = pd.DataFrame(
        imputer.fit_transform(X).astype('int'),
        columns=label_dict
    )
    if print_ == True:
        print('Imputed:')
        print(X.iloc[rowstolookat - 2:rowstolookat + 3, colstolookat - 1:colstolookat + 3], '\n')

    # Reverse encode back into category labels
    X = X.apply(lambda x: label_dict[x.name].inverse_transform(x))
    if print_ == True:
        print('Inverse encoded:')
        print(X.iloc[rowstolookat - 2:rowstolookat + 3, colstolookat - 1:colstolookat + 3], '\n')

    return X[col_to_impute]

# Run function to impute values on the following columns
X['emply'] = impute_missing_feat_vals(X, col_to_impute='emply', value_of_missing_data='9.0',
                                      print_=print__) # 1 # missing
X['age'] = impute_missing_feat_vals(X, col_to_impute='age', value_of_missing_data=99,
                                    print_=print__) # 32 missing
X['first_got_rx'] = impute_missing_feat_vals(X, col_to_impute='first_got_rx', value_of_missing_data='9.0',
                                             print_=print__) # 3 missing
X['understand_health_prob'] = impute_missing_feat_vals(X, col_to_impute='understand_health_prob', value_of_missing_data='9.0',
                                                       print_=print__) # 3 missing
X['mstatus'] = impute_missing_feat_vals(X, col_to_impute='mstatus', value_of_missing_data='9',
                                        print_=print__)  # 6 'Refused', 4 of these non-adherence
X['ownhome'] = impute_missing_feat_vals(X, col_to_impute='ownhome', value_of_missing_data='Unknown/Refused',
                                        print_=print__)  # 10 'Unknown/Refused'


# The variance of some of these after making dummies is very low.
if drop_cols==True:
    X = X.drop(columns=['US_region',
                        'parent',
                        'emply',
                        'mstatus',
                        'ownhome',
                         #'age',
                        'race'])



# Custom ordinal encoder - convert ordinal features to numbers/levels
def encode_ordinals(df):
    #df = pd.DataFrame(df)
    # First started taking an rx on a regular basis
    first_started_taking_vals = {
        'Within the past year': 1,
        '1 to 2 years ago': 2,
        '3 to 5 years ago': 3,
        '6 to 10 years ago': 4,
        'More than 10 years ago': 5
    }
    df['first_got_rx'].replace(first_started_taking_vals, inplace=True)

    # General health
    gen_val = {
        'Excellent': 5,
        'Very good': 4,
        'Good': 3,
        'Fair': 2,
        'Poor': 1
    }
    df['general_health'].replace(gen_val, inplace=True)

    # Income
    income_vals = {
        'No response/Unknown': 0,
        '<$50k': 1,
        '$50k-75k': 2,
        '>$100k': 3
    }
    df['income'].replace(income_vals, inplace=True)

    # Med burden
    med_burden_vals = {
        'Very simple': 1,
        'Somewhat simple': 2,
        'Somewhat complicated': 3,
        'Very complicated': 4
    }
    df['med_burden'].replace(med_burden_vals, inplace=True)

    # Can afford rx
    can_afford_rx_vals = {
        'Very easy':1,
        'Somewhat easy':2,
        'Somewhat difficult':3,
        'Very difficult':4
    }
    df['can_afford_rx'].replace(can_afford_rx_vals, inplace=True)

    # Understand health prob
    understand_health_prob_vals = {
        'A great deal':4,
        'Somewhat': 3,
        'Not so much':2,
        'Not at all':1,
        'Unknown/Refused':0,
    }
    df['understand_health_prob'].replace(understand_health_prob_vals, inplace=True)

    # Education
    educ_vals = {
        'Less than high school':1,
        'High school':2,
        'Some college':3,
        'Technical school/other':4,
        'College graduate':5,
        'Graduate school or more':6,
    }
    df['educ'].replace(educ_vals, inplace=True)

    df = df.reset_index(drop=True)

    return df

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def process_data(X):
    allcols = X.columns
    numeric_cols = ['age','n_prescriptions', 'n_provider_visits']
    ordinal_cols = ['first_got_rx', 'income', 'general_health', 'med_burden', 'can_afford_rx', 'understand_health_prob',
             'educ',]
    categorical_cols = ['sex', 'have_health_insur','have_medicare', 'metro',
                                  #'ownhome',
                                  #'mstatus',
                                  #'emply',
                                  'has_diabetes', 'has_hyperten', 'has_asthma_etc',
                          'has_heart_condition', 'has_hi_cholesterol']

    pipeline = Pipeline([
        ('scale_numeric', ColumnTransformer(
            [('scale', StandardScaler(), numeric_cols),
             ('encode_ord', FunctionTransformer(encode_ordinals), ordinal_cols),
             ('get dummies', OneHotEncoder(drop='first'), categorical_cols),
            ],
            remainder='passthrough'
        ))
    ])

    dummy_col_names = [str(i) for i in pd.get_dummies(X[categorical_cols], drop_first=True).columns]
    columns_ = [numeric_cols + ordinal_cols + dummy_col_names]

    return pd.DataFrame(pipeline.fit_transform(X), columns=columns_[0])

# Process data
X_processed = process_data(X=X)


from imblearn.pipeline import Pipeline

def model_pipeline(X, y, model):
    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    pipeline = Pipeline(
        [('scale', StandardScaler(copy=True, with_mean=True, with_std=True)),
         ('balance', SMOTE(k_neighbors=5, n_jobs=None, random_state=33, sampling_strategy='auto')),
         ('random_forest', model )],
        verbose=True
    )
    #     [
    #     ('scale', StandardScaler()),
    #     ('balance', SMOTE(random_state=33)),
    #     ('logistic_regression', model )
    # ], verbose=True)

    lr_model = pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=predictions))
    print(pd.crosstab(y_test, predictions, rownames=['Actual adherence'], colnames=['Predicted adherence']))
    print('\n')
    print(recall_score(y_true=y_test, y_pred=predictions))

    return lr_model

# Initial logistic regression, before tuning
# rf_model = model_pipeline(X_processed,y,
#                           model = RandomForestClassifier(random_state=42))

# Tuned model
# rf_model = model_pipeline(X_processed,y,
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
#                                         class_weight=None, criterion='gini',
#                                         max_depth=3, max_features='auto',
#                                         max_leaf_nodes=None, max_samples=None,
#                                         min_impurity_decrease=0.0,
#                                         min_impurity_split=None,
#                                         min_samples_leaf=1, min_samples_split=4,
#                                         min_weight_fraction_leaf=0.0,
#                                         n_estimators=1, n_jobs=None,
#                                         oob_score=False, random_state=42,
#                                         verbose=0, warm_start=False)
#                           )


# Tuned model 2
rf_model = model_pipeline(X_processed,y,
                          RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                                 criterion='gini', max_depth=4, max_features='auto',
                                                 max_leaf_nodes=None, max_samples=None,
                                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                                 min_samples_leaf=3, min_samples_split=4,
                                                 min_weight_fraction_leaf=0.0, n_estimators=91,
                                                 n_jobs=None, oob_score=False, random_state=42, verbose=0,
                                                 warm_start=False)
                          )

##

def hyperparam_tune(model):
    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.20, random_state=42, stratify=y)

    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    # Random Forest
    param_grid_RF = {
        'random_forest__max_depth': np.arange(2, 5),
        'random_forest__min_samples_split': np.arange(4, 7),
        'random_forest__min_samples_leaf': np.arange(1, 5),
        'random_forest__n_estimators': range(1, 100, 10),
                     }

    gs = GridSearchCV(model, param_grid=param_grid_RF, cv=kfold, n_jobs=-1, scoring='recall', verbose=True)
    gs.fit(X_train, y_train)
    print('Best score: ', gs.best_score_)
    print('Best params: ', gs.best_params_)
    #print('cv results: ', gs.cv_results_)
    print('Best estimator: ', gs.best_estimator_)

hyperparam_tune(rf_model)

'''
Best score:  0.6946640316205533
Best params:  {'random_forest__max_depth': 3, 'random_forest__min_samples_leaf': 1, 'random_forest__min_samples_split': 4, 'random_forest__n_estimators': 1}
Best estimator:  Pipeline(memory=None,
         steps=[('scale',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('balance',
                 SMOTE(k_neighbors=5, n_jobs=None, random_state=33,
                       sampling_strategy='auto')),
                ('random_forest',
                 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                        class_weight=None, criterion='gini',
                                        max_depth=3, max_features='auto',
                                        max_leaf_nodes=None, max_samples=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=4,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=1, n_jobs=None,
                                        oob_score=False, random_state=42,
                                        verbose=0, warm_start=False))],
         verbose=True)
'''



## ------------------------------------------------------------------------------------------------------------------
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV


def tune(model, grid):
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    gs = GridSearchCV(estimator=model(random_state=42), param_grid=grid, cv=kfold, n_jobs=-1, verbose=True)
    gs.fit(X_train, y_train)
    print('Best score: ', gs.best_score_)
    print('Best params: ', gs.best_params_)
    print('cv results: ', gs.cv_results_)
    print('Best estimator: ', gs.best_estimator_)

# Logistic Regression
param_grid_LR = {
    'penalty': ['l2', 'l1'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'max_iter': [3000],
    'class_weight': [{1: 0.5, 0: 0.5}, {1: 0.9, 0: 0.1}, {1: 0.8, 0: 0.2},
                     {1: 0.95, 0: 0.05}, 'balanced'],
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
}

tune(LogisticRegression, param_grid_LR)


# Random Forest
param_grid_RF = {'max_depth': np.arange(2, 5),
                 'min_samples_split': np.arange(4, 7),
                 'min_samples_leaf': np.arange(1, 5),
                 'n_estimators': range(1, 100, 10),
                 }
# tune(RandomForestClassifier, param_grid_RF)

'''
Best params:  {'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 91}

Best estimator:  RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=4, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=91,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
'''