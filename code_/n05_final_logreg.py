# Setup
import pandas as pd

pd.options.display.max_columns = 30
pd.options.display.width = 120

import pickle
from collections import defaultdict
from termcolor import colored
import statistics

# ML setup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

# Model scoring
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score  # Validation

# ML models
from sklearn.linear_model import LogisticRegression

# Tuning
from sklearn.model_selection import GridSearchCV

# -------------------------------------------------------------------------------------------------------------------
# Import data
# -------------------------------------------------------------------------------------------------------------------
from code_.required_files import ncpa_stopped_or_didnt_fill_file

ncpa_data = pd.read_csv(ncpa_stopped_or_didnt_fill_file)

# Separate target and features
X = ncpa_data.iloc[:, 1:]  # Features
y = ncpa_data.iloc[:, 0]  # Target (adherence)

le = LabelEncoder()
y = le.fit_transform(y)  # ********* 0 = adherence, 1 = non-adherence


# -------------------------------------------------------------------------------------------------------------------
# Pre-processing
# -------------------------------------------------------------------------------------------------------------------

# 1.  Function to impute some missing values. ---------------------------------------------------------------
# Some columns have 'no reponse' to the survey (not a lot).
label_dict = defaultdict(LabelEncoder)
print__ = True  # code switch for testing


def impute_missing_feat_vals(X, col_to_impute, value_of_missing_data, print_=print__):
    """
    Purpose: Impute missing values, allowing selection of what the value of missing data is, and print custom views.
             Accepts columns with categorical/string values. Method:
             LabelEncoder() -> numerics -> impute numerics -> reverse LabelEncoder() back to strings

    params::
    value_of_missing_data:
            Ex.: for 'age', 'No reponse' to the age question = 99.0 --> value_of_missing_data='99.0'
                for 'ownhome', value of missing data is 'Unknown/Refused' --> value_of_missing_data= 'Unknown/Refused'
                Either a number or string is accepted.

    Input: Choose a dataframe, a column, and a missing value.  Imputes missing values across the column via KNN.

    Output: A single column. Use the function to replace the column that's being imputed on.
            Ex.: X['ownhome'] = impute_missing_feat_vals(X, col_to_impute='ownhome', value_of ..............

    Custom views ------------------
    If print_ = True, the function prints out, in this order:
    The original dataframe, numeric-encoded dataframe, imputed numeric dataframe, and inverse encoded dataframe
    so that we can see what the function did.
    **Note: all of the columns will be imputed in the printed results.  Can ignore this.  Only the selected column
    (col_to_impute) is returned and can be replaced into the desired dataframe.
    """

    # Choose a view based on the value to impute. Find a row that contains a value_of_missing_data. If print=True,
    # select and print the 5 rows and 4 columns surrounding the missing value.
    rowstolookat = X.index[X[col_to_impute] == value_of_missing_data].tolist()[0]
    colstolookat = X.columns.get_loc(col_to_impute)

    if print_ == True:
        print('Original dataframe:')
        print(X.iloc[rowstolookat - 2:rowstolookat + 3, colstolookat - 1:colstolookat + 3], '\n')

    # Optional: Uncomment to always use the least abundant value from df[column].value_counts() as the missing val.
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

    # Label Encoding
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
X['emply'] = impute_missing_feat_vals(X, col_to_impute='emply', value_of_missing_data='999',
                                      print_=print__)  # 6 Other/Refused
X['age'] = impute_missing_feat_vals(X, col_to_impute='age', value_of_missing_data=99.0,
                                    print_=print__)  # 32 missing
X['first_got_rx'] = impute_missing_feat_vals(X, col_to_impute='first_got_rx', value_of_missing_data='9.0',
                                             print_=print__)  # 3 missing
X['understand_health_prob'] = impute_missing_feat_vals(X, col_to_impute='understand_health_prob',
                                                       value_of_missing_data='9.0',
                                                       print_=print__)  # 3 missing
X['mstatus'] = impute_missing_feat_vals(X, col_to_impute='mstatus', value_of_missing_data='9',
                                        print_=print__)  # 6 'Refused', 4 of these non-adherence
X['ownhome'] = impute_missing_feat_vals(X, col_to_impute='ownhome', value_of_missing_data='Unknown/Refused',
                                        print_=print__)  # 10 'Unknown/Refused'

# 2. Drop some columns -----------------------------------------------------------------------------------------------
X = X.drop(columns=['US_region', 'mstatus', 'ownhome', 'have_health_insur', 'have_medicare', 'metro', 'race'])


# Save X before processing further - need this for the app
# X.to_csv(r'C:\Users\Kelly\Documents\Python\Insight_AdhereID\output_files\X2.csv')

# 3. Custom ordinal encoder -------------------------------------------------------------------------------------------
def encode_ordinals(df):
    """
    Purpose: Encode the ordinal features explicitly, for sanity's sake. Not all of the features are 1(lowest) to 5(
    highest), some are the reverse.
    """
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
        'Very easy': 1,
        'Somewhat easy': 2,
        'Somewhat difficult': 3,
        'Very difficult': 4
    }
    df['can_afford_rx'].replace(can_afford_rx_vals, inplace=True)

    # Understand health prob
    understand_health_prob_vals = {
        'A great deal': 4,
        'Somewhat': 3,
        'Not so much': 2,
        'Not at all': 1,
        'Unknown/Refused': 0,
    }
    df['understand_health_prob'].replace(understand_health_prob_vals, inplace=True)

    # Education
    educ_vals = {
        'Less than high school': 1,
        'High school': 2,
        'Some college': 3,
        'Technical school/other': 4,
        'College graduate': 5,
        'Graduate school or more': 6,
    }
    df['educ'].replace(educ_vals, inplace=True)

    df = df.reset_index(drop=True)

    return df


# 4. Pipeline for further processing -------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def process_data(X):
    """
    Purpose: Dataframe has continuous numeric, ordinal and categorical features.
    This function will scale the numeric features, encode the ordinal features (using the custom ordinal encoder above),
    and get dummies/OneHotEncode the categorical features, all in one pipeline.

    Input: dataframe X (preferably without any missing values)
    Output: processed dataframe
    """
    numeric_cols = ['age', 'n_prescriptions', 'n_provider_visits']
    ordinal_cols = ['first_got_rx', 'income', 'general_health', 'med_burden', 'can_afford_rx', 'understand_health_prob',
                    'educ', ]
    categorical_cols = ['sex', # Commented out variables were dropped, no impact on model predictions
                        # 'have_health_insur','have_medicare', 'metro',
                        'parent',
                        # 'US_region', 'ownhome', 'mstatus',
                        'emply',
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

    processed_df = pd.DataFrame(pipeline.fit_transform(X), columns=columns_[0])

    return processed_df


# Process data
X_processed = process_data(X=X)

# -------------------------------------------------------------------------------------------------------------------
# Logistic Regression Model - pipeline
# -------------------------------------------------------------------------------------------------------------------

from imblearn.pipeline import Pipeline
import eli5

class Model:
    """
    Purpose: On X and y, train_test_split, fit to pipeline, make predictions, and cross validate.  Putting all of these
    in a class instead of separate functions makes it easier to use in the plots in file n07_model_performance_plots.py.

    Initialize the class first with x = Model(X, y, model name), then can do:
        x.split_() : get training and test set variables out
        x.return_model() : get the model itself out (for saving to file)
        x.predict_on_test() : get predictions out for scoring and plots
        x.cross_val(): print cross-validations scores (recall and f-beta score)
    """
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
        self.pipeline_ = \
            Pipeline(
                [('scale', StandardScaler(copy=True, with_mean=True, with_std=True)),
                 ('balance', SMOTE(k_neighbors=5, n_jobs=None, random_state=33, sampling_strategy='auto')),
                 ('logistic_regression', model)],
                verbose=True
            )

    def split_(self):
        return train_test_split(self.X, self.y, test_size=0.20, random_state=42, stratify=y)

    def return_model(self):
        """
        Train model and return the model (for saving to file)
        """
        X_train, X_test, y_train, y_test = Model.split_(self)
        trained_model = self.pipeline_.fit(X_train, y_train)
        return trained_model

    def predict_on_test(self):
        """
        Get model from the function above, predict on X_test
        Returns: predictions table
        """
        X_train, X_test, y_train, y_test = Model.split_(self)
        # predictions = self.pipeline_.predict(X_test)
        tr_predictions = Model.return_model(self).predict(X_train)
        predictions = Model.return_model(self).predict(X_test)

        # Scoring
        print(eli5.format_as_text(
            eli5.explain_weights_sklearn(Model.return_model(self).named_steps['logistic_regression'],
                                 feature_names=X_processed.columns.to_list())))
        print(classification_report(y_true=y_test, y_pred=predictions))
        print("train  ",classification_report(y_true=y_train, y_pred=tr_predictions))
        print('F-beta score: ', fbeta_score(y_true=y_test, y_pred=predictions, beta=5), '\n')
        print(pd.crosstab(y_test, predictions, rownames=['Actual adherence'], colnames=['Predicted adherence']))
        print('\n')
        print('Overall recall score: ', recall_score(y_true=y_test, y_pred=predictions))
        return predictions

    def cross_val(self):
        """
        Print cross-validation scores - means of recall score and f-beta across all k-folds
        """
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        scoring = {'recall': 'recall',
                   'f-beta': make_scorer(fbeta_score, beta=4)}
        scores = cross_validate(Model.return_model(self),
                       X_processed, y,
                       scoring=scoring,
                       cv=cv,
                       return_train_score=True,
                       )
        print('\n')
        print('scoring', ' ' * 12, 'score value', '\n', '-'*34)
        for score, key in zip(scores, scores.keys()):
            print(key,
                  ' '*(18-len(key)),
                  round(statistics.mean(scores[score]), 3),
                  ' '*(4-len(str(scores[score].mean().round(3)))),
                  '+/-',
                  round(statistics.stdev(scores[score]), 2))



# Initialize instance of Model class using tuned logistic regression model
pipeline = Model(X=X_processed, y=y, model=LogisticRegression(C=0.1,
                                                              class_weight={0: 0.5, 1: 0.5},
                                                              dual=False, fit_intercept=True,
                                                              intercept_scaling=1, l1_ratio=None,
                                                              max_iter=3000, multi_class='auto',
                                                              n_jobs=None, penalty='l1', random_state=42,
                                                              solver='liblinear', tol=0.0001, verbose=0,
                                                              warm_start=False)
                 )

# Return the trained model
lr_model = pipeline.return_model()

# Get predictions on test set
predictions = pipeline.predict_on_test()

# Print cross-validation scores
pipeline.cross_val()

# Save model to file
modelfile = r'code_/lr_model.sav'
with open(modelfile, 'wb') as f:
    pickle.dump(lr_model, f)

'''
Best scores: 

 precision    recall  f1-score   support
           0       0.82      0.61      0.70       135
           1       0.42      0.68      0.52        57
    accuracy                           0.63       192
   macro avg       0.62      0.65      0.61       192
weighted avg       0.70      0.63      0.65       192
Predicted adherence   0   1
Actual adherence           
0                    82  53
1                    18  39

'''


# -------------------------------------------------------------------------------------------------------------------
# Hyperparameter Tuning
# -------------------------------------------------------------------------------------------------------------------

# Hyperparameter tuning the logistic regression model with GridSearchCV
def hyperparam_tune(X, y, model):
    # Scale
    s = StandardScaler()
    X = s.fit_transform(X)

    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    # Logistic Regression
    param_grid_LR = {
        'logistic_regression__penalty': ['l2', 'l1'],
        'logistic_regression__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'logistic_regression__max_iter': [3000],
        'logistic_regression__class_weight': [{1: 0.5, 0: 0.5}, {1: 0.9, 0: 0.1}, {1: 0.8, 0: 0.2},
                                              {1: 0.95, 0: 0.05}, 'balanced'],
        'logistic_regression__solver': ['lbfgs', 'liblinear', 'sag', 'saga']
    }

    gs = GridSearchCV(model, param_grid=param_grid_LR, cv=kfold, n_jobs=-1, scoring='recall', verbose=True)
    gs.fit(X_train, y_train)
    print('Best score: ', gs.best_score_)
    print('Best params: ', gs.best_params_)
    print('Best estimator: ', gs.best_estimator_)


# hyperparam_tune(X_processed, y, lr_model)


# -------------------------------------------------------------------------------------------------------------------
# Compare model performance on subsets of the data
# -------------------------------------------------------------------------------------------------------------------

# All data
# X_processed_1 = X_processed.loc[:, ['age', 'n_prescriptions', 'n_provider_visits', 'first_got_rx', 'income',
#                                     'general_health', 'med_burden', 'can_afford_rx', 'understand_health_prob',
#                                     'educ', 'sex_M', 'parent_Parent',
#                                     'emply_Full-time', 'emply_Homemaker', 'emply_Other/Refused',
#                                     'emply_Part-time', 'emply_Retired', 'emply_Student', 'emply_Temp unemployed',
#                                     'has_diabetes_Yes',
#                                     'has_hyperten_Yes', 'has_asthma_etc_Yes', 'has_heart_condition_Yes',
#                                     'has_hi_cholesterol_Yes']]
'''
Dummy: ROC AUC=0.466

LR: ROC AUC=0.635
RF: ROC AUC=0.630
'''

# Demographic info only
# X_processed_2 = X_processed.loc[:, ['age', 'income', 'educ', 'sex_M', 'parent_Parent',
#
#                                     'emply_Full-time', 'emply_Homemaker', 'emply_Other/Refused', 'emply_Part-time',
#                                     'emply_Retired', 'emply_Student', 'emply_Temp unemployed']]
'''
Dummy: ROC AUC=0.466
LR: ROC AUC=0.650
RF: ROC AUC=0.593
'''

# Health info only
# X_processed_3 = X_processed.loc[:, ['age', 'n_prescriptions', 'n_provider_visits', 'first_got_rx',
#                                     'general_health', 'med_burden', 'can_afford_rx', 'understand_health_prob',
#                                     'has_diabetes_Yes', 'has_hyperten_Yes', 'has_asthma_etc_Yes',
#                                     'has_heart_condition_Yes',
#                                     'has_hi_cholesterol_Yes']]
'''
Dummy: ROC AUC=0.466
LR: ROC AUC=0.633
RF: ROC AUC=0.657
'''
