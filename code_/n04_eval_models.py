# Setup
import pandas as pd
pd.options.display.max_columns = 14
pd.options.display.width = 200
import numpy as np

from collections import OrderedDict
import statistics

# ML setup
from imblearn.over_sampling import SMOTE

# Model scoring
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import StratifiedKFold  # Validation

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier  # Random chance classifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

# Feature selection
from sklearn.feature_selection import RFE

# -------------------------------------------------------------------------------------------------------------------
# Import target and features
from code_.n05_final_logreg import X_processed, y

# -------------------------------------------------------------------------------------------------------------------
# Check Multiple Models
models = []

def append_models():
    models.append(('Dummy Classifier', DummyClassifier(random_state=2)))

    models.append(('LogisticRegression base', LogisticRegression(random_state=42, max_iter=2000)))

    models.append(('LogisticRegression OPTIMIZED', LogisticRegression(C=0.1, class_weight={0: 0.5, 1: 0.5}, dual=False,
                                                                      fit_intercept=True, intercept_scaling=1,
                                                                      l1_ratio=None,
                                                                      max_iter=3000, multi_class='auto', n_jobs=None,
                                                                      penalty='l1',
                                                                      random_state=42, solver='saga', tol=0.0001,
                                                                      verbose=0,
                                                                      warm_start=False)))

    models.append(('Random Forest base', RandomForestClassifier(random_state=42)))

    models.append(("Random Forest OPTIMIZED", RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                                                     criterion='gini', max_depth=4, max_features='auto',
                                                                     max_leaf_nodes=None, max_samples=None,
                                                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                                                     min_samples_leaf=3, min_samples_split=4,
                                                                     min_weight_fraction_leaf=0.0, n_estimators=91,
                                                                     n_jobs=None, oob_score=False, random_state=42,
                                                                     verbose=0,
                                                                     warm_start=False)))

    models.append(('Extra Trees', ExtraTreesClassifier(random_state=42)))

    models.append(('LightGBM', LGBMClassifier(random_state=42)))


append_models()

model_names = [i[0] for i in models]

# -------------------------------------------------------------------------------------------------------------------
# Test models with k-fold cross-validation
# -------------------------------------------------------------------------------------------------------------------
model_stats = []

def eval_metrics(X, y, show_conf_matrix=True, show_feat_importance=True):
    '''
    Evaluate multiple models at once.  For each model:
        - cross-validation (10-fold)
            - SMOTE to balance training set within each fold
            - score each fold and add to an array containing scores for all 10 folds
            - print a classification report based on the mean of the scores across all 10 folds
        (optional):
        - print a confusion matrix
        - print RFE feature importance

    Do SMOTE within each kfold, rather than SMOTE before kfold
    https://stackoverflow.com/questions/37124407/scikit-learn-cross-validation-classification-report
    https://www.researchgate.net/post/should_oversampling_be_done_before_or_within_cross-validation
    '''
    # Cross-validation
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for name, model in models:
        scores_dict_fold = OrderedDict()
        scores_dict_fold['name'] = name

        fold_scores = []

        predictions_overall = np.array([])
        y_test_overall = np.array([])
        conf_matrix_list = np.array([])

        coef_list = np.array([])

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, y_train = X.loc[train_idx], y[train_idx]
            X_test, y_test = X.loc[test_idx], y[test_idx]

            # SMOTE
            sm = SMOTE(random_state=33)
            X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            predictions_overall = np.concatenate([predictions_overall, predictions])
            y_test_overall = np.concatenate([y_test_overall, y_test])

            conf_matrix = confusion_matrix(y_test, predictions)

            f1 = f1_score(y_true=y_test, y_pred=predictions, average='weighted')
            fold_scores.append(round(f1, 3))

            # -----------------------------------------------------------------------------------------------------
            # Custom modification of sklearn.metrics.classification_report source code to allow
            # summary classification report across all k-folds.
            # Idea source: https://stackoverflow.com/questions/37124407/scikit-learn-cross-validation-classification-report?rq=1
            target_names = ['class 0', 'class 1']
            p, r, f1, s = precision_recall_fscore_support(y_test_overall, predictions_overall)

            rows = zip(target_names, p, r, f1, s)

            headers = ["precision", "recall", "f1-score", "support"]
            longest_last_line_heading = 'weighted avg'
            name_width = max(len(cn) for cn in target_names)
            width = max(name_width, len(longest_last_line_heading), 3)
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            report = head_fmt.format('', *headers, width=width)
            report += '\n\n'
            row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
            for row in rows:
                report += row_fmt.format(*row, width=width, digits=3)
            report += '\n'

        scores_dict_fold['f1'] = fold_scores
        scores_dict_fold['f1_avg'] = statistics.mean(fold_scores), ' +/-', round(statistics.stdev(fold_scores), 2)

        model_stats.append(scores_dict_fold)

        print('▬' * 20, 'Model: {}'.format(name), '▬' * 20)
        print(report)
        print('f1: ', statistics.mean(fold_scores), ' +/-', round(statistics.stdev(fold_scores), 2))
        # ------End classification report ---------------------------------------------------------------------------

        # Confusion matrix
        if show_conf_matrix == True:
            print('\n',
                  pd.crosstab(y_test, predictions, rownames=['Actual adherence'], colnames=['Predicted adherence']))

        # Feature importance
        if show_feat_importance == True:
            try:
                print('\n', 'Feature importances:', '\n',
                      pd.DataFrame(list(zip(X, model.feature_importances_))).sort_values([1], ascending=False).head(10)
                      )
            except AttributeError as e:
                print(e)

            if name.startswith('Logistic'):
                rfe = RFE(LogisticRegression())
                rfe_m = rfe.fit(X, y)
                # print(rfe_m.support_, '\n', rfe_m.ranking_)
                print(pd.DataFrame(
                    list(zip(X.columns, rfe.support_, rfe.ranking_))
                ).sample(10).sort_values(by=2))

        print('\n')


eval_metrics(X=X_processed, y=y, show_conf_matrix=True, show_feat_importance=False)
