# Setup
import pandas as pd

pd.options.display.max_columns = 30
pd.options.display.width = 120

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML setup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Model scoring
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold  # Validation
from sklearn.metrics import average_precision_score, precision_recall_curve

# ML models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.dummy import DummyClassifier  # Random chance classifier
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------------------------------------------------------------------------
# Initial model comparison plot (values from running n04_eval_models.py)
#      - F1 score barplot and grouped barplot for recall and precision
# -------------------------------------------------------------------------------------------------------------------

def model_comparison_barplot(f1=False, recall_precision=False):
    sns.set_style('whitegrid')

    # 'Dummy Classifier','Logistic Regression','Random Forest','Extra Trees', 'LightGBM'
    scores = [0.506, 0.602, 0.617, 0.598, 0.617]
    stdev_ = [0.06, 0.04, 0.03, 0.03, 0.04]

    # Percent difference over Dummy Classifier
    scores_diff = [(i / scores[0] - 1) * 100 for i in scores]
    scores_diff = scores_diff[1:]

    precision_on_class_1 = [0.288, 0.396, 0.376, 0.404, 0.394]
    recall_on_class_1 = [0.552, 0.587, 0.175, 0.213, 0.280]

    gr = pd.DataFrame(
        {'Models': ['Dummy Classifier', 'Logistic Regression', 'Random Forest', 'Extra Trees', 'LightGBM'],
         'F1 Scores': scores,
         'Precision': precision_on_class_1,
         'Recall': recall_on_class_1
         })

    f, ax = plt.subplots(figsize=(8, 9))
    sns.set(font_scale=2.0)

    sns.set_style('whitegrid')

    if f1 == True:
        b = sns.barplot('Models', 'F1 Scores', data=gr, palette='pastel')
        b.set_ylabel('F1 score', fontsize=24)
        plt.xticks(rotation=45, ha='right')

    elif recall_precision == True:
        gr = gr.drop(columns=['F1 Scores'])
        gr = pd.melt(gr, id_vars='Models', var_name='Scores', value_name='score')
        b = sns.barplot('Models', 'score', 'Scores', data=gr, palette="pastel")
        b.set_ylabel('Scores (class 1)', fontsize=24)
        plt.xticks(rotation=45, ha='right')

    # sns.set(font_scale=1.8)
    plt.show()


model_comparison_barplot(recall_precision=True)
model_comparison_barplot(f1=True)

# Performance Curves
# ---------------------------------------------------------------------------------------------------------------------
# 1. Precision-Recall curve
# ---------------------------------------------------------------------------------------------------------------------
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/

from code_.n05_final_logreg import X_processed, y, Model


def plot_PRC(X, y, individual_precision_recall_plots=False):
    """
    Purpose:  Plot precision-recall curve for logistic regression and random forest on the same plot. Include
              dashed lines for no-skill model and perfect-skill model.

    """
    plt.grid(True)

    # Split test and train
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Model 1: Logistic Regression -----------------------------------------------------------------------------------
    # Hyperparameter-tuned logistic regression
    lr_pipeline = Model(X, y, model=LogisticRegression(C=0.1,
                                                       class_weight={0: 0.5, 1: 0.5},
                                                       dual=False, fit_intercept=True,
                                                       intercept_scaling=1, l1_ratio=None,
                                                       max_iter=3000, multi_class='auto',
                                                       n_jobs=None, penalty='l1', random_state=42,
                                                       solver='liblinear', tol=0.0001, verbose=0,
                                                       warm_start=False)
                        )
    model1 = lr_pipeline.return_model()
    X_train, X_test, y_train, y_test = lr_pipeline.split_()
    # Fit model and get predictions
    model1.fit(X_train, y_train)

    predictions1 = model1.predict(X_test)

    # Class 0 - adherence
    # y_score = model1.decision_function(X_test) # This works too, is specific to pos_val=1. Doesn't work for rand.forest
    y_score = model1.predict_proba(X_test)[:, 0]  # probability of prediction being class 0
    average_precision_0 = average_precision_score(y_test, y_score)
    print(average_precision_0)
    precision1, recall1, threshold1 = precision_recall_curve(y_test, y_score)

    # Class 1 - nonadherence
    # y_score = model1.decision_function(X_test)
    y_score = model1.predict_proba(X_test)[:, 1]  # probability of prediction being class 1
    average_precision_1 = average_precision_score(y_test, y_score)
    print(average_precision_1)
    precision1, recall1, threshold1 = precision_recall_curve(y_test, y_score, pos_label=1)

    # LR individual plots
    if individual_precision_recall_plots == True:
        # Produces one plot with two lines, one line for recall and one line for precision
        plt.plot(threshold1, precision1[0:-1], 'cornflowerblue', label='LR precision', linestyle='--')
        plt.plot(threshold1, recall1[0:-1], 'royalblue', label='LR recall')
        plt.xlabel('Threshold')
        plt.legend()
    # plt.show()
    else:
        f, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylim(ymin=0.2)
        plt.plot(recall1, precision1, marker='.', lw=2, label='Logistic Regression', color='teal')

    # Model 2: Random Forest -----------------------------------------------------------------------------------------
    # Hyperparameter-tuned random forest
    rf_pipeline = Model(X, y, model=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                                           criterion='gini', max_depth=4, max_features='auto',
                                                           max_leaf_nodes=None, max_samples=None,
                                                           min_impurity_decrease=0.0, min_impurity_split=None,
                                                           min_samples_leaf=3, min_samples_split=4,
                                                           min_weight_fraction_leaf=0.0, n_estimators=91,
                                                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                                                           warm_start=False)
                        )
    model2 = rf_pipeline.return_model()

    # Fit model and get predictions
    model2.fit(X_train, y_train)

    predictions2 = model2.predict(X_test)

    # Class 0
    y_score_rf_0 = model2.predict_proba(X_test)[:, 0]
    average_precision = average_precision_score(y_test, y_score_rf_0)
    precision2, recall2, threshold2 = precision_recall_curve(y_test, y_score_rf_0)

    # Class 1
    y_score_rf_1 = model2.predict_proba(X_test)[:, 1]
    average_precision = average_precision_score(y_test, y_score_rf_1)
    precision2, recall2, threshold2 = precision_recall_curve(y_test, y_score_rf_1)

    # RF individual plots
    if individual_precision_recall_plots == True:
        # Produces one plot with two lines, one line for recall and one line for precision
        plt.plot(threshold2, precision2[0:-1], 'orange', label='RF precision', linestyle='--')
        plt.plot(threshold2, recall2[0:-1], 'darkorange', label='RF recall')
        plt.xlabel('Threshold')
        plt.legend()
        plt.show()
    else:
        plt.plot(recall2, precision2, marker='.', label='Random Forest', color='orange')

    # Plot both LR and RF -------------------------------------------------------------------------------------------
    if individual_precision_recall_plots == False:
        # PRC Baseline - based on the number of majority and minority labels ('No skill' dashed grey line)
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

        # Perfect model - dashed red line
        perfect_skill = 1
        plt.plot([0, 1, 1], [perfect_skill, perfect_skill, no_skill], linestyle='--', label='Perfect performance',
                 color='red')

        # Axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.5, 0.45, 0.45, 0.5), prop={'size': 15})
        plt.show()

plot_PRC(X_processed, y, individual_precision_recall_plots=False)
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
plt.show()



# ---------------------------------------------------------------------------------------------------------------------
# 2. ROC curve
# ---------------------------------------------------------------------------------------------------------------------

def plot_ROC(X, y):
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # SMOTE
    sm = SMOTE(random_state=33)
    X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

    # Dummy Classifier (random chance)
    model_d = DummyClassifier(random_state=42)
    model_d.fit(X_train, y_train)

    predictions_d = model_d.predict(X_test)

    y_score_d = model_d.predict_proba(X_test)[:, 1]  # probability of prediction being class 1
    # Score
    dummy_auc = roc_auc_score(y_test, y_score_d)
    print('Dummy: ROC AUC=%.3f' % (dummy_auc))
    # Plot
    dm_fpr, dm_tpr, _ = roc_curve(y_test, y_score_d)
    plt.plot(dm_fpr, dm_tpr, marker='.', label='Dummy Classifier', color='black')

    # Model 1: Logistic Regression -----------------------------------------------------------------------------------
    cv = StratifiedKFold(random_state=12, n_splits=10)
    model1 = LogisticRegressionCV(cv=cv, Cs=10,
                                  class_weight={0: 0.5, 1: 0.5},
                                  dual=False, fit_intercept=True,
                                  intercept_scaling=1,
                                  max_iter=3000, multi_class='auto',
                                  n_jobs=None, penalty='l1', random_state=42,
                                  solver='liblinear', tol=0.0001, verbose=0,
                                  )
    model1.fit(X_train, y_train)

    predictions1 = model1.predict(X_test)

    y_score = model1.predict_proba(X_test)[:, 1]  # probability of prediction being class 1
    # Score
    lr_auc = roc_auc_score(y_test, y_score)
    print('LR: ROC AUC=%.3f' % (lr_auc))
    # Plot
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_score)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression', color='cornflowerblue')

    # Random Forest
    model2 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                    criterion='gini', max_depth=4, max_features='auto',
                                    max_leaf_nodes=None, max_samples=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=3, min_samples_split=4,
                                    min_weight_fraction_leaf=0.0, n_estimators=91,
                                    n_jobs=None, oob_score=False, random_state=42, verbose=0,
                                    warm_start=False)
    model2.fit(X_train, y_train)

    predictions2 = model2.predict(X_test)

    y_score_rf_1 = model2.predict_proba(X_test)[:, 1]
    # Score
    rf_auc = roc_auc_score(y_test, y_score_rf_1)
    print('RF: ROC AUC=%.3f' % (rf_auc))
    # Plot
    rf_fpr, rf_tpr, _ = roc_curve(y_test, y_score_rf_1)
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest', color='orange')

    # Plot both LR and RF -------------------------------------------------------------------------------------------
    # No skill dashed line
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='grey')

    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


plot_ROC(X_processed, y)

'''
Dummy: ROC AUC=0.466
LR: ROC AUC=0.635
RF: ROC AUC=0.630

'''


