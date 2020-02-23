# Setup
import pandas as pd
pd.options.display.max_columns = 30
pd.options.display.width = 120
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML setup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Model scoring
from sklearn.model_selection import StratifiedKFold # Validation

# ML models
from sklearn.linear_model import LogisticRegression

# Feature selection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFECV
import shap
import eli5


# -------------------------------------------------------------------------------------------------------------------
# Import data
# -------------------------------------------------------------------------------------------------------------------
from code_.n05_final_logreg import X_processed, y

# -------------------------------------------------------------------------------------------------------------------
# Explain model coefficients & plot
# -------------------------------------------------------------------------------------------------------------------
from code_.n05_final_logreg import X_processed, y, pipeline

coefs_ = eli5.format_as_dataframe(
    eli5.explain_weights_sklearn(pipeline.return_model().named_steps['logistic_regression'], top=20,
                                 feature_names=X_processed.columns.to_list()))
print(coefs_)

# Invert the weights to make the graph more readable (a neg coef means a neg contrib to non-adherence)
coefs_['weight'] = [-i for i in coefs_['weight']]
coefs_ = coefs_.drop(coefs_[coefs_['feature']=='<BIAS>'].index)
# Make a new column for coloring
coefs_['is_positive'] = coefs_['weight']>0

# Plot feature importance (coefficients), color by positive/negative
fig, ax = plt.subplots()
sns.barplot(y='feature', x='weight', data=coefs_.sort_values(by='weight', ascending=False),
            palette=sns.diverging_palette(10, 240, n=12),
            label='coef')
plt.xlabel('Contribution towards adherence')
plt.ylabel('Feature')
plt.show()


# -------------------------------------------------------------------------------------------------------------------
# Permutation importance
# -------------------------------------------------------------------------------------------------------------------
from eli5.sklearn import PermutationImportance

def permut_importance(X, y):
    xcols = X.columns

    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.20, random_state=42, stratify=y)

    # SMOTE
    sm = SMOTE(random_state=33)
    X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

    cv = StratifiedKFold(random_state=30, shuffle=True)

    model = LogisticRegression(
                                C=1e10, # for getting coefficients, setting C high nulls regularization
                                class_weight={0: 0.5, 1: 0.5},
                                dual=False, fit_intercept=True,
                                intercept_scaling=1, l1_ratio=None,
                                max_iter=3000, multi_class='auto',
                                n_jobs=None, penalty='l1', random_state=42,
                                solver='liblinear', tol=0.0001, verbose=0,
                                warm_start=False)

    lr_model = model.fit(X_train, y_train)

    perm = PermutationImportance(lr_model, scoring='recall', random_state=12, n_iter=2).fit(X_test, y_test)

    permut_df = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=X_processed.columns.to_list()))
    print(eli5.format_as_text(eli5.explain_weights_sklearn(perm, feature_names=X_processed.columns.to_list())))

    print(permut_df)

    # Shap
    explainer = shap.LinearExplainer(lr_model, X_train)
    shap_values = explainer.shap_values(X_train)
    # shap.summary_plot(shap_values, X_train, feature_names=xcols)
    shap_v = pd.DataFrame(shap_values, columns=xcols)

    # Correlation coefficients
    corr_list = list()
    for i in xcols:
        b = np.corrcoef(shap_v[i], X_train[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(xcols), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')
    print(model.coef_, xcols)

    # Plot
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    k2 = k2.iloc[-15:, ]

    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, figsize=(5, 6), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")

    plt.show()

permut_importance(X_processed, y)


# -------------------------------------------------------------------------------------------------------------------
# Check variance of all features
# -------------------------------------------------------------------------------------------------------------------
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel_var = sel.fit_transform(X_processed)

# sel_df = pd.DataFrame(sel, columns = X.columns)
sel_x = X_processed[X_processed.columns[sel.get_support(indices=True)]]
print(sel_x.columns)

X_processed.var(axis=0)


# -------------------------------------------------------------------------------------------------------------------
# RFE (Recursive Feature Elimination)
# -------------------------------------------------------------------------------------------------------------------
from code_.n05_final_logreg import X_processed, y

def check_RFE(X):
    xcols = X.columns

    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split test and train
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.20, random_state=42, stratify=y)

    # SMOTE
    sm = SMOTE(random_state=33)
    X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

    cv = StratifiedKFold(random_state=30, shuffle=True)

    model = LogisticRegression(
                            C=0.1,
                            # C=1e10, # for getting coefficients, setting C high nulls regularization
                            class_weight={0: 0.5, 1: 0.5},
                            dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None,
                            max_iter=3000, multi_class='auto',
                            n_jobs=None, penalty='l1', random_state=42,
                            solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)

    rfe_cross_val = RFECV(model, cv=10, step=1, scoring='explained_variance', n_jobs=-1, min_features_to_select=3)
    rfe_cross_val.fit(X_processed, y)

    print(rfe_cross_val.n_features_)

    cv_grid_rfecv = np.sqrt(-rfe_cross_val.grid_scores_)
    print(cv_grid_rfecv)

    print(pd.DataFrame(
        list(zip(xcols, rfe_cross_val.support_, rfe_cross_val.ranking_))).sort_values(by=2))


check_RFE(X_processed.iloc[:,0:10])
