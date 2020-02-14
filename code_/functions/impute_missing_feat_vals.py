print__ = True
def impute_missing_feat_vals(X, col_to_impute, value_of_missing_data, print_=print__):
    """
    Purpose: Impute missing values, allowing selection of what the value of missing data is, and print custom views.
             Accepts columns with categorical/string values. Method:
             LabelEncoder() -> numerics -> impute numerics -> reverse LabelEncoder() back to strings

    Input: Choose a dataframe, a column, and a missing value.  Imputes missing values across the column via KNN.

    Args:
    value_of_missing_data:
            Ex.: for 'age', 'No reponse' to the age question = 99.0 --> value_of_missing_data='99.0'
                for 'ownhome', value of missing data is 'Unknown/Refused' --> value_of_missing_data= 'Unknown/Refused'
                Either a number or string is accepted.

    Returns
     A single column. Use the function to replace the column that's being imputed on.
     Ex.: X['ownhome'] = impute_missing_feat_vals(X, col_to_impute='ownhome', value_of ..............

    Custom views ------------------
    If print_ = True, the function prints out, in this order:
    The original dataframe, numeric-encoded dataframe, imputed numeric dataframe, and inverse encoded dataframe
    so that we can see what the function did.

    **Note: all of the columns will be imputed in the printed results.  Can ignore this.  Only the selected column
    (col_to_impute) is returned and can be replaced into the desired dataframe.

    Examples
    X['age'] = impute_missing_feat_vals(X, col_to_impute='age', value_of_missing_data=99.0, print_=print__)
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