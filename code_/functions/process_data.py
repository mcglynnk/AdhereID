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
    categorical_cols = ['sex',
                        # 'have_health_insur','have_medicare', 'metro',
                        'parent',
                        # 'US_region',
                        # 'ownhome',
                        # 'mstatus',
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
