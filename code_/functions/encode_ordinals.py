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