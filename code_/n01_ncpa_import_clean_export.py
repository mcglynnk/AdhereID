# Setup
import pandas as pd
pd.options.display.max_columns = 30
pd.options.display.width = 200

# Read NCPA file
from pyreadstat import read_por  # For reading SPSS/Stat files

from code_.required_files import ncpa_filepath
def load_ncpa():
    ncpa, meta = read_por(ncpa_filepath)
    ncpa
    meta.column_labels
    meta.column_names = [i.lower() for i in meta.column_names]
    ncpa.columns = meta.column_names  # now it's a dataframe
    return ncpa


ncpa = load_ncpa()

# Drop some columns that aren't relevant to the project
def clean_ncpa(df):
    df = df.drop(
        columns=['adults', 'adultm', 'adultf', 'kids1217', 'kids611', 'kids06', 'di22a1', 'di22a2', 'di22a3',
                 'di22a4', 'di22a5', 'di22b1', 'di22b2', 'di22b3', 'di22b4', 'di22b5', 'di22c1', 'di22c2', 'di22c3',
                 'di22c4', 'di22c5', 'nielsen', 'weight', 'week'], axis=1)
    df = df.drop(columns=['affilrot', 'polaffil', 'd3rot', 'd3', 'regvote', 'othtel'])
    df = df.drop(columns=['reli1', 'ident', 'l1', 'l2a', 'c1', 'c2', 'c1a'])

    df = df.reset_index(drop=True)

    return df


ncpa = clean_ncpa(ncpa)

# Combine two columns to make the target column:
# - Stopped taking prescription
# - Didn't fill a prescription
def get_ncpa_stopped_or_didnt_fill():
    df = ncpa.iloc[:, 0:15]

    df['n_prescriptions'] = ncpa['rxqn1']

    df['first_got_rx'] = ncpa['rxqn2']
    df = df[df['first_got_rx'] != 8]

    df['sex'] = ncpa['sex']

    # Age
    df['age'] = ncpa['age']  # 22 ages missing, impute these using KNN later
    # df = df[df['age'] != 99.0] # 99 is no response

    # In the past 12 months, have you ever...? missed a dose of your prescription medication
    # df['missed_dose'] = ncpa['rxqn7b']
    # df['missed_dose_freq'] = ncpa['rxqn7xb']  # how often

    # In the past 12 months, have you ever...? stopped taking a medication entirely without consulting a doctor
    df['stopped_taking'] = ncpa['rxqn7d']  # 896 no, 122 yes
    # df['stopped_taking_freq'] = ncpa['rxqn7xd'] # how often
    df = df[df['stopped_taking'] != 8]  # 2 rows with value 8 (no answer)

    # In the past 12 months, have you ever...? Didn't fill rx
    df['didnt_fill_rx'] = ncpa['rxqn7a']  # 778 no, 241 yes
    # df['didnt_fill_rx_freq'] = ncpa['rxqn7xa'] # how often
    df = df[df['didnt_fill_rx'] != 8]  # 1 row with value 8 (no answer)

    # Survey questions: 1=yes, 2=no
    # Are there patients who have never stopped taking a medication, but said yes to 'Didn't fill rx'?
    # ncpa[(ncpa['rxqn7d'] == 2.0) & (ncpa['rxqn7a'] == 1.0)] # yes! 185 rows. Can combine these two columns.
    # Patients who answered yes to either stopped taking or didn't fill rx:
    # ncpa[(ncpa['rxqn7d']==1.0) | (ncpa['rxqn7a']==1.0)]
    df = df.reset_index(drop=True)

    new_col = []  # If no to EITHER stopped_taking or 'didn't fill rx', replace with 'adherence'
    for idx, i in enumerate(df['stopped_taking']):
        if (i == 1.0) | (df['didnt_fill_rx'][idx] == 1.0):
            new_col.append('non-adherence')
        else:
            new_col.append('adherence')

    df['adherence_stopped_or_didnt_fill'] = new_col

    df['age_range'] = df['age2']

    # Reasons given if person answered yes to has not filled med or not taken as prescribed
    df['bc_side_effects'] = ncpa['rxqn8a']
    df['bc_didnt_need'] = ncpa['rxqn8b']
    df['bc_save_money'] = ncpa['rxqn8f']
    df['bc_busy'] = ncpa['rxqn8c']
    df['bc_forgot'] = ncpa['rxqn9a']

    df['general_health'] = ncpa['rxqn11']
    df = df[df['general_health'] != 8]

    # Health conditions
    df['has_diabetes'] = ncpa['rxqn12a']  # 5 9's, 3 8's; 3 of the 9's are also 9 for next 4 columns; remove these 3
    df = df[(df['has_diabetes'] != 8) & (df['has_diabetes'] != 9)]

    df['has_hyperten'] = ncpa['rxqn12b']

    df['has_asthma_etc'] = ncpa['rxqn12c']
    df = df[df['has_asthma_etc'] != 8]

    df['has_heart_condition'] = ncpa['rxqn12d']
    df = df[df['has_heart_condition'] != 8]  # 4 rows with 8

    df['has_hi_cholesterol'] = ncpa['rxqn12e']
    df = df[df['has_hi_cholesterol'] != 8]  # 2 rows with 8

    df['has_another_chronic_condition'] = ncpa['rxqn12h']  # 6 9's, 6 8's
    df = df[df['has_another_chronic_condition'] != 9]  # 1 row with 9

    df['n_provider_visits'] = ncpa['rxqn13']
    df = df[df['n_provider_visits'] != 999]  # 2 rows have 999
    df = df[df['n_provider_visits'] != 998]  # 16 rows have 999 *** 1 rows is low-adherence

    df['understand_health_prob'] = ncpa['rxqn18a']
    df['med_burden'] = ncpa['rxqn19']

    df['have_health_insur'] = ncpa['demh1']
    df = df[df['have_health_insur'] != 9]  # one row has 9 (= refused)

    df['have_medicare'] = ncpa['demh3']
    df['have_medicare'] = df['have_medicare'].fillna(3.0)  # lots of NA, but NA means ineligible (under 65 years old)
    df = df[df['have_medicare'] != 8]
    df = df[df['have_medicare'] != 9]

    df['can_afford_rx'] = ncpa['rxqn27']  # 5 8's and 3 9's, but none with low adherence
    df = df[df['can_afford_rx'] != 8]  # 8 = don't know
    df = df[df['can_afford_rx'] != 9]  # 9 = refused

    df = df[df['med_burden'] != 8]

    df = df[df['educ'] != 9]

    # Rearrange columns
    df = df[['adherence_stopped_or_didnt_fill',
             'age', 'n_prescriptions', 'n_provider_visits', 'first_got_rx', 'income', 'general_health',
             'med_burden', 'can_afford_rx', 'understand_health_prob', 'educ', 'sex', 'have_health_insur',
             'have_medicare', 'metro', 'ownhome', 'mstatus', 'emply', 'has_diabetes', 'has_hyperten', 'has_asthma_etc',
             'has_heart_condition', 'has_hi_cholesterol', 'division', 'parent', 'race',
             ]]

    # df = df.drop(columns=['caseid'])
    # df = df.drop(columns=['state', 'age_range', 'region'])

    df = df.reset_index(drop=True)

    return df


# Survey questions were recorded as numbers. Change these back to the actual answers (better to do it this way than
# try to use LabelEncoder() directly, for sanity's sake. Not all of the questions are 1(lowest) to 5(highest),
# some are the reverse.
def num_columns_to_categories(df):
    adhere_vals = {
        1: 'Very frequently stops/doesnt fill rx',
        2: 'Somewhat frequently stops/doesnt fill rx',
        3: 'Occasionally stops/doesnt fill rx',
        4: 'Rarely stops/doesnt fill rx',
        5: 'Never stops/doesnt fill rx'
    }
    df['adherence_stopped_or_didnt_fill'].replace(adhere_vals, inplace=True)

    # First started taking an rx on a regular basis
    first_started_taking_vals = {
        1: 'Within the past year',
        2: '1 to 2 years ago',
        3: '3 to 5 years ago',
        4: '6 to 10 years ago',
        5: 'More than 10 years ago'
    }
    df['first_got_rx'].replace(first_started_taking_vals, inplace=True)

    # Sex
    df['sex'] = [str(i).replace(str(1.0), 'M') if i == 1.0 else str(i).replace(str(2.0), 'F') for i in
                 df['sex']]

    # General health
    gen_val = {
        1: 'Excellent',
        2: 'Very good',
        3: 'Good',
        4: 'Fair',
        5: 'Poor',
    }
    df['general_health'].replace(gen_val, inplace=True)

    # Income
    income_new = []
    for idx, i in enumerate(df['income']):
        if i == 1.0 or i == 2.0 or i == 3.0 or i == 4.0 or i == 5.0 or i == 9:
            income_new.append('<$50k')
        elif i == 6 or i == 7 or i == 10:
            income_new.append('$50k-75k')
        elif i == 8 or i == 11 or i == 12 or i == 13 or i == 14 or i == 15:
            income_new.append('>$100k')
        elif i == 98 or i == 99:
            income_new.append('No response/Unknown')
    df = df.drop(columns=['income'])
    df.insert(loc=5, column='income', value=income_new)
    # df['income'] = income_new

    # Med burden
    med_burden_vals = {
        1: 'Very simple',
        2: 'Somewhat simple',
        3: 'Somewhat complicated',
        4: 'Very complicated',
        8: 'Unknown'
    }
    df['med_burden'].replace(med_burden_vals, inplace=True)

    health_insur_vals = {
        1: "Has health insurance",
        2: "No health insurance"
    }
    df['have_health_insur'].replace(health_insur_vals, inplace=True)

    medicare_vals = {
        1: 'Medicare',
        2: 'No medicare',
        3: 'Not eligible',
        8: 'Unknown/Refused',
        9: 'Unknown/Refused'
    }
    df['have_medicare'].replace(medicare_vals, inplace=True)

    can_afford_rx_vals = {
        1: 'Very easy',
        2: 'Somewhat easy',
        3: 'Somewhat difficult',
        4: 'Very difficult'
    }
    df['can_afford_rx'].replace(can_afford_rx_vals, inplace=True)

    understand_health_prob_vals = {
        1: 'A great deal',
        2: 'Somewhat',
        3: 'Not so much',
        4: 'Not at all',
        8: 9.0  # These are KNN imputed later
    }
    df['understand_health_prob'].replace(understand_health_prob_vals, inplace=True)

    metro_vals = {
        1: 'Center City',
        2: 'Center City',
        3: 'Suburban(Metro)',
        4: 'Non-Center City(Metro)',
        5: 'Non-city'
    }
    df['metro'].replace(metro_vals, inplace=True)

    division_vals = {
        1: 'New England',
        2: 'Mid Atlantic',
        3: 'East North Central',
        4: 'West North Central',
        5: 'South Atlantic',
        6: 'East South Central',
        7: 'West South Central',
        8: 'Mountain',
        9: 'Pacific'
    }
    df['division'].replace(division_vals, inplace=True)
    df = df.rename(columns={'division': 'US_region'})

    ownhome_vals = {
        1: 'Homeowner',
        2: "Renter",
        8: 'Unknown/Refused',
        9: 'Unknown/Refused'
    }
    df['ownhome'].replace(ownhome_vals, inplace=True)

    mstatus_vals = {
        1: 'Single',
        2: 'Single partnered',
        3: 'Married',
        4: 'Separated',
        5: 'Widowed',
        6: 'Divorced',
        9: 9
    }
    df['mstatus'].replace(mstatus_vals, inplace=True)

    emply_vals = {
        1: 'Full-time',
        2: 'Part-time',
        3: 'Retired',
        4: 'Homemaker',
        5: 'Student',
        6: 'Unemployed',
        7: 'Disabled',
        8: 999,
        9: 999
    }
    df['emply'].replace(emply_vals, inplace=True)
    emply_combine = {
        'Student': 'Unemployed'
    }
    df['emply'].replace(emply_combine, inplace=True)

    parent_vals = {
        1: 'Parent',
        2: 'Nonparent'
    }
    df['parent'].replace(parent_vals, inplace=True)

    educ_vals = {
        1: 'Less than high school',
        2: 'High school',
        3: 'Some college',
        4: 'College graduate',
        5: 'Graduate school or more',
        6: 'Technical school/other',
        9: 'Unknown/Refused'
    }
    df['educ'].replace(educ_vals, inplace=True)

    race_vals = {
        1: 'White Non-Hispanic',
        2: 'Black Non-Hispanic',
        3: 'White Hispanic',
        4: 'Black Hispanic',
        5: 'Hispanic',
        6: 'Other race',
        7: 'Unknown/Refused'
    }
    df['race'].replace(race_vals, inplace=True)

    has_condition_vals = {
        1.0: 'Yes',
        2.0: 'No'
    }
    df['has_diabetes'].replace(has_condition_vals, inplace=True)
    df['has_hyperten'].replace(has_condition_vals, inplace=True)
    df['has_asthma_etc'].replace(has_condition_vals, inplace=True)
    df['has_heart_condition'].replace(has_condition_vals, inplace=True)
    df['has_hi_cholesterol'].replace(has_condition_vals, inplace=True)

    return df

# Run the data cleaning functions
ncpa_stopped_or_didnt_fill = get_ncpa_stopped_or_didnt_fill()
ncpa_stopped_or_didnt_fill = num_columns_to_categories(ncpa_stopped_or_didnt_fill)
print(ncpa_stopped_or_didnt_fill.sample(10))
print(ncpa_stopped_or_didnt_fill.iloc[:, 0].value_counts())

# Save to csv file
from code_.required_files import ncpa_stopped_or_didnt_fill_file
ncpa_stopped_or_didnt_fill.to_csv(ncpa_stopped_or_didnt_fill_file, index=None)

# Show value counts for all columns
def describe_df(df):
    for column in df:
        print(column, '\n', df[column].value_counts(), '\n', sep='')

describe_df(ncpa_stopped_or_didnt_fill)
