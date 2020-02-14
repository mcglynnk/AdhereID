# Setup
import pandas as pd

pd.options.display.max_columns = 10
pd.options.display.width = 250

# Load PatientsLikeMe data
patientslikeme_df = pd.read_csv(r'C:\Users\Kelly\Documents\Python\Insight_AdhereID\data\raw\patientslikeme_scrape.csv')

def clean_patientslikeme(df):
    '''
    Purpose: clean patientslikeme data

    Input: dataframe of web scraped data

    Output: cleaned dataframe
    '''
    df = df[~df['age'].isna()]
    df = df[~df['sex'].isna()]
    df = df[~df['primary_condition'].isna()]

    # Reorder columns
    df = df[['adherence','drug_name', 'username', 'age', 'sex', 'state', 'country',
             'burden', 'dosage', 'cost', 'side_effects',
             'first_diagnosis_date', 'primary_condition', 'advice',
             'secondary_condition', 'profile_bio']]

    # First diagnosis date
    df['first_diagnosis_date'] = df['first_diagnosis_date'].fillna(0)
    df['first_diagnosis_date'] = df['first_diagnosis_date'].astype(int)

    # Keep north america only...
    # df = df[df['country'].isin(['United States', 'Canada', 'United Kingdom', 'Mexico'])]

    df = df.drop(columns=['profile_bio', 'secondary_condition','advice'], axis=1)
    df = df.drop(columns=['state', 'dosage'], axis=1)

    print('unique users: ', len(df['username'].unique()))

    # print(df.columns)

    # Adherence
    adh_values = {
        'Always': 4,
        'Usually': 3,
        'Sometimes': 2,
        'Never taken as prescribed': 1
    }
    df['adherence'].replace(adh_values, inplace=True)

    df = df.reset_index(drop=True)

    return df


def describe_df(df):
    for column in df[['adherence','cost','burden','age']]:
        print( column,'\n', df[column].value_counts(), '\n', sep='')
    #print(df.info())

# Run cleaning function
patientslikeme_df = clean_patientslikeme(patientslikeme_df)
#print(patientslikeme_df.sample(10))

# Save to csv file
patientslikeme_df.to_csv(r'C:\Users\Kelly\Documents\Python\Insight_AdhereID\data\cleaned\patientslikeme_cl.csv',
                         index=None)

# Drop some columns - ended up using only 9 columns for the app
patientslikeme_df = patientslikeme_df.drop(columns=['adherence','username'])
patientslikeme_df.to_csv(r'C:\Users\Kelly\Documents\Python\Insight_AdhereID\data\cleaned\plm.csv', index=None)









