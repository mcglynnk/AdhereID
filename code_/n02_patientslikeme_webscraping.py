# Setup
import pandas as pd
pd.options.display.max_columns = 20
pd.options.display.width = 200

import traceback  # Error handling
import pickle  # File I/O

import json
from _collections import OrderedDict

'''
DATA SOURCE:
PatientsLikeMe: Find answers, support and a path forward with people like you.
https://www.patientslikeme.com/
accessed 1-20-2020

I used both beautifulsoup and mechanicalsoup in this web scraper. Mechanicalsoup builds on beautifulsoup, but has 
added functionality for logins and page navigation.

'''
import mechanicalsoup

from bs4 import BeautifulSoup
import requests  # requests is used to open urls with beautifulsoup.


# 1. Scrape to get a list of all drugs on the site (74 pages of drug names).  Append drug names to a list.
# From this page: https://www.patientslikeme.com/treatments/browse?cat=1&page=1
drug_list = []
for page_number in range(1, 74):
    url_pg = 'https://www.patientslikeme.com/treatments/browse?cat=1&page={}'.format(page_number)
    # print(url_pg)
    try:
        with requests.get(url_pg) as r:
            soup = BeautifulSoup(r.content).find_all('td')
            for i in soup:
                if i.a is not None:
                    # print(i.a.contents[0])
                    drug_list.append(i.a.contents[0].split(' ')[0].lower())
        print('All drug names collected from page', page_number, ', continuing...', len(drug_list))

    except Exception as e:
        print(e, traceback.format_exc())

# Save drug list to file
# filename = r'C:\Users\Kelly\Documents\Python\Insight_AdhereID\data\patientslikeme_drug_list2.txt'
# with open(filename, 'wb') as f:
#     pickle.dump(drug_list, f)

# Open drug_list file:
filename = r'/\data\patientslikeme_drug_list2.txt'
with open(filename, 'rb') as f:
    drug_list = pickle.load(f)


# 2. Make a mechanicalsoup browser object
browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'html'})

# 3. Initialize empty lists to add data to:
patientslikeme_data = []        # In the scraper below, for each patient's drug review, all of the data (user
                                # name, age, review, etc.) is added to to an OrderedDict (data_each_user =
                                # OrderedDict() below). Each OrderedDict is one row of data. In the loop 'for
                                # each drug', the row is appended to the patientslikeme_data list. At the end,
                                # patientslikeme_data is made into a dataframe.

num_drug_pages = len(drug_list) # For printing out progress below. The section below runs num_drug_pages - 1 for each
                                # loop, to print how many drugs are left to be scraped
num_pages = []


# 4. START SCRAPER
for idx, drug in enumerate(drug_list):  # total: 2190 drug pages
    # Get number of pages of reivews on the url: e.g. gabapentin page has 26 pages of reviews, trazodone has 90 pages,
    # duloxetine has 106
    try:
        url = 'https://www.patientslikeme.com/treatment_evaluations/browse/{}?brand=f&page=1#evaluations'.format(
            drug)
        with browser.open(url) as firstpage:
            soup_firstpage = BeautifulSoup(firstpage.content, features='lxml')
            num_pages = soup_firstpage.find('div', role="navigation").find('a', class_="icon-only is-not-rwd " \
                       "next_page").find_previous_sibling().get_text()
    except AttributeError as e:
        num_pages = 1  # If the page isn't found, set number of pages to 1. It tries to scrape an empty page,
                       # but still works without error (somehow)

    # Print out progress as the scraper runs: will print out text like this as the scraper runs:
        # Scraping methotrexate , 5 pages      (Drugs left: 1780)
        # Scraping... Finished methotrexate, page 1
        # Scraping... Finished methotrexate, page 2
    num_drug_pages = num_drug_pages - 1  # This line is for printing out progress
    print('Scraping', drug, ', {} pages'.format(num_pages),  # Print progress
          '     (Drugs left: {})'.format(num_drug_pages))    # Print progress

    # Start iterating through drug pages:
        # For each drug in drug_list, add the name of the drug to this url:
        # https://www.patientslikeme.com/treatment_evaluations/browse/trazodone?brand=f&page=3#evaluations,
        # replacing 'trazodone' with the name of the drug in each loop.
    for pg_num in range(1, int(num_pages) + 1):
        url = 'https://www.patientslikeme.com/treatment_evaluations/browse/{}?brand=f&page={}#evaluations'.format(
            drug, pg_num)
        if browser.open(url) is None:
            pass # Not sure if this part works; it's supposed to skip a link if the page doesn't exist
        else:
            with browser.open(url) as page: # OPEN URL
                pg = browser.get_current_page()  # for patient/user data section
                soup = BeautifulSoup(page.content)  # for drug data section (need next_sibling functions)

                # User data
                for i in pg.find_all('div', attrs={'data-react-class': 'ProfileSummaryPropsContainer'}):
                    data_each_user = OrderedDict()
                    data_each_user['drug_name'] = drug

                    x = json.loads(i['data-react-props'])  # view all user data fields
                    print(json.dumps(x, indent=2)) # print all user data fields in readable format

                    data_each_user['username'] = x['userName']
                    data_each_user['age'] = x['userAge']
                    data_each_user['sex'] = x['sex']
                    data_each_user['country'] = x['country']
                    data_each_user['state'] = x['state']
                    data_each_user['first_diagnosis_date'] = x["firstDiagnosisDate"]
                    data_each_user['primary_condition'] = x['primaryConditionName']
                    data_each_user['secondary_condition'] = x["secondaryConditionNames"]

                    # Drug data
                    user_root = i.parent.find_next_sibling('div')  # i = patient. next sibling = 'eval-details'
                    user_root_mostrecent = user_root.find(
                        'h3')  # h3 = date of review section (take the most recent only)
                    user_root_mostrecent_eval = user_root_mostrecent.find_next_sibling(
                        'div')  # eval details of first review only
                    user_eval = user_root_mostrecent_eval  # rename to make it shorter, so the part below is more readable

                    # Side Effects
                    data_each_user['side_effects'] = \
                        user_eval.find('span', text='Side effects').parent.find_next_sibling(
                            'div').get_text().split('\n')[2]
                    # Adherence
                    data_each_user['adherence'] = user_eval.find('span', text='Adherence').parent.find_next_sibling(
                        'div').get_text().split('\n')[2]
                    # Burden
                    data_each_user['burden'] = user_eval.find('span', text='Burden').parent.find_next_sibling(
                        'div').get_text().split('\n')[2]

                    # Dosage, Advice, and Cost fields ---------------------------------------------------------
                    h3 = i.parent.find_next_sibling('div').find('h3')  # same as user_root_mostrecent

                    # Dosage
                    if h3.parent.find('strong', text='Dosage:') is not None:
                        dosage = h3.parent.find('strong', text='Dosage:').parent.get_text().split('\n')[2:4]
                        dosage = ' '.join(dosage)
                        data_each_user['dosage'] = dosage
                    else:
                        data_each_user['dosage'] = 'NA'

                    # Advice
                    if h3.parent.find('strong', text='Advice & Tips:') is not None:
                        data_each_user['advice'] = h3.parent.find('strong', text='Advice & Tips:').parent.get_text(
                        ).split('\n')[2].lower()
                    else:
                        data_each_user['advice'] = 'NA'

                    # Cost
                    if h3.parent.find('strong', text='Cost:') is not None:
                        data_each_user['cost'] = h3.parent.find('strong', text='Cost:').parent.get_text().split('\n')[2]
                    else:
                        data_each_user['cost'] = 'NA'

                    data_each_user['profile_bio'] = x['briefBio']

                    patientslikeme_data.append(data_each_user) # Append to the list patientslikeme, from
                                                               # beginning (line 60)

                print('Scraping... Finished {}, page {}'.format(drug, pg_num)) # Print progress

    if (idx == idx): # SAVE RESULTS TO CSV AS THE SCRAPER IS RUNNING; each idx (each drug) contains multiple pages,
        # so save to csv for each one rather than every 10 or something.
        drug_df = pd.DataFrame(patientslikeme_data)
        drug_df.to_csv(r'C:\Users\Kelly\Documents\Python\Insight_AdhereID\data\patientslikeme_scrape.csv', index=False)


# Save the scraped data to a csv
patientslikeme_df = pd.read_csv(r'/\data\patientslikeme_scrape.csv')
patientslikeme_df


