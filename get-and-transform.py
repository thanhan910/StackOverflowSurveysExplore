from bs4 import BeautifulSoup
import requests
import zipfile
import os
import pandas as pd
from tqdm import tqdm
from urllib.parse import urljoin
import shutil
import numpy as np

SESSION = requests.Session()
SURVEYS_DATABASE_PAGE_URL = 'https://survey.stackoverflow.co/'
DATA_FOLDER_PATH = 'local/data'

DOWNLOAD = True

def get_surveydata_links():
    response = SESSION.get(SURVEYS_DATABASE_PAGE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    data_gps_tracks = soup.find_all('a', {'data-gps-track': True, 'data-year': True})
    data_gps_tracks_links = [(urljoin(response.url, a['href']), a['data-year']) for a in data_gps_tracks]
    return data_gps_tracks_links


def download_surveydata_files(url: str, year):
    
    response = SESSION.get(url)
    
    # Extract the file type from the url
    file_type = url.split('.')[-1]

    # Assert that the file type is zip
    assert file_type == 'zip', f'File type {file_type} is not supported'
    
    # Save the file
    file_name = f'{year}.{file_type}'
    with open(file_name, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    path_to_extract = os.path.join(DATA_FOLDER_PATH, year)
    with zipfile.ZipFile(f'{year}.{file_type}', 'r') as zip_ref:
        zip_ref.extractall(path_to_extract)
    
    # Remove the zip file
    os.remove(file_name)

    # Remove __MACOSX folder if it exists
    macosx_folder_path = os.path.join(path_to_extract, '__MACOSX')
    if os.path.exists(macosx_folder_path) and os.path.isdir(macosx_folder_path):
        shutil.rmtree(macosx_folder_path)


def download_all_survey_data():
    data_gps_tracks_links = get_surveydata_links()
    for link, year in tqdm(data_gps_tracks_links):
        download_surveydata_files(link, year)
        # 5s per file
        # Remove __MACOSX folder

if DOWNLOAD:
    download_all_survey_data()

def get_csv_files():
    csv_files : dict[int, list] = {}
    data_folder_path_levels = len(DATA_FOLDER_PATH.split(os.sep))
    for root, dirs, files in os.walk(DATA_FOLDER_PATH):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                folders = root.split(os.sep)
                year = int(folders[data_folder_path_levels])
                if year not in csv_files:
                    csv_files[year] = []
                # Skip __MACOSX
                if '__MACOSX' in file_path:
                    continue
                csv_files[year].append(file_path)

    for year in csv_files:
        if year >= 2017:
            assert len(csv_files[year]) == 2, f'Year {year} has {len(csv_files[year])} files'
            # Assert that the 2 csv files are survey_results_schema.csv and survey_results_public.csv
            for file in csv_files[year]:
                assert file.endswith('survey_results_schema.csv') or file.endswith('survey_results_public.csv'), f'File {file} is not supported'
        else:
            assert len(csv_files[year]) == 1, f'Year {year} has {len(csv_files[year])} files'

    return csv_files

csv_files = get_csv_files()

def get_2016_schema_table():

    def get_readme_2016_filepath():
        for root, dirs, files in os.walk(os.path.join(DATA_FOLDER_PATH, '2016')):
            if '__MACOSX' in root:
                continue
            for file in files:
                if file.endswith('.txt'):
                    readme_file_path = os.path.join(root, file)
                    return readme_file_path

    readme_file_path : str = get_readme_2016_filepath()


    with open(readme_file_path, 'r') as f:
        lines = f.readlines()
    # Get all lines from "Database schema:" to the end
    for i, line in enumerate(lines):
        if 'Database schema:' in line:
            schema_table_lines = [line for line in lines[i+1:] if line.strip() != '']
            break

    schema_table_header = schema_table_lines[0].split(' --- ')
    schema_table_header = [cell.strip().strip("'").strip('"') for cell in schema_table_header]
    schema_table_rows = [line.split(' --- ') for line in schema_table_lines[1:]]
    schema_table_rows = [[cell.strip().strip("'").strip('"') for cell in row] for row in schema_table_rows]
    schema_table_rows = [['' if cell == 'N/A' else cell for cell in row] for row in schema_table_rows]

    schema_table_rows = [['Respondent', '', ''],] + schema_table_rows

    schema_table_df = pd.DataFrame(schema_table_rows, columns=schema_table_header)

    return schema_table_df

def get_dataframes(csv_files):

    dataframes = {}

    for year, files in tqdm(csv_files.items()):
        dataframes[year] = {}
        
        if year <= 2015:
            # If year <= 2015, the csv file has two top rows as headers
            assert len(files) == 1
            file_path = files[0]
            
            try:
                df_schema = pd.read_csv(file_path, nrows=2, header=None)
            except UnicodeDecodeError:
                df_schema = pd.read_csv(file_path, nrows=2, encoding='latin1', header=None)
            try:
                df_content = pd.read_csv(file_path, skiprows=2, header=None, low_memory=False)
            except UnicodeDecodeError:
                df_content = pd.read_csv(file_path, skiprows=2, encoding='latin1', header=None, low_memory=False)

            df_schema = df_schema.transpose().reset_index(drop=False)

            if year < 2015:
                df_schema.rename(columns={0: 'Question', 1: 'Answer', 'index': 'Column'}, inplace=True)
            else: # year == 2015
                df_schema.rename(columns={0: 'Question Type', 1: 'Question', 'index': 'Column'}, inplace=True)
            
            dataframes[year] = {
                'schema': df_schema,
                'content': df_content,
            }

        elif year == 2016:
            file_path = files[0]
            df_schema = get_2016_schema_table()
            df_content = pd.read_csv(file_path, low_memory=False)
            assert df_content['Unnamed: 0'].is_unique
            assert 'Respondent' not in df_content.columns
            df_content.rename(columns={'Unnamed: 0': 'Respondent'}, inplace=True)
            dataframes[year] = {
                'schema': df_schema,
                'content': df_content,
            }
        else:
            for file_path in files:
                if 'survey_results_public' in file_path:
                    content_file_path = file_path
                else:
                    assert 'survey_results_schema' in file_path
                    schema_file_path = file_path
            
            df_schema = pd.read_csv(schema_file_path)
            df_content = pd.read_csv(content_file_path, low_memory=False)
            dataframes[year] = {
                'schema': df_schema,
                'content': df_content,
            }

    return dataframes

# 2011 - 2014
# 2015
# 2016
# 2017 - 2020
# 2021 - 2024

dataframes : dict[int, dict[str, pd.DataFrame]] = get_dataframes(csv_files)

# Transform 2011 - 2014 dataframes
for year in range(2011, 2015):
    schema_rows = dataframes[year]['schema'].to_dict(orient='records')
    it = 0
    for r in schema_rows:
        if(not pd.isna(r['Question'])):
            it += 1
        r['QID'] = f'Q{it}'
    dataframes[year]['schema'] = pd.DataFrame(schema_rows, columns=['QID', 'Column', 'Question', 'Answer'])
    dataframes[year]['questions'] = dataframes[year]['schema'][['QID', 'Question']].drop_duplicates(subset=['QID'], keep='first', inplace=False)
    dataframes[year]['schema'].drop(columns=['Question'], inplace=True)
    dataframes[year]['content'].columns = [f'C{col + 1}' for col in dataframes[year]['content'].columns]
    dataframes[year]['schema']['Column'] = dataframes[2015]['schema']['Column'].apply(lambda x: f'C{x + 1}')

# Transform 2015 dataframes
dataframes[2015]['schema']['Answer'] = dataframes[2015]['schema']['Question'].str.split(': ').apply(lambda x: x[1] if len(x) > 1 else np.nan)
dataframes[2015]['schema']['Question'] = dataframes[2015]['schema']['Question'].str.split(': ').apply(lambda x: x[0])
dataframes[2015]['questions'] = dataframes[2015]['schema'].drop_duplicates(subset=['Question'], keep='first', inplace=False)[['Question', 'Question Type']]
dataframes[2015]['questions'].reset_index(drop=True, inplace=True)
dataframes[2015]['questions'].reset_index(drop=False, inplace=True)
dataframes[2015]['questions'].rename(columns={'index': 'QID'}, inplace=True)
dataframes[2015]['questions']['QID'] = dataframes[2015]['questions']['QID'].apply(lambda x: f'Q{x+1}')
dataframes[2015]['schema'].reset_index(drop=False, inplace=True)
dataframes[2015]['schema'].drop(columns=['Question Type'], inplace=True)
dataframes[2015]['schema'] = pd.merge(dataframes[2015]['schema'], dataframes[2015]['questions'], on='Question', how='left', suffixes=('', '_y'))[['QID', 'Column', 'Answer']]

dataframes[2015]['content'].columns = [f'C{col + 1}' for col in dataframes[2015]['content'].columns]
dataframes[2015]['schema']['Column'] = dataframes[2015]['schema']['Column'].apply(lambda x: f'C{x + 1}')


# Transform 2016 dataframes
dataframes[2016]['schema']['Survey Question'] = dataframes[2016]['schema']['Survey Question'].replace('', np.nan)
dataframes[2016]['questions'] = dataframes[2016]['schema']['Survey Question'].drop_duplicates(keep='first', inplace=False).dropna().reset_index(drop=True, inplace=False).reset_index(drop=False, inplace=False).rename(columns={'index': 'QID'}, inplace=False)
dataframes[2016]['questions']['QID'] = dataframes[2016]['questions']['QID'].apply(lambda x: f'Q{x+1}')
dataframes[2016]['schema'] = dataframes[2016]['schema'].merge(dataframes[2016]['questions'], left_on='Survey Question', right_on='Survey Question', how='left', suffixes=('', '_y'))[['QID', 'Column Name', 'Note (if any)']]


def match_unmatched_qnames(year, df_schema, df_content):
    """
    Some qnames in the schema dataframe are not found among the columns in the content dataframe because the columns contains the qnames with an answer as a suffix.
    This function matches the unmatched qnames with the columns in the content dataframe.
    """
    assert year >= 2021
    unmatched_qnames = sorted(set(df_schema['qname']) - set(df_content.columns))
    unmatched_columns = sorted(set(df_content.columns) - set(df_schema['qname']))
    i = 0
    j = 0
    min_j_assessed = 0
    i_isfound = False
    columns_qname = {col:  None for col in unmatched_columns}
    qname_columns = {qname: [] for qname in unmatched_qnames}
    while (i < len(unmatched_qnames)) and (min_j_assessed < len(unmatched_columns)):
        if unmatched_columns[j].startswith(unmatched_qnames[i]):
            i_isfound = True
            qname_columns[unmatched_qnames[i]].append(unmatched_columns[j])
            columns_qname[unmatched_columns[j]] = unmatched_qnames[i]
            j += 1
            min_j_assessed = j
        else:
            if i_isfound:
                i += 1
                i_isfound = False
            else:
                j += 1
            if not i_isfound and j == len(unmatched_columns):
                j = min_j_assessed
                i += 1
    return qname_columns, columns_qname

def get_qname_to_columns_df(year, df_schema, df_content):
    qname_columns, columns_qname = match_unmatched_qnames(year, df_schema, df_content)
    unmatched_columns = [col for col in columns_qname if columns_qname[col] is None]
    unmatched_qnames = [qname for qname in qname_columns if qname_columns[qname] == []]
    
    # Some assertions
    assert unmatched_columns == ['ConvertedCompYearly', 'ResponseId']

    qname_columns_csv = [(qname, '|'.join([col.removesuffix(qname) for col in matched_columns])) for qname, matched_columns in qname_columns.items()]
    
    return pd.DataFrame(qname_columns_csv, columns=['qname', 'columns'])

# Transform 2021 - 2024 dataframes
for year in range(2021, 2025):
    dataframes[year]['questions'] = dataframes[year]['schema'][['qid', 'qname']].drop_duplicates(subset=['qid'], keep='first', inplace=False).reset_index(drop=True, inplace=False)
    dataframes[year]['qname_columns'] = get_qname_to_columns_df(year, dataframes[year]['schema'], dataframes[year]['content'])


# Output the dataframes to csv files
output_dir = 'local/output'
for year in range(2011, 2025):
    os.makedirs(os.path.join(output_dir, str(year)), exist_ok=True)
    for name, df in dataframes[year].items():
        df.to_csv(os.path.join(output_dir, str(year), f'{name}.csv'), index=False)