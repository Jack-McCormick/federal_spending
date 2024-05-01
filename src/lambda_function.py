import zipfile
import os
import pandas as pd
import configparser
from bs4 import BeautifulSoup
from dateutil import parser
from dateutil.tz import gettz
import requests
import boto3
from langchain.embeddings.bedrock import BedrockEmbeddings
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, ArrayType

def download_latest_xml(url):
    data_folder = '../data'

    # Clear existing files in the data folder
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(data_folder, exist_ok=True)

    print("Old files deleted.")

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'usa-table'})
    rows = table.find_all('tr')
    latest_xml_url = None
    latest_datetime = None

    tzinfos = {'EDT': gettz('EDT')}

    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) >= 3:
            file_url = cells[0].find('a')['href']
            datetime_str = cells[2].text.strip()
            datetime_obj = parser.parse(datetime_str, tzinfos=tzinfos)

            if latest_datetime is None or datetime_obj > latest_datetime:
                latest_xml_url = file_url
                latest_datetime = datetime_obj

    response = requests.get(latest_xml_url)

    if response.status_code == 200:
        zip_filename = latest_xml_url.split("/")[-1]

        # Save the downloaded file to the 'data' folder
        zip_file_path = os.path.join(data_folder, zip_filename)
        with open(zip_file_path, "wb") as file:
            file.write(response.content)
        print(f"File '{zip_filename}' downloaded successfully to folder.")

        # Unzip the file to the 'data' folder
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

        print(f"File '{zip_filename}' unzipped to folder.")

        # Delete the downloaded ZIP file
        os.remove(zip_file_path)
        print(f"Deleted the downloaded ZIP file '{zip_filename}'.")

    else:
        print('file not downloaded')

def push_to_snowflake(url):
    
    download_latest_xml(url)
    
    data_folder = '../data'

    file_name = os.listdir(data_folder)
    latest_xml_file = os.path.join(data_folder, file_name[0])

    config = configparser.ConfigParser()
    config_path = os.path.join("..", "credentials.ini")
    config.read(config_path)

    # Create a Snowflake session
    session = Session.builder.configs({
        'account': config.get('bonterra_snowflake', 'Account'),
        'user': config.get('bonterra_snowflake', 'User'),
        'password': config.get('bonterra_snowflake', 'Password'),
        'role': config.get('bonterra_snowflake', 'Role'),
        'warehouse': config.get('bonterra_snowflake', 'Warehouse'),
        'database': config.get('bonterra_snowflake', 'Database'),
        'schema': config.get('bonterra_snowflake', 'Schema')
    }).create()

    existing_table = session.read.table("FEDERAL_GRANT_OPS")
    existing_id_list = [row[0] for row in existing_table.select('OPPORTUNITY_ID').collect()] # GET LIST OF CURRENT OPPORTUNITY IDS

    col_dict = {
    'OpportunityID': 'OPPORTUNITY_ID',
    'OpportunityTitle': 'OPPORTUNITY_TITLE',
    'OpportunityNumber': 'OPPORTUNITY_NUMBER',
    'OpportunityCategory': 'OPPORTUNITY_CATEGORY',
    'FundingInstrumentType': 'FUNDING_INSTRUMENT_TYPE',
    'CategoryOfFundingActivity': 'CATEGORY_OF_FUNDING_ACTIVITY',
    'CategoryExplanation': 'CATEGORY_EXPLANATION',
    'CFDANumbers': 'CFDA_NUMBERS',
    'EligibleApplicants': 'ELIGIBLE_APPLICANTS',
    'AdditionalInformationOnEligibility': 'ADDITIONAL_INFORMATION_ON_ELIGIBILITY',
    'AgencyCode': 'AGENCY_CODE',
    'AgencyName': 'AGENCY_NAME',
    'PostDate': 'POST_DATE',
    'CloseDate': 'CLOSE_DATE',
    'LastUpdatedDate': 'LAST_UPDATED_DATE',
    'AwardCeiling': 'AWARD_CEILING',
    'AwardFloor': 'AWARD_FLOOR',
    'EstimatedTotalProgramFunding': 'ESTIMATED_TOTAL_PROGRAM_FUNDING',
    'ExpectedNumberOfAwards': 'EXPECTED_NUMBER_OF_AWARDS',
    'Description': 'DESCRIPTION',
    'Version': 'VERSION',
    'CostSharingOrMatchingRequirement': 'COST_SHARING_OR_MATCHING_REQUIREMENT',
    'ArchiveDate': 'ARCHIVE_DATE',
    'GrantorContactEmail': 'GRANTOR_CONTACT_EMAIL',
    'GrantorContactEmailDescription': 'GRANTOR_CONTACT_EMAIL_DESCRIPTION',
    'GrantorContactText': 'GRANTOR_CONTACT_TEXT',
    'AdditionalInformationURL': 'ADDITIONAL_INFORMATION_URL',
    'AdditionalInformationText': 'ADDITIONAL_INFORMATION_TEXT',
    'CloseDateExplanation': 'CLOSE_DATE_EXPLANATION',
    'OpportunityCategoryExplanation': 'OPPORTUNITY_CATEGORY_EXPLANATION',
    'EstimatedSynopsisPostDate': 'ESTIMATED_SYNOPSIS_POST_DATE',
    'FiscalYear': 'FISCAL_YEAR',
    'EstimatedSynopsisCloseDate': 'ESTIMATED_SYNOPSIS_CLOSE_DATE',
    'EstimatedSynopsisCloseDateExplanation': 'ESTIMATED_SYNOPSIS_CLOSE_DATE_EXPLANATION',
    'EstimatedAwardDate': 'ESTIMATED_AWARD_DATE',
    'EstimatedProjectStartDate': 'ESTIMATED_PROJECT_START_DATE',
    'GrantorContactName': 'GRANTOR_CONTACT_NAME',
    'GrantorContactPhoneNumber': 'GRANTOR_CONTACT_PHONE_NUMBER'
    }
    
    df = pd.read_xml(latest_xml_file) # READ LOCAL XML FILE
    df = df.rename(columns=col_dict)
    time_window = pd.Timestamp.now() - pd.DateOffset(days=7)
    df = df[pd.to_datetime(df['POST_DATE'], format='%m%d%Y') > time_window] # ONLY LOOK AT AWARDS THAT HAVE BEEN AWARDED IN THE LAST 7 DAYS
    df['concat_text'] = df[['OPPORTUNITY_TITLE','AGENCY_NAME','ADDITIONAL_INFORMATION_ON_ELIGIBILITY','DESCRIPTION']].apply(lambda row: ' '.join(str(val) for val in row), axis=1) # CREATE TEXT COLUMN FOR EMBEDDING
    df['concat_text'] = df['concat_text'].apply(lambda x: x[:2048]) # MAX LENGTH FOR COHERE EMBEDDINGS
    
    # FILTER FOR ONLY ROWS THAT DOESN'T EXIST IN THE CURRENT TABLE
    only_new_df = df[~df['OPPORTUNITY_ID'].isin(existing_id_list)].reset_index(drop=True)
    print(len(only_new_df),"new rows will be added.")

    # ADD EMBEDDINGS FOR THOSE ROWS
    embed_session = boto3.session.Session(
        aws_access_key_id=config["AWS"]["AWS_ACCESS_KEY"],
        aws_secret_access_key=config["AWS"]["AWS_SECRET_ACCESS_KEY"]
        )
    
    embed_client = embed_session.client("bedrock-runtime", region_name=config["AWS"]["AWS_REGION"])

    embedder = BedrockEmbeddings(client=embed_client, model_id="cohere.embed-english-v3")

    embeddings = embedder.embed_documents(only_new_df['concat_text'])

    only_new_df.loc[:, 'EMBEDDED_TEXT'] = embeddings

    only_new_df = only_new_df.drop(columns=['concat_text'])

    # ORDER COLUMNS TO MATCH EXISTING TABLE
    only_new_df = only_new_df[existing_table.columns]

    # CREATE SNOWPARK DATAFRAME
    snowpark_df = session.create_dataframe(only_new_df)

    # COPY INTO STAGE
    snowpark_df.write.copy_into_location("@fed_grant_stage/new_funding_ops", overwrite=True, single=True)
    print("Rows pushed to stage.")

    csv_schema = StructType([
        StructField("OPPORTUNITY_ID", StringType()),
        StructField("OPPORTUNITY_TITLE", StringType()),
        StructField("OPPORTUNITY_NUMBER", StringType()),
        StructField("OPPORTUNITY_CATEGORY", StringType()),
        StructField("FUNDING_INSTRUMENT_TYPE", StringType()),
        StructField("CATEGORY_OF_FUNDING_ACTIVITY", StringType()),
        StructField("CATEGORY_EXPLANATION", StringType()),
        StructField("CFDA_NUMBERS", StringType()),
        StructField("ELIGIBLE_APPLICANTS", StringType()),
        StructField("ADDITIONAL_INFORMATION_ON_ELIGIBILITY", StringType()),
        StructField("AGENCY_CODE", StringType()),
        StructField("AGENCY_NAME", StringType()),
        StructField("POST_DATE", StringType()),
        StructField("CLOSE_DATE", StringType()),
        StructField("LAST_UPDATED_DATE", StringType()),
        StructField("AWARD_CEILING", IntegerType()),
        StructField("AWARD_FLOOR", IntegerType()),
        StructField("ESTIMATED_TOTAL_PROGRAM_FUNDING", IntegerType()),
        StructField("EXPECTED_NUMBER_OF_AWARDS", StringType()),
        StructField("DESCRIPTION", StringType()),
        StructField("VERSION", StringType()),
        StructField("COST_SHARING_OR_MATCHING_REQUIREMENT", StringType()),
        StructField("ARCHIVE_DATE", StringType()),
        StructField("GRANTOR_CONTACT_EMAIL", StringType()),
        StructField("GRANTOR_CONTACT_EMAIL_DESCRIPTION", StringType()),
        StructField("GRANTOR_CONTACT_TEXT", StringType()),
        StructField("FISCAL_YEAR", IntegerType()),
        StructField("ESTIMATED_SYNOPSIS_CLOSE_DATE", StringType()),
        StructField("ESTIMATED_AWARD_DATE", StringType()),
        StructField("ESTIMATED_PROJECT_START_DATE", StringType()),
        StructField("GRANTOR_CONTACT_NAME", StringType()),
        StructField("GRANTOR_CONTACT_PHONE_NUMBER", StringType()),
        StructField("ESTIMATED_SYNOPSIS_CLOSE_DATE_EXPLANATION", StringType()),
        StructField("OPPORTUNITY_CATEGORY_EXPLANATION", StringType()),
        StructField("CLOSE_DATE_EXPLANATION", StringType()),
        StructField("ADDITIONAL_INFORMATION_TEXT", StringType()),
        StructField("ESTIMATED_SYNOPSIS_POST_DATE", StringType()),
        StructField("ADDITIONAL_INFORMATION_URL", StringType()),
        StructField("EMBEDDED_TEXT", ArrayType())
    ])

    new_rows_from_staging = session.read.schema(csv_schema).csv("@fed_grant_stage/new_funding_ops")

    new_rows_from_staging.copy_into_table('federal_grant_ops')
    print("New rows added to existing federal ops table.")

    # REMOVE FILE FROM STAGE
    session.sql("remove @fed_grant_stage/new_funding_ops").collect()
    print("New rows removed from stage.")

if __name__ == '__main__':
    push_to_snowflake('https://www.grants.gov/xml-extract.html')