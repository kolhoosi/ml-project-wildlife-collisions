import numpy as np
import pandas as pd


def parse_data():
    df = pd.read_csv('koeti_riista_tau_001_fi.csv', encoding='latin-1')  # reading the data into a pandas dataframe
    '''For some reason the encoding has to be in latin-1 and not UTF. See this issue for more info:
    https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte'''

    # Renaming columns to English
    df.columns = ['id', 'datetime', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'region_name',
                  'road_type_name', 'road_type', 'road_number', 'admin', 'admin_name', 'species', 'species_name']

    # Remove any rows from dataframe "df" which contain missing values
    df = df.dropna(axis=0)

    # Drop unnecessary columns: only datetime, region and road type information are needed
    data = df.drop(['id', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'road_number', 'admin',
                    'admin_name', 'species', 'species_name'], axis=1)

    # rework datetime into separate date and time columns
    data[['date', 'time']] = data['datetime'].str.split('T', 1, expand=True)

    # TODO: format date and time correctly, drop datetime column
    # TODO: derive speed limits from road type information


def main():
    parse_data()

    # TODO: linear regression for data


if __name__ == "__main__":
    main()