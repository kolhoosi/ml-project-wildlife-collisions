import numpy as np
import pandas as pd


def speed_limits(df):
    """Calculates speed limits based on road type"""

    ''' These are MAX speed limits for each road type. It is possible and likely that many of the roads have a lower 
        speed limit than this, however finding data about the speed limit for each road in Finland is outside the scope 
        of this project. Therefore I will be using maximum speed limits instead.
        https://vayla.fi/-/nopeusrajoituksilla-turvallisempaa-sujuvampaa-ja-haitattomampaa-liikennetta'''

    MAANTIE = 80        # Seututie, kantatie and 'muu maantie' can have a max limit of 80: let's call them all 'maantie'
    VALTATIE = 100      # most common speed limit for this road type (main roads)
    MOOTTORITIE = 120   # most common speed limit for highways in Uusimaa
    TAAJAMA = 50        # kunnan tie = streets inside municipalities within 'taajama' urban areas with max limit of 50

    def categorize(row):
        """Assigning the correct maximum speed limit based on road type"""

        if row['road_type'] == 'Muu maantie' or row['road_type'] == 'Kantatie' or row['road_type'] == 'Seututie':
            return MAANTIE
        elif row['road_type'] == 'Valtatie':
            return VALTATIE
        elif row['road_type'] == 'Moottoritie':
            return MOOTTORITIE
        else:
            return TAAJAMA

    df['road_type'] = df.apply(lambda row: categorize(row), axis=1)
    df = df.rename(columns={'road_type': 'limit'})

    return df


def parse_data():
    df = pd.read_csv('koeti_riista_tau_001_fi.csv', encoding='latin-1')  # reading the data into a pandas dataframe
    '''For some reason the encoding has to be in latin-1 and not UTF. See this issue for more info:
    https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte'''

    # Renaming columns to English
    df.columns = ['id', 'datetime', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'region_name',
                  'road_type_id', 'road_type', 'road_number', 'admin', 'admin_name', 'species', 'species_name']

    # Remove any rows from dataframe "df" which contain missing values
    df = df.dropna(axis=0)

    # Drop unnecessary columns: only datetime, region and road type information are needed
    data = df.drop(['id', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'road_type_id', 'road_number', 'admin',
                    'admin_name', 'species', 'species_name'], axis=1)

    # Rework datetime into separate date and time columns
    data[['date', 'time']] = data['datetime'].str.split('T', 1, expand=True)
    # For the purposes of this projet we need to know the time only at a precision of 60 minutes
    data['time'] = data['time'].str.slice(0, 2)

    '''Narrowing the selection to only Uusimaa region and
       choosing years 2021 (for prediction) and 2020 (for training)'''
    data = data[data['date'].str.contains("2020|2021")]
    data = data[data['region_name'] == "Uusimaa"]

    data = data.drop(['region_name', 'datetime'], axis=1)  # Drop unnecessary columns
    data = speed_limits(data)  # Deriving speed limits from road type information

    return data


def main():
    data = parse_data()

    # TODO: linear regression for data


if __name__ == "__main__":
    main()
