import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


def speed_limits(df):
    """Calculates speed limits based on road type"""

    ''' These are MAX speed limits for each road type. It is possible and likely that many of the roads have a lower 
        speed limit than this, however finding data about the speed limit for each road in Finland is outside the scope 
        of this project. Therefore I will be using maximum speed limits instead.
        https://vayla.fi/-/nopeusrajoituksilla-turvallisempaa-sujuvampaa-ja-haitattomampaa-liikennetta'''

    MAANTIE = 80  # Seututie, kantatie and 'muu maantie' can have a max limit of 80: let's call them all 'maantie'
    VALTATIE = 100  # most common speed limit for highways and other main roads
    TAAJAMA = 50  # kunnan tie = streets inside municipalities within 'taajama' urban areas with max limit of 50

    def categorize(row):
        """Assigning the correct maximum speed limit based on road type"""

        if row['road_type'] == 'Muu maantie' or row['road_type'] == 'Kantatie' or row['road_type'] == 'Seututie':
            return MAANTIE
        elif row['road_type'] == 'Valtatie' or row['road_type'] == 'Moottoritie':
            return VALTATIE
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
    data = df.drop(['id', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'road_type_id', 'road_number',
                    'admin', 'admin_name', 'species', 'species_name'], axis=1)

    # Rework datetime into separate date and time columns
    data[['date', 'time']] = data['datetime'].str.split('T', 1, expand=True)
    # For the purposes of this projet we need to know the time only at a precision of 60 minutes
    data['time'] = data['time'].str.slice(0, 5)

    # Choosing years 2021 (for validation and testing) and 2020 (for training)
    data = data[data['date'].str.contains("2020|2021")]
    data = data[data['region_name'] == "Uusimaa"]  # Narrowing the selection to only Uusimaa region

    data = data.drop(['region_name', 'datetime'], axis=1)  # Drop unnecessary columns
    data = speed_limits(data)  # Deriving speed limits from road type information

    return data


def poly_regr(x_tr, y_tr, x_val, y_val):
    """Polynomial regression training for the dataset"""

    degrees = [5, 8, 10, 12]  # these degrees were a result of trial and error
    tr_errors = []
    val_errors = []

    x_tr = x_tr.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)

    # rename x labels
    positions = (0, 12, 24, 36, 48)
    labels = ("00:00", "06:00", "12:00", "18:00", "24:00")

    for i, degree in enumerate(degrees):
        # plt.subplot(len(degrees), 1, i + 1)

        lin_regr = LinearRegression()
        poly = PolynomialFeatures(degree=degree)
        x_train_poly = poly.fit_transform(x_tr)
        lin_regr.fit(x_train_poly, y_tr)

        y_pred_train = lin_regr.predict(x_train_poly)
        tr_error = mean_squared_error(y_tr, y_pred_train) * 100  # dataset has been normalized
        X_val_poly = poly.fit_transform(x_val)
        y_pred_val = lin_regr.predict(X_val_poly)
        val_error = mean_squared_error(y_val, y_pred_val) * 100  # dataset has been normalized

        tr_errors.append(tr_error)
        val_errors.append(val_error)

        X_fit = np.linspace(-25, 50, 100)
        plt.tight_layout()
        plt.xlim([0, 48])
        plt.ylim([0, 1])
        plt.xticks(positions, labels)
        plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))),
                 label="Model")  # plot the polynomial regression model
        plt.scatter(x_tr, y_tr, color="b", s=10, label="Train Datapoints")
        # plot a scatter plot of y(maxtmp) vs. X(mintmp) with color 'blue' and size '10'
        plt.scatter(x_val, y_val, color="r", s=10, label="Validation Datapoints")
        # do the same for validation data with color 'red'
        plt.xlabel('time of day')  # set the label for the x/y-axis
        plt.ylabel('accidents')
        plt.legend(loc="best")  # set the location of the legend
        plt.title(f'Polynomial degree = {degree}\nTraining error = {tr_error:.5}\nValidation error = {val_error:.5}')
        # set the title

        plt.show()


def main():
    data = parse_data()  # data now has 4815 rows

    # Split data into training and test/validation sets into 50/50 split
    s_train = data[:][data.date < '2020-12-31']
    s_test = data[:][data.date > '2020-12-31']

    # Sort values
    train = s_train.sort_values(by=['time'])
    test = s_test.sort_values(by=['time'])

    # Create starting and end datetime object from string
    start = datetime.strptime("00:00", "%H:%M")
    end = datetime.strptime("23:59", "%H:%M")

    # min_gap
    min_gap = 10

    # compute datetime interval
    arr = [(start + timedelta(hours=min_gap * i / 60)).strftime("%H:%M")
           for i in range(int((end - start).total_seconds() / 60.0 / min_gap))]

    d_train = np.zeros((len(arr) + 2))
    d_test = np.zeros((len(arr) + 2))

    for i in range(len(arr)):
        if i == 0:
            d_train[i] = len(train[train['time'] <= str(arr[i])])
            d_test[i] = len(test[test['time'] <= str(arr[i])])

        else:
            d_train[i] = len(train[(train['time'] <= str(arr[i])) & (train['time'] > str(arr[i - 1]))])
            d_test[i] = len(test[(test['time'] <= str(arr[i])) & (test['time'] > str(arr[i - 1]))])

    # d_train[-2] = len(train[(train['time'] <= '23:55') & (train['time'] > '23:50')])
    # d_train[-1] = len(train[(train['time'] <= '23:59') & (train['time'] > '23:55')])

    # d_test[-2] = len(test[(test['time'] <= '23:55') & (test['time'] > '23:50')])
    # d_test[-1] = len(test[(test['time'] <= '23:59') & (test['time'] > '23:55')])

    # removing outliers
    #d_train = d_train[abs(d_train - np.mean(d_train)) < 2 * np.std(d_train)]
    #d_test = d_test[abs(d_test - np.mean(d_test)) < 2 * np.std(d_test)]

    # normalizing
    d_train_max = d_train.max()
    n_d_train = d_train / d_train_max

    d_test_max = d_test.max()
    n_d_test = d_test / d_test_max


    times_train = np.arange(0, len(d_train), 1)  # values from 0 to 47 corresponding to times between 00:00 to 23:30
    times_test = np.arange(0, len(d_test), 1)

    poly_regr(times_train, n_d_train, times_test, n_d_test)


if __name__ == "__main__":
    main()
