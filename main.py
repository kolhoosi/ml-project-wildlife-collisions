import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


def parse_data():
    df = pd.read_csv('koeti_riista_tau_001_fi.csv', encoding='latin-1')  # reading the data into a pandas dataframe
    '''For some reason the encoding has to be in latin-1 and not UTF. See this issue for more info:
    https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte'''

    # Renaming columns to English
    df.columns = ['id', 'datetime', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'region_name',
                  'road_type_id', 'road_type', 'road_number', 'admin', 'admin_name', 'species', 'species_name']

    # Remove any rows from dataframe "df" which contain missing values
    df = df.dropna(axis=0)

    # Drop unnecessary columns: only datetime and region information are needed
    data = df.drop(['id', 'year', 'month', 'x', 'y', 'city', 'city_name', 'region', 'road_type', 'road_type_id',
                    'road_number', 'admin', 'admin_name', 'species', 'species_name'], axis=1)

    # Rework datetime into separate date and time columns
    data[['date', 'time']] = data['datetime'].str.split('T', 1, expand=True)
    # For the purposes of this projet we need to know the time only at a precision of 60 minutes
    data['time'] = data['time'].str.slice(0, 5)

    # Choosing years 2021 (for validation and testing) and 2020 (for training)
    data = data[data['date'].str.contains("2020|2021")]
    data = data[data['region_name'] == "Uusimaa"]  # Narrowing the selection to only Uusimaa region

    data = data.drop(['region_name', 'datetime'], axis=1)  # Drop unnecessary columns

    return data


def poly_regr(x_tr, y_tr, x_test, y_test, x_val, y_val):
    """Polynomial regression training for the dataset"""

    degrees = [5, 8, 10]  # these degrees were a result of trial and error
    train_errors = []
    test_errors = []
    val_errors = []

    x_tr = x_tr.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)

    # rename x labels
    positions = (0, 36, 72, 108, 144)
    labels = ("00:00", "06:00", "12:00", "18:00", "24:00")

    for degree in degrees:

        lin_regr = LinearRegression()
        poly = PolynomialFeatures(degree=degree)
        x_train_poly = poly.fit_transform(x_tr)
        lin_regr.fit(x_train_poly, y_tr)

        y_pred_train = lin_regr.predict(x_train_poly)
        tr_error = mean_squared_error(y_tr, y_pred_train) * 100  # dataset has been normalized

        X_test_poly = poly.fit_transform(x_test)
        y_pred_test = lin_regr.predict(X_test_poly)
        test_error = mean_squared_error(y_test, y_pred_test) * 100  # dataset has been normalized

        X_val_poly = poly.fit_transform(x_val)
        y_pred_val = lin_regr.predict(X_val_poly)
        val_error = mean_squared_error(y_val, y_pred_val) * 100  # dataset has been normalized

        train_errors.append(tr_error)
        test_errors.append(test_error)
        val_errors.append(val_error)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(x_tr, y_tr, color="r", s=10, label="Train Datapoints")
        ax[0].plot(x_tr, y_pred_train, "black", label="Polynomial regression", linewidth=2.0)
        ax[0].legend(loc="best")
        ax[0].set_xlabel('time of day')  # set the label for the x/y-axis
        ax[0].set_ylabel('accidents')
        ax[0].set_title(f'Training error = {tr_error:.5}')
        ax[0].set_xticks(positions)
        ax[0].set_xticklabels(labels)

        ax[1].scatter(x_test, y_test, color="g", s=10, label="Test Datapoints")
        ax[1].plot(x_test, y_pred_test, "black", label="Polynomial regression", linewidth=2.0)
        ax[1].legend(loc="best")
        ax[1].set_xlabel('time of day')  # set the label for the x/y-axis
        ax[1].set_ylabel('accidents')
        ax[1].set_title(f'Testing error = {test_error:.5}')
        ax[1].set_xticks(positions)
        ax[1].set_xticklabels(labels)

        ax[2].scatter(x_val, y_val, color="b", s=10, label="Validation Datapoints")
        ax[2].plot(x_val, y_pred_val, "black", label="Polynomial regression", linewidth=2.0)
        ax[2].legend(loc="best")
        ax[2].set_xlabel('time of day')  # set the label for the x/y-axis
        ax[2].set_ylabel('accidents')
        ax[2].set_title(f'Validation error = {val_error:.5}')
        ax[2].set_xticks(positions)
        ax[2].set_xticklabels(labels)

        fig.suptitle(f'Polynomial regression with polynomial degree = {degree}\n')
        plt.savefig('poly'+str(degree))
        plt.show()


def linear_regr(x_tr, y_tr, x_test, y_test, x_val, y_val):
    """Linear regression training for the dataset"""

    train_errors = []
    test_errors = []
    val_errors = []

    x_tr = x_tr.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)

    # rename x labels
    positions = (0, 36, 72, 108, 144)
    labels = ("00:00", "06:00", "12:00", "18:00", "24:00")

    lin_regr = LinearRegression()
    lin_regr.fit(x_tr, y_tr)
    y_pred_train = lin_regr.predict(x_tr)
    tr_error = mean_squared_error(y_tr, y_pred_train) * 100  # dataset has been normalized

    y_pred_test = lin_regr.predict(x_test)
    test_error = mean_squared_error(y_test, y_pred_test) * 100  # normalized

    y_pred_val = lin_regr.predict(x_val)
    val_error = mean_squared_error(y_val, y_pred_val) * 100  # normalized

    train_errors.append(tr_error)
    test_errors.append(test_error)
    val_errors.append(val_error)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(x_tr, y_tr, color="r", s=10, label="Train Datapoints")
    ax[0].plot(x_tr, y_pred_train, "black", label="Linear regression", linewidth=2.0)
    ax[0].legend(loc="best")
    ax[0].set_xlabel('time of day')  # set the label for the x/y-axis
    ax[0].set_ylabel('accidents')
    ax[0].set_title(f'Training error = {tr_error:.5}')
    ax[0].set_xticks(positions)
    ax[0].set_xticklabels(labels)

    ax[1].scatter(x_test, y_test, color="g", s=10, label="Test Datapoints")
    ax[1].plot(x_test, y_pred_test, "black", label="Linear regression", linewidth=2.0)
    ax[1].legend(loc="best")
    ax[1].set_xlabel('time of day')  # set the label for the x/y-axis
    ax[1].set_ylabel('accidents')
    ax[1].set_title(f'Testing error = {test_error:.5}')
    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(labels)

    ax[2].scatter(x_val, y_val, color="b", s=10, label="Validation Datapoints")
    ax[2].plot(x_val, y_pred_val, "black", label="Linear regression", linewidth=2.0)
    ax[2].legend(loc="best")
    ax[2].set_xlabel('time of day')  # set the label for the x/y-axis
    ax[2].set_ylabel('accidents')
    ax[2].set_title(f'Validation error = {val_error:.5}')
    ax[2].set_xticks(positions)
    ax[2].set_xticklabels(labels)

    fig.suptitle(f'Linear regression with ordinary least squares loss\n')
    plt.savefig('linear')
    plt.show()


def huber_regr(x_tr, y_tr, x_test, y_test, x_val, y_val):
    """Huber regression training for the dataset"""

    epsilon_values = [1, 1.5, 1.75]  # these degrees were a result of trial and error
    train_errors = []
    test_errors = []
    val_errors = []

    x_tr = x_tr.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_val = x_val.reshape(-1, 1)

    # rename x labels
    positions = (0, 36, 72, 108, 144)
    labels = ("00:00", "06:00", "12:00", "18:00", "24:00")

    for i, epsilon in enumerate(epsilon_values):
        # plt.subplot(len(degrees), 1, i + 1)

        lin_regr = LinearRegression()
        lin_regr.fit(x_tr, y_tr)
        huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
        huber.fit(x_tr, y_tr)
        coef = huber.coef_ * x_tr + huber.intercept_

        y_pred_train = huber.predict(x_tr)
        tr_error = mean_squared_error(y_tr, y_pred_train) * 100  # dataset has been normalized

        y_pred_test = huber.predict(x_test)
        test_error = mean_squared_error(y_test, y_pred_test) * 100  # normalized

        y_pred_val = huber.predict(x_val)
        val_error = mean_squared_error(y_val, y_pred_val) * 100  # normalized

        train_errors.append(tr_error)
        test_errors.append(test_error)
        val_errors.append(val_error)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(x_tr, y_tr, color="r", s=10, label="Train Datapoints")
        ax[0].plot(x_tr, y_pred_train, "black", label="Linear regression", linewidth=2.0)
        ax[0].legend(loc="best")
        ax[0].set_xlabel('time of day')  # set the label for the x/y-axis
        ax[0].set_ylabel('accidents')
        ax[0].set_title(f'Training error = {tr_error:.5}')
        ax[0].set_xticks(positions)
        ax[0].set_xticklabels(labels)

        ax[1].scatter(x_test, y_test, color="g", s=10, label="Test Datapoints")
        ax[1].plot(x_test, y_pred_test, "black", label="Linear regression", linewidth=2.0)
        ax[1].legend(loc="best")
        ax[1].set_xlabel('time of day')  # set the label for the x/y-axis
        ax[1].set_ylabel('accidents')
        ax[1].set_title(f'Testing error = {test_error:.5}')
        ax[1].set_xticks(positions)
        ax[1].set_xticklabels(labels)

        ax[2].scatter(x_val, y_val, color="b", s=10, label="Validation Datapoints")
        ax[2].plot(x_val, y_pred_val, "black", label="Linear regression", linewidth=2.0)
        ax[2].legend(loc="best")
        ax[2].set_xlabel('time of day')  # set the label for the x/y-axis
        ax[2].set_ylabel('accidents')
        ax[2].set_title(f'Validation error = {val_error:.5}')
        ax[2].set_xticks(positions)
        ax[2].set_xticklabels(labels)

        fig.suptitle(f'Linear regression with huber loss, epsilon={epsilon} \n')
        plt.savefig('huber_eps'+str(epsilon)+'.png')
        plt.show()


def main():
    data = parse_data()  # data now has 4815 rows

    # Split data into training and remaining sets by making 2020 training data
    s_train = data[:][data.date < '2020-12-31']
    s_rem = data[:][data.date > '2020-12-31']

    # randomly split s_rem into testing and validation sets
    s_test, s_val = train_test_split(s_rem, test_size=0.5)

    # Sort values
    train = s_train.sort_values(by=['time'])
    test = s_test.sort_values(by=['time'])
    valid = s_val.sort_values(by=['time'])

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
    d_val = np.zeros(len(arr) + 2)

    for i in range(len(arr)):
        if i == 0:
            d_train[i] = len(train[train['time'] <= str(arr[i])])
            d_test[i] = len(test[test['time'] <= str(arr[i])])
            d_val[i] = len(valid[valid['time'] <= str(arr[i])])

        else:
            d_train[i] = len(train[(train['time'] <= str(arr[i])) & (train['time'] > str(arr[i - 1]))])
            d_test[i] = len(test[(test['time'] <= str(arr[i])) & (test['time'] > str(arr[i - 1]))])
            d_val[i] = len(valid[(valid['time'] <= str(arr[i])) & (valid['time'] > str(arr[i - 1]))])

    # d_train[-2] = len(train[(train['time'] <= '23:55') & (train['time'] > '23:50')])
    # d_train[-1] = len(train[(train['time'] <= '23:59') & (train['time'] > '23:55')])

    # d_test[-2] = len(test[(test['time'] <= '23:55') & (test['time'] > '23:50')])
    # d_test[-1] = len(test[(test['time'] <= '23:59') & (test['time'] > '23:55')])

    # removing outliers
    # d_train = d_train[abs(d_train - np.mean(d_train)) < 2 * np.std(d_train)]
    # d_test = d_test[abs(d_test - np.mean(d_test)) < 2 * np.std(d_test)]

    # normalizing
    d_train_max = d_train.max()
    n_d_train = d_train / d_train_max

    d_test_max = d_test.max()
    n_d_test = d_test / d_test_max

    d_val_max = d_val.max()
    n_d_val = d_val / d_val_max

    times_train = np.arange(0, len(d_train), 1)  # values from 0 to 47 corresponding to times between 00:00 to 23:30
    times_test = np.arange(0, len(d_test), 1)
    times_val = np.arange(0, len(d_val), 1)

    poly_regr(times_train, n_d_train, times_test, n_d_test, times_val, n_d_val)
    linear_regr(times_train, n_d_train, times_test, n_d_test, times_val, n_d_val)
    huber_regr(times_train, n_d_train, times_test, n_d_test, times_val, n_d_val)


if __name__ == "__main__":
    main()
