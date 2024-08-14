import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.stattools as ts


def dftest(timeseries):
    test = ts.adfuller(
        timeseries,
    )
    dfoutput = pd.Series(
        test[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Observations Used']
    )
    for key, value in test[4].items():
        dfoutput[f'Critical Value ({key})%'] = value
    print(dfoutput)
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plot rolling statistics:
    _ = plt.plot(timeseries, color='blue', label='Original')
    _ = plt.plot(rolmean, color='red', label='Rolling Mean')
    _ = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.grid()
    plt.show(block=False)


def generate_future_timestamps(start_time, num_predictions, interval_hours):
    future_times = pd.date_range(
        start=start_time, periods=num_predictions + 1, freq=f'{interval_hours}T'
    )[1:]
    return future_times
