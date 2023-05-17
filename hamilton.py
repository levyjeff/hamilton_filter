import numpy as np
import statsmodels.api as sm

def hamilton_filter(data, h=8, p=4):
    #implementation of the Hamilton (2017) alternative to the HP filter

    def shift(orig_series, n):
        #implements efficient (positive) shifting for non-Series dtypes
        new_series = np.empty_like(orig_series)
        new_series[:n] = np.NaN
        new_series[n:] = orig_series[:-n]
        return new_series

    new_cols = [shift(data, s) for s in range(h, h+p)]

    exog = sm.add_constant(np.array(new_cols).transpose())
    model = sm.GLM(endog=data, exog=exog, missing='drop')
    res = model.fit()

    trend = res.fittedvalues
    rand = data - shift(data, h)
    if isinstance(data, pd.Series):
        cycle = pd.Series(res.resid_pearson, 
                          index=trend.index, 
                          name=f'{data.name}.cycle')
        trend.name = f'{data.name}.trend'
        rand.name = f'{data.name}.rand'
    else:
        cycle = res.resid_pearson

    return cycle, trend, rand


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    gdpc1 = pd.read_csv('gdpc1.csv', parse_dates=['DATE'], index_col='DATE')
    gdpc1['GDPC1'] = 100*np.log(gdpc1['GDPC1'])
    c, t, r = hamilton_filter(gdpc1['GDPC1'], h=8, p=4)
    df = pd.concat([gdpc1, t, c, r], axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(12,6))
    axs[0].plot(c.index, c.values, 'b-', label=c.name)
    axs[1].plot(t.index, t.values, 'r-', label=t.name)
    fig.legend(ncols=2, loc='upper center')
    plt.show()