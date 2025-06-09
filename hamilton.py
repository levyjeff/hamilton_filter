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
        #puts the correct NaNs into the series
        d = pd.concat([data, cycle, trend, rand], axis=1)
        cycle, trend, rand = d[f'{data.name}.cycle'], d[f'{data.name}.trend'], d[f'{data.name}.rand']
    else:
        cycle = res.resid_pearson

    return cycle, trend, rand


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.filters.hp_filter import hpfilter

    gdpc1 = pd.read_csv(r'data\gdpc1.csv', parse_dates=['DATE'], index_col='DATE')
    gdpc1['GDPC1'] = 100*np.log(gdpc1['GDPC1'])
    c, t, r = hamilton_filter(gdpc1['GDPC1'], h=8, p=4)
    df = pd.concat([gdpc1, t, c, r], axis=1)

    hp_c, hp_t = hpfilter(gdpc1['GDPC1'], lamb=129600)

    fig, axs = plt.subplots(3, 1, figsize=(12,6), sharex=True)
    axs[0].plot(gdpc1.index, gdpc1['GDPC1'], 'k-')
    axs[1].plot(c.index, c.values, 'b-', label='Hamilton Filter')
    axs[1].plot(hp_c.index, hp_c.values, 'r-', label='HP Filter')
    axs[2].plot(t.index, t.values, 'b-')
    axs[2].plot(hp_t.index, hp_t.values, 'r-')
    axs[0].set_ylabel('GDPC1')
    axs[1].set_ylabel('Cycle')
    axs[2].set_ylabel('Trend')
    fig.legend(ncols=3, loc='upper center')
    plt.show()