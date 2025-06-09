"""
Copyright (C) 2022 Jeffrey A Levy

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd

def hamilton_filter(data: pd.Series, h: int=8, p: int=4) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Implementation of the Hamilton (2017) alternative to the HP filter.
    "It is also desirable with seasonal data to have both p and h be integer 
     multiples of the number of observations in a year. Hence, for quarterly 
     data, my recommendation is p = 4 and h = 8" - Hamilton 2018

    Arguments:
    data -- a Pandas Series (required)
    h -- look-ahead horizon based on two year business cycles (default 8 for quarterly data)
    p -- lags, corresponding to seasonality component of h (default 4 for quarterly data)

    Returns:
    cycle, trend, and random components matching the dtype of data 
    """

    name = data.name
    data = data.to_frame()

    for i in range(p):
        data[name+f'_{i}'] = data[name].shift(h+i)

    model = sm.GLM(endog=data.iloc[:,0], exog=sm.add_constant(data.iloc[:,1:]), missing='drop')
    res = model.fit()

    trend = res.fittedvalues
    rand = data[name] - data[name].shift(h)

    cycle = pd.Series(res.resid_pearson, 
                        index=trend.index, 
                        name=f'{name}.cycle')
    trend.name = f'{name}.trend'
    rand.name = f'{name}.rand'
    #puts the correct NaNs into the series
    d = pd.concat([data, cycle, trend, rand], axis=1)
    cycle, trend, rand = d[f'{name}.cycle'], d[f'{name}.trend'], d[f'{name}.rand']

    return cycle, trend, rand

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from statsmodels.tsa.filters.hp_filter import hpfilter

    gdpc1 = pd.read_csv(r'data\gdpc1.csv', parse_dates=['DATE'], index_col='DATE')
    gdpc1['GDPC1'] = 100*np.log(gdpc1['GDPC1'])
    c, t, r = hamilton_filter(gdpc1['GDPC1'], h=8, p=4)
    df = pd.concat([gdpc1, t, c, r], axis=1)
    # identical to results in Hamilton, compare to R package: 
    # https://cran.r-project.org/web/packages/neverhpfilter/vignettes/Reproducing-Hamilton.html

    hp_c, hp_t = hpfilter(gdpc1['GDPC1'], lamb=1600)

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