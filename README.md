# Hamilton Filter
Implements the alternative to the Hodrick-Prescott Filter proposed by James Hamilton (2018). Splits a data series into three components: the trend, the cycle, and random fluctuations.  

Currently only works with a Pandas Series object, though it's not much of a stretch to extend it to Numpy arrays.

# References
Hamilton, James D. "Why you should never use the Hodrick-Prescott filter." _Review of Economics and Statistics_ 100, no. 5 (2018): 831-843.

Shea, Justin M. "Reproducing Hamilton." _CRAN_, https://cran.r-project.org/web/packages/neverhpfilter/vignettes/Reproducing-Hamilton.html
