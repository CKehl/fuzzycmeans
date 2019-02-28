# fuzzycmeans
A fast implementation of fuzzy c-means clustering algorithm using numpy.

The implementation was based on the paper:
"Prediction of stock index futures prices based on fuzzy sets and multivariate fuzzy time series,
Neurocomputing, BaiQing Sun, Haifeng Guo, Hamid Reza Karimi, Yuanjing Ge, Shan Xiong".


## How to use

- First install the numpy library and clone the repository.
- Import cmeans.py
- call cmeans function

Parameters:
- data: must be a numpy array
- n_c: number of clusteres
- m: fuzzyness coefficient
- epsilon: minimum error
- max_steps: max steps to finish the execution
```
