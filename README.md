# timeseries
Forecasting project

1) We load the training dataset and transform it's indices to timedate format for a timeseries analysis considers only the date and time of a data and other orders are irrelevant.
2) We plot the data - some varying trend and seasonality exists with regularity between adjacent curves and then sharp dip or spike.
3) Next applying statistical methods to verify stationarity of data- 
  -> The DF Test says the timeseries is not difference stationary (even after differencing not stationary which is true as I have tried differencing and it did not make the data stationary). 
  -> The KPSS test also tells us that data is NOT TREND stationary.

4) Decomposed data into it's trend & seasonality components, leaving residuals that should not vary periodically with time.
5) Plotted ACF and Partial ACF to look at the best lag value where most correlation exist in order to use for modelling data.
   -> Both plots agree that lag value is 1.
6) We create a BASE model with the most popular ARIMA model to try and see how it performs on the data. Since the training data has varying trends and seasonality, used iterative method to find the best parameters instead of heuristically assigning them.
 -> ARIMA (1, 0 1) fails to capture the seasonality and trends of the original data, gives a flat curve.
 -> Potted residual- info were not captured by the model (residuals should not exhibit any pattern).
 -> ARIMA(1, 0 1) = AR(1), MA(1), I(0) as data is NOT difference stationary.

7) Since transformation behind the scenes did not work efficiently to create stationary data, we move on to SARIMA (seasonal ARIMA) to try and capture trend & seasonality.
8) Used iterative method again to find best parameters where the second triplet of parameters are for seasonal components of ARIMA parameters.
 -> SARIMA(1, 1, 1) (0,1, 1) 13 where 13 is the seasonal lag.
  
 9) Plotted predicted values to show a better fit than ARIMA.
 10) Plotted residuals to see if it is truly random, but there is still some information left to model.
 11) Plotted residual ACF & PACF to find autocorrelation still exists in the stochastic terms, meaning we have left out some information while fitting data.
 13) Plotted histogram to see normality of residuals- it shows no bias which is good.
 14) Written prediction values to a csv file.
