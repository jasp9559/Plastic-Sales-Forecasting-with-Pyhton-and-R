library(readxl)
Plastic <- read.csv(file.choose()) # read the Plastic data
View(Plastic) # Seasonality 12 months

# Pre Processing
# input t
Plastic["t"] <- c(1:60)
View(Plastic)

Plastic["t_square"] <- Plastic["t"] * Plastic["t"]
Plastic["log_Sales"] <- log(Plastic["Sales"])

# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 60), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

PlasticSales <- cbind(Plastic, X)
colnames(PlasticSales)

View(PlasticSales)
## Pre-processing completed

attach(PlasticSales)

# partitioning
train <- PlasticSales[1:48, ]
test <- PlasticSales[49:60, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Sales ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))
rmse_linear <- sqrt(mean((test$Sales - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_Sales ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Sales - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Sales ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Sales-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Sales - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_Sales ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Sales - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Sales ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Sales - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(PlasticSales, file = "PlasticSales.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Sales ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = PlasticSales)
summary(Add_sea_Quad_model_final)

#Lets get the Residuals
resid <- residuals(Add_sea_Quad_model_final)
resid[1:10]

windows()
hist(resid)

windows()
acf(resid,lag.max = 10)
# Taking the lag as 1

k <- arima(resid, order=c(1,0,0))

windows()
acf(k$residuals,lag.max = 15)
pred_res<- predict(arima(resid, order = c(1,0,0)), n.ahead = 12)
str(pred_res)
pred_res$pred
acf(k$residuals)

#Lets Build a ARIMA model for whole dataset. ARIMA(Auto Regression Integrated Moving Average)

#Lets convert the data into Time Series data

library(tseries)
library(forecast)

Plastic_ts <- ts(Plastic$Sales, frequency = 12, start = c(49)) #Create a Time Series data
View(Plastic_ts)
plot(Plastic_ts) #Plots the data into a Line chart By default as the data is a Time Series Data

#For Building ARIMA model we the AR coefficient i.e p-value then Integration Coefficient i.e d and Moving Average coefficient i.e q-value
#Lets find p-value, p-valueis Obtained by pacf
pacf(Plastic_ts) #Lets Consider it as 0.2

#Lets Find the q-value by acf
acf(Plastic_ts) #Lets Consider this as 0.1

#Also lets Consider the d-value as 1
#now lets build an ARIMA model
a <- arima(Plastic_ts, order = c(0.2, 0.2, 1), method = "ML")
a
#Lets plot the forecast using the ARIMA model
plot(forecast(a, h = 12), xaxt = "n")

#Seeing the plot, we get to know that the forecast done was not accurate. This happens when we dont provide the right
# p-value and q-value to the model.

#If we dont know the p-vale and q-value for the model then we can build the model using auto.arima() function.
#This function will analyse the p and q value and build a proper model. Lets Build the Model

ab <- auto.arima(Plastic_ts)

windows()
plot(forecast(ab, h = 12), xaxt = "n")
#So now we can see that the forecast was accurate 

prediction <- forecast(ab, h = 12) #This will predict for the next 12 months

prediction

###########################################################################
###########Applying Exponential smoothing model############################
library(forecast)
library(fpp)
library(smooth) # for smoothing and MAPE
library(tseries)

library(readxl)
Plastic_Sales_Rawdata <- read.csv("C:/Data Science/Data Science/Assignments/Ass24. Forecasting/Plastic/PlasticSales_raw.csv")
View(Plastic_Sales_Rawdata)

# Converting data into time series object
tsSales <- ts(Plastic_Sales_Rawdata$Sales, frequency = 12, start = c(49))
View(tsSales)

# dividing entire data into training and testing data 
train <- tsSales[1:48]
test <- tsSales[49:60]
# Considering 12 months of data for testing because data
# seasonal data

# converting time series object
train <- ts(train, frequency = 12)
test <- ts(test, frequency = 12)

# Plotting time series data
plot(tsSales)
# Visualization shows that it has level, trend, seasonality => Additive seasonality

###### MOVING AVERAGE ######
ma_model1 <- sma(train)
unlist(ma_model1)
ma_pred <- data.frame(predict(ma_model1, h = 12))
ma_pred

plot(forecast(ma_model1))
ma_mape <- MAPE(ma_pred$Point.Forecast, test)*100
ma_mape


#### USING HoltWinters function ################
# Optimum values
# with alpha = 0.2 which is default value
# Assuming time series data has only level parameter
hw_a <- HoltWinters(train, alpha = 0.2, beta = F, gamma = F)
hw_a
hwa_pred <- data.frame(predict(hw_a, n.ahead = 12))
hwa_pred
# By looking at the plot the forecasted values are not showing any characters of train data 
plot(forecast(hw_a, h = 12))
hwa_mape <- MAPE(hwa_pred$fit, test)*100

# with alpha = 0.2, beta = 0.15
# Assuming time series data has level and trend parameter 
hw_ab <- HoltWinters(train, alpha = 0.2, beta = 0.15, gamma = F)
hw_ab
hwab_pred <- data.frame(predict(hw_ab, n.ahead = 12))
# by looking at the plot the forecasted values are still missing some characters exhibited by train data
plot(forecast(hw_ab, h = 12))
hwab_mape <- MAPE(hwab_pred$fit,test)*100

# with alpha = 0.2, beta = 0.15, gamma = 0.05 
# Assuming time series data has level,trend and seasonality 
hw_abg <- HoltWinters(train, alpha = 0.2, beta = 0.15, gamma = 0.05)
hw_abg
hwabg_pred <- data.frame(predict(hw_abg, n.ahead = 12))
# by looking at the plot the characters of forecasted values are closely following historical data
plot(forecast(hw_abg, h = 12))
hwabg_mape <- MAPE(hwabg_pred$fit, test)*100

# With out optimum values 
hw_na <- HoltWinters(train, beta = F, gamma = F)
hw_na
hwna_pred <- data.frame(predict(hw_na, n.ahead = 12))
hwna_pred
plot(forecast(hw_na, h = 12))
hwna_mape <- MAPE(hwna_pred$fit,test)*100


hw_nab <- HoltWinters(train, gamma = F)
hw_nab
hwnab_pred <- data.frame(predict(hw_nab, n.ahead = 12))
hwnab_pred
plot(forecast(hw_nab, h = 12))
hwnab_mape <- MAPE(hwnab_pred$fit, test)*100

hw_nabg <- HoltWinters(train)
hw_nabg
hwnabg_pred <- data.frame(predict(hw_nabg, n.ahead = 12))
hwnabg_pred
plot(forecast(hw_nabg, h = 12))
hwnabg_mape <- MAPE(hwnabg_pred$fit, test)*100


df_mape <- data.frame(c("hwa_mape","hwab_mape","hwabg_mape","hwna_mape","hwnab_mape","hwnabg_mape"),c(hwa_mape,hwab_mape,hwabg_mape,hwna_mape,hwnab_mape,hwnabg_mape))
colnames(df_mape)<-c("MAPE","VALUES")
View(df_mape)

# Based on the MAPE value we choose holts winter exponential tecnique which assumes the time series
# Data level, trend, seasonality characters with default values of alpha, beta and gamma

new_model <- HoltWinters(tsSales)
new_model

plot(forecast(new_model, n.ahead = 12))
plot(forecast())
# Forecast values for the next 5 years
forecast_new <- data.frame(predict(new_model, n.ahead = 60))

forecast_new
