rm(list = ls())
#install.packages('dsa')
# Load the required packages
library(readxl)
library(stats)
library(AER)
library(fpp3)
library(forecast)
library(tis)
library(ggplot2)
library(tsibble)
library(feasts)
library(tidyverse)
library(astsa)
library(zoo)
library(xts)
library(reshape)
library(tibble)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggseas)
library(dsa)
library(tseries)
library(urca)
library(strucchange)
library(rdrr)




#Setting the working directory 
setwd("F:/")

# Reading the dataset and renaming columns
data <- read_excel("Rejsekortrejser (1).xlsx")
colnames(data)
new_column_names <- c("Date", "Num_Passengers")
colnames(data) <- new_column_names
head(data)
nrow(data)

#Cleaning time series removing additional entries
data_cleaned <- data %>%
  mutate(Date = dmy(Date)) %>%
  filter(!is.na(Date))

#Checking if the noise rows are cleaned from top and overall dataset
head(data_cleaned)
nrow(data_cleaned)

#Converting to time series
ts_data <- ts(data_cleaned$Num_Passengers, start = c(2013,1), frequency = 365)  
length(ts_data)

ts_data

#Looking at the summaries 
summary(ts_data)

autoplot(ts_data)
plot(decompose(ts_data))

#converting to xts for weekly monthly and quarterly analysis
ts_xts <- ts2xts(ts_data)

#Checking the quarterly properties
quarterly_sums <- apply.quarterly(ts_xts, sum)
ts_quarterly <- ts(quarterly_sums, start = c(2013, 1), frequency = 4)
ggseasonplot(ts_quarterly)+
  labs(y="Passengers", title="Quarter")

#
gg_subseries(as_tsibble(ts_quarterly))

#Checking the Monthly properties
monthly_sums <- apply.monthly(ts_xts, sum)
ts_monthly <- ts(monthly_sums, start = c(2013, 1), frequency = 12)
ggseasonplot(ts_monthly)+
  labs(y="Passengers", title="Month")

gg_subseries(as_tsibble(ts_monthly))



#Checking Structural Breaks with

pd <- c(1:length(ts_data))

qlr <- Fstats(ts_data~pd, from = 0.01)

sctest(qlr, type='supF')
plot(qlr)

breakpoints <- breakpoints(qlr, alpha = 0.01)

breakpoints

plot(ts_data)
lines(breakpoints)





#Subsetting the data due to Covid
subset_data <- ts(window(ts_data, start = 2021), start = c(2021, 1), frequency = 365)


#looking at the growth rate
plot(diff(log(subset_data)))

#Growth rate has some seasonality




# Split the data into train and test sets
total_obs <- length(subset_data)
train_size <- round(0.8 * total_obs)

train <- ts(subset_data[1:train_size], start = c(2021), frequency = 365)



test<- ts(subset_data[(train_size + 1):total_obs], start = c(2022,222), frequency = 365)

ggtsdisplay(log(train))




#Checking if log transformation is necessary
Lambda<- BoxCox.lambda(train)
Lambda

#Checking for Deterministic trend

# Using lags = 1 for low power

adf_test <- ur.df(train, lags = 1)

# Print the ADF test results
summary(adf_test)

ggtsdisplay(diff(train))


#suggests no deterministic trend


#Checking if Data is stationary
#ADF test with trend
summary(ur.df(diff(train), type = "trend"))
# t-value < cv => Can reject H0 => data is potentially stationary with trend

#ADF test with drift
summary(ur.df(diff(train), type = "drift"))
# t-value < cv => Can reject H0 => data is potentially stationary with drift

#ADF test with drift
summary(ur.df(diff(train), type = "none"))

# t-value < cv => Can reject H0 => data is stationary

#let's move to a KPSS
summary(ur.kpss(diff(train), type = "tau"))
# test stat < CV at 1pct => Can't reject H0 => data is stationary.

summary(ur.kpss(diff(train), type = "mu"))
# test stat < CV at 1pct => Can't reject H0 => data is stationary.



#data is stationary after the first differentiation



#removing seasonality

ggtsdisplay(diff(train))


ts_data2 <- diff((diff(train,lag=7)),1)
ggtsdisplay(ts_data2)


#Rechecking Structural Breaks with

pd <- c(1:length(train))

qlr <- Fstats(train~pd, from = 0.01)

sctest(qlr, type='supF')
plot(qlr)


#IS

breakpoints <- breakpoints(qlr, h = 0)  # h=0 for automatic selection of the number of breaks

# Print the breakpoints
print(breakpoints)

#breaks Ignored

###############################
#Fitting Seasonal ARIMA model

ts_mod0<- auto.arima(train, seasonal=TRUE, stepwise=TRUE, approximation=FALSE, seasonal.test="ch")

forecast_values0 <- forecast(ts_mod0, h = 365)

# Plot the forecast
plot(forecast_values0, main = "Forecast  Years 2023", xlab = "Time", ylab = "Value",col='blue')

# Add the original data to the plot
lines(subset_data, col = "black")  # Original data in blue


checkresiduals(ts_mod0)

summary(ts_mod0)

accuracy(forecast_values0,test)

#Arima Guessed
ts_mod1 <-arima(train,c(1,1,0),seasonal = list(order=c(1,1,1),period=7))
forecast_values1 <- forecast(ts_mod1, h = 150)

# Plot the forecast
plot(forecast_values1, main = "Forecast Year 2023", xlab = "Time", ylab = "Value",col='blue')

# Add the original data to the plot
lines(subset_data, col = "black")  # Original data in blue


checkresiduals(ts_mod1)

summary(ts_mod1)


accuracy(forecast_values1,test)

