# Required Libraries


library(Metrics)
library(nortest)
library(dplyr)
library(tidyverse)
library(caret)
library(Metrics)
library(caTools)
library(e1071)
library(glmnet)
library(randomForest)
library(xgboost)
library(data.table)
library(lubridate)
library(carData)
library(car)
library(lattice)
library(lmtest)
library(zoo)
library(ggplot2)
library(corrplot)
library(knitr)
library(kableExtra)


## Analysis 2: Sale Price  

## Restate the problem


# Load the data
train_data <- read.csv("C:/Users/andre/Documents/SMU Information/DS 6371/Unit 14/train.csv")
test_data <- read.csv("C:/Users/andre/Documents/SMU Information/DS 6371/Unit 14/test.csv")

# Handling Missing Values
# Imputing missing values for numerical columns with median and categorical columns with mode
num_cols <- sapply(train_data, is.numeric)
cat_cols <- sapply(train_data, is.character)
train_data[num_cols] <- lapply(train_data[num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
train_data[cat_cols] <- lapply(train_data[cat_cols], function(x) ifelse(is.na(x), names(sort(table(x), decreasing = TRUE))[1], x))

# Transforming Variables
# Converting categorical variables to factors
train_data[cat_cols] <- lapply(train_data[cat_cols], as.factor)

# Removing Unnecessary Columns
# Dropping 'Id' column
train_data <- train_data %>% select(-Id)

# Checking and Handling Outliers
# Example with 'GrLivArea' for Multiple Linear Regression
Q1 <- quantile(train_data$GrLivArea, 0.25)
Q3 <- quantile(train_data$GrLivArea, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
train_data <- train_data %>% 
  filter(GrLivArea >= lower_bound & GrLivArea <= upper_bound)

# Preparing dataset for Simple Linear Regression
# Selecting a single explanatory variable, e.g., 'LotArea'
slr_data <- train_data %>% 
  select(SalePrice, LotArea)

# Preparing dataset for Multiple Linear Regression
# Including 'GrLivArea' and 'FullBath' as explanatory variables
mlr_data <- train_data %>% 
  select(SalePrice, GrLivArea, FullBath)

# Feature Engineering for additional Multiple Linear Regression model
# Example: Creating a new feature 'TotalBathrooms'
train_data$TotalBathrooms <- train_data$FullBath + train_data$HalfBath

# Preparing dataset for Additional Multiple Linear Regression
# Selecting explanatory variables for the model
additional_mlr_data <- train_data %>% 
  select(SalePrice, GrLivArea, FullBath, TotalBathrooms)


# Building a Simple Linear Regression Model
# Using 'LotArea' as the explanatory variable
slr_model <- lm(SalePrice ~ LotArea, data = train_data)

# Building a Multiple Linear Regression Model with 'GrLivArea' and 'FullBath'
mlr_model_1 <- lm(SalePrice ~ GrLivArea + FullBath, data = train_data)

# Building an Additional Multiple Linear Regression Model
# Building a Multiple Linear Regression Model with 'GrLivArea' and 'FullBath' and "YearRemodAdd"
mlr_model_2 <- lm(SalePrice ~ GrLivArea + FullBath + YearRemodAdd, data = train_data)

# Building an Additional Multiple Linear Regression Model
# Selecting additional explanatory variables, e.g., 'YearRemodAdd' and 'TotalRooms'
mlr_model_3 <- lm(SalePrice ~ GrLivArea + FullBath + YearRemodAdd + TotRmsAbvGrd, data = train_data)

# Making Predictions on Test Data
# For Simple Linear Regression
predictions_slr <- predict(slr_model, newdata = test_data)

# For Multiple Linear Regression Model 1
predictions_mlr_1 <- predict(mlr_model_1, newdata = test_data)

# For Additional Multiple Linear Regression Model
predictions_mlr_2 <- predict(mlr_model_2, newdata = test_data)

# For Additional Multiple Linear Regression Model
predictions_mlr_3 <- predict(mlr_model_3, newdata = test_data)

# The predictions can now be used for further analysis or evaluation
}

# Adjusted R² for each model
adj_r2_slr <- summary(slr_model)$adj.r.squared
adj_r2_mlr_1 <- summary(mlr_model_1)$adj.r.squared
adj_r2_mlr_2 <- summary(mlr_model_2)$adj.r.squared
adj_r2_mlr_3 <- summary(mlr_model_3)$adj.r.squared

# Function to calculate CV Press
cv_press <- function(model, data) {
  residuals <- resid(model)
  leverage <- hatvalues(model)
  press <- sum((residuals / (1 - leverage))^2)
  return(press)
}
# CV Press for each model
cv_press_slr <- cv_press(slr_model, train_data)
cv_press_mlr_1 <- cv_press(mlr_model_1, train_data)
cv_press_mlr_2 <- cv_press(mlr_model_2, train_data)


#change to submission formate

predictions_slr <- as.data.frame(as.matrix(predictions_slr)) 
predictions_mlr_1 <- as.data.frame(as.matrix(predictions_mlr_1)) 
predictions_mlr_2 <- as.data.frame(as.matrix(predictions_mlr_2)) 
predictions_mlr_3 <- as.data.frame(as.matrix(predictions_mlr_3)) 

res1 = data.table(Id = test_data$Id, SalePrice = predictions_slr$V1)
res2 = data.table(Id = test_data$Id, SalePrice = predictions_mlr_1$V1)
res3 = data.table(Id = test_data$Id, SalePrice = predictions_mlr_2$V1)
res4 = data.table(Id = test_data$Id, SalePrice = predictions_mlr_3$V1)


#create cvs for submission

write.csv(res1, file = "res1.csv",row.names = F)
write.csv(res2, file = "res2.csv",row.names = F)
write.csv(res3, file = "res3.csv",row.names = F)
write.csv(res4, file = "res4.csv",row.names = F)


