library(xgboost)
library(caret)
library(car)
library(Matrix)

x_train <- read.csv("input/train.csv")
y_train <- x_train['loss']
x_train <- x_train[-grep('loss', colnames(x_train))]
x_test <- read.csv("input/test.csv")

params <- {}
params["objective"] <- "reg:linear"
params["eta"] <- 0.01
params["min_child_weight"] <- 7
params["subsample"] <- 0.7
params["colsample_bytree"] <- 0.7
params["scale_pos_weight"] <- 0.8
params["silent"] <- 0
params["max_depth"] <- 4
params["eval_metric"] <- "merror"
plst = as.list(params)
mtrain <- as.matrix(sapply(x_train, as.numeric))
mlabel <- as.matrix(sapply(y_train, as.numeric))
xgtrain <- xgb.DMatrix(mtrain,label=mlabel, missing=NaN)
mtest <- as.matrix(sapply(x_test, as.numeric))
xgtest <- xgb.DMatrix(mtest, missing=NaN)
num_rounds <- 30
set.seed(0)
xgmodel_cv <- xgb.cv(plst, xgtrain, num_rounds, nfold=6)
