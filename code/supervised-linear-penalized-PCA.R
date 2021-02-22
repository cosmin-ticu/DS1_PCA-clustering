library(tidyverse)
library(datasets)
library(MASS)
library(ISLR)
library(caret)
library(Hmisc)
library(dplyr)
library(GGally)
library(glmnet)
library(generics)

# load helper functions from DA3
source("code/helper/da_helper_functions.R")

# more info about the data here: https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
data <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>%
  mutate(logTotalValue = log(TotalValue)) %>%
  drop_na()


# EDA ---------------------------------------------------------------------

describe(data)

# some occurences are too sparse
table(data$LotType)

to_filter <- data %>% 
  group_by(LotType) %>% 
  summarise(n = n()) %>% 
  filter(n < 50)

data <- data %>% mutate(LotType = 
                          ifelse(LotType %in% to_filter$LotType, 
                                 "Other Lot Type", LotType))

table(data$HealthArea)

to_filter <- data %>% 
  group_by(HealthArea) %>% 
  summarise(n = n()) %>% 
  filter(n < 50)

data <- data %>% mutate(HealthArea = 
                          ifelse(HealthArea %in% to_filter$HealthArea, 
                                 "Other", HealthArea))

# remove variables with too many missing values

data$ZoneDist4 <- NULL
data$ZoneDist3 <- NULL
data$ZoneDist2 <- NULL

# convert binary variables to factor

data <- mutate(data,
  IrregularLot = as.factor(ifelse(IrregularLot == "Yes",1,0)),
  Landmark = as.factor(ifelse(Landmark == "Yes",1,0)),
  HistoricDistrict = as.factor(ifelse(HistoricDistrict == "Yes" ,1,0)),
  High = as.factor(ifelse(High == T,1,0))) 


data <- mutate(data,
  Council = as.factor(Council),
  PolicePrct = as.factor(PolicePrct),
  HealthArea = as.factor(HealthArea)
)

data[, sapply(data, class) == 'factor'] # all factor variables are in order
  
# coerce easements to factor (it should ideally be factor for ease of interpretation)

data <- mutate(data, Easements = as.factor(ifelse(Easements > 0,1,0)))

# identify the sets of variables for more structured EDA

data <- data %>% rename('Health_local_code' = HealthArea)

area_level_vars <- data[ , grepl( ".Area" , names( data ) ) ] %>% colnames()

technical_level_vars <- c("BldgDepth", "BldgFront", "LotDepth", "LotFront", "UnitsTotal",
                          "UnitsRes", "NumFloors", "NumBldgs")

distance_vars <- c("ZoneDist1", "BuiltFAR", "ResidFAR", "CommFAR", "FacilFAR",
                   "Proximity")

zone_vars <- c("SchoolDistrict", "Council", "FireService", "PolicePrct", "Health_local_code",
               "HistoricDistrict", "Landmark")

building_vars <- c("Class", "LandUse", "Easements", "OwnerType", "Extension",
                   "IrregularLot", "BasementType", "Built", "High")

## plot distribution of numeric variables

# outcome variable
ggplot(data, aes(x = TotalValue)) + geom_density()+scale_x_continuous(labels=scales::comma)
ggplot(data, aes(x = logTotalValue)) + geom_density()

# numeric variables on technical measures -> take log to reduce skew
ggplot(data = data, aes(x = LotDepth))+
  geom_histogram()

ggplot(data = data, aes(x = log(LotDepth + 1) ))+
  geom_histogram()

# numeric variables on area measures -> take log to reduce skew
ggplot(data = data, aes(x = LotArea))+
  geom_histogram()

ggplot(data = data, aes(x = log(LotArea + 1)))+
  geom_histogram()

# numeric variables on distance measures -> don't take log (most entries too sparse)
ggplot(data = data, aes(x = BuiltFAR))+
  geom_histogram()

ggplot(data = data, aes(x = log(BuiltFAR + 1)))+
  geom_histogram()

ggplot(data = data, aes(x = ResidFAR))+
  geom_histogram()

ggplot(data = data, aes(x = log(ResidFAR + 1)))+
  geom_histogram()

# add the logs of the designated numeric variables variables
ln_vars <- c(area_level_vars, technical_level_vars)

# flag 0 values for these variables
data <- data %>% 
  mutate_at(vars(ln_vars), funs("flag" = ifelse( . == 0, 1, 0 )))

flag_vars <- NULL
for (i in ln_vars){
  new <- paste0(i, "_flag")
  flag_vars <- c(flag_vars, new)
}

# add logs and replace with NA if it is below or equal to 0
data <- data %>% 
  mutate_at(vars(ln_vars), funs("log" = ifelse( . <= 0, NA, log(.))))

ln_vars2 <- NULL
for (i in ln_vars){
  new <- paste0(i, "_log")
  ln_vars2 <- c(ln_vars2, new)
}

# replace NAs with half of the minimum value of the given column
data <- data %>% 
  mutate_at(vars(ln_vars2), funs(ifelse(is.na(.), min(., na.rm = T)/2, .)))

# check all correlations
ggcorr(data[, colnames(data) %in% ln_vars2])

# include a few noticeable correlations -> potential interactions
interactions <- c("ResArea_log*NumFloors_log","ResArea_log*UnitsRes_log","ResArea_log*UnitsTotal_log","BldgArea_log*NumFloors_log", 
                  "RetailArea_log*ComArea_log","OtherArea_log*ResArea_log")

# Modeling Choices --------------------------------------------------------

# define sets of predictors
X1 <- paste0(" ~ ",paste0(c(area_level_vars, building_vars) , collapse = " + "))
X2 <- paste0(" ~ ",paste0(c(ln_vars2, flag_vars, building_vars) ,collapse = " + "))
X3 <- paste0(" ~ ",paste0(c(ln_vars2, flag_vars, building_vars, distance_vars, zone_vars)  ,collapse = " + "))
X4 <- paste0(" ~ ",paste0(c(ln_vars2, flag_vars, building_vars, distance_vars, zone_vars, interactions) ,collapse = " + "))

# Work vs holdout sets
set.seed(1234)
train_indices <- as.integer(createDataPartition(data$logTotalValue, p = 0.3, list = FALSE))
data_work <- data[train_indices, ]
data_holdout <- data[-train_indices, ]

# set train control for caret

train_control <- trainControl(method = "cv",number = 10,verboseIter = FALSE)  

# Linear Regression -------------------------------------------------------

n_folds=10
# Create the folds
set.seed(1234)

folds_i <- sample(rep(1:n_folds, length.out = nrow(data_work) ))
# Create results
model_results_cv <- list()


for (i in (1:4)){
  model_name <-  paste0("X",i)
  model_pretty_name <- paste0("(",i,")")
  
  yvar <- "logTotalValue"
  xvars <- eval(parse(text = model_name))
  formula <- formula(paste0(yvar,xvars))
  
  # Initialize values
  rmse_train <- c()
  rmse_test <- c()
  
  model_work_data <- lm(formula,data = data_work)
  BIC <- BIC(model_work_data)
  nvars <- model_work_data$rank -1
  r2 <- summary(model_work_data)$r.squared
  
  # Do the k-fold estimation
  for (k in 1:n_folds) {
    test_i <- which(folds_i == k)
    # Train sample: all except test_i
    data_train <- data_work[-test_i, ]
    # Test sample
    data_test <- data_work[test_i, ]
    # Estimation and prediction
    model <- lm(formula,data = data_train)
    prediction_train <- predict(model, newdata = data_train)
    prediction_test <- predict(model, newdata = data_test)
    
    # Criteria evaluation
    rmse_train[k] <- mse_lev(prediction_train, data_train[,yvar] %>% pull)**(1/2)
    rmse_test[k] <- mse_lev(prediction_test, data_test[,yvar] %>% pull)**(1/2)
    
  }
  
  model_results_cv[[model_name]] <- list(yvar=yvar,xvars=xvars,formula=formula,model_work_data=model_work_data,
                                         rmse_train = rmse_train,rmse_test = rmse_test,BIC = BIC,
                                         model_name = model_pretty_name, nvars = nvars, r2 = r2)
}

t1 <- imap(model_results_cv,  ~{
  as.data.frame(.x[c("rmse_test", "rmse_train")]) %>%
    dplyr::summarise_all(.funs = mean) %>%
    mutate("model_name" = .y , "model_pretty_name" = .x[["model_name"]] ,
           "nvars" = .x[["nvars"]], "r2" = .x[["r2"]], "BIC" = .x[["BIC"]])
}) %>%
  bind_rows()
t1

knitr::kable(t1)

# the model of choice is model 4 with all of the variables and the interactions in place
# complexity is very high

model_4_vars <- paste0(c(ln_vars2, flag_vars, building_vars, distance_vars, zone_vars, interactions) ,collapse = " + ")
model_4_formula <- formula(paste0("logTotalValue", " ~ ",paste0(c(ln_vars2, flag_vars, building_vars, distance_vars, zone_vars, interactions) ,collapse = " + ")))


# Penalized models --------------------------------------------------------

# ## RIDGE
# 
# # glmnet needs inputs as a matrix. model.matrix: handles factor variables
# # -1: we do not need the intercept as glment will automatically include it
# 
# features <- setdiff(names(data), c("logTotalValue","ID",ln_vars))
# 
# x_train <- model.matrix( ~ . -1, data_work[, features, with = FALSE])
# dim(x_train)
# 
# # standardization of variables is automatically done by glmnet
# 
# # how much penalty do we want to apply? select with CV
# lambda_grid <- 10^seq(2,-5,length=100)  
# 
# set.seed(1234)
# system.time({
#   ridge_model <- cv.glmnet(
#     x = x_train, y = data_work[["logTotalValue"]], 
#     lambda = lambda_grid,
#     family = "gaussian", # for continuous response
#     alpha = 0,  # the ridge model
#     nfolds = 10
#   )
# })
# 
# plot(ridge_model, xvar = "lambda", main = "Collapsing coefficient values with RIDGE")
# 
# best_lambda <- ridge_model$lambda.min
# message(paste0("The optimally chosen penalty parameter: ", best_lambda))
# 
# highest_good_enough_lambda <- ridge_model$lambda.1se
# message(paste0("The highest good enough penalty parameter: ", highest_good_enough_lambda))
# 
# # we can see that RMSE is much lower than with the linear regression
# 
# ## LASSO
# 
# set.seed(1234)
# system.time({
#   lasso_model <- cv.glmnet(
#     x = x_train, y = data_work[["logTotalValue"]], 
#     lambda = lambda_grid,
#     family = "gaussian", # for continuous response
#     alpha = 1,  # the LASSO model
#     nfolds = 10
#   )
# })
# 
# plot(lasso_model, xvar = "lambda", main = "Reducing dimensionality in LASSO")
# 
# best_lambda <- lasso_model$lambda.min
# message(paste0("The optimally chosen penalty parameter: ", best_lambda))
# 
# highest_good_enough_lambda <- lasso_model$lambda.1se
# message(paste0("The highest good enough penalty parameter: ", highest_good_enough_lambda))

# we can see that RMSE is much lower than with the linear regression

# CARET Penalized Models ------------------------------------------------------------

## LASSO 

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = 10^seq(2,-5,length=100)
)

set.seed(1234)
system.time({
  lasso_fit <- caret::train(
    model_4_formula,
    data = data_work,
    method = "glmnet",
    preProcess = c("center", "scale"),
    tuneGrid = lasso_tune_grid,
    trControl = train_control
  )
})

ggplot(lasso_fit) + scale_x_log10()

## RIDGE

ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = 10^seq(2,-5,length=100)  
)

set.seed(1234)
system.time({
  ridge_fit <- caret::train(
    model_4_formula,
    data = data_work,
    method = "glmnet",
    preProcess = c("center", "scale"),
    tuneGrid = ridge_tune_grid,
    trControl = train_control
  )
})

ggplot(ridge_fit) + scale_x_log10()


## Elastic Net
enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(1234)
system.time({
  enet_fit <- caret::train(
    model_4_formula,
    data = data_work,
    method = "glmnet",
    preProcess = c("center", "scale"),
    tuneGrid = enet_tune_grid,
    trControl = train_control
  )  
})

ggplot(enet_fit) + scale_x_log10()

lasso_fit$bestTune
ridge_fit$bestTune
enet_fit$bestTune

## Compare all models

resample_profile <- resamples(
  list("ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit
  )
) 

summary(resample_profile)


# Simplest one that is still good enough? ---------------------------------

train_control_1se <- trainControl(method = "cv",number = 10,
                                  verboseIter = FALSE,
                                  selectionFunction = "oneSE")

## LASSO 

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = 10^seq(2,-5,length=100)
)

set.seed(1234)
system.time({
  lasso_fit_1se <- caret::train(
    model_4_formula,
    data = data_work,
    method = "glmnet",
    preProcess = c("center", "scale"),
    tuneGrid = lasso_tune_grid,
    trControl = train_control_1se
  )
})

ggplot(lasso_fit_1se) + scale_x_log10()

## RIDGE

ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = 10^seq(2,-5,length=100)  
)

set.seed(1234)
system.time({
  ridge_fit_1se <- caret::train(
    model_4_formula,
    data = data_work,
    method = "glmnet",
    preProcess = c("center", "scale"),
    tuneGrid = ridge_tune_grid,
    trControl = train_control_1se
  )
})

ggplot(ridge_fit_1se) + scale_x_log10()


## Elastic Net
enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(1234)
system.time({
  enet_fit_1se <- caret::train(
    model_4_formula,
    data = data_work,
    method = "glmnet",
    preProcess = c("center", "scale"),
    tuneGrid = enet_tune_grid,
    trControl = train_control_1se
  )  
})

ggplot(enet_fit_1se) + scale_x_log10()

lasso_fit_1se$bestTune
ridge_fit_1se$bestTune
enet_fit_1se$bestTune

## Compare all models - We see an overall increase in RMSE, which is expected
# but we all see an increase in the R-squared as less penalties are applied overall, meaning that
# more coefficients are kept, thus better fitting the overall pattern

resample_profile_final <- resamples(
  list("ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit,
       "ridge 1se" = ridge_fit_1se,
       "lasso 1se" = lasso_fit_1se,
       "elastic net 1se" = enet_fit_1se
  )
) 

summary(resample_profile_final)


# PCA ---------------------------------------------------------------------

pre_process <- preProcess(data_work, method = c("center", "scale", "pca"))

pre_process # 27 PCs are found to be optimal

pre_process$rotation

preProcess(data_work, method = c("center", "scale", "pca"), thresh = 0.999) # 53 PCs are found to be optimal at this threshold

# one extremely pressing issue with PCA is that cannot use factor variables
# as such, we will need to transpose all of the factor variables into dummy variables

#before converting everything to dummy, must get rid of ID and duplicate TotalValue
# I filtered out some variables
data$ID <- NULL
data$TotalValue <- NULL

# drop unused factor levels
data <- data %>%
  mutate_at(vars(colnames(data)[sapply(data, is.factor)]), funs(fct_drop))

# Using dummyVars to make factor variables -> make new dataframe basically
dummies <- dummyVars(model_4_formula, data = data)
tdf <- cbind("logTotalValue" = data$logTotalValue,predict(dummies,newdata = data) %>% as.data.frame())

tdf$PolicePrct.52 <- NULL # drop one dummy with 0 variance

# Work vs holdout sets - once again for the new dataframe
set.seed(1234)
train_indices_dummies <- as.integer(createDataPartition(tdf$logTotalValue, p = 0.3, list = FALSE))
data_work_dummies <- tdf[train_indices_dummies, ]
data_holdout_dummies <- tdf[-train_indices_dummies, ]

pre_process <- preProcess(data_work_dummies, method = c("center", "scale", "pca"))

pre_process # optimal components found at 140 to explain 95% variance

pre_process$rotation

preProcess(data_work_dummies, method = c("center", "scale", "pca"), thresh = 0.8) # optimal at 97 PCs to explain 80% of variance

# Re-run OLS with PCA
set.seed(1234)
lm_model_pca_50 <- caret::train(logTotalValue ~ .,
                      data = data_work_dummies, 
                      method = "lm", 
                      trControl = trainControl(
                        method = "cv", 
                        number = 10,
                        preProcOptions = list(pcaComp = 50)),
                      preProcess = c("center", "scale", "nzv", "pca")
)
lm_model_pca_50

set.seed(1234)
lm_model_pca_90 <- caret::train(logTotalValue ~ .,
                                data = data_work_dummies, 
                                method = "lm", 
                                trControl = trainControl(
                                  method = "cv", 
                                  number = 10,
                                  preProcOptions = list(pcaComp = 90)),
                                preProcess = c("center", "scale", "nzv", "pca")
)
lm_model_pca_90
# running lm model prediction with preprocessing from PCA results in only 91 principal components
# being computed, while in reality there could be up to 248 PCs

# using PCR function to search for the optimal number of features

trctrlWithPCA <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
tune_grid <- data.frame(ncomp = 60:150)


set.seed(1234)
lm_model_pca_selftune <- caret::train(
  logTotalValue ~ .,
  data = data_work_dummies,
  method = "pcr",
  preProcess = c("center", "scale"),
  tuneGrid = tune_grid,
  trControl = trctrlWithPCA)
lm_model_pca_selftune
# ncomp = 150 is chosen by model as optimal... perhaps a lower threshold would be better
# when using PCR method, all of the predictors are counted as principal components,
# while in the case of lm and glmnet models, only 91 variables are taken

# Pre-process PCA & penalized models --------------------------------------

# Test if applying PCA prior to estimating penalized models result in a better fit
# LASSO PCA with 80 principal components - safe bet

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = 10^seq(2,-5,length=100))

set.seed(1234)
system.time({
  lasso_fit_pca <- caret::train(
    logTotalValue ~ .,
    data = data_work_dummies,
    method = "glmnet",
    preProcess = c("center", "scale", "nzv", "pca"),
    tuneGrid = lasso_tune_grid,
    trControl = trainControl(method = "cv",number = 10,preProcOptions = list(pcaComp = 80), verboseIter = T)
  )
})

lasso_fit_pca

# Final model evaluation --------------------------------------------------

resample_profile_allmodels <- resamples(
  list("ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit,
       "ridge 1se" = ridge_fit_1se,
       "lasso 1se" = lasso_fit_1se,
       "elastic net 1se" = enet_fit_1se,
       "linear model with 50 PCA"=lm_model_pca_50,
       "linear model with 90 PCA"=lm_model_pca_90,
       "linear model with self-tune PCA"=lm_model_pca_selftune,
       "LASSO with PCA"=lasso_fit_pca
       
  )
) 

summary(resample_profile_allmodels)

# the best model is still ultimately the LASSO, but the initial linear model
# comes extremely close in ranking, as can be seen in the initial table comparing
# all of the linear models;

# PCA penalized & OLS models performed worse than their respective regular counterparts

# the test set (original, without dummy variables for PCA) will be evaluated with the LASSO model without PCA

data_holdout$pred_reg_LASSO <- predict(lasso_fit, newdata = data_holdout)

final_RMSE <- RMSE(data_holdout$pred_reg_LASSO,data_holdout$logTotalValue) %>% round(4)
final_MAE <-  MAE(data_holdout$pred_reg_LASSO,data_holdout$logTotalValue) %>% round(4)

# predict also on the dummy dataset using the PCA LASSO
data_holdout_dummies$pred_reg_LASSO <- predict(lasso_fit_pca, newdata = data_holdout_dummies)

final_RMSE_PCA <- RMSE(data_holdout_dummies$pred_reg_LASSO,data_holdout_dummies$logTotalValue) %>% round(4)
final_MAE_PCA <-  MAE(data_holdout_dummies$pred_reg_LASSO,data_holdout_dummies$logTotalValue) %>% round(4)

HoldoutSumms <- rbind(cbind(final_RMSE,final_MAE),cbind(final_RMSE_PCA,final_MAE_PCA)) %>% as.data.frame()

rownames(HoldoutSumms) <- c("Best_Elastic_Net","Elastic_Net_PCA")
HoldoutSumms
