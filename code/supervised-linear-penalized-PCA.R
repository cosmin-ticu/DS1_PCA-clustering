library(tidyverse)
library(datasets)
library(MASS)
library(ISLR)
library(caret)

# more info about the data here: https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
data <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>%
  mutate(logTotalValue = log(TotalValue)) %>%
  drop_na()


# EDA ---------------------------------------------------------------------


# Modeling Choices --------------------------------------------------------


# Linear Regression -------------------------------------------------------


# Penalized models --------------------------------------------------------


# PCA ---------------------------------------------------------------------


# Pre-process PCA & penalized models --------------------------------------


# Final model evaluation --------------------------------------------------


