



# EDA ---------------------------------------------------------------------


# Optimal # of clusters ---------------------------------------------------



# K-means clustering ------------------------------------------------------


# PCA ---------------------------------------------------------------------

pca_result <- prcomp(data, scale = TRUE)
first_two_pc <- as_tibble(pca_result$x[, 1:2])