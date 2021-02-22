### 2) Clustering on the USArrests dataset ----
#   Task: Apply Clustering -> make sense of clusters with PCA
#   Data used in Class

library(tidyverse)
library(caret)
library(skimr)
library(janitor)
library(factoextra) # provides nice functions for visualizing the output of PCA
library(NbClust) # for choosing the optimal number of clusters
library(knitr)
library(kableExtra)
library(data.table)
library(gridExtra)

data_arrest <- as.data.table(USArrests)
skimr::skim(data_arrest)

# data_arrest <- USArrests

str(data_arrest$Murder) # all variables are numeric

# EDA ---------------------------------------------------------------------

GGally::ggpairs(data_arrest, title = "USA arrests data scatters, Densities & correlations")

# apart from a correlation matrix which does show some significant association
# between the murder and the assault variables. it is, however, below the
# threshold of 90% correlation (usually the benchmark for multicollinearity)

# because k-means clustering uses distance to establish similarities and optimal
# clustering values, we can proceed by using all of the variables

# nothing seems entirely skewed

# scaling will happen through PCA, thus no need for now

# Optimal # of clusters ---------------------------------------------------

# run k-means to identify optimal number of clusters
nb <- NbClust(data_arrest, method = "kmeans",
              min.nc = 2, max.nc = 48, index = "all")

# visualize this better by reducing the maximum number of clusters
nb <- NbClust(data_arrest, method = "kmeans", 
              min.nc = 2, max.nc = 10, index = "all")

fviz_nbclust(nb)

# according to the Hubert Index, the most significant peak between two points
# is observed when going from 3 clusters to 2 clusters
# interestingly, we see a smaller peak when forming 5 clusters
# we will stick with 2 clusters here

# K-means clustering ------------------------------------------------------

set.seed(1122)
km <- kmeans(data_arrest, centers = 2)
km

data_w_clusters <- cbind(data_arrest, 
                         data.table("cluster" = factor(km$cluster)))

# Explore significant differences in clustering between Rape and UrbanPop
ggplot(data_w_clusters, aes(x = UrbanPop, y = Rape, color = cluster)) +
  geom_point()

# Explore significant differences in clustering between Assault and UrbanPop
ggplot(data_w_clusters, aes(x = UrbanPop, y = Assault, color = cluster)) +
  geom_point()

# Explore significant differences in clustering between Murder and UrbanPop
ggplot(data_w_clusters, aes(x = UrbanPop, y = Murder, color = cluster)) +
  geom_point()

# Plot with respective cluster centers

centers <- data.table(km$centers)

centers[, cluster := factor("center", levels = c(1, 2, "center"))]

data_w_clusters_centers <- rbind(data_w_clusters, centers)

ggplot(data_w_clusters_centers, 
       aes(x = Murder, y = UrbanPop,
           color = cluster,
           size = ifelse(cluster == "center", 2, 1.5))) + 
  geom_point()+theme(legend.position = "none")

# What is the overall trend?
cluster_table_pca <- data_w_clusters %>% group_by(cluster) %>%
  summarize(Count_States = n(), 
            Mean_Murder = mean(Murder), 
            Mean_Assault = mean(Assault),
            Mean_UrbanPop = mean(UrbanPop), 
            Mean_Rape = mean(Rape))

# PCA ---------------------------------------------------------------------

pca_result <- prcomp(data_arrest, scale = TRUE)
first_two_pc <- as_tibble(pca_result$x[, 1:2])

fviz_contrib(pca_result, "var", axes = 1) # All of the crime related variables take a share
fviz_contrib(pca_result, "var", axes = 2) # UrbanPop prevail here

data_w_clusters_pca <- cbind(data_w_clusters, first_two_pc)

data_w_clusters_pca_states <- data.frame(State = row.names(USArrests), data_w_clusters_pca)

# Plot k-means identified clusters along coordinates of PC1 & PC2
ggplot(data_w_clusters_pca_states, aes(PC1, PC2,color = cluster)) + 
  modelr::geom_ref_line(h = 0) +
  modelr::geom_ref_line(v = 0) +
  geom_text(aes(label = State), size = 3) +
  xlab("First Principal Component") + 
  ylab("Second Principal Component") + 
  ggtitle("First Two Principal Components of USArrests Data on level clusters")
# + theme(legend.position = "none")


# exploring variance explained by PCs
Variance <- pca_result$sdev^2
percentage_variance <- Variance / sum(Variance)
percentage_variance


# Cumulative PVE plot
qplot(c(1:4), cumsum(percentage_variance)) +
  geom_line() +
  xlab("# Principal Component") +
  ylab(NULL) +
  ggtitle("Cumulative Explained Variance") +
  ylim(0,1)
