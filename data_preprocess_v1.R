library(fields)
library(reshape2)
library(caTools)

source('version_final/utils.R')

#
# Load Data
#

# data files
in_flnms <- c("/Volumes/DATA/Project-3-Data/NT/ahi_NT_10_minute_2015_1Sep_23Sep_20190208.csv",
              "/Volumes/DATA/Project-3-Data/NT/ahi_NT_h_161_w_105_ids_and_majorroads_20181023.csv")

# read in csv file
data <- read.csv(in_flnms[1])  # freq=10 min, dates=9/1/2015-9/23/2015
locs <- read.csv(in_flnms[2])  # region=NT, dim=161x105 pixels

#
# Data Cleaning
#

# remove edge polygons from data
data <- data[which(data$AHI_ID %in% locs$AHI_ID), ]

# remove time points that don't have any data
w1        <- which(substr(unique(data$TIMEPOINT), 10, 14) == "0240")
tpts_miss <- c(as.character(unique(data$TIMEPOINT)[w1]), "20150904_0040", "20150909_0500")
w2        <- which(data$TIMEPOINT %in% tpts_miss)
data      <- data[-w2, ]

# replace FRP NA's with 0's
data$FRP[is.na(data$FRP)] <- 0

# incorporate cloudy flag variable
data$CLOUD_MASK_TYPE_VALID[data$FLAGS_CLOUD != 0 & 
                             !(data$CLOUD_MASK_TYPE_VALID %in% c(101, 111, 23, 27, 33, 37 , 100, 110))] <- 9

#
# Replace Missing Values w/Neighborhood Value
#

# find 1st non-missing value, and then take the average of the next k pixels
k <- 15

# create a matrix of nearest neighbors (rows = indices of nearest neighbors in order)
myLocs         <- as.matrix(locs[, c("x", "y")])
dist           <- rdist.earth(myLocs, myLocs)
colnames(dist) <- locs[, "AHI_ID"]; rownames(dist) <- locs[, "AHI_ID"]
dist_ordered   <- apply(dist, 1, order) 

# B5
for (t in unique(data$TIMEPOINT)) {
  b5        <- data$B5[data$TIMEPOINT == t]
  na_inds   <- which(is.na(b5))
  if (length(na_inds) > 0) {                            
    for (i in 1:length(na_inds)) {                      
      my_ind     <- na_inds[i] 
      ind1       <- which(!is.na(b5[dist_ordered[my_ind, ]]))[1]
      knn_ind    <- dist_ordered[my_ind, ind1:(ind1 + k)]
      b5[my_ind] <- mean(b5[knn_ind], na.rm = T)
    }
  }
  print(paste("Day: ", t, "; percent missing = ", round(100*(length(na_inds)/length(b5)), 1), "%", sep = ""))
  data$B5[data$TIMEPOINT == t] <- b5
}

# cloud_mask_type_valid
for (t in unique(data$TIMEPOINT)) {
  yy      <- data$CLOUD_MASK_TYPE_VALID[data$TIMEPOINT == t]  
  na_inds <- which(is.na(yy))                           
  if (length(na_inds) > 0) {                            
    for (i in 1:length(na_inds)) {                      
      my_ind     <- na_inds[i]
      ind1       <- which(!is.na(yy[dist_ordered[my_ind, ]]))[1]
      knn_ind    <- dist_ordered[my_ind, ind1:(ind1 + k)]
      yy[my_ind] <- Mode(yy[knn_ind])
    }
  }
  print(paste("Day ", t, "; percent missing: ", round(100*(length(na_inds)/length(yy)), 1), "%", sep = ""))
  data$CLOUD_MASK_TYPE_VALID[data$TIMEPOINT == t] <- yy
}
#save(data, file="version_g/data_preprocessed_20190215.RData")
save(data, file="version_g/data_preprocessed.RData")

# print means and sd for normalizing the inputs
apply(data[, c("B1", "B2", "B3", "B4", "B5", "TMPR_B14", "FRP")], 2, mean)
apply(data[, c("B1", "B2", "B3", "B4", "B5", "TMPR_B14", "FRP")], 2, sd)

#
# Binary Ground Truth Variable
#

load("version_g/data_preprocessed.RData")
smoke <- as.integer(c(101, 111, 23, 27, 33, 37, 100, 110))
tpts  <- unique(data$TIMEPOINT)
m     <- length(unique(data$TIMEPOINT))  # 975 timepoints
n     <- length(unique(data$AHI_ID))     # 16905 pixels = 161*105
gt1   <- matrix(0, m, n)                 # pixel-wise
gt2   <- rep(0, m)                       # image-wise
for (i in 1:m) {
  print(paste(tpts[i]))
  w <- which(data$TIMEPOINT == tpts[i])
  if (any(data[w, "CLOUD_MASK_TYPE_VALID"] %in% smoke)) {
    k         <- which(data[w, "CLOUD_MASK_TYPE_VALID"] %in% smoke)
    gt1[i, k] <- 1
    gt2[i]    <- 1
  }
}
save(gt2, file="version_g/gt2.RData")

#
# Data for Python Script
#

# concatenate ground truth with each covariate
variable_nms       <- names(data)[c(6:11, 13)]
data_mat           <- gt1
colnames(data_mat) <- sprintf("labels%s", 1:ncol(data_mat))
ranges             <- list(c(1, n))
inputs             <- list()
for (i in 1:length(variable_nms)) {
  print(paste(variable_nms[i]))
  n1                 <- ncol(data_mat)
  v                  <- variable_nms[i]
  mat                <- dcast(data, TIMEPOINT ~ AHI_ID, value.var=v)
  inputs[[i]]        <- mat
  new_data           <- mat[, 2:(n+1)]
  colnames(new_data) <- sprintf(paste0(v, "_%s"), 1:ncol(new_data))
  data_mat           <- cbind(data_mat, new_data)
  ranges[[i + 1]]    <- c(n1 + 1, ncol(data_mat)) 
}
#save(data_mat, file="version_g/data_preprocessed_py_20190215.RData")
save(data_mat, file="version_g/data_preprocessed_py.RData")

#
# Train/Test Sets
#

# split labels into train and test sets
set.seed(1319)
set_assign <- ifelse(sample.split(gt2, 0.7), "train", "test")
train      <- as.data.frame(data_mat[set_assign == "train", ])
test       <- as.data.frame(data_mat[set_assign == "test", ])

table(gt2, set_assign)

# write as csv file
write.csv(train, file = "version_g/train.csv", row.names = F)
write.csv(test, file = "version_g/test.csv", row.names = F)




