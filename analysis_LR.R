# do cross-validation and keep images whole.
# refer to lr_analysis (for format to feed to python) and lr_analysis3.R in version_final

# Split the images into train and test sets and built the lr model on the training set of images
# tested on the remaining. 5-fold cross validation.


library(tidyr)
library(reshape2)
library(caTools)
library(nnet)
library(fields)

source("/Volumes/DATA/Project 3/Code_Smoke Detection/version_final/utils.R")

##
##
## logistic regression:
##   Yt ~ b1_t + b2_t + b3_t + b4_t + b5_t + temp_t + frp_t
##
##

# read in data
load("/Volumes/DATA/Project 3/Code_Smoke Detection/version_g/data_preprocessed.RData")

# save data dimemsions
m <- length(unique(data$TIMEPOINT))  # 1079 timepoints
n <- length(unique(data$AHI_ID))     # 16905 pixels = 161*105

# create image IDs
data$IMAGE_ID <- rep(1:m, each=n)

# create binary ground truth variable
y      <- matrix(0, nrow = n*m, ncol = 1)
xx     <- data$CLOUD_MASK_TYPE %in% as.integer(c(101, 111, 23, 27, 33, 37, 100, 110))
y[xx]  <- 1
data$y <- as.factor(y)

# create folds for the cross-validation 
set.seed(1234)
im_shuffle <- sample(m, m, replace = F)               # shuffle image indices
n_folds    <- 5                                       # number of folds 
folds      <- split(im_shuffle, as.factor(1:n_folds)) # split evenly into n_folds groups

# empty variables for the loop
conf_mat    <- list()
iou_mn      <- NULL
iou_tp      <- NULL
iou_tn      <- NULL
pixel_acc   <- NULL
roc_data    <- NULL

# logistic regression variables
lr_vrbs  <- c("y", "B1", "B2", "B3", "B4", "B5", "TMPR_B14", "FRP")

# loop through each fold; k-fold cross-validation, LR
for (k in 1:n_folds) {
  
  # keep track of place in loop
  print(k)
  
  # split data into train and test sets
  test_ind <- data$IMAGE_ID %in% folds[[k]]
  df_train <- data[!test_ind, lr_vrbs]
  df_test  <- data[test_ind, lr_vrbs]
  
  # run lr on train set
  reg_model <- multinom(y ~ ., data = df_train, trace = F)
  
  # make predictions on test data
  predicted_vals <- predict(reg_model, newdata = df_test)
  
  # compute and save performance metrics
  tab            <- table(predicted_vals, df_test$y)
  conf_mat[[k]]  <- tab
  iou_out        <- iou(tab)
  iou_mn[k]      <- iou_out[1]
  iou_tn[k]      <- iou_out[2]
  iou_tp[k]      <- iou_out[3]
  pixel_acc[k]   <- mean(as.character(na.omit(predicted_vals)) == as.character(na.omit(df_test)$y))
  
  # save roc data
  roc_data <- rbind(roc_data, cbind(as.numeric(as.character(predicted_vals)),
                                    as.numeric(as.character(df_test$y))))
}

res <- c(mean(pixel_acc), mean(iou_mn), mean(iou_tp), mean(iou_tn))

round(res, 3)
# [1] 0.57 0.31 0.06 0.56

# numbers in thesis:
# 0.976708  (pixel acc.)
# 0.489237  (IoU)
# 0.001766  (IoU TP)
# 0.976707  (IoU TN)
# 0.954551  (f.w. IoU)
# 0.000040  (f.w. IoU TP)
# 0.954511  (f.w. IoU TN)

write.csv(roc_data, "version_g/roc_data.csv")

#write.csv(roc_data, "output/roc_data.csv")
