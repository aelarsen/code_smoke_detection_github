
library(tidyr)
library(reshape2)
library(caTools)
library(nnet)
library(fields)
library(ggplot2)


source('version_final/utils.R')
source('version_final/map1.R')

##
##
## Read in data files
##
##

in_flnms <- c("/Volumes/DATA/Project-3-Data/NT/ahi_NT_10_minute_2015_1Sep_23Sep_20190130.csv",
              "/Volumes/DATA/Project-3-Data/NT/ahi_NT_h_161_w_105_ids_and_majorroads_20181023.csv")

# read in csv file
data <- read.csv(in_flnms[1])  # freq=10 min, dates=9/1/2015-9/23/2015
locs <- read.csv(in_flnms[2])  # region=NT, dim=161x105 pixels

# remove edge polygons from data
data <- data[which(data$AHI_ID %in% locs$AHI_ID), ]

# remove time points that don't have any data
w1        <- which(substr(unique(data$TIMEPOINT), 10, 14) == "0240")
tpts_miss <- c(as.character(unique(data$TIMEPOINT)[w1]), "20150904_0040", "20150909_0500")
w2        <- which(data$TIMEPOINT %in% tpts_miss)
data      <- data[-w2, ]

# groups
smoke_group      <- c("smoke (fresh)", "smoke (aged)", "bright smoke (fresh)", "bright smoke (aged)")
cloud_group      <- c("cirrus_-20", "cirrus_-50", "cirrus_-50_small",  "cumulus_cont_clean", "cumulus_cont_poll", "cumulus_maritime", "stratus_cont", "stratus_maritime")
absorptive_group <- c("absorptive", "bright absorptive")
other_group      <- c("23", "27", "33", "37")

n <- 16905
m <- 1079

##
##
## plot the type variable
##
##

# lons, lats, cloud_mask_type, cloud_mask_type_valid, class, type

# cloud_mask_type m x n 
df1    <- dcast(data, TIMEPOINT ~ AHI_ID, value.var="CLOUD_MASK_TYPE")
codes1 <- unique(na.omit(data$CLOUD_MASK_TYPE))[order(unique(na.omit(data$CLOUD_MASK_TYPE)))]
nms1   <- c("cirrus_-20", "cirrus_-50", "cirrus_-50_small", "cumulus_cont_clean", "cumulus_cont_poll", "cumulus_maritime", "fog", "stratus_cont", "stratus_maritime", "23", "27", "33", "37", "smoke (fresh)", "smoke (aged)", "dust", "absorptive", "bright smoke (fresh)", "bright smoke (aged)", "bright absorptive")
df11   <- as.character(melt(df1[, 2:ncol(df1)])$value)
groups <- df11; total_count <- 0
for (j in 1:length(codes1)) {
  code <- codes1[j]
  val  <- nms1[j]
  if (val %in% smoke_group) {
    nm <- "smoke"
  } else if (val %in% cloud_group) {
    nm <- "cloud"
  } else if (val %in% absorptive_group) {
    nm <- "absorptive"
  } else if (val %in% other_group) {
    nm <- "other"
  } else {
    nm <- val
  }
  
  count       <- length(which(df11 == code))
  perc        <- round(100 * (count / (n*m)), 1)
  total_count <- total_count + count
  
  print(paste(code, ", ", val, ", ", nm, ", perc = ", perc, "%", sep = ""))
  groups[which(df11 == code)] <- nm
  
}
print("code, val, nm, count")
cloud_mask_type <- matrix(group, nrow=m, ncol=n)


# cloud_mask_type_valid m x n
df2    <- dcast(data, TIMEPOINT ~ AHI_ID, value.var="CLOUD_MASK_TYPE_VALID")
codes2 <- unique(na.omit(data$CLOUD_MASK_TYPE_VALID))[order(unique(na.omit(data$CLOUD_MASK_TYPE_VALID)))]
nms2   <- c("0", "cirrus_-20", "cirrus_-50", "cirrus_-50_small", "cumulus_cont_clean", "cumulus_cont_poll", "cumulus_maritime", "fog", "stratus_cont", "stratus_maritime", "23", "27", "33", "37", "smoke (fresh)", "smoke (aged)", "dust", "absorptive", "bright smoke (fresh)", "bright smoke (aged)", "bright absorptive")
df22   <- as.character(melt(df2[, 2:ncol(df2)])$value)
groups <- df22; total_count <- 0
for (j in 1:length(codes2)) {
  code <- codes2[j]
  val  <- nms2[j]
  if (val %in% smoke_group) {
    nm <- "smoke"
  } else if (val %in% cloud_group) {
    nm <- "cloud"
  } else if (val %in% absorptive_group) {
    nm <- "absorptive"
  } else if (val %in% other_group) {
    nm <- "other"
  } else {
    nm <- val
  }
  
  count       <- length(which(df22 == code))
  perc        <- round(100 * (count / (n*m)), 1)
  total_count <- total_count + count
  
  print(paste(code, ", ", val, ", ", nm, ", perc = ", perc, "%", sep = ""))
  groups[which(df22 == code)] <- nm
  
}
print("code, val, nm, count")
cloud_mask_type_valid <- matrix(groups, nrow=m, ncol=n)

save(cloud_mask_type_valid, file="mydat4.RData")


# raw type classes = {1, 0, NA}
group1   <- "smoke"
group0   <- c("cloud", "fog", "other", "dust", "absorptive")
myNA     <- c("0", NA)
groups11 <- groups
groups11[groups %in% group1] <- 1
groups11[groups %in% group0] <- 0
groups11[groups %in% myNA]   <- NA
type_raw <- matrix(as.numeric(groups11), m, n)

save(type_raw, file="mydat3.RData")

# TODO: type (type_interpolated) variables.
myLocs         <- as.matrix(locs[, c("x", "y")])
dist           <- rdist.earth(myLocs, myLocs)
colnames(dist) <- locs[, "AHI_ID"]
rownames(dist) <- locs[, "AHI_ID"]
dist_ordered   <- apply(dist, 1, order)      
nbhd_size_all1 <- NULL; type_df <- type_raw                           
for (t in 1:m) {
  yy  <- type_df[t, ]
  nas <- which(is.na(yy))                           # indices of missing values
  nbhd_size <- NULL                                 # nbhd sizes on day t
  if (length(nas) > 0) {                            # if there is more than one NA
    for (i in 1:length(nas)) {                      # loop through each NA
      mypt_ind <- nas[i]                            # index of missing value i
      new_val  <- NA                                # set new_val to NA
      k        <- 1                                 # set nbhd size to 1
      while(is.na(new_val) & k < 131) {             # while new_val is NA
        k <- k+1                                    # increase nbhd size
        knn_ind <- dist_ordered[mypt_ind, 1:k]      # indices of locs of knn's of mypt
        new_val <- Mode(yy[knn_ind])                # mode of the knn Y values
      }
      yy[mypt_ind] <- new_val                       # save mode of the Y values in the kth order nbhd
      nbhd_size    <- c(nbhd_size, k)               # save the nhbd size used for this NA
    }
    nbhd_size_all1 <- c(nbhd_size_all1, nbhd_size)    # save all of the nhbd sizes for day t
  }
  print(paste("Day: ", t, "; avg neighborhood size = ", round(mean(nbhd_size), 1), "; percent missing = ", round(100*(length(nas)/length(yy)), 1), "%", sep = ""))
  type_df[t, ] <- yy   # replace Y values on day t with yy
}

save(type_df, file="mydat.RData")
save(nbhd_size_all1, file="mydat1.RData")
print(summary(nbhd_size_all1))

##
##
## data for plotting
##
##

  
for (t in 1:n_tpts) {
  
  plotdata <- data.frame("lons" = x,
                         "lats" = y,
                         "cloud_mask_type" = cloud_mask_type[t, ],
                         "cloud_mask_type_valid" = cloud_mask_type_valid[t, ],
                         "class" = type_raw[t, ])
  
}



# lat, lon, group_raw (cloud_mask_type grouped), group_valid (cloud_mask_type_valid grouped), class (type_raw), 
#   type (type_interpolated)
#   - put rings around the values that are interpolated.  (class = {1, 0, NA}, type = {0, 1})

# plot 1: group_valid with grey pixels to indicate NA and white to indicate "0"
# plot 2: type_raw (red, blue, grey)
# plot 3: type (red, blue)






# lat, lon, B1, B2, B3, B4, B5_raw, B5, temp, hotspot_raw, hotspot, od (split gradient @ cutoff) 
