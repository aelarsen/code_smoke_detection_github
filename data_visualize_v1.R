# -------------------------------------------------------------------------
# Upload the data and visualize
#
# -------------------------------------------------------------------------

install.packages("reshape2")
install.packages("caTools")
install.packages("nnet")
install.packages("fields")
install.packages("ggplot2")

library(tidyr)
library(reshape2)
library(caTools)
library(nnet)
library(fields)
library(ggplot2)

source('utils.R')

##
##
## Read in the data files
##
##

in_flnms <- c("/Volumes/DATA/Project-3-Data/NT/ahi_NT_10_minute_2015_1Sep_23Sep_20190130.csv",
              "/Volumes/DATA/Project-3-Data/NT/ahi_NT_h_161_w_105_ids_and_majorroads_20181023.csv")

# read in csv file
data <- read.csv(in_flnms[1])  # 10 min, 9/1/2015-9/21/2015
locs <- read.csv(in_flnms[2])  # NT, 161 x 105

# remove edge polygons from data
data <- data[which(data$AHI_ID %in% locs$AHI_ID), ]

# remove time points that don't have any data
w1        <- which(substr(unique(data$TIMEPOINT), 10, 14) == "0240")
tpts_miss <- c(as.character(unique(data$TIMEPOINT)[w1]), "20150904_0040", "20150909_0500")
w2        <- which(data$TIMEPOINT %in% tpts_miss)
data      <- data[-w2, ]


##
##
## Type variable - raw
##
##

type_df <- melt(dcast(data[, c("CLOUD_MASK_TYPE", "TIMEPOINT")], TIMEPOINT ~ CLOUD_MASK_TYPE), 
                id = c("TIMEPOINT"))

colnames(type_df) <- c("time_point", "type", "n_pixels")

tpts <- as.Date(as.character(type_df$time_point), format="%Y%m%d_%H%M")
type_df$date <- tpts

ggplot(data = type_df, aes(x = time_point, y = n_pixels, fill = type)) + 
  geom_bar(stat = 'identity')


##
##
## Type variable - grouped
##
##

type_codes <- unique(type_df$type)
type_nms   <- c("0", "cirrus_-20", "cirrus_-50", "currus_-50_small",
                "cumulus_cont_clean", "cumulus_cont_poll", "cumulus_maritime", "fog",
                "stratus_cont", "stratus_maritime", "23", "27", "33", "37",
                "smoke (fresh)", "smoke (aged)", "dust", "absorptive", "bright smoke (fresh)",
                "bright smoke (aged)", "bright absorptive", "NA")


new_type <- as.character(type_df$type)
for (j in 1:length(type_codes)) {
  
  print(j)
  
  code <- type_codes[j]
  nm   <- type_nms[j]
  new_type[which(type_df$type == code)] <- nm
  
}


smoke_group <- c("smoke (fresh)", "smoke (aged)", "bright smoke (fresh)",
                 "bright smoke (aged)")

cloud_group <- c("cirrus_-20", "cirrus_-50", "currus_-50_small",
                 "cumulus_cont_clean", "cumulus_cont_poll", 
                 "cumulus_maritime", "stratus_cont", "stratus_maritime")

absorptive_group <- c("absorptive", "bright absorptive")

other_group <- c("23", "27", "33", "37")

group <- new_type
for (i in 1:length(new_type)){
  
  val <- new_type[i]
  
  if (val %in% smoke_group) {
    group[i] <- "smoke"
  } else if (val %in% cloud_group) {
    group[i] <- "cloud"
  } else if (val %in% absorptive_group) {
    group[i] <- "absorptive"
  } else if (val %in% other_group) {
    group[i] <- "other"
  }
}

type_df$group <- group

type_df2 <- type_df[which(type_df$group != "other"), ]

ggplot(data = type_df, aes(x = date, y = n_pixels, fill = group)) + 
  geom_bar(stat = 'identity') + ggtitle("Pixel Types During the 09/2015 NT Fire - Dataset 1")






