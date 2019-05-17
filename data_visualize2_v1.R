
#
# Load Data
#

# load pre-processed data
load("/Volumes/DATA/Project 3/Code_Smoke Detection/version_g/data_preprocessed_20190215.RData")
load("/Volumes/DATA/Project 3/Code_Smoke Detection/version_g/data_preprocessed_py_20190215.RData")

# select timepoint
tpts  <- unique(data$TIMEPOINT)
day_i <- which(tpts == "20150911_0650")

dat <- data_mat[day_i, 1:length(unique(data$AHI_ID))]

write.csv(dat, file="/Volumes/Data/Project-3-Data/NT/data_day_20150911_0650.csv", row.names =F)
