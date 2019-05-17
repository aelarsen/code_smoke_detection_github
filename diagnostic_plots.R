library(tidyr)
library(reshape2)
library(caTools)
library(nnet)
library(fields)
library(ggplot2)
library(viridis)
library(gridExtra)

# TODO version_final -> version_g

# load data
load("version_g/data_preprocessed_py.RData")
load("version_g/gt2.RData")

# make train/test splits and save
train_size <- seq(.1, .9, length.out = 10)
for(i in 1:length(train_size)) {
  print(i)
  
  # split labels into train and test sets
  set.seed(1319)
  set_assign <- ifelse(sample.split(gt2, train_size[i]), "train", "test")
  train      <- as.data.frame(data_mat[set_assign == "train", ])
  test       <- as.data.frame(data_mat[set_assign == "test", ])
  
  # write as csv file
  write.csv(train, file = paste("/Volumes/DATA/Project-3-Data/NT/Diagnostics/Set", i, "/train.csv", sep=""), row.names = F)
  write.csv(test, file = paste("/Volumes/DATA/Project-3-Data/NT/Diagnostics/Set", i, "/test.csv", sep=""), row.names = F)
}


##
## results
##

# nrow = 10 = train set size; ncol = 18 = epochs
loss <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/loss.csv", header=F)
acc <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/acc.csv", header=F)
i_mn <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/i_mn.csv", header=F)
i_tp <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/i_tp.csv", header=F)
i_tn <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/i_tn.csv", header=F)
wi_tp <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/wi_tp.csv", header=F)
wi_tn <- read.csv("/Volumes/DATA/Project-3-Data/NT/Diagnostics/wi_tn.csv", header=F)

# function to make plots
make_plot <- function(dat, dat_title) {
  
  colnames(dat) <- 1:ncol(dat)
  rownames(dat) <- c("10:90", "19:81", "28:72", "37:63", "46:54", "54:46", "63:37", "72:28", "81:19", "90:10")
  plot_loss <- melt(t(dat))
  colnames(plot_loss) <- c("epoch", "train_test", "value")
  p <- ggplot(plot_loss, aes(x=epoch, y=value, colour=train_test, group=train_test)) + 
    geom_line() + 
    geom_point() +
    scale_color_viridis(discrete=TRUE) + 
    xlab("Training Epoch") + ylab(dat_title) + labs(color = "Train:Test") +
    theme_bw()
  
  return(p)
  
}

# plot
p1 <- make_plot(loss, "Average Loss")
p1
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/loss.png")

p2 <- make_plot(acc, "Average % Accuracy")
p2
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/acc.png")

p3 <- make_plot(i_mn, "Mean IoU")
p3
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/i_mn.png")

p4 <- make_plot(i_tp, "TP IoU")
p4
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/i_tp.png")

p5 <- make_plot(i_tn, "TN IoU")
p5
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/t_tn.png")

p6 <- make_plot(wi_tp, "W-TP IoU")
p6
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/wi_tp.png")

p7 <- make_plot(wi_tn, "W-TN IoU")
p7
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/wi_tn.png")



p8 <- grid.arrange(p1, p2, p3, p4, p5, p6, p7, nrow = 4, ncol = 2)
ggsave("/Volumes/DATA/Project-3-Data/NT/Diagnostics/metrics.png", p8, width=11, height=8.5)
