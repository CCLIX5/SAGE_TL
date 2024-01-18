library(reshape2)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tidyverse)
#df_rank1 = read.csv("NN_source_local.csv")
df_rank2 = read.csv("NN_target_local.csv")
df_rank3 = read.csv("NN_target_TL.csv",header = TRUE, sep = ",")
#df_rank1 <- df_rank1[-c(1),]
#df_rank1 <- df_rank1[,-c(1)]


df_rank2 <- df_rank2[-c(1),]
df_rank2 <- df_rank2[,-c(1)]

df_rank3 <- df_rank3[-c(1),]
df_rank3 <- df_rank3[,-c(1)]

head(df_rank3)
#df1 <- melt(df_rank1)
#df1 <- df1 %>% mutate(method = "Source_Local",
#                     sig = TRUE)
df2 <- melt(df_rank2)
df2 <- df2 %>% mutate(method = "Local",
                      sig = TRUE)
df3 <- melt(df_rank3)
df3 <- df3 %>% mutate(method = "TL",
                      sig = TRUE)
df3 <- df3[order(df3$value),]
head(df3)

df <- rbind(df3,df2)
df_new = df %>% mutate(var=gsub("\\."," ",variable))
#df <- rbind(df,df1)
head(df)


bubble_for_rank(data = df_new, y_name = "var", x_name = "method",
                rank_name = "value", highlight_name = "sig", 
                arrange_rows_by = "TL",
                x_order = c("TL", "Local")
)
