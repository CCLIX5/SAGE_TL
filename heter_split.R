library(dplyr)
library(tableone)
set.seed(5)

prob.gen <- function(value, cutoff, min = 0.6, max = 1){
  if(value>cutoff){x = runif(1, min, max)}
  else{x = runif(1, 0, 1-min+0.2)}
  return(x)
}

data_split <- function(dat, prob, cutoff){
  ind = ifelse(prob>cutoff, 1, 0)
  return(list(dat1 = dat[which(ind==1), ],
              dat2 = dat[-which(ind==1), ]))
}


d <- getwd()
print(d)
dat0 <- read.csv("master_dataset.csv", 
                 header = TRUE, stringsAsFactors = FALSE, na.strings = c("NA", "", " "))
dat0 <- dat0 %>%filter(ethnicity=="ASIAN")
dat0$gender[dat0$gender=="F"] <- 0
dat0$gender[dat0$gender=="M"] <- 1
dat0$outcome_inhospital_mortality[dat0$outcome_inhospital_mortality=="FALSE"] <- 0
dat0$outcome_inhospital_mortality[dat0$outcome_inhospital_mortality=="TRUE"] <- 1
dat0$chiefcom_fever_chills[dat0$chiefcom_fever_chills=="FALSE"] <- 0
dat0$chiefcom_fever_chills[dat0$chiefcom_fever_chills=="TRUE"] <- 1

dat_pick <- dat0 %>% select(
  age, gender, triage_pain,triage_sbp, triage_dbp, cci_MI, cci_CHF, 
  cci_PVD, cci_Stroke, cci_Dementia, cci_Pulmonary, cci_Rheumatic, cci_PUD, cci_Liver1, 
  cci_Liver2, cci_DM1, cci_DM2, cci_Paralysis, eci_Renal, 
  eci_BloodLoss, eci_Drugs, eci_Coagulopathy, eci_WeightLoss, eci_Depression, 
  ed_heartrate_last, ed_temperature_last, ed_resprate_last,
  ed_o2sat_last,chiefcom_fever_chills,n_ed_90d, outcome_inhospital_mortality
)

dat <- na.omit(dat_pick)
nrow(dat)
prob = lapply(dat$age, prob.gen, cutoff = 75) %>% unlist
hist(prob)
dat_split = data_split(dat, prob, 0.7) #.6
dat_split$dat1 %>% nrow
dat_split$dat2 %>% nrow

dat_split$dat1$age %>% hist
dat_split$dat2$age %>% hist

dat_split$dat1$Renal %>% table
dat_split$dat2$Renal %>% table

# slightly change the data structure:
dat1 = dat_split$dat1
dat1$site = "site1"
dat2 = dat_split$dat2
dat2$site = "site2"
dat_all = rbind(dat1, dat2)
vars = colnames(dat_split$dat1)
# use tableone(), reformat table and set stratas;
CreateTableOne(data = dat_all, strata = "site", vars = vars, test = F)
res = CreateTableOne(data = dat_all, strata = "site", vars = vars, test = F) %>% print()
write.csv(res, "T_S.csv")

dat1$age %>% range
dat2$age %>% range

dat1$age %>% hist
dat2$age %>% hist

# save data
write.csv(dat1, "Target.csv")
write.csv(dat2, "Source.csv")


