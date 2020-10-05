#Author: Linh Dinh 

########################################INSTALLING AND LOADING PACKAGES########################################
#install.packages("dplyr")
#install.packages("caret")

library(dplyr)
library(caret)

########################################CREATING PSEUDO SESSION DATA########################################
set.seed(123)
sample_size = 100000
visitorID = sample(100000:200000, sample_size, replace=T)
visitorID_count = length(unique(visitorID))
rand_datetime <- function(N, st="2018-10-01", et="2018-11-30") {
  st <- as.POSIXct(as.Date(st))
  et <- as.POSIXct(as.Date(et))
  dt <- as.numeric(difftime(et,st,unit="secs"))
  ev <- runif(N, 0, dt)
  rt <- st + ev
  return(rt)
}
visitStartTime = rand_datetime(sample_size)
conversion_count = sample(c(0, 1, 2, 3, 4), size = sample_size, 
                          replace = TRUE, prob = c(0.750, 0.150, 0.065, 0.025, 0.010))
channel = sample(c("OS", "BPS", "GPS", "Direct", "Display", "Email", "Social", "Referral", "Other"), 
                 size = sample_size, replace = TRUE, 
                 prob = c(0.28, 0.20, 0.13, 0.12, 0.06, 0.07, 0.02, 0.05, 0.07))
#"OS" = "Organic Search; "BPS" = "Branded Paid Search"; "GPS" = "Generic Paid Search"

data_session = data.frame(fullVisitorId = visitorID, visitStartTime = visitStartTime, 
                          conversion_count = conversion_count, channelGroup = channel)

########################################DATA CLEANING########################################
df_data = data_session %>% 
  mutate(visitDate = as.Date(as.POSIXct(visitStartTime, "CET"))) %>% 
  group_by(fullVisitorId) %>% 
  arrange(visitStartTime) %>% 
  mutate(date_interval = as.numeric(visitDate-lag(visitDate))) %>% 
  ungroup()
df_data$date_interval[is.na(df_data$date_interval)] = 0

all_channels = unique(df_data$channelGroup)
number_channels = length(all_channels)
number_visitors = length(unique(df_data$fullVisitorId))
number_sessions = length(df_data$fullVisitorId)

get_path_no = function(x,y) {
  newx = ifelse(x>0,1,0)
  pathID = cumsum(newx)
  for (i in 1:length(newx)) {
    if(newx[i] == 0 || y[i] > 30 && !(newx[i] == 1 && y[i] > 30)) {pathID[i] = pathID[i] + 1}
  }
  return(pathID)
}

df_data_TOTAL = df_data %>% group_by(fullVisitorId) %>% 
  filter(!(is.null(channelGroup))) %>%
  mutate(path_no = get_path_no(conversion_count,date_interval), total_conversion = conversion_count) %>% 
  ungroup()

get_journey_data = function(data_ss) {
  journey_data = data_ss %>% group_by(fullVisitorId, path_no) %>% 
    summarise(journey = paste(channelGroup, collapse = " > "), journey_len= n(), conv = sum(total_conversion), 
              is_conv = 1-as.integer(all(total_conversion==0)), conv_null = as.integer(all(total_conversion==0))) %>%
    ungroup() 
  return(journey_data)
}

df_path_TOTAL = get_journey_data(df_data_TOTAL) 

### Stratified K-fold Cross Validation ###
### Function to create fold ID for K-fold CV
kfold_num = 10
Kfolds = function(data, num_fold = kfold_num){
  folds = createFolds(factor(data$is_conv), k = num_fold, list = FALSE)
  data$fold = folds
  return(data)
}

df_path_fold = Kfolds(df_path_TOTAL, num_fold = kfold_num)

### Function to create plot of ROC (False Positive Rate vs True Positive Rate)
plot_kfold_roc = function(fpr_list, tpr_list, kfold = kfold_num, type) {
  df_plot = NULL
  for (i in 1:kfold) {
    df_temp = data.frame(fpr = fpr_list[[i]], tpr = tpr_list[[i]], fold=rep(i:i, each=length(fpr_list[[i]]))) 
    df_plot <- rbind(df_plot,df_temp)
  }
  #plot ROC for all k-fold
  ggplot(df_plot,aes(x = fpr, y = tpr, group = fold, colour = factor(fold))) + 
    geom_line() + 
    labs(x = "False Positive Rate" , y = "True Positive Rate") +
    ggtitle(paste("ROC of", type, "attribution")) 
}

########################################DATA STATISTICS########################################
##### Channel statistics in the session data #####
df_channel_stat = df_data %>% 
  group_by(channelGroup) %>% 
  summarise(count = n(), count_conv = sum(ifelse(conversion_count > 0, 1, 0))) %>% 
  ungroup()
df_channel_stat$total_sessions = number_sessions
df_channel_stat$total_conv_sessions = sum(df_channel_stat$count_conv)
df_channel_stat = df_channel_stat %>% 
  mutate(share_sessions = count/total_sessions, share_conv_sessions = count_conv/total_conv_sessions, freq_conv = count_conv/count) %>% 
  select(-c(total_sessions, total_conv_sessions))

##### Channel statistics in the session data #####
number_journeys = length(df_path_TOTAL$journey)
number_journeys
df_path_TOTAL %>% group_by(journey_len) %>% filter(journey_len == 1) %>% count() %>% ungroup()
df_count_test = df_path_TOTAL %>% 
  group_by(journey_len) %>% filter(journey_len > 1) %>%
  count() %>% ungroup() 
sum(df_count_test$n)
df_count = df_path_TOTAL %>% 
  group_by(journey_len) %>% #filter(journey_len > 1) %>%
  count() %>% ungroup()
max(df_count$journey_len)
avg_len = sum(df_count$n*df_count$journey_len)/number_journeys
avg_len
number_conversion = sum(df_path_TOTAL$conv)
number_conversion
conv_rate_journey = number_conversion/number_journeys
conv_rate_journey

##### Graph: The number of journeys of the length from 1 to 10 #####
df_count_TOTAL = df_path_TOTAL %>% group_by(journey_len) %>% count() %>% filter(journey_len < 11) %>%
  rename(count_TOTAL = n) %>% 
  ungroup()
journey_length = df_count_TOTAL$journey_len
values = c(df_count_TOTAL$count_TOTAL)
level = c(rep("TOTAL", length(journey_length)))
df_count <- data.frame(journey_length, values, level)
ggplot(data = df_count, aes(journey_length, values)) + 
  geom_bar(stat = "identity", aes(fill = level), position = "dodge") + 
  xlab("Journey length") + ylab("Number of journeys") + scale_x_continuous(breaks=seq(0, 10, 2)) 