#Author: Linh Dinh 

########################################INSTALLING AND LOADING PACKAGES########################################

# install.packages("caret")
# install.packages("dplyr")
# install.packages("lubridate")
# install.packages("tidyverse")
# install.packages("pROC")

library(caret)
library(dplyr)
library(lubridate)
library(tidyverse)
library(pROC)

########################################LOADING NECESSARY SCRIPT(S)########################################
getwd()
source('Channel Attribution - Simulating data.R', echo=TRUE)
set.seed(123)

########################################HEURISTIC ATTRIBUTION########################################
##### Attribution ##### 
##### Input
### Single-touch models 
df_single_input = df_path_TOTAL %>%
  mutate(first_channel = sub(' >.*','', journey),
         last_channel = sub('.*> ', '', journey))
df_single_input$total_conv =  sum(df_single_input$conv)

df_single_fold = left_join(df_single_input, df_path_fold, 
                           by = c("fullVisitorId" = "fullVisitorId", "path_no" = "path_no")) %>%
  dplyr::select(fullVisitorId, path_no, first_channel, last_channel, conv.x, conv_null.x, is_conv.x, fold) %>% 
  rename(conv = conv.x, conv_null = conv_null.x, is_conv = is_conv.x)

### Linear-touch model
df_linear_input =  df_path_TOTAL %>% 
  group_by(fullVisitorId,path_no) %>% 
  separate_rows(journey, sep = " > ") %>% 
  mutate(weight_conv = conv/n(), weight_null = conv_null/n()) %>% 
  rename(channel_name = journey) %>% 
  ungroup()
df_linear_input$total_conv =  sum(df_linear_input$weight_conv)

df_linear_fold = left_join(df_linear_input, df_path_fold, 
                           by = c("fullVisitorId" = "fullVisitorId", "path_no" = "path_no")) %>%
  dplyr::select(fullVisitorId, path_no, channel_name, conv.x, conv_null.x, weight_conv, weight_null, is_conv.x, fold) %>% 
  rename(conv = conv.x, conv_null = conv_null.x, is_conv = is_conv.x)

##### Attribution function 
### Single-touch models 
single_touch_attr = function(data, channel_type = "first") {
  if(channel_type == "last" | channel_type == "first") {
    df_fit = data %>% 
      filter(conv > 0) %>% 
      rename(channel_name = ifelse(channel_type == "last", "last_channel", "first_channel")) %>%
      group_by(channel_name) %>%
      summarise(conversion = sum(conv), attribution =  round(sum(conv)/max(total_conv),6))
    df_dummy = data.frame(channel_name = all_channels, dummy_col = rep(0, number_channels))
    combined <- sort(union(levels(df_fit$channel_name), levels(df_dummy$channel_name)))
    df_result = left_join(mutate(df_dummy, channel_name=factor(channel_name, levels=combined)),
                          mutate(df_fit, channel_name=factor(channel_name, levels=combined)), 
                          by = c("channel_name" = "channel_name")) %>% 
      dplyr::select(-dummy_col) %>% replace_na(list(attribution = 0, conversion = 0))
    return(df_result)
  }
  else{print("Not available channel_type!")}
}

### Linear-touch model
linear_touch_attr = function(data) {
  df_fit = data %>%
    filter(conv > 0) %>%
    group_by(channel_name) %>%
    summarise(conversion = sum(weight_conv), attribution = round(sum(weight_conv)/max(total_conv),6)) %>% 
    ungroup()
  df_dummy = data.frame(channel_name = all_channels, dummy_col = rep(0, number_channels))
  combined <- sort(union(levels(df_fit$channel_name), levels(df_dummy$channel_name)))
  df_result = left_join(mutate(df_dummy, channel_name=factor(channel_name, levels=combined)),
                        mutate(df_fit, channel_name=factor(channel_name, levels=combined)), 
                        by = c("channel_name" = "channel_name")) %>% 
    dplyr::select(-dummy_col) %>% replace_na(list(attribution = 0, conversion = 0))
  return(df_result)
}

##### Attribution Results for all heuristics models
options(pillar.sigfig=7)

### First-touch model
df_first_touch = single_touch_attr(df_single_input, channel_type = "first") %>% 
  rename(first_touch_conversion = conversion, first_touch_attribution = attribution)

### Last-touch model
df_last_touch = single_touch_attr(df_single_input, channel_type = "last") %>% 
  rename(last_touch_conversion = conversion, last_touch_attribution = attribution)

### Linear-touch model
df_linear_touch = linear_touch_attr(df_linear_input) %>% 
  rename(linear_conversion = conversion, linear_attribution = attribution)

### All heuristics models
df_heuristics = Reduce(function(x, y) merge(x = x, y = y, by.x = "channel_name", by.y = "channel_name", all = TRUE), 
                       list(df_first_touch, df_last_touch, df_linear_touch)) %>% 
  dplyr::select(-c(first_touch_conversion, last_touch_conversion, linear_conversion))
df_heuristics[is.na(df_heuristics)] = 0

##### Predictive power ##### 
##### Predictive functions 
### Single-touch models
predict_single_touch = function(data_train, data_test, channel_type = "last") {
  if(channel_type == "last" | channel_type == "first") {
    data_fit = data_train %>% 
      rename(channel_name = ifelse(channel_type == "last", "last_channel", "first_channel")) %>%
      group_by(channel_name) %>%
      summarise(conversion_rate = sum(conv) /(sum(conv) + sum(conv_null))) %>%
      ungroup()
    data_test = data_test %>%      
      rename(channel_name = ifelse(channel_type == "last",  "last_channel", "first_channel"))
    data_predict = left_join(data_test, data_fit, by = c("channel_name" = "channel_name")) %>%
      dplyr::select(channel_name, is_conv, conversion_rate)
    return(data_predict)
  }
  else {print("Not available channel_type!")}
}

### Linear-touch models
predict_linear_touch = function(data_train, data_test) {
  data_fit = data_train %>% 
    group_by(channel_name) %>%
    summarise(conversion_rate = (sum(weight_conv)) /(sum(weight_conv) + sum(weight_null))) %>% 
    ungroup()
  data_predict = left_join(data_test, data_fit, by = c("channel_name" = "channel_name")) %>% 
    dplyr::select(fullVisitorId, path_no, channel_name, is_conv, conversion_rate) %>% 
    group_by(fullVisitorId, path_no) %>% 
    summarise(is_conv = ifelse(sum(is_conv) == 0, 0, 1), conversion_rate = sum(conversion_rate)) %>% 
    ungroup()
  return(data_predict)
}

##### Get the prediction accuracy and robustness for all the heuristic models 
### Initializing
auc_first = rep(0,kfold_num);attr_first = list(list()); fpr_first = list(list()); tpr_first = list(list())
auc_last = rep(0,kfold_num); attr_last = list(list()); fpr_last = list(list()); tpr_last = list(list())
auc_linear = rep(0, kfold_num); attr_linear = list(list()); fpr_linear = list(list()); tpr_linear = list(list())

### Running k-fold CV
for (i in 1:kfold_num){
  ###Single-touch method: split train and test set
  train_set = df_single_fold %>% filter(fold != i)
  test_set = df_single_fold %>% filter(fold == i)
  # test_set = train_set
  train_set$total_conv =  sum(train_set$conv)
  
  #First-touch method
  attr_first[[i]] = single_touch_attr(train_set, channel_type = "first")[,-2]
  first_touch_predict = predict_single_touch(train_set, test_set, channel_type = "first")
  first_touch_roc = roc(is_conv ~ conversion_rate, data = first_touch_predict)
  auc_first[i] = first_touch_roc$auc
  fpr_first[[i]] = 1-first_touch_roc$specificities
  tpr_first[[i]] = first_touch_roc$sensitivities
  
  #Last-touch method
  attr_last[[i]] = single_touch_attr(train_set, channel_type = "last")[,-2]
  last_touch_predict = predict_single_touch(train_set, test_set, channel_type = "last")
  last_touch_roc = roc(is_conv ~ conversion_rate, data = last_touch_predict)
  auc_last[i] = last_touch_roc$auc
  fpr_last[[i]] = 1-last_touch_roc$specificities
  tpr_last[[i]] = last_touch_roc$sensitivities
  
  ###Linear-touch method: split train and test set
  train_set_lin = df_linear_fold %>% filter(fold != i) 
  test_set_lin = df_linear_fold %>% filter(fold == i)
  # test_set_lin = train_set_lin
  train_set_lin$total_conv =  sum(train_set_lin$weight_conv)
  
  #Linear-touch method
  attr_linear[[i]] = linear_touch_attr(train_set_lin)[,-2]
  linear_touch_predict = predict_linear_touch(train_set_lin, test_set_lin)
  linear_touch_roc = roc(is_conv ~ conversion_rate, data = linear_touch_predict)
  auc_linear[i] = linear_touch_roc$auc
  fpr_linear[[i]] = 1-linear_touch_roc$specificities
  tpr_linear[[i]] = linear_touch_roc$sensitivities
}

### Test AUC 
cat("First touch test AUC has mean (coefficient of variation):", mean(auc_first), "(",sd(auc_first)/mean(auc_first), ")")
cat("Last touch test AUC has mean (coefficient of variation):", mean(auc_last), "(",sd(auc_last)/mean(auc_last), ")")
cat("Linear touch test AUC has mean (coefficient of variation):", mean(auc_linear), "(",sd(auc_linear)/mean(auc_linear), ")")

### Plot ROC 
plot_kfold_roc(fpr_first, tpr_first, kfold = kfold_num, type = "first touch")
plot_kfold_roc(fpr_last, tpr_last, kfold = kfold_num, type = "last touch")
plot_kfold_roc(fpr_linear, tpr_linear, kfold = kfold_num, type = "linear touch")

#Attribution coefficient of variation per channel 
all_attr_first = attr_first %>% reduce(left_join, by = "channel_name")%>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_last = attr_last %>% reduce(left_join, by = "channel_name") %>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_linear = attr_linear %>% reduce(left_join, by = "channel_name") %>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_first[,c(1,14)]
all_attr_last[,c(1,14)]
all_attr_linear[,c(1,14)]

#Attribution average coefficient of variation
cat("First touch attribution average CV:", mean(all_attr_first$CV))
cat("Last touch attribution average CV:", mean(all_attr_last$CV))
cat("Linear touch attribution average CV:", mean(all_attr_linear$CV))
