#Author: Linh Dinh 

########################################INSTALLING AND LOADING PACKAGES########################################

# install.packages("caret")
# install.packages("dplyr")
# install.packages("lubridate")
# install.packages("tidyverse")
# install.packages("pROC")
# install.packages("GameTheoryAllocation")

library(caret)
library(dplyr)
library(lubridate)
library(tidyverse)
library(pROC)
library(GameTheoryAllocation)

########################################LOADING NECESSARY SCRIPT(S)########################################
getwd()
source('Channel Attribution - Simulating data.R', echo=TRUE)
set.seed(123)

########################################SHAPLEY VALUE METHOD########################################
##### Attribution #####
##### Input
df_shapley_input = df_path_TOTAL %>%
  group_by(fullVisitorId, path_no) %>%
  mutate(OS = ifelse(str_count(journey,"OS") > 0,1,0),
         BPS = ifelse(str_count(journey,"BPS") > 0,1,0),
         GPS = ifelse(str_count(journey,"GPS") > 0,1,0),
         Direct = ifelse(str_count(journey,"Direct") > 0 ,1,0),
         Display = ifelse(str_count(journey,"Display") > 0,1,0),
         Email = ifelse(str_count(journey,"Email") > 0,1,0),
         Social = ifelse(str_count(journey,"Social") > 0,1,0),
         Referral = ifelse(str_count(journey,"Referral") > 0,1,0),
         Other = ifelse(str_count(journey,"Other") > 0,1,0)
  ) %>% 
  ungroup()

df_shapley_fold = inner_join(df_shapley_input, df_path_fold, 
                             by = c("fullVisitorId" = "fullVisitorId", "path_no" = "path_no")) %>% 
  dplyr::select(-c(journey_len.x, conv_null.x, journey.y, journey_len.y, conv.y, conv_null.y, is_conv.y)) %>% 
  rename(journey = journey.x, is_conv = is_conv.x, conv = conv.x)

##### Attribution Function 
shapley_attribution = function (data, num_channels = number_channels) {
  # Sum up the number of sequences and conversions for each combination of marketing channels
  df_convert_rates = data %>%
    group_by(Other,
             BPS,
             Direct,
             Display,
             Email,
             GPS, 
             OS,
             Referral,
             Social) %>%
    summarise(conversions = sum(conv), total_sequences = n()) %>% 
    ungroup()
  
  # The coalitions function is a handy function from the GameTheoryALlocation library 
  # that creates a binary matrix to which you can fit your characteristic function 
  df_touch_comb = as.data.frame(coalitions(num_channels)$Binary)
  names(df_touch_comb) = c("OS",
                           "BPS",
                           "GPS",
                           "Direct",
                           "Display",
                           "Email",
                           "Social", 
                           "Referral",
                           "Other")
  
  # Join the previous summary results with the binary matrix.
  df_touch_comb_conv_rate = left_join(df_touch_comb, df_convert_rates, 
                                      by = c("OS" = "OS",
                                             "BPS" = "BPS",
                                             "GPS" = "GPS",
                                             "Direct" = "Direct",
                                             "Display" = "Display",
                                             "Email" = "Email",
                                             "Social" = "Social",
                                             "Referral" = "Referral",
                                             "Other" = "Other"))
  
  df_touch_comb_conv_rate = df_touch_comb_conv_rate %>%
    mutate_all(funs(ifelse(is.na(.),0,.))) %>%
    mutate(conv_rate = ifelse(total_sequences > 0, conversions/total_sequences, 0))
  
  # Building Shapley Values for each channel combination
  df_shapley_val = as.data.frame(coalitions(num_channels)$Binary)
  #names(df_shapley_val) = all_channels
  names(df_shapley_val) = c("OS",
                            "BPS",
                            "GPS",
                            "Direct",
                            "Display",
                            "Email",
                            "Social", 
                            "Referral",
                            "Other")
  coalition_matrix = df_shapley_val
  df_shapley_val[2^num_channels,] = Shapley_value(df_touch_comb_conv_rate$conv_rate, game="profit")
  
  for(i in 2:(2^num_channels-1)){
    if(sum(coalition_matrix[i,]) == 1){
      df_shapley_val[i,which(df_shapley_val[i,]==1)] = df_touch_comb_conv_rate[i,"conv_rate"]
    }
    else if(sum(coalition_matrix[i,]) > 1){
      if(sum(coalition_matrix[i,]) < num_channels){
        channels_of_interest = which(coalition_matrix[i,] == 1)
        char_func = data.frame(rates = df_touch_comb_conv_rate[1,"conv_rate"])
        for(j in 2:i){
          if(sum(coalition_matrix[j,channels_of_interest])>0 & 
             sum(coalition_matrix[j,-channels_of_interest])==0) {
            char_func = rbind(char_func,df_touch_comb_conv_rate[j,"conv_rate"])
          }
        }
        df_shapley_val[i,channels_of_interest] = Shapley_value(char_func$rates, game="profit")
      }
    }
  }
  
  # Apply Shapley Values as attribution weighting
  order_distribution = df_shapley_val * df_touch_comb_conv_rate$total_sequences
  shapley_total = t(t(round(colSums(order_distribution),4)))
  df_shapley_conversion = data.frame(channel_name = row.names(shapley_total), 
                                     conversions = as.numeric(shapley_total))
  df_shapley_attr = df_shapley_conversion %>% 
    mutate(attribution = round(conversions/sum(conversions) , 6)) %>% 
    dplyr::select(-conversions)
  return(list(shapley_conversion = df_shapley_conversion, shapley_attribution = df_shapley_attr, conversion_rate = df_touch_comb_conv_rate))
}

##### Attribution Results
Shapley = shapley_attribution(df_shapley_input, num_channels = number_channels)
shapley_conversion_rate = Shapley$conversion_rate
df_shapley_attribution = Shapley$shapley_attribution

##### Predictive power #####
##### Predict function 
### Function to get the test predicted results depended on the train conversion rate
predict_shapley = function(data_test, conversion_rate) {
  df_shapley_combined = left_join(data_test,conversion_rate,  
                                  by = c("Other" = "Other",
                                         "BPS" = "BPS",
                                         "Direct" = "Direct",
                                         "Display" = "Display",
                                         "Email" = "Email",
                                         "GPS" = "GPS",
                                         "OS" = "OS",
                                         "Referral" = "Referral",
                                         "Social" = "Social"))
  
  data_predict = df_shapley_combined %>%
    mutate(conv_rate = ifelse(conv_rate >= 1, 1, conv_rate))  %>% 
    dplyr::select(journey,is_conv,conv_rate)
  return(data_predict)
}

##### Get the prediction accuracy and robustness for all the markov models 
### Initializing
auc_shapley = rep(0,kfold_num); attr_shapley = list(list()); fpr_shapley = list(list()); tpr_shapley = list(list())

### Running k-fold CV
for (i in 1:kfold_num){
  ###Getting train and test sets
  train_set = df_shapley_fold %>% filter(fold != i) 
  test_set = df_shapley_fold %>% filter(fold == i)
  # test_set = train_set
  
  ###Shapley Value Method  
  #Fitting
  Shapley = shapley_attribution(train_set, number_channels)
  attr_shapley[[i]] = Shapley$shapley_attribution
  #Predicting
  train_conversion_rate = Shapley$conversion_rate
  shapley_predict = predict_shapley(test_set, train_conversion_rate)
  shapley_roc = roc(is_conv ~ conv_rate, data = shapley_predict)
  auc_shapley[i] = shapley_roc$auc
  fpr_shapley[[i]] = 1 - shapley_roc$specificities
  tpr_shapley[[i]] = shapley_roc$sensitivities 
}

#Test AUC 
cat("Shapley Value test AUC has mean (coefficient of variation):", mean(auc_shapley), "(",sd(auc_shapley)/mean(auc_shapley), ")")

#Plot ROC 
plot_kfold_roc(fpr_shapley, tpr_shapley, kfold = kfold_num, type = "Shapley Value")

#Attribution coefficient of variation per channel  
all_attr_shapley = attr_shapley %>% reduce(left_join, by = "channel_name")%>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_shapley[,c(1,14)]

#Attribution average coefficient of variation
cat("Shapley Value attribution average CV:", mean(all_attr_shapley$CV))
