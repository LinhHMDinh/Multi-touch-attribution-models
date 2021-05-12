#Author: Linh Dinh 

########################################INSTALLING AND LOADING PACKAGES########################################

# install.packages("caret")
# install.packages("dplyr")
# install.packages("lubridate")
# install.packages("tidyverse")
# install.packages("pROC")
# install.packages("GameTheoryAllocation")
# install.packages("glmnet")
# install.packages("margins")

library(caret)
library(dplyr)
library(lubridate)
library(tidyverse)
library(pROC)
library(GameTheoryAllocation)
library(glmnet)
library(margins)

########################################LOADING NECESSARY SCRIPT(S)########################################
getwd()
source('Channel Attribution - Simulating data.R', echo=TRUE)
set.seed(123)

########################################LOGISTIC REGRESSION########################################
##### Estimation #####
##### Input 
### Simple logistic regression
df_logit_input = df_path_TOTAL %>% 
  group_by(fullVisitorId,path_no) %>% 
  mutate(OrganicSearch = str_count(journey,"OS"),
         BrandedPaidSearch = str_count(journey,"BPS"),
         GenericPaidSearch = str_count(journey,"GPS"),
         Direct = str_count(journey,"Direct"),
         Display = str_count(journey,"Display"),
         Email = str_count(journey,"Email"),
         Social = str_count(journey,"Social"), 
         Referral = str_count(journey,"Referral"), 
         Other = str_count(journey,"Other")) %>%  #conversions = sum(conv)
  ungroup()

df_logit_fold = inner_join(df_logit_input, df_path_fold, 
                           by = c("fullVisitorId" = "fullVisitorId", "path_no" = "path_no")) %>% 
  dplyr::select(-c(journey.x, journey_len.x, conv = conv.x, conv_null.x, 
                   journey.y, journey_len.y, conv.y, conv_null.y, is_conv.y)) %>% 
  rename(is_conv = is_conv.x)

# ### Dynamic logistic regression 1 (with the last channel)
# # The regressors are the number of touches of channel that are not the last channel 
# # The dummies are created for the last channel, excluding one channel (Direct) to avoid perfect multicollinearity 
# df_logit_dy1_input = df_path_TOTAL %>% group_by(fullVisitorId,path_no) %>% 
#   mutate(journey = strsplit(journey, " > ")) %>% 
#   mutate(last_channel = last(journey[[1]]), 
#          leftover = ifelse(journey_len > 1, list(journey[[1]][-journey_len]), list("None"))) %>%  
#   mutate(OrganicSearch_lag1 = ifelse(last_channel == "OrganicSearch", 1, 0),
#          BrandedPaidSearch_lag1 = ifelse(last_channel == "BrandedPaidSearch", 1, 0), 
#          GenericPaidSearch_lag1 = ifelse(last_channel == "GenericPaidSearch", 1, 0),
#          # Direct_lag1 = ifelse(last_channel == "Direct", 1, 0),
#          Display_lag1 = ifelse(last_channel == "Display", 1, 0),
#          Email_lag1 = ifelse(last_channel == "Email", 1, 0),
#          Social_lag1 = ifelse(last_channel == "Social", 1, 0), 
#          Referral_lag1 = ifelse(last_channel == "Referral", 1, 0),
#          Other_lag1 = ifelse(last_channel == "Other", 1, 0), 
#          
#          OrganicSearch_left= ifelse(last_channel == "OrganicSearch", 0, length(grep("OrganicSearch", leftover[[1]]))),
#          BrandedPaidSearch_left = ifelse(last_channel == "BrandedPaidSearch", 0, length(grep("BrandedPaidSearch", leftover[[1]]))),
#          GenericPaidSearch_left = ifelse(last_channel == "GenericPaidSearch", 0, length(grep("GenericPaidSearch", leftover[[1]]))),
#          Direct_left = ifelse(last_channel == "Direct", 0, length(grep("Direct", leftover[[1]]))),
#          Display_left = ifelse(last_channel == "Display", 0, length(grep("Display", leftover[[1]]))),
#          Email_left = ifelse(last_channel == "Email", 0, length(grep("Email", leftover[[1]]))),
#          Social_left = ifelse(last_channel == "Social", 0, length(grep("Social", leftover[[1]]))), 
#          Referral_left =  ifelse(last_channel == "Referral", 0, length(grep("Referral", leftover[[1]]))),
#          Other_left = ifelse(last_channel == "Other", 0, length(grep("Other", leftover[[1]])))) %>% 
#   ungroup() %>%
#   dplyr::select(-c(journey,journey_len,conv,conv_null,last_channel,leftover))
# 
# df_logit_dy1_fold = inner_join(df_logit_dy1_input, df_path_fold, by = c("fullVisitorId" = "fullVisitorId", "path_no" = "path_no")) %>% 
#   dplyr::select(-c(journey, journey_len, conv, conv_null, Other_lag1, is_conv.y)) %>% 
#   rename(is_conv = is_conv.x)

# ### Dynamic logistic regression 2 (with two last channels)
# # The regressors are the number of touches of channel that are not the last channel nor the second last channel 
# # The dummies are created for the last channel and the second last channel, excluding one channel (Direct) to avoid perfect multicollinearity
# df_logit_dy2_input = df_path_TOTAL %>% group_by(fullVisitorId,path_no) %>% 
#   mutate(journey = strsplit(journey, " > ")) %>% 
#   mutate(last_channel = last(journey[[1]]), 
#          lag2_channel = ifelse(journey_len > 1, nth(journey[[1]], -2), "None" ), 
#          leftover = ifelse(journey_len > 2, list(journey[[1]][-c(journey_len-1, journey_len)]), list("None"))) %>%  
#   mutate(OrganicSearch_last = ifelse(last_channel == "OrganicSearch", 1, 0),
#          BrandedPaidSearch_lag1 = ifelse(last_channel == "BrandedPaidSearch", 1, 0),
#          GenericPaidSearch_lag1 = ifelse(last_channel == "GenericPaidSearch", 1, 0), 
#          # Direct_lag1 = ifelse(last_channel == "Direct", 1, 0),
#          Display_lag1 = ifelse(last_channel == "Display", 1, 0),
#          Email_lag1 = ifelse(last_channel == "Email", 1, 0),
#          Social_lag1 = ifelse(last_channel == "Social", 1, 0),
#          Referral_lag1 = ifelse(last_channel == "Referral", 1, 0),
#          Other_lag1 = ifelse(last_channel == "Other", 1, 0), 
#          
#          OrganicSearch_lag2 = ifelse(lag2_channel == "OrganicSearch", 1, 0),
#          BrandedPaidSearch_lag2 = ifelse(lag2_channel == "BrandedPaidSearch", 1, 0),
#          GenericPaidSearch_lag2 = ifelse(lag2_channel == "GenericPaidSearch", 1, 0),
#          # Direct_lag2 = ifelse(lag2_channel == "Direct", 1, 0),
#          Display_lag2 = ifelse(lag2_channel == "Display", 1, 0),
#          Email_lag2 = ifelse(lag2_channel == "Email", 1, 0),
#          Social_lag2 = ifelse(lag2_channel == "Social", 1, 0),
#          Referral_lag2 = ifelse(lag2_channel == "Referral", 1, 0),
#          Other_lag2 = ifelse(lag2_channel == "Other", 1, 0), 
#     
#          OrganicSearch_left= ifelse(last_channel == "OrganicSearch" && lag2_channel == "OrganicSearch", 0,
#                                     length(grep("OrganicSearch", leftover[[1]]))),
#          BrandedPaidSearch_left = ifelse(last_channel == "BrandedPaidSearch" && lag2_channel == "BrandedPaidSearch", 0,
#                                          length(grep("BrandedPaidSearch", leftover[[1]]))),
#          GenericPaidSearch_left = ifelse(last_channel == "GenericPaidSearch" && lag2_channel == "GenericPaidSearch", 0,
#                                          length(grep("GenericPaidSearch", leftover[[1]]))),
#          Direct_left = ifelse(last_channel == "Direct" && lag2_channel == "Direct", 0,
#                               length(grep("Direct", leftover[[1]]))),
#          Display_left = ifelse(last_channel == "Display" && lag2_channel == "Display", 0, 
#                                length(grep("Display", leftover[[1]]))),
#          Email_left = ifelse(last_channel == "Email" && lag2_channel == "Email", 0,
#                              length(grep("Email", leftover[[1]]))),
#          Social_left = ifelse(last_channel == "Social" && lag2_channel == "Social", 0,
#                               length(grep("Social", leftover[[1]]))),
#          Referral_left =  ifelse(last_channel == "Referral"&& lag2_channel == "Referral", 0,
#                                  length(grep("Referral", leftover[[1]]))),
#          Other_left = ifelse(last_channel == "Other" && lag2_channel == "Other", 0,
#                              length(grep("Other", leftover[[1]]))) ) %>% 
#   ungroup() %>% 
#   dplyr::select(-c(journey,journey_len,conv,conv_null,last_channel,lag2_channel,leftover))
# 
# df_logit_dy2_fold = inner_join(df_logit_dy2_input, df_path_fold, by = c("fullVisitorId" = "fullVisitorId", "path_no" = "path_no")) %>% 
#   dplyr::select(-c(journey, journey_len, conv, conv_null, Other_lag1, Other_lag2, is_conv.y)) %>% 
#   rename(is_conv = is_conv.x)

##### Estimation functions
### A. Orginal Logistic Regression:
### Function to fit the logistic regression for the standard model the and dynamic models 
fit_logit = function(data) {
  df_input = data %>% dplyr::select(-c(fullVisitorId, path_no, fold)) 
  glm_fit = glm(is_conv ~ ., data = df_input, family = "binomial")
  coef = as.data.frame(summary(glm_fit)$coefficients) %>% 
    dplyr::select(Estimate)
  coef_result = c(coef[,1][-1],coef[,1][1])
  names(coef_result) = c(rownames(coef)[-1],rownames(coef)[1])
  return(list(coeficient = coef_result, fit = glm_fit))
}

### Function to obtain the marginal effect for the logitstic regression
marginal_effect = function(data) {
  df_input = data %>% dplyr::select(-c(fullVisitorId, path_no, fold)) 
  glm_fit = glm(is_conv ~ ., data = df_input, family = "binomial")
  mar = margins(glm_fit, type = "response")
  return(mar)
}

### B. Regularized Logistic Regression: 
### Function to fit logistic regression for the regularized model 
fit_reg = function(data) { 
  df_input = data %>% dplyr::select(-c(fullVisitorId, path_no, fold))
  y = as.vector(ifelse(df_input$is_conv == 1, 1, 0))
  X = model.matrix(is_conv~., df_input)
  glmnet_cv = cv.glmnet(X, y, family = "binomial", alpha = 0) #alpha=0 the ridge regularization
  lambda_cv = glmnet_cv$lambda.min
  glmnet_fit = glmnet(X, y, family = "binomial", alpha = 0, lambda=lambda_cv) 
  coef_glmnet = coef(glmnet_fit) 
  coef_result = c(coef_glmnet[,1][-c(1,2)],coef_glmnet[,1][1])
  return(list(coeficient = coef_result, fit = glmnet_fit, lambda = lambda_cv))
}

### Function to fit logistic regression for the regularized model with different lambdas 
logitreg_full = function(data) {
  df_input = data %>% dplyr::select(-c(fullVisitorId, path_no, fold))
  y = as.vector(ifelse(df_input$is_conv == 1, 1, 0))
  X = model.matrix(is_conv~., df_input)  
  glmnet_fit = glmnet(X, y, family = "binomial", alpha = 0)
  return(glmnet_fit)
}

### Function to add legend to plot for the coefficients vs different log lambdas
add_legend = function(fit) {
  par(mar=c(5, 4, 4, 8), xpd=TRUE)
  L = length(fit$lambda)
  x = log(fit$lambda[L])
  y = fit$beta[, L]
  labs = names(y)
  legend("topright", inset = c(-0.2, 0), legend=labs, col= factor(1:length(labs)), lty=1)
}


##### Estimation results
### A. Orginal Logistic Regression:
# Result of fit_logit
fitresult_logit = fit_logit(df_logit_fold)
coef_logit = fitresult_logit$coeficient
summary(fitresult_logit$fit)

# Result of marginal_effect
ME = marginal_effect(df_logit_fold) 
df_ME = summary(ME) %>% select(factor,AME,lower,upper)
ggplot(df_ME, aes(x=factor, y=AME)) + 
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.4) +
  geom_line() +
  geom_point() + 
  theme_bw() +
  labs(y= "AME", x = "") +
  geom_hline(yintercept=0, linetype="dashed", color = "red") +
  theme(axis.text.x = element_text(angle = 90)) 

### B. Regularized Logistic Regression: 
# Result of fit_reg
fitresult_logitreg = fit_reg(df_logit_fold)
fit_logitreg = fitresult_logitreg$fit
lambda_logitreg = fitresult_logitreg$lambda
coef_logitreg = fitresult_logitreg$coeficient

# Result of logitreg_full
logitreg_full = logitreg_full(df_logit_fold)
# Plot coefficients vs log lambdas
dev.off()
plot(logitreg_full, xvar="lambda",label=TRUE, col=as.factor(1:dim(coef(logitreg_full))[1]))
add_legend(logitreg_full)

### C. Dynamic Logistic Regression:
# #Result of fit_logit for dynamic logit 1
# fitresult_logitdy1 = fit_logit(df_logit_dy1_fold)
# coef_logitdy1 = fitresult_logitdy1$coeficient
# fit_logitdy1 = fitresult_logitdy1$fit

# #Result of fit_logit for dynamic logit 2
# fitresult_logitdy2 = fit_logit(df_logit_dy2_fold)
# coef_logitdy2 = fitresult_logitdy2$coeficient
# fit_logitdy2 = fitresult_logitdy2$fit

##### Attribution #####
##### Attribution function
logit_attribution = function (data, fit_coef,  num_channels = number_channels) {
  # Convert logit input data to Shapley input data
  df_input = data %>%
    group_by(fullVisitorId, path_no) %>%
    mutate(OrganicSearch = ifelse(OrganicSearch > 0,1,0),
           BrandedPaidSearch = ifelse(BrandedPaidSearch > 0,1,0),
           GenericPaidSearch = ifelse(GenericPaidSearch > 0,1,0),
           Direct = ifelse(Direct > 0 ,1,0),
           Display = ifelse(Display > 0,1,0),
           Email = ifelse(Email > 0,1,0),
           Social = ifelse(Social > 0,1,0),
           Referral = ifelse(Referral > 0,1,0),
           Other = ifelse(Other > 0,1,0)) %>% 
    ungroup()
  
  # Sum up the number of sequences for each combination of marketing channels
  df_convert_rates = df_input %>%
    group_by(Other,
             BrandedPaidSearch,
             Direct,
             Display,
             Email,
             GenericPaidSearch,
             OrganicSearch,
             Referral,
             Social) %>%
    summarise(total_sequences = n()) %>%
    ungroup()
  
  # The coalitions function is a handy function from the GameTheoryALlocation library 
  # that creates a binary matrix to which you can fit your characteristic function 
  df_touch_comb = as.data.frame(coalitions(num_channels)$Binary)
  coef_name = names(fit_coef)
  names(df_touch_comb) = coef_name[-length(coef_name)]
  
  # Include the intercept in the binary matrix created above
  df_touch_comb$Intercept = 1
  df_touch_comb$Intercept[1] = 0
  mat_touch_comb = as.matrix(df_touch_comb)
  
  # Get the predicted log odds for each combination of the channels
  pred_log_odds = mat_touch_comb %*% fit_coef
  df_touch_comb$predict_log_odds = pred_log_odds 
  
  df_touch_comb_conv_rate = left_join(df_touch_comb, df_convert_rates,
                                      by = c("OrganicSearch" = "OrganicSearch",
                                             "BrandedPaidSearch" = "BrandedPaidSearch",
                                             "GenericPaidSearch" = "GenericPaidSearch",
                                             "Direct" = "Direct",
                                             "Display" = "Display",
                                             "Email" = "Email",
                                             "Social" = "Social",
                                             "Referral" = "Referral",
                                             "Other" = "Other"))
  
  # df_touch_comb_conv_rate = df_touch_comb_conv_rate %>%
  #   mutate_all(funs(ifelse(is.na(.),0,.)))
  df_touch_comb_conv_rate[is.na(df_touch_comb_conv_rate)] = 0
  
  # Create a new table to to get the predicted conversion rate for each combination 
  df_touch_comb_conv_rate = df_touch_comb_conv_rate %>% 
    mutate(conv_rate = exp(predict_log_odds) / (1+exp(predict_log_odds))) %>% # OR conv_rate = 1 / (1+exp(-predict_log_odds)))
    dplyr::select(-c(Intercept,predict_log_odds))
  df_touch_comb_conv_rate$conv_rate[1] = 0
  
  # Build Shapley Values for each channel combination
  df_shapley_val = as.data.frame(coalitions(num_channels)$Binary)
  names(df_shapley_val) = coef_name[-length(coef_name)]
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
            add_rate = as.numeric(df_touch_comb_conv_rate[j,"conv_rate"])
            char_func = rbind(char_func,add_rate)
          }
        }
        df_shapley_val[i,channels_of_interest] = Shapley_value(char_func$rates, game="profit")
      }
    }
  }
  # Apply Shapley Values as attribution weighting
  df_order_distribution = df_shapley_val * df_touch_comb_conv_rate$total_sequences
  df_shapley_total = t(t(round(colSums(df_order_distribution),4)))
  df_shapley_conversion = data.frame(channel_name = row.names(df_shapley_total), 
                                     conversions = as.numeric(df_shapley_total))
  df_shapley_attr = df_shapley_conversion %>% 
    mutate(attribution = round(conversions/sum(conversions) , 6)) %>% 
    dplyr::select(-conversions)
  return(list(conversion = df_shapley_conversion, attribution = df_shapley_attr))
}

##### Attribution results 
### Logistic model
logit_attr = logit_attribution(df_logit_fold, coef_logit)
logit_attr$attribution
# logit_attr$conversion

### Regularized Logitstic model
logitreg_attr = logit_attribution(df_logit_fold, coef_logitreg)
logitreg_attr$attribution
# logitreg_attr$conversion

##### Predictive power #####
##### Predict function 
### Function to get the roc curve of the predicted probability and the real binary conversion depended on the train coeficients
roc_logitreg = function(test_set, train_fit, train_lambda) {
  df_test = test_set %>% dplyr::select(-c(fullVisitorId, path_no, fold))
  X_test = model.matrix(is_conv ~., df_test)
  y_test = as.vector(ifelse(df_test$is_conv == 1, 1, 0))
  predict_prob = as.numeric(predict(train_fit, newx= X_test, s = train_lambda, type="response"))
  roc = roc(y_test, predict_prob)
  return(roc)
}

##### Get the prediction accuracy and robustness for all the markov models 
### Initializing
auc_logit = rep(0,kfold_num); attr_logit = list(list()); fpr_logit = list(list()); tpr_logit = list(list())
auc_logit_reg = rep(0,kfold_num); attr_logit_reg = list(list()); fpr_logit_reg = list(list()); tpr_logit_reg = list(list())
# auc_logit_dy1 = rep(0,kfold_num); attr_logit_dy1 = list(list()); fpr_logit_dy1 = list(list()); tpr_logit_dy1 = list(list())
# auc_logit_dy2 = rep(0,kfold_num); attr_logit_dy2 = list(list()); fpr_logit_dy2 = list(list()); tpr_logit_dy2 = list(list())

### Running k-fold CV
for (i in 1:kfold_num){
  ### Getting train and test sets
  train_set_logit = df_logit_fold %>% filter(fold != i)
  test_set_logit = df_logit_fold %>% filter(fold == i)
  # test_set_logit = train_set_logit
  
  ### Logistic regression
  #Fitting
  logit_estimation = fit_logit(train_set_logit)
  train_fit_logit = logit_estimation$fit
  train_coef_logit = logit_estimation$coeficient
  # #Attribution
  attr_logit[[i]] = logit_attribution(train_set_logit, train_coef_logit)$attribution
  #Predicting
  predict_prob = plogis(predict(train_fit_logit, test_set_logit))
  roc_logit = roc(test_set_logit$is_conv, predict_prob) #roc(is_conv ~ predict_prob, data = test_set_logit)
  auc_logit[i] = roc_logit$auc
  fpr_logit[[i]] = 1 - roc_logit$specificities
  tpr_logit[[i]] = roc_logit$sensitivities
  
  ### Regularized logistic regression
  #Fitting
  logitreg_estimation = fit_reg(train_set_logit)
  train_fit_logitreg = logitreg_estimation$fit
  train_coef_logitreg = logitreg_estimation$coeficient
  train_lambda_logitreg = logitreg_estimation$lambda
  #Attribution
  attr_logit_reg[[i]] = logit_attribution(train_set_logit, train_coef_logitreg)$attribution
  #Predicting
  roc_reg =  roc_logitreg(test_set_logit, train_fit_logitreg, train_lambda_logitreg)
  auc_logit_reg[i] = roc_reg$auc
  fpr_logit_reg[[i]] = 1 - roc_reg$specificities
  tpr_logit_reg[[i]] = roc_reg$sensitivities
  
  # ### Dynamic logistic regression 1
  # #Getting train and test sets
  # train_set_dy1 = df_logit_dy1_fold %>% filter(fold != i)
  # test_set_dy1 = df_logit_dy1_fold %>% filter(fold == i)
  # #test_set_dy1 = train_set_dy1
  # #Fitting
  # train_fit_dy1 = fit_logit(train_set_dy1)$fit
  # #Predicting
  # test_set_dy1$predict_prob  = plogis(predict(train_fit_dy1, test_set_dy1))
  # dy1_roc = roc(is_conv ~ predict_prob, data = test_set_dy1)
  # auc_logit_dy1[i] = dy1_roc$auc
  # fpr_logit_dy1[[i]] = 1 - dy1_roc$specificities
  # tpr_logit_dy1[[i]] = dy1_roc$sensitivities
  # 
  # ### Dynamic logistic regression 2
  # #Getting train and test sets
  # train_set_dy2 = df_logit_dy2_fold %>% filter(fold != i)
  # test_set_dy2 = df_logit_dy2_fold %>% filter(fold == i)
  # #test_set_dy2 = train_set_dy2
  # #Fitting
  # train_fit_dy2 = fit_logit(train_set_dy2)$fit
  # #Predicting
  # test_set_dy2$predict_prob  = plogis(predict(train_fit_dy2, test_set_dy2))
  # dy2_roc = roc(is_conv ~ predict_prob, data = test_set_dy2)
  # auc_logit_dy2[i] = dy2_roc$auc
  # fpr_logit_dy2[[i]] = 1 - dy2_roc$specificities
  # tpr_logit_dy2[[i]] = dy2_roc$sensitivities
}

#Test AUC 
cat("Logistic regression test AUC has mean (sd):", mean(auc_logit), "(",sd(auc_logit)/mean(auc_logit), ")")
cat("Regularized Logistic regression test AUC has mean (sd):", mean(auc_logit_reg), "(",sd(auc_logit_reg)/mean(auc_logit_reg), ")")
# cat("Dynamic Logistic regression 1 test AUC has mean (sd):", mean(auc_logit_dy1), "(",sd(auc_logit_dy1), ")")
# cat("Dynamic Logistic regression 2 test AUC has mean (sd):", mean(auc_logit_dy2), "(",sd(auc_logit_dy2), ")")

#Plot ROC 
plot_kfold_roc(fpr_logit, tpr_logit, kfold = kfold_num, type = "Logistic regression")
plot_kfold_roc(fpr_logit_reg, tpr_logit_reg, kfold = kfold_num, type = "Regularized Logistic regression")
# plot_kfold_roc(fpr_logit_dy1, tpr_logit_dy1, kfold = kfold_num, type = "Dynamic Logistic regression 1")
# plot_kfold_roc(fpr_logit_dy2, tpr_logit_dy2, kfold = kfold_num, type = "Dynamic Logistic regression 2")

#Attribution coefficient of variation per channel  
all_attr_logit = attr_logit %>% reduce(left_join, by = "channel_name")%>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_logit_reg = attr_logit_reg %>% reduce(left_join, by = "channel_name")%>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_logit[,c(1,14)]
all_attr_logit_reg[,c(1,14)]

#Attribution average coefficient of variation
cat("Logistic attribution average CV:", mean(all_attr_logit$CV))
cat("Regularized logistic attribution average CV:", mean(all_attr_logit_reg$CV))
