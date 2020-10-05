#Author: Linh Dinh 

########################################INSTALLING AND LOADING PACKAGES########################################

# install.packages("caret")
# install.packages("dplyr")
# install.packages("lubridate")
# install.packages("tidyverse")
# install.packages("pROC")
# install.packages("reshape2")
# install.packages("ChannelAttribution")
# install.packages("markovchain")

library(caret)
library(dplyr)
library(lubridate)
library(tidyverse)
library(pROC)
library(reshape2)
library(ChannelAttribution)
library(markovchain)

########################################LOADING NECESSARY SCRIPT(S)########################################
getwd()
source('Channel Attribution - Simulating data.R', echo=TRUE)
set.seed(123)

########################################MARKOV CHAIN MODEL########################################
##### Attribution ##### 
##### Attribution functions
### Attribution function using two methods: matrix multiplication (return_more == TRUE) and simulation (return_more == FALSE)
markov_attribution = function(data = df_path_TOTAL, journey_var = 'journey', 
                              convert_var = 'conv', noconvert_var = 'conv_null', 
                              return_more = TRUE, markov_order = 1) {
  ###Fitting the markov model 
  markov = markov_model(data, var_path = journey_var, var_conv = convert_var, var_null = noconvert_var, 
                        out_more = TRUE, order = markov_order)
  
  ###Get the transition matrix of the model
  #Transition probability
  df_trans <- markov$transition_matrix
  
  #Add extra nodes 
  df_extra <- data.frame(channel_from = c('(start)', '(conversion)', '(null)'),
                         channel_to = c('(start)', '(conversion)', '(null)'),
                         transition_probability = c(0, 1, 1))
  df_trans_matrix <- rbind(df_trans, df_extra)
  
  #Order channels
  df_trans_matrix$channel_from <- factor(df_trans_matrix$channel_from,
                                         levels = unique(c(as.character(df_trans_matrix$channel_from), as.character(df_trans_matrix$channel_to))))
  df_trans_matrix$channel_to <- factor(df_trans_matrix$channel_to,
                                       levels =unique(c(as.character(df_trans_matrix$channel_from), as.character(df_trans_matrix$channel_to))))
  df_trans_matrix <- dcast(df_trans_matrix, channel_from ~ channel_to, value.var = 'transition_probability')
  
  #Creat the markovchain object
  trans_matrix <- matrix(data = as.matrix(df_trans_matrix[, -1]),
                         nrow = nrow(df_trans_matrix[, -1]), ncol = ncol(df_trans_matrix[, -1]),
                         dimnames = list(c(as.character(df_trans_matrix[, 1])), c(colnames(df_trans_matrix[, -1]))))
  trans_matrix[is.na(trans_matrix)] <- 0
  len_trans_matrix = nrow(trans_matrix)
  trans_matrix_final <- new("markovchain", transitionMatrix = trans_matrix)
  
  #Plot the graph
  # plot(trans_matrix_final, edge.arrow.size = 0.35)
  
  ###Get the attribution result of the model
  #by matrix multiplication 
  if(return_more == TRUE)
  {
    absorb_states_og = c(len_trans_matrix-1,len_trans_matrix) #Absorbing states are "(conversion)" and "(null)", which are at the last two rows of trans_matrix
    r = length(absorb_states_og) # number of absorbing states
    t = len_trans_matrix - r     # number of transient states 
    Q = trans_matrix_final[-absorb_states_og,-absorb_states_og] 
    I = diag(1,t,t)
    N = solve(I - Q)
    R = trans_matrix_final[-absorb_states_og,absorb_states_og]
    B = N %*% R
    prob_convert_og = B['(start)','(conversion)'] 
    
    all_states = trans_matrix_final@states[-c(1,len_trans_matrix-1,len_trans_matrix)]
    RE = data.frame(channel_name = all_channels, Prob_without = rep(0,length(all_channels)), Removal_effect = rep(0,length(all_channels)))
    for (i in 1:length(all_channels)){
      channel_of_interest = as.character(all_channels[i])
      trans_matrix_new = trans_matrix_final@transitionMatrix
      removed_states = c()
      for (j in 1:length(all_states)) {
        count_same_channel = str_count(all_states[j],channel_of_interest)
        if(count_same_channel > 0) {
          trans_matrix_new[,len_trans_matrix] = trans_matrix_new[,len_trans_matrix] + trans_matrix_new[,j+1]
          trans_matrix_new[j+1,] = 0
          trans_matrix_new[,j+1] = 0
          #trans_matrix_new[j+1,j+1] = 1
          removed_states = c(removed_states,j+1)
        }
      } 
      absorb_states = c(len_trans_matrix-1,len_trans_matrix)
      t_new = len_trans_matrix - length(absorb_states) - length(removed_states) # number of transient states 
      Q_new = trans_matrix_new[-c(absorb_states,removed_states),-c(absorb_states,removed_states)]  # t_new x t_new matrix of prob from transient to transient state
      I_new = diag(1,t_new, t_new)                                  # t_new x t_new identity matrix
      N_new = solve(I_new - Q_new)                                  # t_new x t_new fundamental matrix 
      R_new = trans_matrix_new[-c(absorb_states,removed_states),absorb_states]   # t_new x r_new matrix of prob from transient to absorbing state
      B_new = N_new %*% R_new                                   # t_new x r_new matrix of prob from starting at transient to ending at absorbing
      RE[i,'Prob_without'] = B_new['(start)','(conversion)']
      RE[i,'Removal_effect'] = 1 - (RE[i,'Prob_without']/prob_convert_og) 
    }
    df_markov_attribution = RE %>%
      mutate(Markov_attribution = round(Removal_effect/sum(Removal_effect),6)) %>% 
      dplyr::select(channel_name, Markov_attribution)
  }
  
  #by simulation  
  else 
  {
    df_markov_attribution = markov$removal_effect %>% 
      mutate(Markov_attribution = round(removal_effects/sum(removal_effects),6)) %>% 
      dplyr::select(channel_name, Markov_attribution)
  }
  return(list(Attribution = df_markov_attribution, 
              Transition_Matrix = trans_matrix_final, Transition_Probability = df_trans))
}

### Function to rename attribution result according to order number
rename_markov = function(result_markov, order) { 
  colnames(result_markov)[2] = paste("Markov_order_",order, sep = "") 
  return(result_markov)
}

##### Attribution Results for all markov models
### Order 1
attr_markov1 = markov_attribution(data = df_path_TOTAL, return_more = TRUE, markov_order = 1)
trans_mat1 = attr_markov1$Transition_Matrix@transitionMatrix
trans_prob1 = attr_markov1$Transition_Probability
res_markov1 = attr_markov1$Attribution
result_markov1 = rename_markov(res_markov1, order = 1)

#Plot the transition graph
plot(attr_markov1$Transition_Matrix, edge.arrow.size = 0.35)
cols <- c("#e7f0fa", "#c9e2f6", "#95cbee", "#0099dc", "#4ab04a", "#ffd73e", "#eec73a",
          "#e29421", "#e29421", "#f05336", "#ce472e")
t <- max(trans_prob1$transition_probability)

ggplot(trans_prob1, aes(y = channel_from, x = channel_to, fill = transition_probability)) +
  theme_minimal() +
  geom_tile(colour = "white", width = .9, height = .9) +
  scale_fill_gradientn(colours = cols, limits = c(0, t),
                       breaks = seq(0, t, by = t/4),
                       labels = c("0", round(t/4*1, 2), round(t/4*2, 2), round(t/4*3, 2), round(t/4*4, 2)),
                       guide = guide_colourbar(ticks = T, nbin = 50, barheight = .5, label = T, barwidth = 10)) +
  geom_text(aes(label = round(transition_probability, 2)), fontface = "bold", size = 4) +
  theme(legend.position = 'bottom',
        legend.direction = "horizontal",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(size = 20, face = "bold", vjust = 2, color = 'black', lineheight = 0.8),
        axis.title.x = element_text(size = 24, face = "bold"),
        axis.title.y = element_text(size = 24, face = "bold"),
        axis.text.y = element_text(size = 8, face = "bold", color = 'black'),
        axis.text.x = element_text(size = 8, angle = 90, hjust = 0.5, vjust = 0.5, face = "plain")) +
  ggtitle("Transition matrix heatmap")

### Order 2
attr_markov2 = markov_attribution(data = df_path_TOTAL, return_more = TRUE, markov_order = 2)
trans_mat2 = attr_markov2$Transition_Matrix@transitionMatrix
res_markov2 = attr_markov2$Attribution
result_markov2 = rename_markov(res_markov2, order = 2)

### Order 3
attr_markov3 = markov_attribution(data = df_path_TOTAL, return_more = TRUE, markov_order = 3)
trans_mat3 = attr_markov3$Transition_Matrix@transitionMatrix
res_markov3 = attr_markov3$Attribution
result_markov3 = rename_markov(res_markov3, order = 3)

### Order 4
attr_markov4 = markov_attribution(data = df_path_TOTAL, return_more = TRUE, markov_order = 4)
trans_mat4 = attr_markov4$Transition_Matrix@transitionMatrix
res_markov4 = attr_markov4$Attribution
result_markov4 = rename_markov(res_markov4, order = 4)

### All the models
df_markov_all = Reduce(function(x, y) merge(x = x, y = y, by.x = "channel_name", by.y = "channel_name", all = TRUE), 
                       list(result_markov1, result_markov2, result_markov3, result_markov4)) 

##### Predictive Power #####
##### Predict functions
### Function to get the last k-channels (choosing k depends on the order number)
get_last_k_channels = function(journey, markov_order = 1) {
  all_C = unlist(strsplit(journey, " > "))
  last_k_channels = paste(tail(all_C, markov_order), collapse = ",")
  return(last_k_channels)
}

### Function to get the predicted probability of the last k-channels
predict_Mprob = function (last_k_channels, trans_matrix) {
  trans = trans_matrix[[1]]
  pred_prob = ifelse(last_k_channels %in% colnames(trans), trans[last_k_channels,"(conversion)"], 0)
  return(pred_prob)
}

### Function to get the test predicted results depend on the train trainsition matrix and the order number
predict_markov = function(data_train, data_test, transit_matrix, markov_order = 1) {
  data_train = data_train %>% dplyr::select(-conv, -conv_null)
  data_train$trans_matrix = list(transit_matrix) 
  data_fit = data_train %>% 
    group_by(fullVisitorId, path_no) %>% 
    mutate(last_k_channels = get_last_k_channels(journey, min(markov_order,journey_len)), 
           predict_prob = predict_Mprob(last_k_channels, trans_matrix)) %>% 
    ungroup() %>%
    dplyr::select(last_k_channels, predict_prob)
  data_fit = unique(data_fit)
  data_test = data_test %>%
    group_by(fullVisitorId, path_no) %>% 
    mutate(last_k_channels = get_last_k_channels(journey,min(markov_order,journey_len))) 
  data_predict = left_join(data_test, data_fit, by = c("last_k_channels" = "last_k_channels")) 
  return(data_predict)
} 

##### Get the prediction accuracy and robustness for all the markov models 
### Initializing
auc_markov1 = rep(0,kfold_num); attr_markov1 = list(list()); fpr_markov1 = list(list()); tpr_markov1 = list(list())
auc_markov2 = rep(0,kfold_num); attr_markov2 = list(list()); fpr_markov2 = list(list()); tpr_markov2 = list(list())
auc_markov3 = rep(0, kfold_num); attr_markov3 = list(list()); fpr_markov3 = list(list()); tpr_markov3 = list(list())
auc_markov4 = rep(0, kfold_num); attr_markov4 = list(list()); fpr_markov4 = list(list()); tpr_markov4 = list(list())

### Running k-fold CV
for (i in 1:kfold_num){
  ###Getting train and test sets
  train_set = df_path_fold %>% filter(fold != i) 
  test_set = df_path_fold %>% filter(fold == i)
  # test_set = train_set
  
  ###Order 1
  #Fitting
  markov1 = markov_attribution(data = train_set, return_more = TRUE, markov_order = 1)
  t_mat1 = markov1$Transition_Matrix@transitionMatrix
  attr_markov1[[i]] = markov1$Attribution
  #Predicting
  Markov_predict1 = predict_markov(train_set,test_set, transit_matrix = t_mat1, markov_order = 1)
  roc_markov1 = roc(is_conv~predict_prob, data = Markov_predict1)
  auc_markov1[i] = roc_markov1$auc
  fpr_markov1[[i]] = 1 - roc_markov1$specificities
  tpr_markov1[[i]] = roc_markov1$sensitivities
  
  ###Order 2
  #Fitting
  markov2 = markov_attribution(data = train_set, return_more = TRUE, markov_order = 2)
  t_mat2 = markov2$Transition_Matrix@transitionMatrix
  attr_markov2[[i]] = markov2$Attribution
  #Predicting
  Markov_predict2 = predict_markov(train_set,test_set, transit_matrix = t_mat2, markov_order = 2)
  roc_markov2 = roc(is_conv~predict_prob, data = Markov_predict2)
  auc_markov2[i] = roc_markov2$auc
  fpr_markov2[[i]] = 1 - roc_markov2$specificities
  tpr_markov2[[i]] = roc_markov2$sensitivities
  
  ###Order 3
  #Fitting
  markov3 = markov_attribution(data = train_set, return_more = TRUE, markov_order = 3)
  t_mat3 = markov3$Transition_Matrix@transitionMatrix
  attr_markov3[[i]] = markov3$Attribution
  #Predicting
  Markov_predict3 = predict_markov(train_set,test_set, transit_matrix = t_mat3, markov_order = 3)
  roc_markov3 = roc(is_conv~predict_prob, data = Markov_predict3)
  auc_markov3[i] = roc_markov3$auc
  fpr_markov3[[i]] = 1 - roc_markov3$specificities
  tpr_markov3[[i]] = roc_markov3$sensitivities
  
  ###Order 4
  #Fitting
  markov4 = markov_attribution(data = train_set, return_more = TRUE, markov_order = 4)
  t_mat4 = markov4$Transition_Matrix@transitionMatrix
  attr_markov4[[i]] = markov4$Attribution
  #Predicting
  Markov_predict4 = predict_markov(train_set,test_set, transit_matrix = t_mat4, markov_order = 4)
  roc_markov4 = roc(is_conv~predict_prob, data = Markov_predict4)
  auc_markov4[i] = roc_markov4$auc
  fpr_markov4[[i]] = 1 - roc_markov4$specificities
  tpr_markov4[[i]] = roc_markov4$sensitivities
}

#Test AUC 
cat("Markov (order 1) test AUC has mean (coefficient of variation):", mean(auc_markov1), "(",sd(auc_markov1)/mean(auc_markov1), ")")
cat("Markov (order 2) test AUC has mean (coefficient of variation):", mean(auc_markov2), "(",sd(auc_markov2)/mean(auc_markov2), ")")
cat("Markov (order 3) test AUC has mean (coefficient of variation):", mean(auc_markov3), "(",sd(auc_markov3)/mean(auc_markov3), ")")
cat("Markov (order 4) test AUC has mean (coefficient of variation):", mean(auc_markov4), "(",sd(auc_markov4)/mean(auc_markov4), ")")

#Plot ROC 
plot_kfold_roc(fpr_markov1, tpr_markov1, kfold = kfold_num, type = "Markov model (first order)")
plot_kfold_roc(fpr_markov2, tpr_markov2, kfold = kfold_num, type = "Markov model (second order)")
plot_kfold_roc(fpr_markov3, tpr_markov3, kfold = kfold_num, type = "Markov model (third order)")
plot_kfold_roc(fpr_markov4, tpr_markov4, kfold = kfold_num, type = "Markov model (fourth order)")

#Attribution coefficient of variation per channel  
all_attr_markov1 = attr_markov1 %>% reduce(left_join, by = "channel_name") %>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_markov2 = attr_markov2 %>% reduce(left_join, by = "channel_name") %>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_markov3 = attr_markov3 %>% reduce(left_join, by = "channel_name") %>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_markov4 = attr_markov4 %>% reduce(left_join, by = "channel_name") %>% 
  mutate(average = apply(.[-1],1,mean), SD = apply(.[-1],1,sd)) %>% mutate(CV = SD/average)
all_attr_markov1[,c(1,14)]
all_attr_markov2[,c(1,14)]
all_attr_markov3[,c(1,14)]
all_attr_markov4[,c(1,14)]

#Attribution average coefficient of variation
cat("Markov model order 1 attribution average CV:", mean(all_attr_markov1$CV))
cat("Markov model order 2 attribution average CV:", mean(all_attr_markov2$CV))
cat("Markov model order 3 attribution average CV:", mean(all_attr_markov3$CV))
cat("Markov model order 4 attribution average CV:", mean(all_attr_markov4$CV))
