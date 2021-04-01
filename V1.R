##########################Project 2-ML Code###################################
## Setting WD ####
setwd("~/Desktop/Intoduction to ML and DM (02450)/Project 2/Project 2-ML")
## Loading packages ####
library(markdown)
library(tidyverse)
library(caret)
library(dplyr)
library(doFuture)
library(doParallel)
library(earth)
library(gbm)
library(gam)
library(ggplot2)
library(glmnet)
library(grid)
library(gridExtra)
library(hexbin)
library(ipred)
library(labeling)
library(MASS)
library(neuralnet)
library(NeuralNetTools)
library(NeuralNetworkVisualization)
library(nnet)
library(pdp)
library(plotmo)
library(randomForest)
library(ranger)
library(reshape2)
library(rlang)
library(rpart.plot)
library(rsample)
library(shape)
library(splines)
library(xgboost)
library(pROC)
library(caTools)
library(adabag)
library(reshape2)
library(caret)
library(recipes)
library(tidymodels)
library(parsnip)


## Creating data sets and Merging the two datasets ####
math=read.table("Data/math.csv",sep=";",header=TRUE)
port=read.table("Data/port.csv",sep=";",header=TRUE)
port$id<-seq(1:nrow(port))
merged=merge(port,math,by=c("school","sex","age","address","famsize","Pstatus","Medu",
                            "Fedu","Mjob","Fjob","reason","nursery","internet","guardian"
                            ,"traveltime","studytime","activities","higher","romantic"
                            ,"goout","Dalc","Walc","famrel","freetime","famsup","schoolsup","health"),all.x = TRUE)
#merge by the personal attributes (i.e NOT the course specific attributes)

sum(is.na(merged$G3.y))

##adding indicator variable to portugese data set (1 if enrolled in math else 0)
math_ind<-rep(0,nrow(merged))
for(i in 1:nrow(merged)){
  ifelse(is.na(merged$failures.y[i]),math_ind[i]<-0,math_ind[i]<-1)
}
merged$math_ind=math_ind
merged<-merged[order(merged$id),]
port<-merged[,-c(34:40)]
names(port)[28:33]<-c("failures","paid","absences","G1","G2","G3")
sum(port$math_ind) #370
port$math_ind<-as.factor(port$math_ind)
## Structure of the variables####
str(port)
################ Variable transformation ###############################
## Changing integers to factors and ordering factors####
port<-transform(port,
                studytime=factor(studytime,labels=c('<2 h','2-5 h','5-10 h','>10 h'),ordered=F),
                traveltime=factor(traveltime,labels=c('<15 min','15-30 min','30-60 min','>60 min'),ordered=F),
                Fedu=factor(Fedu,labels=c('none','1st-4th grade','5th-9th grade','10th-12th grade','higher'),ordered=F),
                Medu=factor(Medu,labels=c('none','1st-4th grade','5th-9th grade','10th-12th grade','higher'),ordered=F),
                freetime=factor(freetime,labels=c('very low','low','medium','high','very high'),ordered=F),
                goout=factor(goout,labels=c('very low','low','medium','high','very high'),ordered=F),
                Dalc=factor(Dalc,labels=c('very low','low','medium','high','very high'),ordered=F),
                Walc=factor(Walc,labels=c('very low','low','medium','high','very high'),ordered=F),
                health=factor(health,labels=c('very bad','bad','medium','good','very good'),ordered=F),
                famrel=factor(famrel,labels=c('very bad','bad','medium','good','very good'),ordered=F)
)
levels(port$famsize)<-c("LE3","GT3")
port$famsize<-factor(port$famsize,ordered=F)

str(port)

## Combining levels for sparse levels##
table(port$Medu)
table(port$Fedu)
table(port$Dalc)
table(port$Walc)

levels(port$Medu)<-c("None/1st-4th","None/1st-4th","5th-9th","10th-12th","higher")
levels(port$Fedu)<-c("None/1st-4th","None/1st-4th","5th-9th","10th-12th","higher")
levels(port$Dalc)<-c('very low','low','medium','high/very high','high/very high')
levels(port$Walc)<-c('very low','low','medium','high/very high','high/very high')
str(port)


###########Regression part a ##############
options(dplyr.print_max = 1e9)
### Cross-Validation
train<-port
cv_folds <- rsample::vfold_cv(train, v = 5)
cv_folds_twolayer <- rsample::nested_cv(train,
                                       outside=vfold_cv(),
                                       inside=vfold_cv())  
## Recipe 
rec_1 <- recipe(G3 ~ ., data = train) %>%
  step_normalize(all_numeric(),-all_outcomes())

rec_2<-recipe(G3 ~ ., data = train) %>%
  step_normalize(all_numeric(),-all_outcomes())%>%
  step_dummy(names(Filter(is.factor, port))) #uses polynomial contrasts for ordered factors
  
baked_data_1<-bake(prep(rec_1),new_data = train)
baked_data_2<-bake(prep(rec_2),new_data = train)
## Fitting a reg.regression model tuning lambda##

RegReg <- function(rec, cv_folds){
  
  glmnet_spec <- parsnip::linear_reg(
    penalty = tune(),
    mixture = 0.5
  ) %>%
    set_engine("glmnet")
  
  lmbda_mixtr_grid <- grid_regular(
    penalty(c(-3,0)), 
    levels = 50
  )
  
  wf <- workflow() %>%
    add_recipe(rec) %>% 
    add_model(glmnet_spec)
 
  doParallel::registerDoParallel() 
  model_tuned <- tune::tune_grid(
    wf %>% update_model(glmnet_spec),
    resamples = cv_folds,
    grid = lmbda_mixtr_grid,
    metrics = yardstick::metric_set(rmse)
  )
  
  plot<-model_tuned %>%
    collect_metrics() %>%
    ggplot(aes(penalty, mean, color = .metric)) +
    geom_point()+
    facet_wrap(~.metric, scales = "free", nrow = 2) +
    scale_x_log10() +
    theme(legend.position = "none")
  
  
  lowest_rmse <- model_tuned %>%
    select_best("rmse", maximize = FALSE) #select the model with the lowest RMSE
  
  final_model <- finalize_workflow(  #Define the best fit model
    wf %>% update_model(glmnet_spec),
    lowest_rmse
  )
  
  RMSE_best<-show_best(model_tuned, "rmse", n = 1) #Cross-Validation RMSE for best tuned model

  coef<-final_model %>%
    fit(train) %>%
    pull_workflow_fit() %>%
    tidy()%>% 
    print(n = Inf)


  return(list(RMSE_best,final_model,plot,coef)) 
}
RegReg(rec_2,cv_folds=cv_folds)
