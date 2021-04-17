##########################Project 2-ML Code###################################
## Setting WD ####
setwd("~/Desktop/Intoduction to ML and DM (02450)/Project 2/Project 2-ML")
library(keras)
library(formattable)
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
library(cvTools)


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
t1<-data.frame(table(port$Medu))
names(t1)[1]<-"Medu"
t2<-data.frame(table(port$Fedu))
names(t2)[1]<-"Fedu"
grid.arrange(formattable(t1), formattable(t2))


table(port$Dalc)
table(port$Walc)

levels(port$Medu)<-c("None/1st-4th","None/1st-4th","5th-9th","10th-12th","higher")
levels(port$Fedu)<-c("None/1st-4th","None/1st-4th","5th-9th","10th-12th","higher")
levels(port$Dalc)<-c('very low','low','medium','high/very high','high/very high')
levels(port$Walc)<-c('very low','low','medium','high/very high','high/very high')
str(port)


###########Regression part a ##############
port$intercept<-rep(1,nrow(port)) #including intercept term
### Cross-Validation
cv_folds <- rsample::vfold_cv(port,v = 5)

## Recipe 

rec_1<-recipe(G3 ~ ., data = port) %>%
  step_normalize(all_numeric(),- c(all_outcomes(),intercept))%>%
  step_dummy(names(Filter(is.factor, port))) 


baked_data_1<-bake(prep(rec_1),new_data = port)
## Fitting a reg.regression model tuning lambda##

RegReg <- function(rec,cv_folds,x){
  
  glmnet_spec <- parsnip::linear_reg(
    penalty = tune(),
    mixture=0.5
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
    wf,
    resamples = cv_folds,
    grid = lmbda_mixtr_grid,
    metrics = yardstick::metric_set(yardstick::rmse)
  )
  
  plot<-model_tuned %>%
    collect_metrics() %>%
    ggplot(aes(penalty, mean, color = .metric)) +
    geom_point()+
    scale_x_log10() +
    theme(legend.position = "none")
  
  
  lowest_rmse <- model_tuned %>%
    select_best("rmse", maximize = FALSE) #select the parameters for the best model (in our case the penalty term)
  
  final_model <- finalize_workflow(  #Updates the workflow with best fit model
    wf,
    lowest_rmse
  )
  
  RMSE_best<-show_best(model_tuned, "rmse", n = 1) #Cross-Validation RMSE for best tuned model
  
  coef<-final_model %>%    #Extracting non zero parameter estimates for the best fit model fitted to the whole data set
    #pull_workflow_fit returns the model fit
    fit(x) %>%
    pull_workflow_fit() %>%
    tidy()%>% 
    print(n = Inf)%>%filter(estimate!=0)
  
  
  return(list(RMSE_best,final_model,lowest_rmse,plot,coef)) 
}
RegReg(rec=rec_1,cv_folds=cv_folds,x=port)



#### Regression part b ####



##CV algorithm for Neural network algorithm ####
crossvalidation <- function(model,data,size) {
  n <- nrow(data)
  group <- sample(rep(1:5, length.out = n))
  err <- list()
  for(i in 1:5){
    d1 <- data[group != i, ]
    d2 <- data[group == i, ]
    m <- model(d1,size)
    p <- predict(m, d2)
    err[i] <- list((p - d2$y_train)^2)
  }
  mean(unlist(err))
}

##Two-layer Cross-validation####
X<-port[-33]
y<-port$G3
N = nrow(X)
M = ncol(X)
K=5
y_true = matrix(, 0,1)
y_hat = matrix(, 0,3)
r = matrix(, 0,3)
lambda=matrix(, 0,1)
h_unit=matrix(, 0,1)

## set the seed to make your partition reproducible
set.seed(123)

CV <- cvFolds(N, K=K)    
for(k in 1:K){ # For each  outer fold
  print(paste('Crossvalidation fold ', k, '/', CV$NumTestSets, sep=''))
  # Extract training and test set
  X_train <- X[CV$subsets[CV$which!=k], ];
  y_train <- y[CV$subsets[CV$which!=k]];
  X_test <- X[CV$subsets[CV$which==k], ];
  y_test <- y[CV$subsets[CV$which==k]];
  
  Xdatframe_train <- X_train
  Xdatframe_test = X_test
  
  
  ##Reg.Regression and baseline##
  data_train<-data.frame(cbind(y_train,Xdatframe_train))
  
  rec<-recipe(y_train ~ ., data = data_train) %>%
    step_normalize(all_numeric(),- c(all_outcomes(),intercept))%>%   #standardize and dummy encode recipe
    step_dummy(names(Filter(is.factor,data_train)))
  cv_folds_new <- rsample::vfold_cv(data_train, v = 5)
  
  best_fit_A<-RegReg(rec=rec,cv_folds=cv_folds_new,x=data_train) #lin.reg function from part a) using CV (inner fold)
  opt_Lin<-best_fit_A[[2]]$fit$actions$model$spec[[1]]$penalty[[2]] #Optimal lambda
  final_model_A<-glmnet(x=Xdatframe_train,y=y_train,alpha=0.5,standardize=FALSE) 
  
  
  ##neural network ##
  
  rec_1<-recipe(y_train ~ ., data =data.frame(cbind(y_train,Xdatframe_train)) ) %>%
    step_normalize(all_numeric(),- c(all_outcomes(),intercept))%>%   #standardize and dummy encode recipe
    step_dummy(names(Filter(is.factor,data.frame(cbind(y_train,Xdatframe_train)))))
  
  dummy_train<-bake(prep(rec_1),Xdatframe_train)
  
  colnames<-colnames(dummy_train)
  fmla <- as.formula(paste("y_train ~ ", paste(colnames, collapse= "+")))
  n1 <- function(d,sz){nnet(fmla,data=d,size=sz,linout=T)}
  n1error<-rep(0,10)
  for(i in 1:10){
    n1error[i]<-crossvalidation(model=n1, data=cbind(y_train,dummy_train),size=i) #10 fold cross validation error for 10 different values of h (tuning grid)
  }
  error<-data.frame(n1error)
  opt_NN<-which(error==min(error))
  final_model_C<-n1(d=dummy_train,sz=opt_NN)
  
  ##Predictions
  yhat_A<-predict(final_model_A,newx=data.matrix(Xdatframe_test),s=opt_Lin)
  yhat_B<-data.frame(rep(mean(y_train),length(y_test)))
  
  ## Dummy encoding test data just as we did for train data s.t NN model can be used to predict values for this data set
  rec_2<-recipe(y_test ~ ., data =data.frame(cbind(y_test,Xdatframe_test)) ) %>%
    step_normalize(all_numeric(),- c(all_outcomes(),intercept))%>%   #standardize and dummy encode recipe
    step_dummy(names(Filter(is.factor,data.frame(cbind(y_test,Xdatframe_test)))))
  
  dummy_test<-bake(prep(rec_2),Xdatframe_test)
  yhat_C<-predict(final_model_C,data.matrix(dummy_test))
  
  
  
  dyhat = cbind(yhat_A,yhat_B,yhat_C)
  y_hat <- rbind( y_hat, dyhat)
  
  dr_Lin = colMeans( ( yhat_A-y_test )^2)
  dr_Bas=colMeans((yhat_B-y_test)^2)
  dr_NN=colMeans((yhat_C-y_test)^2)
  dr = cbind(dr_Lin,dr_Bas,dr_NN)
  r = rbind(r, dr)
  
  lambda<-rbind(lambda,opt_Lin)
  h_unit<-rbind(h_unit,opt_NN)
  y_true<- rbind( y_true, matrix(y_test))
}


print(list(r,lambda,h_unit))

## Estimating the gen.error of a regression model ##
z<-abs(y_hat-y_true)
colnames(z)<-c("El.net","baseline","NN")
z_Eb<-z$El.net-z$baseline
z_En<-z$El.net-z$NN
z_bn<-z$NN-z$baseline
hist(z_bn,breaks=100)
t.test(z_bn, alternative = "two.sided", alpha=0.05)
