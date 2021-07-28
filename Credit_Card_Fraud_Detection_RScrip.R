set.seed(1, sample.kind="Rounding")
if(!require(tidyverse))  install.packages("tidyverse")
if(!require(randomForest))  install.packages("randomForest")
if(!require(imbalance))  install.packages("imbalance")
if(!require(caret))  install.packages("caret")
if(!require(e1071))  install.packages("e1071")
if(!require(Metrics))  install.packages("Metrics")


library(tidyverse)
library(randomForest)
library(imbalance)
library(caret)
library(e1071)
library(Metrics)

gc()
credit_card_data <- read.table(file = "creditcard.csv", sep = ",", header=TRUE)
glimpse(credit_card_data)

n_fraud_not_fraud <- credit_card_data %>% count(Class) 
print(n_fraud_not_fraud)

credit_card_data_noTime <- subset (credit_card_data, select = -Time)
new_fraud_data <- pdfos(credit_card_data_noTime, numInstances = 
                          n_fraud_not_fraud$n[1] - n_fraud_not_fraud$n[2])
new_credit_card_data <- rbind(credit_card_data_noTime, new_fraud_data)
new_n_fraud_not_fraud <- new_credit_card_data %>% count(Class) 
rm(credit_card_data, credit_card_data_noTime, new_fraud_data)
glimpse(new_credit_card_data)

new_credit_card_data$Class <- factor(new_credit_card_data$Class)

test_index <- createDataPartition(y = new_credit_card_data$Class, 
                                  times = 1, p = 0.1, list = FALSE)
training_data <- new_credit_card_data[-test_index,]
test_data <- new_credit_card_data[test_index,]
rm(new_credit_card_data)

if(!require(rpart))  install.packages("rpart")
library(rpart)

fit <- rpart(Class~., data = training_data, method = 'class')

if(!require(rattle))  install.packages("rattle")
if(!require(rpart.plot))  install.packages("rpart.plot")
if(!require(RColorBrewer))  install.packages("RColorBrewer")

library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(fit, caption = NULL)

pred_DT <- predict(object= fit, newdata = test_data, type = 'class')
conf_DT <- confusionMatrix(pred_DT,test_data[['Class']])
overall_accuracy <- conf_DT$overall[['Accuracy']]
sensitivity <- conf_DT$byClass[['Sensitivity']]
specificity <- conf_DT$byClass[['Specificity']]

print(conf_DT)
print(paste('Overall accuracy: ',overall_accuracy))
print(paste('Sensitivity: ',sensitivity))
print(paste('Specificity: ',specificity))

mtry_i <- c(1:6,rep(3,4))
ntree_i <- c(rep(50,6),seq(10,70,by = 20))
sensitivity_i <- numeric(10)
specificity_i <- numeric(10)
overall_accuracy <- numeric(10)
for (i in 1:6){
  gc()
  rf_classifier <- randomForest(formula = Class ~ ., data = 
                                  training_data,ntree=ntree_i[i],
                                mtry= mtry_i[i], importance = TRUE)
  pred <- predict(rf_classifier, test_data)
  conf <- confusionMatrix(pred,test_data[['Class']])
  overall_accuracy[i] <- conf$overall[['Accuracy']]
  sensitivity_i[i] <- conf$byClass[['Sensitivity']]
  specificity_i[i] <- conf$byClass[['Specificity']]
  number_of_trees <- 1:ntree_i[i]
}

tab_1 <- data.frame(mtry = mtry_i[1:6],ntree= rep(50,6),
                    overall_accuracy = overall_accuracy[1:6],
                    Sensitivity = sensitivity_i[1:6]
                    , Specificity = specificity_i[1:6])
print(tab_1)

gc()
for (i in 7:10){
  gc()
  rf_classifier <- randomForest(formula = Class ~ ., data =
                                  training_data,ntree=ntree_i[i],
                                mtry= mtry_i[i], importance = TRUE)
  pred <- predict(rf_classifier, test_data)
  conf <- confusionMatrix(pred,test_data[['Class']])
  overall_accuracy[i] <- conf$overall[['Accuracy']]
  sensitivity_i[i] <- conf$byClass[['Sensitivity']]
  specificity_i[i] <- conf$byClass[['Specificity']]
  number_of_trees <- 1:ntree_i[i]
}

tab_2 <- data.frame(mtry = mtry_i,ntree= ntree_i,
                    overall_accuracy = overall_accuracy,
                    Sensitivity = sensitivity_i, Specificity = specificity_i)

print(tab_2)
gc()