install.packages("tidyr")
install.packages("caret")
install.packages("pROC")
library(tidyr)
library(caret)
library(pROC)

loan <- read.csv("loansub.csv")
data <- loan[,c("addr_state","grade","emp_length","int_rate","dti",
                   "installment","avg_cur_bal","tot_hi_cred_lim",
                   "bc_open_to_buy","revol_util","annual_inc",
                   "total_bc_limit","bc_util","total_rev_hi_lim",
                   "revol_bal","tot_cur_bal","total_bal_ex_mort","loan_outcome")]

data$loan_outcome <- as.numeric(data$loan_outcome) - 1

sample_size = floor(0.7*nrow(data))
index <- sample(nrow(data),size=sample_size,replace=F)
train <- data[index,]
test <- data[-index,]

fit <- glm(loan_outcome~., train, family = binomial(link = 'logit'))
summary(fit)

preds <- predict(fit, test, type = 'response')
ggplot(data.frame(preds), aes(preds)) +
  geom_density(fill = 'lightblue', alpha = 0.4) +
  labs(x = 'Predicted Probabilities on test dataset')

k <- 0
accuracy <- c()
sensitivity <- c()
specificity <- c()
threshold <- seq(from = 0.51, to = 0.98, by = 0.01)
for(i in threshold){
  k <- k+1
  preds_binominal <- ifelse(preds > i, 1, 0)
  confmat <- table(preds_binominal, test$loan_outcome)
  accuracy[k] <- sum(diag(confmat))/sum(confmat)
  sensitivity[k] <- confmat[1,1]/sum(confmat[,1])
  specificity[k] <- confmat[2,2]/sum(confmat[,2])
}
comparison <- data.frame(threshold, accuracy, sensitivity, specificity)
head(comparison)

ggplot(gather(comparison, key = 'Metric', value = 'Value', 2:4),
       aes(x = threshold, y = Value, color = Metric)) +
  geom_line(size=1.5)

preds_by80 <- ifelse(preds > 0.8, 1, 0)
evaluation <- data.frame(test$loan_outcome,preds_by80)
colnames(evaluation) <- c("actual", "predict")
evaluation$actual <- as.factor(evaluation$actual)
evaluation$predict <- as.factor(evaluation$predict)
confusionMatrix(evaluation$predict,evaluation$actual)

roc_log <- roc(evaluation$actual,as.numeric(evaluation$predict),levels = c("0","1"))
plot(roc_log, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"),
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,
     xlab= "sensitivity", ylab= "Specificity",main='LR ROC')