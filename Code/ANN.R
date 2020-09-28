install.packages("neuralnet")
install.packages("caret")
install.packages("sampling")
install.packages("pROC")
library(neuralnet)
library(caret)
library(sampling)
library(pROC)

loan <- read.csv("loansub.csv")
loansub <- loan[,c("addr_state","grade","emp_length","int_rate","dti",
                   "installment","avg_cur_bal","tot_hi_cred_lim",
                   "bc_open_to_buy","revol_util","annual_inc",
                   "total_bc_limit","bc_util","total_rev_hi_lim",
                   "revol_bal","tot_cur_bal","total_bal_ex_mort","loan_outcome")]
str(loansub) 
# From here, you can see that the first three variables are categorical.
# One hot encoding
addr_state <- as.data.frame(model.matrix(~addr_state-1,loansub))
grade <- as.data.frame(model.matrix(~grade-1,loansub))
emp_length <- as.data.frame(model.matrix(~emp_length-1,loansub))
# Normalization
maxs <- apply(loansub[,4:17], 2, max)
mins <- apply(loansub[,4:17], 2, min)
scaled.data <- as.data.frame(scale(loansub[,4:17], center=mins,
                                   scale = maxs - mins))
# 0/1 the outcome: 0-Default, 1-Nondefault
outcome <- as.numeric(loansub$loan_outcome) - 1

data <- cbind(addr_state,grade,emp_length,scaled.data,outcome)

sample_size = floor(0.7*nrow(data))
index <- sample(nrow(data),size=sample_size,replace=F)
train <- data[index,]
test <- data[-index,]

baseline <- nrow(loansub[loansub$loan_outcome == "NonDefault",])/nrow(loansub)
#index2 <- sample(nrow(train),size=10000,replace=F)
#train <- train[index2,]
train <- strata(train,stratanames = "outcome",
                size=c(floor(5000*baseline),5000-floor(5000*baseline)),
                       method = "srswor", description = FALSE)
train<- data.frame(getdata(data,train))[,1:85]

features <- names(scaled.data)
f <- paste(features, collapse = ' + ')
f <- paste('outcome ~', f)
f <- as.formula(f)

nn <- neuralnet(f, train, hidden = 8, linear.output = FALSE)
predicted <- compute(nn, test[1:84])
ggplot(data.frame(predicted$net.result), aes(predicted$net.result)) +
  geom_density(fill = 'lightblue', alpha = 0.4) +
  labs(x = 'Predicted Probabilities on test dataset')

k <- 0
accuracy <- c()
sensitivity <- c()
specificity <- c()
threshold <- seq(from = 0.51, to = 0.98, by = 0.01)
for(i in threshold){
  k <- k+1
  preds_binominal <- ifelse(predicted$net.result > i, 1, 0)
  confmat <- table(preds_binominal, test$outcome)
  accuracy[k] <- sum(diag(confmat))/sum(confmat)
  sensitivity[k] <- confmat[1,1]/sum(confmat[,1])
  specificity[k] <- confmat[2,2]/sum(confmat[,2])
}
comparison <- data.frame(threshold, accuracy, sensitivity, specificity)
head(comparison)

ggplot(gather(comparison, key = 'Metric', value = 'Value', 2:4),
       aes(x = threshold, y = Value, color = Metric)) +
  geom_line(size=1.5)

predicted$net.result <- ifelse(predicted$net.result > 0.8, 1, 0)
evaluation <- data.frame(test$outcome, predicted$net.result)
colnames(evaluation) <- c("actual", "predict")
table (evaluation$predict, evaluation$actual)
evaluation$correct <- ifelse(evaluation$actual == evaluation$predict, 1, 0)
sum(evaluation$correct)/nrow(evaluation)
evaluation$actual <- as.factor(evaluation$actual)
evaluation$predict <- as.factor(evaluation$predict)
confusionMatrix(evaluation$predict,evaluation$actual)

roc_log <- roc(test$outcome,as.numeric(evaluation$predict),levels = c("0","1"))
plot(roc_log, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"),
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,
     xlab= "sensitivity", ylab= "Specificity",main='ANN ROC')

plot(nn,intercept = F)

