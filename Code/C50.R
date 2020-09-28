# C50
install.packages('C50')
install.packages("caret")
install.packages("pROC")
library(caret)
library(C50)
library(pROC)

loansub <- read.csv("loansub.csv")
data <- loansub[,c("addr_state","grade","emp_length","int_rate","dti",
                   "installment","avg_cur_bal","tot_hi_cred_lim",
                   "bc_open_to_buy","revol_util","annual_inc",
                   "total_bc_limit","bc_util","total_rev_hi_lim",
                   "revol_bal","tot_cur_bal","total_bal_ex_mort","loan_outcome")]
sample_size = floor(0.7*nrow(data))
index <- sample(nrow(data),size=sample_size,replace=F)
train <- data[index,]
test <- data[-index,]

k <- 0
accuracy <- c()
sensitivity <- c()
specificity <- c()
cost <- c(4,8,12,16,20)
for (i in cost){
  k <- k + 1
  error_cost <- matrix(c(0, i, 1, 0), nrow = 2)
  #ctrl <- C5.0Control(winnow=T)
  ctree <- C5.0((subset(train, select = -loan_outcome )),train$loan_outcome,costs=error_cost,trials=4)
  preC50 <- predict(ctree, test)
  confmat <- table(preC50, test$loan_outcome)
  accuracy[k] <- sum(diag(confmat))/sum(confmat)
  sensitivity[k] <- confmat[1,1]/sum(confmat[,1])
  specificity[k] <- confmat[2,2]/sum(confmat[,2])
}

comparison <- data.frame(cost, accuracy, sensitivity, specificity)
head(comparison)

ggplot(gather(comparison, key = 'Metric', value = 'Value', 2:4),
       aes(x = cost, y = Value, color = Metric)) +
  geom_line(size=1.5)

error_cost <- matrix(c(0, 12, 1, 0), nrow = 2)
#ctrl <- C5.0Control(winnow=T)
ctree <- C5.0((subset(train, select = -loan_outcome )),train$loan_outcome,costs=error_cost,trials=4)
print(ctree)
preC50 <- predict(ctree, test)

confusionMatrix(preC50, test$loan_outcome)

roc_log <- roc(test$loan_outcome,as.numeric(preC50),levels = c("Default","NonDefault"))
plot(roc_log, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"),
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,
     xlab= "sensitivity", ylab= "Specificity",main='C5.0 ROC')