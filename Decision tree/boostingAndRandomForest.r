

setwd("/home/mazen/FMODHB3/")
setwd("C:/Users/Mazen/Desktop/MEMOT/FMODHB3/")
set.seed(70)

# Reading the data
Data <- read.csv("6867Data.csv")
hours <- rep(c(0:23), 365)
Data <- cbind(Data, hours)

colnames(Data) <- c("Time", "Load", "DryTemp", "DewTemp", "Humidity", "Pressure", "Illuminance", "Wind", "Hour")
obs <- c(1:nrow(Data))

# Generating day variables
Day <- (ceiling(obs/24) %% 7)
Fri <- 1*(Day==1)
Sat <- 1*(Day==2)
Sun <- 1*(Day==3)
Mon <- 1*(Day==4)
Tue <- 1*(Day==5)
Wed <- 1*(Day==6)
Thurs <- 1*(Day==0)

Data <- cbind(Data, Fri, Sat, Sun, Mon, Tue, Wed, Thurs)

# Generating the first lag of the load
t1 <- append(NA,Data$Load[1:(length(Data$Load)-1)])
Data <- cbind(Data, t1)
Data <- Data[complete.cases(Data),	]
obs <- 1:nrow(Data)

# Making sure that the plotted days are always in the test data (Feb 1st and 2nd and Aug 1st and 2nd)
alwaystest <- obs %in% append(c(744:791),c(5088:5135))*1
DataTr <- Data[alwaystest==0,]
DataTs <- Data[alwaystest==1,]

BoostY<- NULL
RFY <- NULL
MAPERF <- NULL
MAPEBOOST <- NULL

# Looping over 10 iterations of boosting and random forests
for(k in 1:10){
# Splitting the data into training (60%) and test (40%)
trainobs <- sample(nrow(DataTr), 5255)
TrainingData <-  DataTr[trainobs,]
TestData <- DataTr[-trainobs,]
TestData <- rbind(TestData, DataTs)

options(warn=-1,message=-1)	
#Using the gbm library for boosting
library(gbm)	
Y <- TrainingData$Load
X <- TrainingData[,3:16]

# Optimizing the parameters (minimum number of obs in a leaf by cross validation)
u <- NULL
for(i in c(5,10,15,20,25)){
Boost=gbm(Y~.,data=as.data.frame(X),distribution= "gaussian",n.trees=500, interaction.depth=5, cv.folds=5, n.minobsinnode=i)
u <- append(u, mean(Boost$cv.error))}	

# Choosing the parameter with the least MSE
i=c(5,10,15,20,25)[which.min(u)]

#Using library randomForest for RF
library(randomForest)	

# Using the same procedure for random forests
v <- NULL
for(i in c(5,10,15,20,25)){
RandomForestFit  =  randomForest(X,Y, ntree=500, nodesize=i, cv.fold=5) 	
v <- append(v, mean(RandomForestFit$mse))}	
j=c(5,10,15,20,25)[which.min(v)]

# Fitting the final models
Boost=gbm(Y~.,data=as.data.frame(X),distribution= "gaussian",n.trees=5000, interaction.depth=5,n.minobsinnode=i)	
RandomForestFit  =  randomForest(X,Y, ntree=5000, nodesize=j) 	

TestY <- TestData$Load
TestX <- TestData[,3:16]

# Calculating MAPE on the test data
yhat.boost.test =predict(Boost, TestX,n.trees=5000)	
BoostMAPE <- mean(abs((TestY - yhat.boost.test)/TestY))*100
MAPEBOOST <- append(MAPEBOOST, BoostMAPE)

yhat.RF.test <- predict(RandomForestFit, newdata=TestX)
RFMAPE <- mean(abs((TestY - yhat.RF.test)/TestY))*100
MAPERF <- append(MAPERF, RFMAPE)

BoostY <- cbind(BoostY, yhat.boost.test)
RFY <- cbind(RFY, yhat.RF.test)
print(k)}