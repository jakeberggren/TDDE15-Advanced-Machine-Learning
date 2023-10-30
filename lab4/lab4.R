# Lab 4 course TDDE15, Advanced Machine Learning, Linkoping University, Sweden


########### Libraries #############
library(kernlab)
library(AtmRay)
########### Libraries #############

############ Functions ############

# Covariance function
SquaredExpKernel <- function(x1, x2, sigmaF = 1, l = 3) {
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA, n1, n2)
  for (i in 1:n2) {
    K[, i] <- sigmaF ^ 2 * exp(-0.5 * ((x1 - x2[i]) / l) ^ 2)
  }
  return(K)
}
############ Functions ############

###################################

# Q1. Implementing GP Regression.

# Q1.1: Write your own code for simulating from the posterior distribution of
# f using the squared exponential kernel. The function should return a vector
# with the posterior mean and variance of f, both evaluated at a set of
# x-values (X*). You can assume that the prior mean of f is zero for all x.

posteriorGP <- function(X, y, XStar, sigmaNoise, k, ...) {
  
  # Covariance matrices:
  K <- k(X, X, ...)
  KStar <- k(X, XStar, ...)
  
  L <- t(chol(K + (sigmaNoise ** 2) * diag(length(X)))) # Cholesky decomposition
  
  alpha <- solve(t(L), solve(L, y))
  fstar <- t(KStar) %*% alpha # Predictive mean
  
  v <- solve(L, KStar)
  V <- k(XStar, XStar, ...) - t(v) %*% v # Predictive variance
  
  return(list(predMean = fstar, predVar = V))
}

# Q1.2: let the prior hyperparameters be sigmaf = 1 and l = 0.3. Update this
# prior with a single observation: (x, y) = (0.4, 0.719). Assume that sigman = 0.1. 
# Plot the posterior mean of f over the interval x ∈ [−1, 1].
# Plot also 95 % probability (pointwise) bands for f.

# Prior hyperparameters
sigmaF <- 1
l <- 0.3

observation <- data.frame(x = 0.4, y = 0.719) # single observation: (x, y) = (0.4, 0.719).

sigmaNoise <- 0.1
interval <- seq(-1, 1, length = 100) # Plot the posterior mean of f over the interval [-1, 1]

postSim <- posteriorGP(X = observation$x, y = observation$y, interval, sigmaNoise, k = SquaredExpKernel, sigmaF, l)

plot(interval, 
     postSim$predMean, 
     type = "l", 
     ylim = c(-3, 3), 
     xlab = "", 
     ylab = "Predictive Mean",
     main = "Posterior mean of f, single observation")
lines(interval, postSim$predMean - 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)
lines(interval, postSim$predMean + 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)

# Q1.3: Update your posterior from (2) with another observation: (x,y) = (−0.6,−0.044).
# Plot the posterior mean of f over the interval x ∈ [−1, 1]. Plot also 95 % 
# probability (point-wise) bands for f.

observations <- data.frame( x = c(0.4, -0.6), y = c(0.719, -0.044))

postSim <- posteriorGP(X = observations$x, y = observations$y, interval, sigmaNoise, k = SquaredExpKernel, sigmaF, l)

plot(interval, 
     postSim$predMean, 
     type = "l", 
     ylim = c(-3, 3), 
     xlab = "", 
     ylab = "Predictive Mean",
     main = "Posterior mean of f, two observations")
lines(interval, postSim$predMean - 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)
lines(interval, postSim$predMean + 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)

# Q1.4: Compute the posterior distribution of f using all the five data points in the table

x = c(-1.0, -0.6, -0.2, 0.4, 0.8)
y = c(0.768, -0.044, -0.940, 0.719, -0.664)
observations <- data.frame(x = x, y = y)

postSim <- posteriorGP(X = observations$x, y = observations$y, interval, sigmaNoise, k = SquaredExpKernel, sigmaF, l)

plot(interval, 
     postSim$predMean, 
     type = "l", 
     ylim = c(-3, 3), 
     xlab = "", 
     ylab = "Predictive Mean",
     main = "Posterior mean of f, five observations")
lines(interval, postSim$predMean - 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)
lines(interval, postSim$predMean + 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)

# Q1.5: Repeat (4), with hyperparameters σf = 1 and l = 1. Compare the results.

# New hyperparameters
sigmaF <- 1
l <- 1

postSim <- posteriorGP(X = observations$x, y = observations$y, interval, sigmaNoise, k = SquaredExpKernel, sigmaF, l)

plot(interval, 
     postSim$predMean, 
     type = "l", 
     ylim = c(-3, 3), 
     xlab = "", 
     ylab = "Predictive Mean",
     main = expression(paste("Posterior mean of f, five observations, ", sigma[f] == 1, " and ", l == 1)))
lines(interval, postSim$predMean - 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)
lines(interval, postSim$predMean + 1.96 * sqrt(diag(postSim$predVar)), col = "blue", lwd = 2)

###################################

# Q2. GP Regression with kernlab.

data <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/TempTullinge.csv", header=TRUE, sep=";")

# Structure data according to lab instructions.

# time = 1, 2, ..., 365 x 6, only use every fifth observation
time <- seq(from = 1, to = 365 * 6, by = 5)
# day = 1, 2, . . ., 365, 1, 2, . . ., 365, only use every fifth observation
day <- rep(seq(from = 1, to = 365, by = 5), times = 6)

temperatures <- data$temp[time] # Temperatures for each (fifth) day.

# Q2.1 Familiarization with kernlab

# Define your own square exponential kernel function with parameters l and sigmaf.
SE <- function(l = 1, sigmaf = 1) {
  r <- function (X, XStar) {
    n1 <- length(X)
    n2 <- length(XStar)
    K <- matrix(NA, n1, n2)
    for (i in 1:n2) {
      K[, i] <- sigmaf ^ 2 * exp(-0.5 * ((X - XStar[i]) / l) ^ 2)
    }
    return(K)
  }
  class(r) <- 'kernel' # Return as class kernel.
  return (r)
}

X <- matrix(c(1,3,4)) # Simulating some data.
Xstar <- matrix(c(2,3,4))

# Own SE kernel
SEkernel <- SE(l = 1, sigmaf = 1)
SEkernel(1,2) # Just a test - evaluating the kernel in the points x=1 and x'=2.
# Computing the whole covariance matrix K from the kernel. Just a test.
kernelMatrix(kernel = SEkernel, x = X, y = Xstar) # So this is K(X,Xstar).

# Q2.2

# Consider first the following model: temp=f(time)+εwithε∼N(0,σn2)andf ∼GP(0,k(time,time))
# Let σn2 be the residual variance from a simple quadratic regression fit 
# (using the lm function in R). Estimate the above Gaussian process regression 
# model using the squared exponential function from (1) with σf = 20 and l = 0.2. 
# Use the predict function in R to compute the posterior mean at every data point 
# in the training dataset. Make a scatterplot of the data and superimpose 
# the posterior mean of f as a curve

# Let sigmaN be the residual variance from a simple quadratic regression fit 
lmfit <- lm(temperatures ~ time + time^2)
sigmaN <- sd(lmfit$residuals)

# Estimate the above Gaussian process regression model using 
# the squared exponential function from (1) using sigmaF = 20 and l = 0.2.

GP <- gausspr(time, temperatures, kernel=SE(l=0.2, sigmaf=20), var=sigmaN^2)

# Use the predict function in R to compute the posterior mean 
# at every data point in the training dataset.
postMean <- predict(GP, newdata=time)

# Make a scatterplot of the data and superimpose the posterior mean of f as a curve
plot(time, temperatures, main="Temperatures over time")
lines(time, postMean, type="l", col="blue")

# Q2.3

# Do your own computations for the posterior variance of f
post <- posteriorGP(scale(time), scale(temperatures), scale(time), sigmaN, k = SquaredExpKernel, 20, 0.2)

plot(time, temperatures, main="Temperatures over time")
# plot the 95 % probability (pointwise) bands for f
lines(time, post$predMean * sqrt(var(temperatures)) + mean(temperatures) - 1.96 * sqrt(diag(post$predVar)), col = "red", lwd = 2)
lines(time, post$predMean * sqrt(var(temperatures)) + mean(temperatures) + 1.96 * sqrt(diag(post$predVar)), col = "red", lwd = 2)
# Superimpose with Posterior mean from Q2.2
lines(time, postMean, type="l", col="blue") 

# Q2.4

# Estimate the model using the squared exponential function with sigmaf = 20 and l = 0.2.
GP <- gausspr(day, temperatures, kernel=SE(l = 0.2, sigmaf = 20), var = sigmaN^2)
postMean2 <- predict(GP, newdata=day)

plot(time, temperatures, main="Temperatures over time")
# Superimpose the posterior mean from this model on the posterior mean from the model in (2).
lines(time, postMean, type="l", col="blue")
lines(time, postMean2, type="l", col="green")

# Q2.5

# Implement the periodic kernel.
PeriodicKernel <- function(sigmaf, l1, l2, d) {
  r <- function(X, XStar) {
    t1 <- -((2 * sin(pi * abs(X - XStar)^2 / d)) / l1^2)
    t2 <- -(0.5 * (abs(X - XStar)^2) / l2^2)
    return(sigmaf^2 * exp(t1) * exp(t2))
  }
  class(r) <- 'kernel' # Return as class kernel.
  return (r)
}

# Estimate the GP model using the time variable with this kernel
GPPeriodic <- gausspr(time, 
              temperatures, 
              kernel=PeriodicKernel(sigmaf = 20, l1 = 1, l2 = 10, d = 365/sd(time)), 
              var = sigmaN^2)

postMeanPeriodic <- predict(GPPeriodic, newdata = time)

# Compare the fit to the previous two models.
plot(time, temperatures, main="Temperatures over time")
lines(time, postMean, type="l", col="blue")
lines(time, postMean2, type="l", col="green")
lines(time, postMeanPeriodic, type="l", col="red")

# Question 3: GP Classification with kernlab.

# Read data
data <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/banknoteFraud.csv", header=FALSE, sep=",") 
names(data) <- c("varWave","skewWave","kurtWave","entropyWave","fraud") 
data[,5] <- as.factor(data[,5])

# Train and Test
set.seed(111)
SelectTraining <- sample(1:dim(data)[1], size = 1000, replace = FALSE)
train <- data[SelectTraining, ]
test <- data[-SelectTraining, ]

# Q3.1

# Fit a Gaussian process classification model for fraud on the training data.
# Start only using covariates varWave and skewWave in the model.
GPClass <- gausspr(fraud ~ varWave + skewWave, data = train) # Use default kernel and hyperparameters
fraudPrediction <- predict(GPClass, newdata=train)
table(fraudPrediction, train$fraud) # Confusion Matrix
sum(diag(table(fraudPrediction, train$fraud))) / 
  sum(table(fraudPrediction, train$fraud)) # Accuracy
  
# class probabilities 
probPreds <- predict(GPClass, train, type="probabilities")
x1 <- seq(min(train$varWave),max(train$varWave),length=100)
x2 <- seq(min(train$skewWave),max(train$skewWave),length=100)
gridPoints <- meshgrid(x1, x2)
gridPoints <- cbind(c(gridPoints$x), c(gridPoints$y))

gridPoints <- data.frame(gridPoints)
names(gridPoints) <- names(train[1:2])
probPreds <- predict(GPClass, gridPoints, type="probabilities")

# Plotting for Prob(fraud)
contour(x1,x2,matrix(probPreds[,1],100,byrow = TRUE), 20, xlab = "varWave", ylab = "skewWave", main = "Countor Plot Fraud")
points(train[train[,5]==1,1],train[train[,5]==1,2],col="red")
points(train[train[,5]==0,1],train[train[,5]==0,2],col="blue")

# Q3.2

# Make predictions for the test set. 
fraudPredictionTest <- predict(GPClass, newdata=test)

# Compute the confusion matrix and the classifier accuracy
table(fraudPredictionTest, test$fraud) # Confusion Matrix Test
sum(diag(table(fraudPredictionTest, test$fraud))) /
  sum(table(fraudPredictionTest, test$fraud)) # Accuracy Test

# Q3.3

# Train model using all four covariates.
GPClass2 <- gausspr(fraud ~ ., data = train)

# Make predictions for the test set. 
fraudPredictionTest2 <- predict(GPClass2, newdata=test)

# Confusion Matrix and Accuracy
table(fraudPredictionTest2, test$fraud) # Confusion Matrix Test
sum(diag(table(fraudPredictionTest2, test$fraud))) /
  sum(table(fraudPredictionTest2, test$fraud)) # Accuracy Test