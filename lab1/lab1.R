# Assignment 1, Lab 1 in course TDDE15,
# Advanced Machine Learning, Linkoping University, Sweden


########### Libraries #############
library(bnlearn)
library(gRain)
########### Libraries #############

###################################

# Q1. Show that multiple runs of the hill-climbing algorithm can return 
# non-equivalent Bayesian network (BN) structures.

# Load data
data("asia")

# Two BNs using hill climb algorithm.
set.seed(12345) # Ensure we use the same seed 
bn1 <- hc(x = asia, restart = 10)

set.seed(12345)
# Using different score function.
bn2 <- hc(x = asia, restart = 10, score = 'aic')

# Two DAGs are equivalent if and only if they have the same
# adjacencies and unshielded colliders.

# Check unshielded colliders. We will find they are different.
unshielded.colliders(bn1)
unshielded.colliders(bn2)

# Convert to CPDAGs to standardize representation and check for equivalence.
bn1 <- cpdag(bn1)
bn2 <- cpdag(bn2)

# Recall, hill climb is not asymptotically correct under faithfulness, 
# i.e. it may get trapped in local optima. -> This is the reason why 
# we get non-equivalent BN structures.

all.equal(bn1, bn2)
graphviz.plot(bn1)
graphviz.plot(bn2)

###################################

# Q2. Learn a BN from 80 % of the Asia dataset. Learn both the structure and 
# the parameters. Use any learning algorithm and settings that you consider 
# appropriate. Use the BN learned to classify the remaining 20 % of the Asia 
# dataset in two classes: S = yes and S = no. Use exact or approximate inference 
# with the help of the bnlearn and gRain packages, i.e. you are not allowed to 
# use functions such as predict. Report the confusion matrix. Compare your 
# results with those of the true Asia BN.

# Again, Load the data.
data("asia")

# Split into 80/20 train test.
n <- dim(asia)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.8))
train <- asia[id, ]
test  <- asia[-id, ]

rm(n, id) # Remove unnecessary variables.

# Learning BN structure using HC.
set.seed(12345)
bn_struct <- hc(x = train, restart = 10)
true_struct <- model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]") # True Asia BN

# Learn parameters
bn_param <- bn.fit(x = bn_struct, data = train)
true_param <- bn.fit(x = true_struct, data = train)

# Transform into gRain object and compile.
grain_obj <- as.grain(bn_param)
bn_compiled <- compile(grain_obj)

true_grain_obj <- as.grain(true_param) 
true_bn_compiled <- compile(true_grain_obj) 

# Custom predict function
predict_bn <- function(obj) {
  pred <- c()
  for (i in 1:nrow(test)) {
    evidence <- setEvidence(object = obj, 
                            nodes = c("A", "T", "L", "B", "E", "X", "D"), 
                            states = as.character(unlist(test[i,-2])))
    query <- querygrain(evidence, nodes = c("S"))$S
    pred[i] <- ifelse(query["yes"] > 0.5, "yes", "no")
  }
  return (pred)
}

# Confusion matrices
table(predict_bn(bn_compiled), test$S)
table(predict_bn(true_bn_compiled), test$S) # CM from True Asia BN

###################################

# Q3. Classify S given observations only for the so-called Markov blanket of S,
# i.e. its parents plus its children plus the parents of its children minus S 
# itself. Report the confusion matrix.

mb <- mb(bn_param, c("S"))
mb_true <- mb(true_param, c("S"))

# Custom predict function
predict_mb <- function(obj, nodes) {
  pred <- c()
  for (i in 1:nrow(test)) {
    evidence <- setEvidence(object = obj, 
                            nodes = nodes, 
                            states = as.character(unlist(
                              subset(test, select = nodes)[i,])))
    query <- querygrain(evidence, nodes = c("S"))$S
    pred[i] <- ifelse(query["yes"] > 0.5, "yes", "no")
  }
  return (pred)
}

# Predict and create confusion matrices.
table(predict_mb(bn_compiled, mb), test$S)
table(predict_mb(true_bn_compiled, mb_true), test$S)

###################################

# Q4. Repeat Q2. using a naive Bayes classifier, i.e. the predictive variables 
# are independent given the class variable. Model the naive Bayes classifier 
# as a BN. You have to create the BN by hand, i.e. you are not allowed to use 
# the function naive.bayes from the bnlearn package.

# Naive Bayes Structure
nb <- model2network("[S][A|S][B|S][D|S][E|S][L|S][T|S][X|S]")
nb <- bn.fit(x = nb, data = train) # Parameters from training data

# Transform into gRain object and compile.
nb <- as.grain(nb)
nb <- compile(nb)

# Confusion matrix
table(predict_bn(nb), test$S)

