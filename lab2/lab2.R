# Lab 2 course TDDE15, Advanced Machine Learning, Linkoping University, Sweden


########### Libraries #############
library(HMM)
library(entropy)
########### Libraries #############

###################################

### Scenario ###
# Model the behavior of a robot that walks around a ring. The ring is divided 
# into 10 sectors. At any given time point, the robot is in one of the sectors 
# and decides with equal probability to stay in that sector or move to the next 
# sector. You do not have direct observation of the robot. However, the robot 
# is equipped with a tracking device that you can access. The device is not 
# very accurate though: If the robot is in the sector i, then the device will 
# report that the robot is in the sectors [i − 2, i + 2] with equal probability.
### Scenario ###


# Q1. Build a hidden Markov Model (HMM) for the scenario described.

states <- 1:10 # Each Sector represent a state.
symbols <- 1:10 

# Transition probabilities as described.
transition_probs <- matrix(c(0.5,0.5,0,0,0,0,0,0,0,0,
                             0,0.5,0.5,0,0,0,0,0,0,0,
                             0,0,0.5,0.5,0,0,0,0,0,0,
                             0,0,0,0.5,0.5,0,0,0,0,0,
                             0,0,0,0,0.5,0.5,0,0,0,0,
                             0,0,0,0,0,0.5,0.5,0,0,0,
                             0,0,0,0,0,0,0.5,0.5,0,0,
                             0,0,0,0,0,0,0,0.5,0.5,0,
                             0,0,0,0,0,0,0,0,0.5,0.5,
                             0.5,0,0,0,0,0,0,0,0,0.5),
                           nrow = length(states), 
                           ncol = length(states), 
                           byrow = TRUE)

# Emission probabilities as described.
emission_probs <- matrix(c(0.2,0.2,0.2,0,0,0,0,0,0.2,0.2,
                           0.2,0.2,0.2,0.2,0,0,0,0,0,0.2,
                           0.2,0.2,0.2,0.2,0.2,0,0,0,0,0,
                           0,0.2,0.2,0.2,0.2,0.2,0,0,0,0,
                           0,0,0.2,0.2,0.2,0.2,0.2,0,0,0,
                           0,0,0,0.2,0.2,0.2,0.2,0.2,0,0,
                           0,0,0,0,0.2,0.2,0.2,0.2,0.2,0,
                           0,0,0,0,0,0.2,0.2,0.2,0.2,0.2,
                           0.2,0,0,0,0,0,0.2,0.2,0.2,0.2,
                           0.2,0.2,0,0,0,0,0,0.2,0.2,0.2),
                          nrow = length(states),
                          ncol = length(symbols),
                          byrow = TRUE)

# Equal probability for each state to start.
start_probs <- rep(0.1, length(states))

hmm <- initHMM(states, symbols, start_probs, transition_probs, emission_probs)

###################################

# Q2. Simulate the HMM for 100 time steps.

set.seed(1234) # Set seed to get repeatable results
sim <- simHMM(hmm, length = 100)

###################################

# Q3. Discard the hidden states from the sample obtained above. Use the remaining
# observations to compute the filtered and smoothed probability distributions
# for each of the 100 time points. Compute also the most probable path.

# To compute smoothed and filtered probability distribution, we first need alpha
# and Beta (forward and backward probs) obtained by the forward-backward algorithm.
alpha <- prop.table(exp(forward(hmm, sim$observation)))
beta <- prop.table(exp(backward(hmm, sim$observation)))

# Compute filtered and smoothed distributions.
filtered <- t(apply(alpha, MARGIN = 1, "/", apply(alpha, MARGIN = 2, sum)))
smoothed <- t(apply(alpha * beta, MARGIN = 1, "/", apply(alpha*beta, MARGIN = 2, sum)))

# Compute most probable path with viterbi algorithm.
viterbi <- viterbi(hmm, sim$observation) 

###################################

# Q4. Compute the accuracy of the filtered and smoothed probability
# distributions, and of the most probable path. That is, compute the
# percentage of the true hidden states that are guessed by each method.

# Predicitions for filtered and smoothed distributions.
pred_filtered <- apply(filtered, MARGIN = 2, FUN = which.max)
pred_smoothed <- apply(smoothed, MARGIN = 2, FUN = which.max)

# Compute accuracy
sum(sim$states == pred_filtered) / 100 # Accuracy Filtered Distribution
sum(sim$states == pred_smoothed) / 100 # Accuracy Smoothed Distribution
sum(sim$states == viterbi) / 100       # Accuracy Most Probable Path

###################################

# Q5. Repeat the previous exercise with different simulated samples. In general,
# the smoothed distributions should be more accurate than the filtered 
# distributions. Why ? In general, the smoothed distributions should be more 
# accurate than the most probable paths, too. Why ?

# Function to generate different simulated samples and compute accuracies.
GenerateSims <- function(sim, hmm) {
  
  # Alpha and Beta
  alpha <- prop.table(exp(forward(hmm, sim$observation)))
  beta <- prop.table(exp(backward(hmm, sim$observation)))
  
  # Filtered and smoothed Distributions
  filtered2 <- t(apply(alpha, MARGIN = 1, "/", apply(alpha, MARGIN = 2, sum)))
  smoothed2 <- t(apply(alpha * beta, MARGIN = 1, "/", apply(alpha*beta, MARGIN = 2, sum)))
  
  # Compute accuracies and return as array.
  return(c(sum(sim$states == apply(filtered2, MARGIN = 2, FUN = which.max)) / 100,
           sum(sim$states == apply(smoothed2, MARGIN = 2, FUN = which.max)) / 100,
           sum(sim$states == viterbi(hmm, sim$observation)) / 100))
}

# Generate simulations and compute accuracies
accuracies <- matrix(nrow = 100, ncol = 3)
for (i in 1:nrow(accuracies)) {
  accuracies[i,] <- GenerateSims(simHMM(hmm, length = 100), hmm)
}

# Visualize data with Boxplot
boxplot(accuracies, 
        main = "Accuracy Distribution for Different Methods",
        ylab = "Accuracy",
        col = c("skyblue", "lightgreen", "lightcoral"),
        names = c("Filtered", "Smoothed", "Viterbi"))

# Add Legend and horizontal gridlines
legend("topright", legend = c("Filtered", "Smoothed", "Viterbi"),
       fill = c("skyblue", "lightgreen", "lightcoral"))
abline(h = seq(0, 1, by = 0.1), col = "gray", lty = 2)

###################################

# Q6. Is it always true that the later in time (i.e., the more observations
# you have received) the better you know where the robot is?

# Compute entropy of the filtered distributions and visualize.
entropy <- apply(filtered, MARGIN = 2, entropy.empirical)
plot(entropy, 
     type = "l", 
     xlab = "Observations", 
     ylab = "Entropy value", 
     main="Entropy vs. Nr of Observations")

###################################

# Q7. Consider any of the samples above of length 100. Compute the 
# probabilities of the hidden states for the time step 101.

# Multiply transition probabilities with the last filtered step.
transition_probs %*% filtered[ ,100]

