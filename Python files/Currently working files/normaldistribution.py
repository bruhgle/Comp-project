import statistics
from scipy.stats import norm
from decimal import Decimal, getcontext
import numpy as np

def mean_and_std_dev(lst):
    mean = statistics.mean(lst)
    std_dev = statistics.stdev(lst)
    return mean, std_dev

def probability_above_below(x, mean, std_dev):
    getcontext().prec = 30  # Set precision
    mean = Decimal(str(mean))
    std_dev = Decimal(str(std_dev))
    x = Decimal(str(x))
    z_score = (x - mean) / std_dev
    prob_below = Decimal(norm.cdf(float(z_score)))
    prob_above = Decimal(1) - prob_below
    return prob_below, prob_above

# Example usage
data = np.load('50moids1000down.npy').tolist()
mean, std_dev = mean_and_std_dev(data)
print("Mean:", mean)
print("Standard Deviation:", std_dev)

num = float(input("Enter a number: "))
prob_below, prob_above = probability_above_below(num, mean, std_dev)
print("Probability of being below", num, ":", prob_below)
print("Probability of being above", num, ":", prob_above)
