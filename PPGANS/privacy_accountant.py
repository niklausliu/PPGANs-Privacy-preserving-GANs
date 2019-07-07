import math

def epsilon(sensitivity=1, sigma=0.001, delta=0.001, iterations=1):
    """
    Given injected Gaussian noise with fixed covariance matrix sig^2, find the epsilon privacy parameters.
    We are comparing between two distribution N(mu_1,sigma^2 I) and N(mu_2,sigma^2 I) where
    mu = mu_2 - mu_1 is sensitivity when we do a descent step.
    - sensitivity: 2-norm of sensitivity.
    This is usually the product of the clipping value (bounding two norm of gradient) and learning rate. By default: 1.0
    - sigma: standard deviation of Gaussian noise with covariance matrix sigma^2 I_d to be added to comparing distributions.
    This is usually the deviation of Gaussian to the gradient multiplied by learning rate. By default: 0.001
    - delta: privacy parameter. Must be between 0 and 1. By default: 0.001
    - iterations: number of iterations that the algorithm has run this Gaussian mechanism. By default: 1
    """
    mu = sensitivity
    t = iterations
    print(math.sqrt(2 * t * math.log(1 / delta) / (sigma ** 2)) * mu + t * (mu ** 2) / (2 * sigma ** 2))
    return math.sqrt(2 * t * math.log(1 / delta) / (sigma ** 2)) * mu + t * (mu ** 2) / (2 * sigma ** 2)

