import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numba import njit, prange

# Numba-optimized Monte Carlo simulation (without control variate adjustment)
@njit(parallel=True)
def monte_carlo_simulation(S0, sigma_0, mu, r, T, K, v, z_max, sigma_min, num_paths, num_steps):
    delta_t = T / num_steps
    sqrt_delta_t = np.sqrt(delta_t)

    # Initialize arrays
    S_paths = np.zeros((num_paths * 2, num_steps + 1))  # Antithetic paths
    sigma_paths = np.zeros((num_paths * 2, num_steps + 1))
    S_paths[:, 0] = S0
    sigma_paths[:, 0] = sigma_0

    # Initialize random number arrays for antithetic variates
    Z1 = np.zeros((num_paths, num_steps))
    Z2 = np.zeros((num_paths, num_steps))
    
    # Generate random numbers for antithetic variates using Numba-compatible method
    for i in range(num_paths):
        for t in range(num_steps):
            Z1[i, t] = np.random.normal()
            Z2[i, t] = np.random.normal()

    # Monte Carlo simulation with antithetic variates
    for t in prange(1, num_steps + 1):
        # Original paths
        Z1_clipped = np.clip(Z1[:, t-1], -z_max, z_max)
        Z2_clipped = np.clip(Z2[:, t-1], -z_max, z_max)

        # Volatility update
        sigma_next = np.maximum(sigma_min, sigma_paths[:num_paths, t-1] + v * sigma_paths[:num_paths, t-1] * sqrt_delta_t * Z2_clipped)
        sigma_paths[:num_paths, t] = sigma_next

        # Asset price update
        log_S_next = (
            np.log(S_paths[:num_paths, t-1])
            + (mu - 0.5 * sigma_next**2) * delta_t
            + sigma_next * sqrt_delta_t * Z1_clipped
        )
        S_paths[:num_paths, t] = np.exp(log_S_next)

        # Antithetic paths
        sigma_next_antithetic = np.maximum(sigma_min, sigma_paths[num_paths:, t-1] + v * sigma_paths[num_paths:, t-1] * sqrt_delta_t * (-Z2_clipped))
        sigma_paths[num_paths:, t] = sigma_next_antithetic

        log_S_next_antithetic = (
            np.log(S_paths[num_paths:, t-1])
            + (mu - 0.5 * sigma_next_antithetic**2) * delta_t
            + sigma_next_antithetic * sqrt_delta_t * (-Z1_clipped)
        )
        S_paths[num_paths:, t] = np.exp(log_S_next_antithetic)

    # Calculate Asian option payoff
    average_prices = np.zeros(num_paths * 2)  # To store average prices
    for i in range(num_paths * 2):
        average_prices[i] = np.mean(S_paths[i, 1:])  # Mean of the path (excluding S_paths[:, 0])

    payoffs = np.maximum(average_prices - K, 0)      # Payoff for each path
    discounted_payoffs = np.exp(-r * T) * payoffs    # Discounted to present value

    # Compute geometric mean manually
    geometric_average_prices = np.zeros(num_paths * 2)
    for i in range(num_paths * 2):
        log_sum = 0.0
        for t in range(1, num_steps + 1):  # Ignore S_paths[:, 0] (initial value)
            log_sum += np.log(S_paths[i, t])
        geometric_average_prices[i] = np.exp(log_sum / num_steps)  # Geometric mean

    return discounted_payoffs, geometric_average_prices




# Analytical solution for geometric average Asian option
def geometric_asian_option(S0, K, T, r, sigma):
    b_avg = (r - 0.5 * sigma**2) / 2
    sigma_avg = sigma / np.sqrt(3)
    d1 = (np.log(S0 / K) + (b_avg + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
    d2 = d1 - sigma_avg * np.sqrt(T)
    call_price = S0 * np.exp((b_avg - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Parameters
S0 = 100
sigma_0 = 0.2
mu = 0.05
r = 0.05
T = 1.0
K = 100
v = 0.1
z_max = 3.0
sigma_min = 0.1
num_steps = 100

# List of number of paths to test
num_paths_list = [10**3, 10**4, 10**5, 10**6]

# Store results
mean_prices = []
ci_lower = []
ci_upper = []

# Run simulation for different numbers of paths
for num_paths in num_paths_list:
    discounted_payoffs, geometric_average_prices = monte_carlo_simulation(S0, sigma_0, mu, r, T, K, v, z_max, sigma_min, num_paths, num_steps)
    
    # Calculate geometric payoffs and discounted payoffs
    geometric_payoffs = np.maximum(geometric_average_prices - K, 0)
    discounted_geometric_payoffs = np.exp(-r * T) * geometric_payoffs

    # Compute results
    mean_price = np.mean(discounted_payoffs)  # No control variate adjustment
    std_dev = np.std(discounted_payoffs, ddof=1)
    z = 1.96  # For 95% confidence interval
    ci = (
        float(mean_price - z * std_dev / np.sqrt(num_paths * 2)),
        float(mean_price + z * std_dev / np.sqrt(num_paths * 2))
    )

    mean_prices.append(mean_price)
    ci_lower.append(ci[0])
    ci_upper.append(ci[1])
    print(f"Paths: {num_paths}, Mean Price: {mean_price:.4f}, CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(num_paths_list, mean_prices, marker='o', linestyle='-', color='b', label='Mean Price')
plt.fill_between(num_paths_list, ci_lower, ci_upper, color='b', alpha=0.2, label='95% CI')
plt.xscale('log')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Monte Carlo Convergence Analysis (No Control Variates)')
plt.legend()
plt.grid()
plt.show()
