import numpy as np
from scipy.stats import norm

def monte_carlo_simulation(S0, sigma_0, mu, r, T, K, v, z_max, sigma_min, num_paths, num_steps):
    """
    Monte Carlo simulation for Asian option pricing with antithetic variates and control variates.

    Parameters:
        S0: Initial asset price.
        sigma_0: Initial volatility.
        mu: Drift term.
        r: Risk-free interest rate.
        T: Time to maturity.
        K: Strike price.
        v: Volatility shift parameter.
        z_max: Clipping value for normal samples.
        sigma_min: Minimum allowed volatility.
        num_paths: Number of simulated paths.
        num_steps: Number of time steps.

    Returns:
        mean_price: Estimated mean option price.
        ci: 95% confidence interval for the option price.
    """
    delta_t = T / num_steps
    sqrt_delta_t = np.sqrt(delta_t)

    # Initialize arrays
    S_paths = np.zeros((num_paths * 2, num_steps + 1))  # Antithetic paths
    sigma_paths = np.zeros((num_paths * 2, num_steps + 1))
    S_paths[:, 0] = S0
    sigma_paths[:, 0] = sigma_0

    # Generate random numbers for antithetic variates
    Z1 = np.random.normal(size=(num_paths, num_steps))
    Z2 = np.random.normal(size=(num_paths, num_steps))

    # Monte Carlo simulation with antithetic variates
    for t in range(1, num_steps + 1):
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
    average_prices = np.mean(S_paths[:, 1:], axis=1)  # Average price across time steps
    payoffs = np.maximum(average_prices - K, 0)      # Payoff for each path
    discounted_payoffs = np.exp(-r * T) * payoffs    # Discounted to present value

    # Control variate: Geometric average Asian option
    geometric_average_prices = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
    geometric_payoffs = np.maximum(geometric_average_prices - K, 0)
    discounted_geometric_payoffs = np.exp(-r * T) * geometric_payoffs

    # Control variate adjustment
    cov = np.cov(discounted_payoffs, discounted_geometric_payoffs)[0, 1]
    var_control = np.var(discounted_geometric_payoffs, ddof=1)
    control_variate_coeff = cov / var_control

    # Adjusted payoffs
    adjusted_payoffs = discounted_payoffs - control_variate_coeff * (discounted_geometric_payoffs - geometric_asian_option(S0, K, T, r, sigma_0))

    # Compute results
    mean_price = np.mean(adjusted_payoffs)
    std_dev = np.std(adjusted_payoffs, ddof=1)
    z = 1.96  # For 95% confidence interval
    ci = (
        float(mean_price - z * std_dev / np.sqrt(num_paths * 2)),
        float(mean_price + z * std_dev / np.sqrt(num_paths * 2))
    )

    return mean_price, ci, np.mean(average_prices)

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
num_paths = 10**5
num_steps = 100

# Run simulation
mean_price, ci, avg_price = monte_carlo_simulation(S0, sigma_0, mu, r, T, K, v, z_max, sigma_min, num_paths, num_steps)
print(f"Mean Price: {mean_price:.4f}")
print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"Average Asset Price: {avg_price:.4f}")