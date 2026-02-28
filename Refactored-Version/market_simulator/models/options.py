import math


def norm_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2


def norm_pdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_price(S, K, T, r, sigma, option_type):
    """Black-Scholes price + greeks. Returns (price, delta, gamma, theta, vega). T in years."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        delta = (1.0 if S > K else 0.0) if option_type == 'call' else (-1.0 if S < K else 0.0)
        return intrinsic, delta, 0.0, 0.0, 0.0

    sigma = max(sigma, 1e-4)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = (-(S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1
        theta = (-(S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365

    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm_pdf(d1) * math.sqrt(T) / 100  # per 1% vol move

    return max(price, 0), delta, gamma, theta, vega
