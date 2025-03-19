"""
Benchmark script for comparing order book implementations:
1. Original OrderBook (order_book.py + order_level.py)
2. Reworked OrderBook (reworked_order_book.py)
3. Claude's Optimized OrderBook (claude_optimized_ob.py)
"""

import time
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up mocks for dependencies
# Mock portfolio utils to avoid actual calculations
import sys
from unittest.mock import MagicMock

# Mock expensive operations - this must happen BEFORE importing the order books
import market_simulator.portfolio_utils
import market_simulator.price_server

# Mock the portfolio calculation function to return a simple dictionary
market_simulator.portfolio_utils.calculate_portfolio_status = MagicMock(
    return_value={
        "cash": 10000.0,
        "portfolio_value": 15000.0,
        "total_value": 25000.0,
        "positions": {}
    }
)

# Mock the emit functions to do nothing
market_simulator.portfolio_utils.emit_portfolio_update = MagicMock(return_value=None)
market_simulator.price_server.emit_trade_update = MagicMock(return_value=None)

# Now it's safe to import the OrderBook implementations
from market_simulator.models.order_book import OrderBook as OriginalOrderBook
from market_simulator.models.reworked_order_book import OrderBook as ReworkedOrderBook
from market_simulator.models.claude_optimized_ob import OrderBook as OptimizedOrderBook

# Setup mock accounts for testing
from market_simulator.utils.market_utils import accounts

# Create a proper mock Account class with all required methods
class MockAccount:
    def __init__(self):
        self.positions = {}
        self.cash = 1000000.0
    
    def addPosition(self, asset, qty):
        if asset not in self.positions:
            self.positions[asset] = 0
        self.positions[asset] += qty
        return
    
    def tradeAtPrice(self, asset, price, qty, direction):
        if asset not in self.positions:
            self.positions[asset] = 0
        self.positions[asset] += qty * direction
        self.cash -= price * qty * direction
        return
    
    def getPosition(self, asset):
        return self.positions.get(asset, 0)
    
    def getCash(self):
        return self.cash

# Create test accounts with our proper mock
for i in range(1, 101):
    account_id = f"TEST_ACCOUNT_{i}"
    accounts[account_id] = MockAccount()

# Create account for market maker
accounts["MARKET MAKER"] = MockAccount()

class BenchmarkResults:
    def __init__(self):
        self.results = {
            "Original": {},
            "Reworked": {},
            "Claude Optimized": {}
        }
    
    def add_result(self, impl, test_name, duration, ops_per_sec):
        self.results[impl][test_name] = {
            "duration": duration,
            "ops_per_sec": ops_per_sec
        }
    
    def print_summary(self):
        print("\n===== BENCHMARK RESULTS =====\n")
        
        # Get all test names
        all_tests = set()
        for impl_results in self.results.values():
            all_tests.update(impl_results.keys())
        
        # Create DataFrame for comparison
        data = []
        for test in sorted(all_tests):
            row = {"Test": test}
            for impl in ["Original", "Reworked", "Claude Optimized"]:
                if test in self.results[impl]:
                    row[f"{impl} (ops/sec)"] = f"{self.results[impl][test]['ops_per_sec']:.2f}"
                    row[f"{impl} (time)"] = f"{self.results[impl][test]['duration']:.6f}s"
                else:
                    row[f"{impl} (ops/sec)"] = "N/A"
                    row[f"{impl} (time)"] = "N/A"
            data.append(row)
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("\n")
        
        # Add relative performance section
        print("===== RELATIVE PERFORMANCE =====\n")
        for test in sorted(all_tests):
            print(f"Test: {test}")
            base_time = None
            base_impl = None
            
            # Find the fastest implementation to use as baseline
            for impl in ["Original", "Reworked", "Claude Optimized"]:
                if test in self.results[impl]:
                    if base_time is None or self.results[impl][test]['duration'] < base_time:
                        base_time = self.results[impl][test]['duration']
                        base_impl = impl
            
            if base_impl:
                print(f"  Fastest: {base_impl} ({base_time:.6f}s)")
                
                for impl in ["Original", "Reworked", "Claude Optimized"]:
                    if impl != base_impl and test in self.results[impl]:
                        relative = self.results[impl][test]['duration'] / base_time
                        print(f"  {impl}: {relative:.2f}x slower")
            
            print()
    
    def generate_charts(self, filename="benchmark_results.png"):
        # Get all test names
        all_tests = set()
        for impl_results in self.results.values():
            all_tests.update(impl_results.keys())
        
        all_tests = sorted(all_tests)
        
        # Prepare data for plotting
        implementations = ["Original", "Reworked", "Claude Optimized"]
        ops_per_sec = np.zeros((len(implementations), len(all_tests)))
        
        for i, impl in enumerate(implementations):
            for j, test in enumerate(all_tests):
                if test in self.results[impl]:
                    ops_per_sec[i, j] = self.results[impl][test]["ops_per_sec"]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 10))
        
        x = np.arange(len(all_tests))
        width = 0.25
        
        for i, impl in enumerate(implementations):
            ax.bar(x + (i - 1) * width, ops_per_sec[i], width, label=impl)
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_tests, rotation=45, ha="right")
        ax.set_xlabel("Test")
        ax.set_ylabel("Operations per Second")
        ax.set_title("Order Book Performance Comparison")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Chart saved to {filename}")
        
        # Create a second chart showing relative performance
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Find the fastest implementation for each test
        best_perf = np.max(ops_per_sec, axis=0)
        relative_perf = np.zeros_like(ops_per_sec)
        
        for i in range(ops_per_sec.shape[0]):
            for j in range(ops_per_sec.shape[1]):
                if best_perf[j] > 0:
                    relative_perf[i, j] = ops_per_sec[i, j] / best_perf[j]
        
        # Plot relative performance
        for i, impl in enumerate(implementations):
            ax.bar(x + (i - 1) * width, relative_perf[i], width, label=impl)
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_tests, rotation=45, ha="right")
        ax.set_xlabel("Test")
        ax.set_ylabel("Relative Performance (higher is better)")
        ax.set_title("Order Book Relative Performance Comparison")
        ax.set_ylim(0, 1.2)  # Max is 1.0 (best), plus some margin
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("benchmark_relative.png")
        plt.close()
        
        print(f"Relative performance chart saved to benchmark_relative.png")

def run_benchmark(order_book_class, test_name, operations, num_ops, setup_func=None):
    """Run a benchmark test for a specific order book implementation"""
    # Create fresh order book
    order_book = order_book_class("TEST_ASSET", 100.0)
    
    # Run setup if provided
    if setup_func:
        setup_func(order_book)
    
    # Force garbage collection before test
    gc.collect()
    
    # Run the benchmark
    start_time = time.time()
    for _ in tqdm(range(num_ops), desc=f"{test_name}", leave=False):
        operations(order_book)
    end_time = time.time()
    
    duration = end_time - start_time
    ops_per_sec = num_ops / duration
    
    return duration, ops_per_sec

def benchmark_all_implementations(test_configs):
    """Run all benchmark tests for all implementations"""
    results = BenchmarkResults()
    
    implementations = {
        "Original": OriginalOrderBook,
        "Reworked": ReworkedOrderBook,
        "Claude Optimized": OptimizedOrderBook
    }
    
    for test_name, config in test_configs.items():
        print(f"\nRunning benchmark: {test_name}")
        
        for impl_name, impl_class in implementations.items():
            try:
                print(f"  Implementation: {impl_name}")
                
                duration, ops_per_sec = run_benchmark(
                    impl_class,
                    f"{impl_name}-{test_name}",
                    config["operations"],
                    config["num_ops"],
                    config.get("setup")
                )
                
                results.add_result(impl_name, test_name, duration, ops_per_sec)
                print(f"    Time: {duration:.6f}s, Ops/sec: {ops_per_sec:.2f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                # Continue with next implementation
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define all benchmark tests
    benchmark_tests = {
        # "Add Limit Orders": {
        #     "num_ops": 50000,
        #     "operations": lambda ob: ob.addOrder(
        #         "buy" if random.random() < 0.5 else "sell",
        #         random.uniform(95.0, 105.0),
        #         random.randint(1, 100),
        #         "limit",
        #         f"TEST_ACCOUNT_{random.randint(1, 100)}"
        #     )
        # },
        # "Add Market Orders": {
        #     "num_ops": 50000,
        #     "operations": lambda ob: ob.addOrder(
        #         "buy" if random.random() < 0.5 else "sell",
        #         100.0,
        #         random.randint(1, 100),
        #         "market",
        #         f"TEST_ACCOUNT_{random.randint(1, 100)}"
        #     )
        # },
        "Matching Orders": {
            "num_ops": 100000,
            "operations": lambda ob: (
                ob.addOrder(
                    "buy" if random.random() < 0.5 else "sell",
                    100.0,  # Ensure price crosses the book
                    random.randint(1, 10),
                    "limit",
                    f"TEST_ACCOUNT_{random.randint(1, 100)}"
                ),
                ob.matchBooks()  # Explicitly call matchBooks
            )[0],  # Return the result of addOrder to maintain the same return type
            "setup": lambda ob: [
                ob.addOrder(
                    "buy", 
                    99.5, 
                    1000, 
                    "limit", 
                    "MARKET MAKER"
                ),
                ob.addOrder(
                    "sell", 
                    100.5, 
                    1000, 
                    "limit", 
                    "MARKET MAKER"
                )
            ]
        }
        # "Matching Market Orders": {
        #     "num_ops": 100000,
        #     "operations": lambda ob: (
        #         ob.addOrder(
        #             "buy" if random.random() < 0.5 else "sell",
        #             100.0,  # Ensure price crosses the book
        #             random.randint(1, 10),
        #             "market",
        #             f"TEST_ACCOUNT_{random.randint(1, 100)}"
        #         ),
        #         ob.matchBooks()  # Explicitly call matchBooks
        #     )[0],  # Return the result of addOrder to maintain the same return type
        #     "setup": lambda ob: [
        #         ob.addOrder(
        #             "buy", 
        #             99.5, 
        #             1000, 
        #             "limit", 
        #             "MARKET MAKER"
        #         ),
        #         ob.addOrder(
        #             "sell", 
        #             100.5, 
        #             1000, 
        #             "limit", 
        #             "MARKET MAKER"
        #         )
        #     ]
        # },
        # "Cancel Orders": {
        #     "num_ops": 10000,
        #     "operations": lambda ob: ob.cancelOrdersByAccount(f"TEST_ACCOUNT_{random.randint(1, 100)}"),
        #     "setup": lambda ob: [
        #         ob.addOrder(
        #             "buy" if random.random() < 0.5 else "sell",
        #             random.uniform(95.0, 105.0),
        #             random.randint(1, 100),
        #             "limit",
        #             f"TEST_ACCOUNT_{random.randint(1, 100)}"
        #         ) for _ in range(5000)
        #     ]
        # },
        # "Mixed Workload": {
        #     "num_ops": 10000,
        #     "operations": lambda ob: random.choice([
        #         # 60% add orders
        #         lambda: ob.addOrder(
        #             "buy" if random.random() < 0.5 else "sell",
        #             random.uniform(95.0, 105.0),
        #             random.randint(1, 100),
        #             random.choice(["limit", "limit", "limit", "market"]),  # 75% limit, 25% market
        #             f"TEST_ACCOUNT_{random.randint(1, 100)}"
        #         ),
        #         # 20% cancel orders
        #         lambda: ob.cancelOrdersByAccount(f"TEST_ACCOUNT_{random.randint(1, 100)}"),
        #         # 20% get book info
        #         lambda: (ob.bestBid, ob.bestAsk, ob.getBidSize(), ob.getAskSize())
        #     ])()
        # },
        # "High-Frequency Trading": {
        #     "num_ops": 1000,
        #     "operations": lambda ob: (
        #         # Add small order
        #         ob.addOrder(
        #             "buy" if random.random() < 0.5 else "sell",
        #             random.uniform(99.90, 100.10),
        #             random.randint(1, 5),
        #             "limit",
        #             f"TEST_ACCOUNT_{random.randint(1, 100)}"
        #         ),
        #         # Cancel an order
        #         ob.cancelOrdersByAccount(f"TEST_ACCOUNT_{random.randint(1, 10)}")
        #     )[0]  # Return first result from tuple to avoid error on None return from cancelOrdersByAccount
        # }
    }
    
    # Run all benchmarks
    results = benchmark_all_implementations(benchmark_tests)
    
    # Print summary and generate charts
    results.print_summary()
    results.generate_charts()