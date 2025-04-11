import time
import random
import numpy as np
from market_simulator.models.order_book import OrderBook as OrderBook
from market_simulator.models.reworked_order_book import OrderBook as ReOrderBook

def generate_random_orders(n_orders):
    """Generate random order parameters for testing"""
    orders = []
    for _ in range(n_orders):
        direction = random.choice(["buy", "sell"])
        price = round(random.uniform(90, 110), 2)
        quantity = random.randint(1, 100)
        order_type = random.choice(["limit", "market"])
        account_id = f"test_account_{random.randint(1, 10)}"
        orders.append((direction, price, quantity, order_type, account_id))
    return orders

def benchmark_orderbook(OrderBookClass, orders, name):
    """Benchmark a specific orderbook implementation"""
    results = {}
    
    # Initialize orderbook
    start_time = time.perf_counter()
    ob = OrderBookClass("TEST", 100.0)
    init_time = time.perf_counter() - start_time
    results["init"] = init_time

    # Test order addition
    order_ids = []
    start_time = time.perf_counter()
    for order in orders:
        order_id = ob.addOrder(*order)
        if order_id:  # Some market orders return 0
            order_ids.append(order_id)
    add_time = time.perf_counter() - start_time
    results["add"] = add_time

    # Test order cancellation
    start_time = time.perf_counter()
    for order_id in order_ids[:len(order_ids)//2]:  # Cancel half the orders
        ob.cancelOrder(order_id)
    cancel_time = time.perf_counter() - start_time
    results["cancel"] = cancel_time

    # Test book matching
    start_time = time.perf_counter()
    ob.matchBooks()
    match_time = time.perf_counter() - start_time
    results["match"] = match_time

    # Test getting bid/ask pairs
    start_time = time.perf_counter()
    for _ in range(1000):
        ob.getBidAskPairs()
    query_time = time.perf_counter() - start_time
    results["query"] = query_time

    print(f"\nResults for {name}:")
    print(f"Initialization time: {results['init']:.6f} seconds")
    print(f"Adding {len(orders)} orders: {results['add']:.6f} seconds")
    print(f"Canceling {len(order_ids)//2} orders: {results['cancel']:.6f} seconds")
    print(f"Matching books: {results['match']:.6f} seconds")
    print(f"1000 bid/ask queries: {results['query']:.6f} seconds")
    
    return results

def run_benchmarks(n_orders=10000, n_trials=5):
    """Run multiple trials of benchmarks and compute statistics"""
    rere_results = []
    re_results = []
    
    print(f"Running {n_trials} trials with {n_orders} orders each...")
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        orders = generate_random_orders(n_orders)
        
        # Benchmark ReReOrderBook
        rere_trial = benchmark_orderbook(OrderBook, orders, "Orderbook")
        rere_results.append(rere_trial)
        
        # Benchmark ReOrderBook
        re_trial = benchmark_orderbook(ReOrderBook, orders, "ReOrderBook")
        re_results.append(re_trial)

    # Compute and display statistics
    operations = ["init", "add", "cancel", "match", "query"]
    print("\nAverage times across trials:")
    print(f"{'Operation':<15} {'ReReOrderBook':>15} {'ReOrderBook':>15} {'Speedup':>10}")
    print("-" * 55)
    
    for op in operations:
        rere_mean = np.mean([r[op] for r in rere_results])
        re_mean = np.mean([r[op] for r in re_results])
        speedup = re_mean / rere_mean if rere_mean > 0 else float('inf')
        
        print(f"{op:<15} {rere_mean:>15.6f} {re_mean:>15.6f} {speedup:>10.2f}x")

if __name__ == "__main__":
    # Run benchmarks with 10,000 orders and 5 trials
    run_benchmarks(n_orders=10000, n_trials=5)