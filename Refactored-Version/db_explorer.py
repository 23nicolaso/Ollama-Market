import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def explore_transactions():
    # Connect to the database
    conn = sqlite3.connect('market_history.db')
    
    while True:
        print("\n=== Market History Explorer ===")
        print("1. View recent transactions")
        print("2. View transactions by asset")
        print("3. View transactions by account")
        print("4. View transaction summary")
        print("5. Plot price history")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            # View recent transactions
            limit = int(input("How many recent transactions to view? "))
            query = f"""
                SELECT timestamp, asset, account_id, direction, quantity, price, order_type
                FROM transactions
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            print("\nRecent Transactions:")
            print(df.to_string())
            
        elif choice == "2":
            # View transactions by asset
            asset = input("Enter asset symbol: ").upper()
            limit = int(input("How many transactions to view? "))
            query = f"""
                SELECT timestamp, account_id, direction, quantity, price, order_type
                FROM transactions
                WHERE asset = '{asset}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            print(f"\nTransactions for {asset}:")
            print(df.to_string())
            
        elif choice == "3":
            # View transactions by account
            account = input("Enter account ID: ")
            limit = int(input("How many transactions to view? "))
            query = f"""
                SELECT timestamp, asset, direction, quantity, price, order_type
                FROM transactions
                WHERE account_id = '{account}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            print(f"\nTransactions for {account}:")
            print(df.to_string())
            
        elif choice == "4":
            # View transaction summary
            query = """
                SELECT asset, 
                       COUNT(*) as total_trades,
                       SUM(CASE WHEN direction = 'buy' THEN quantity ELSE 0 END) as total_buys,
                       SUM(CASE WHEN direction = 'sell' THEN quantity ELSE 0 END) as total_sells,
                       AVG(price) as avg_price
                FROM transactions
                GROUP BY asset
            """
            df = pd.read_sql_query(query, conn)
            print("\nTransaction Summary:")
            print(df.to_string())
            
        elif choice == "5":
            # Plot price history
            asset = input("Enter asset symbol: ").upper()
            query = f"""
                SELECT timestamp, price
                FROM transactions
                WHERE asset = '{asset}'
                ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn)
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['price'])
            plt.title(f'Price History for {asset}')
            plt.xlabel('Timestamp')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()
            
        elif choice == "6":
            break
            
        else:
            print("Invalid choice. Please try again.")
    
    conn.close()

if __name__ == "__main__":
    explore_transactions()
