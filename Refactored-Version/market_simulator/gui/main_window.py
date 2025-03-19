import tkinter as tk
from tkinter import ttk
from market_simulator.gui.charts import ChartFrame
from market_simulator.gui.news_feed import NewsFeedFrame
from market_simulator.utils.market_utils import assets, last_prices, estimateUnderlyingValue, reset_markets, price_history, initial_prices
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread
import tkinter.messagebox as messagebox

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Market Simulator")
        self.root.geometry("2000x800")

        # Create top frame for controls
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add asset selector
        self.asset_var = tk.StringVar(value=assets[0])
        self.asset_selector = ttk.Combobox(
            self.control_frame, 
            textvariable=self.asset_var,
            values=assets
        )
        self.asset_selector.pack(side=tk.LEFT, padx=5)
        self.asset_selector.bind('<<ComboboxSelected>>', self.on_asset_change)

        # Add number selector
        self.number_var = tk.StringVar(value="10")
        self.number_selector = ttk.Entry(
            self.control_frame,
            textvariable=self.number_var,
            width=10
        )
        self.number_selector.pack(side=tk.LEFT, padx=5)

        # Add wipe database button
        self.wipe_button = ttk.Button(
            self.control_frame,
            text="Wipe DB",
            command=self.wipe_database
        )
        self.wipe_button.pack(side=tk.RIGHT, padx=5)

        # Create main frame
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create the dropdown menu
        self.asset_var = tk.StringVar()
        self.asset_dropdown = ttk.Combobox(self.frame, textvariable=self.asset_var, values=assets)
        self.asset_dropdown.set(assets[0])  # Set default value
        self.asset_dropdown.grid(row=0, column=0, rowspan=1, columnspan=1)

        # Create a number selection menu
        self.chart_length_label = ttk.Label(self.frame, text="Chart Length")
        self.chart_length_label.grid(row=0, column=1, rowspan=1, columnspan=1)

        self.number_var = tk.IntVar(value=10)
        self.number_spinbox = tk.Spinbox(self.frame, from_=1, to=1000, textvariable=self.number_var)
        self.number_spinbox.grid(row=0, column=2, rowspan=1, columnspan=1)

        # Add generate news and chat buttons
        self.gen_news_button = ttk.Button(self.frame, text="Generate News", command=generate_news_thread)
        self.gen_news_button.grid(row=0, column=3, rowspan=1, columnspan=1)
        
        self.gen_chat_button = ttk.Button(self.frame, text="Generate Chat", command=generate_chat_thread)
        self.gen_chat_button.grid(row=0, column=4, rowspan=1, columnspan=1)

        # Create chart frame
        self.chart_frame = ChartFrame(self.frame)
        self.chart_frame.grid(row=1, column=0, rowspan=3, columnspan=3)

        # Create news feed frame
        self.news_feed_frame = NewsFeedFrame(self.frame)
        self.news_feed_frame.grid(row=1, column=3, rowspan=3, columnspan=2)

        # Create price table
        self.create_price_table()

        # Create sentiment table
        self.create_sentiment_table()

        # Bind events
        self.asset_dropdown.bind("<<ComboboxSelected>>", self.on_asset_change)

    def create_price_table(self):
        # Create a frame for the price table
        self.price_frame = ttk.Frame(self.frame)
        self.price_frame.grid(row=1, column=5, rowspan=1, columnspan=1, sticky="nsew")

        # Create and set up the treeview for price display
        self.price_tree = ttk.Treeview(self.price_frame, columns=("Asset", "Price", "Fair Value"), show="headings")
        self.price_tree.heading("Asset", text="Asset")
        self.price_tree.heading("Price", text="Price")
        self.price_tree.heading("Fair Value", text="Fair Value")
        self.price_tree.column("Asset", width=150, anchor="center")
        self.price_tree.column("Price", width=100, anchor="center")
        self.price_tree.column("Fair Value", width=100, anchor="center")
        self.price_tree.pack(fill=tk.BOTH, expand=True)

    def create_sentiment_table(self):
        # Create a frame for the sentiment table
        self.sentiment_frame = ttk.Frame(self.frame)
        self.sentiment_frame.grid(row=2, column=5, rowspan=2, columnspan=1, sticky="nsew")

        # Create and set up the treeview for sentiment display
        self.sentiment_tree = ttk.Treeview(self.sentiment_frame, columns=("Asset", "Sentiment"), show="headings")
        self.sentiment_tree.heading("Asset", text="Asset")
        self.sentiment_tree.heading("Sentiment", text="Sentiment")
        self.sentiment_tree.column("Asset", width=150, anchor="center")
        self.sentiment_tree.column("Sentiment", width=200, anchor="center")
        self.sentiment_tree.pack(fill=tk.BOTH, expand=True)
        
    def update_prices(self):
        for item in self.price_tree.get_children():
            self.price_tree.delete(item)
        for asset in assets:
            price = last_prices[asset]
            fair_value = estimateUnderlyingValue(asset)
            self.price_tree.insert("", "end", values=(asset, f"{price:.2f}", f"{fair_value:.2f}"))

    def update_sentiments(self, retail_trader):
        for item in self.sentiment_tree.get_children():
            self.sentiment_tree.delete(item)
        for asset in assets:
            sentiment = retail_trader.retailSentimentScore[asset]
            self.sentiment_tree.insert("", "end", values=(asset, f"{sentiment:.2f}"))

    def on_asset_change(self, event):
        self.chart_frame.update_chart(self.asset_var.get(), self.number_var.get())
        self.news_feed_frame.clear_chat()

    def update(self):
        self.chart_frame.update_chart(self.asset_var.get(), self.number_var.get())
        self.root.update()

    def wipe_database(self):
        """Wipes the price history database and resets markets after confirmation"""
        if messagebox.askyesno("Confirm", "Are you sure you want to wipe the price history database and reset all markets?"):
            from market_simulator.utils.db_utils import wipe_db
            from market_simulator.utils.market_utils import reset_markets, price_history, last_prices, initial_prices
            
            # Wipe database
            wipe_db()
            
            # Reset all markets and prices
            reset_markets()
            
            # Reset price history and last prices to initial values
            for asset in price_history:
                price_history[asset].clear()
                last_prices[asset] = initial_prices[asset]
            
            messagebox.showinfo("Success", "Database has been wiped and markets reset") 