import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from zoneinfo import ZoneInfo # Required by db_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import colorsys

# Attempt to import db_utils
try:
    import db_utils
except ImportError:
    messagebox.showerror("Error", "Could not import db_utils.py. Make sure it's in the same directory or Python path.")
    exit()
except Exception as e:
     messagebox.showerror("Error", f"An error occurred during db_utils import: {e}")
     exit()

# Ensure the database manager is initialized (creates the singleton)
# This implicitly starts the background thread if not already started.
# It doesn't init the schema here, only the manager object.
try:
    db_man = db_utils.DatabaseManager()
    # Optionally initialize the DB schema if it might not exist
    # db_man.init_db() # Uncomment if you want the GUI to ensure table exists
except Exception as e:
     messagebox.showerror("Error", f"Failed to initialize DatabaseManager: {e}")
     exit()


class DbExplorerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Market History Explorer")
        self.geometry("1200x800")

        # --- Configuration ---
        self.datetime_format = "%Y-%m-%d %H:%M:%S"
        self.display_tz = ZoneInfo('UTC') # Display times in UTC to match DB

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.price_frame = ttk.Frame(self.notebook)
        self.trade_frame = ttk.Frame(self.notebook)
        self.impact_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.price_frame, text="Price History")
        self.notebook.add(self.trade_frame, text="Trade History")
        self.notebook.add(self.impact_frame, text="Price Impact Analysis")

        # --- Widgets ---
        self._create_price_widgets()
        self._create_trade_widgets()
        self._create_impact_widgets()
        self._populate_assets() # Populate asset list initially
        self.agent_colors = {}  # Dictionary to store agent colors
        self._initialize_agent_colors()

    def _initialize_agent_colors(self):
        """Initialize a set of distinct colors for agents"""
        # Generate a set of distinct colors using HSV color space
        num_colors = 20  # Maximum number of distinct colors
        for i in range(num_colors):
            # Use golden ratio to distribute colors evenly in HSV space
            golden_ratio = 0.618033988749895
            h = (i * golden_ratio) % 1.0
            # Convert to RGB with high saturation and value for good contrast
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.8)
            # Convert to hex color
            color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            self.agent_colors[f'AGENT_{i}'] = color

    def _get_agent_color(self, agent_id):
        """Get a color for an agent, generating a new one if needed"""
        if agent_id not in self.agent_colors:
            # Generate a new color if we run out of predefined ones
            h = random.random()
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.8)
            color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            self.agent_colors[agent_id] = color
        return self.agent_colors[agent_id]

    def _create_price_widgets(self):
        # Frame for controls
        control_frame = ttk.Frame(self.price_frame, padding="10")
        control_frame.pack(fill=tk.X)

        # Asset Selection
        ttk.Label(control_frame, text="Asset:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.asset_combo = ttk.Combobox(control_frame, width=15)
        self.asset_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        # Add a refresh button for assets
        refresh_btn = ttk.Button(control_frame, text="Refresh Assets", command=self._populate_assets)
        refresh_btn.grid(row=0, column=2, padx=5, pady=5)


        # Date Range
        ttk.Label(control_frame, text="Start (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.start_entry = ttk.Entry(control_frame, width=25)
        self.start_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(control_frame, text="End (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.end_entry = ttk.Entry(control_frame, width=25)
        self.end_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # Fetch Button
        fetch_button = ttk.Button(control_frame, text="Fetch Price History", command=self._fetch_price_data)
        fetch_button.grid(row=3, column=0, columnspan=3, pady=10)

        control_frame.columnconfigure(1, weight=1) # Make entry/combo expand

        # --- Results Area ---
        result_frame = ttk.Frame(self.price_frame, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview for displaying results
        columns = ('price', 'timestamp')
        self.price_tree = ttk.Treeview(result_frame, columns=columns, show='headings')
        self.price_tree.heading('price', text='Price')
        self.price_tree.heading('timestamp', text='Timestamp (UTC)')

        # Adjust column widths (optional)
        self.price_tree.column('price', width=100, anchor=tk.E) # Align price right
        self.price_tree.column('timestamp', width=200)

        # Scrollbars for Treeview
        vsb = ttk.Scrollbar(result_frame, orient="vertical", command=self.price_tree.yview)
        hsb = ttk.Scrollbar(result_frame, orient="horizontal", command=self.price_tree.xview)
        self.price_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.price_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)

    def _create_trade_widgets(self):
        # Frame for controls
        control_frame = ttk.Frame(self.trade_frame, padding="10")
        control_frame.pack(fill=tk.X)

        # Asset Selection
        ttk.Label(control_frame, text="Asset:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.trade_asset_combo = ttk.Combobox(control_frame, width=15)
        self.trade_asset_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # Account ID filter
        ttk.Label(control_frame, text="Account ID:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.account_entry = ttk.Entry(control_frame, width=15)
        self.account_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        # Date Range
        ttk.Label(control_frame, text="Start (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.trade_start_entry = ttk.Entry(control_frame, width=25)
        self.trade_start_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(control_frame, text="End (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.trade_end_entry = ttk.Entry(control_frame, width=25)
        self.trade_end_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky=tk.EW)

        # Fetch Button
        fetch_button = ttk.Button(control_frame, text="Fetch Trade History", command=self._fetch_trade_data)
        fetch_button.grid(row=3, column=0, columnspan=4, pady=10)

        control_frame.columnconfigure(1, weight=1)

        # --- Results Area ---
        result_frame = ttk.Frame(self.trade_frame, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview for displaying results
        columns = ('price', 'quantity', 'buyer', 'seller', 'timestamp')
        self.trade_tree = ttk.Treeview(result_frame, columns=columns, show='headings')
        self.trade_tree.heading('price', text='Price')
        self.trade_tree.heading('quantity', text='Quantity')
        self.trade_tree.heading('buyer', text='Buyer')
        self.trade_tree.heading('seller', text='Seller')
        self.trade_tree.heading('timestamp', text='Timestamp (UTC)')

        self.trade_tree.column('price', width=100, anchor=tk.E)
        self.trade_tree.column('quantity', width=100, anchor=tk.E)
        self.trade_tree.column('buyer', width=150)
        self.trade_tree.column('seller', width=150)
        self.trade_tree.column('timestamp', width=200)

        # Configure tag colors
        self.trade_tree.tag_configure('price_up', foreground='#26a69a')  # Green
        self.trade_tree.tag_configure('price_down', foreground='#ef5350')  # Red

        # Scrollbars for Treeview
        vsb = ttk.Scrollbar(result_frame, orient="vertical", command=self.trade_tree.yview)
        hsb = ttk.Scrollbar(result_frame, orient="horizontal", command=self.trade_tree.xview)
        self.trade_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.trade_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)

    def _create_impact_widgets(self):
        # Frame for controls
        control_frame = ttk.Frame(self.impact_frame, padding="10")
        control_frame.pack(fill=tk.X)

        # Asset Selection
        ttk.Label(control_frame, text="Asset:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.impact_asset_combo = ttk.Combobox(control_frame, width=15)
        self.impact_asset_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # Date Range
        ttk.Label(control_frame, text="Start (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.impact_start_entry = ttk.Entry(control_frame, width=25)
        self.impact_start_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(control_frame, text="End (UTC, YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.impact_end_entry = ttk.Entry(control_frame, width=25)
        self.impact_end_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # Window Size (in minutes)
        ttk.Label(control_frame, text="Impact Window (minutes):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.window_size = ttk.Entry(control_frame, width=10)
        self.window_size.insert(0, "30")  # Default 30 minutes
        self.window_size.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # Fetch Button
        fetch_button = ttk.Button(control_frame, text="Analyze Price Impact", command=self._analyze_price_impact)
        fetch_button.grid(row=4, column=0, columnspan=3, pady=10)

        control_frame.columnconfigure(1, weight=1)

        # --- Results Area ---
        result_frame = ttk.Frame(self.impact_frame, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure for the plot
        self.impact_figure = Figure(figsize=(10, 6))
        self.impact_canvas = FigureCanvasTkAgg(self.impact_figure, master=result_frame)
        self.impact_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Table for detailed results
        self.impact_tree = ttk.Treeview(result_frame, columns=('agent', 'total_volume', 'avg_price_impact', 'max_price_impact'), show='headings')
        self.impact_tree.heading('agent', text='Agent')
        self.impact_tree.heading('total_volume', text='Total Volume')
        self.impact_tree.heading('avg_price_impact', text='Avg Price Impact')
        self.impact_tree.heading('max_price_impact', text='Max Price Impact')

        self.impact_tree.column('agent', width=150)
        self.impact_tree.column('total_volume', width=100, anchor=tk.E)
        self.impact_tree.column('avg_price_impact', width=100, anchor=tk.E)
        self.impact_tree.column('max_price_impact', width=100, anchor=tk.E)

        self.impact_tree.pack(fill=tk.BOTH, expand=True)

    def _populate_assets(self):
        """Fetches distinct asset names from the DB and populates the comboboxes."""
        try:
            session = db_man.Session()
            # Query distinct assets directly using SQLAlchemy
            asset_tuples = session.query(db_utils.PriceHistory.asset).distinct().order_by(db_utils.PriceHistory.asset).all()
            assets = [a[0] for a in asset_tuples] # Extract string names
            self.asset_combo['values'] = assets
            self.trade_asset_combo['values'] = assets
            self.impact_asset_combo['values'] = assets
            if assets:
                self.asset_combo.current(0) # Select the first asset by default
                self.trade_asset_combo.current(0)
                self.impact_asset_combo.current(0)
        except Exception as e:
            # Handle case where table might not exist yet
            if "no such table: price_history" in str(e).lower():
                 messagebox.showwarning("DB Warning", "Price history table not found. Run init_db() or store some data first.")
                 self.asset_combo['values'] = [] # Clear list if table doesn't exist
                 self.trade_asset_combo['values'] = []
                 self.impact_asset_combo['values'] = []
            else:
                messagebox.showerror("DB Error", f"Failed to fetch assets: {e}")
                self.asset_combo['values'] = []
                self.trade_asset_combo['values'] = []
                self.impact_asset_combo['values'] = []
        finally:
            if 'session' in locals() and session:
                db_man.Session.remove() # Use the scoped session's remove method

    def _parse_datetime(self, dt_string):
        """Parses user input string into a timezone-aware datetime object (UTC)."""
        if not dt_string:
            return None
        try:
            # Parse as naive datetime first
            naive_dt = datetime.strptime(dt_string, self.datetime_format)
            # Assume the user entered UTC time and make it timezone-aware
            return naive_dt.replace(tzinfo=ZoneInfo('UTC'))
        except ValueError:
            messagebox.showerror("Input Error", f"Invalid datetime format: '{dt_string}'.\nPlease use YYYY-MM-DD HH:MM:SS.")
            return None # Indicate parsing failure

    def _fetch_price_data(self):
        """Fetches price history data and displays it in the Treeview."""
        asset = self.asset_combo.get()
        start_str = self.start_entry.get()
        end_str = self.end_entry.get()

        if not asset:
            messagebox.showwarning("Input Missing", "Please select or enter an asset.")
            return

        start_time = self._parse_datetime(start_str)
        # If start time parsing failed, _parse_datetime shows error and returns None
        if start_str and start_time is None:
             return

        end_time = self._parse_datetime(end_str)
        # If end time parsing failed, _parse_datetime shows error and returns None
        if end_str and end_time is None:
            return

        # Clear previous results
        for item in self.price_tree.get_children():
            self.price_tree.delete(item)

        try:
            # Use the convenience function from db_utils
            history = db_utils.get_price_history(asset, start_time, end_time)

            if not history:
                messagebox.showinfo("No Results", f"No price history found for '{asset}' within the specified criteria.")
                return

            # Populate the Treeview
            for price, timestamp in history:
                # Format timestamp for display
                # Convert to display timezone if needed, here we keep UTC
                display_ts = timestamp.astimezone(self.display_tz).strftime(self.datetime_format)
                # Ensure price is formatted reasonably
                price_str = f"{price:.8f}" # Adjust precision as needed
                self.price_tree.insert('', tk.END, values=(price_str, display_ts))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while fetching data: {e}")

    def _fetch_trade_data(self):
        """Fetches trade history data and displays it in the Treeview."""
        asset = self.trade_asset_combo.get()
        account_id = self.account_entry.get()
        start_str = self.trade_start_entry.get()
        end_str = self.trade_end_entry.get()

        if not asset:
            messagebox.showwarning("Input Missing", "Please select or enter an asset.")
            return

        start_time = self._parse_datetime(start_str)
        if start_str and start_time is None:
             return

        end_time = self._parse_datetime(end_str)
        if end_str and end_time is None:
            return

        # Clear previous results
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)

        try:
            trades = db_utils.get_trade_history(asset, start_time, end_time, account_id if account_id else None)

            if not trades:
                messagebox.showinfo("No Results", f"No trade history found for '{asset}' within the specified criteria.")
                return

            # Get price history for comparison
            prices = db_utils.get_price_history(asset, start_time, end_time)
            price_dict = {p[1]: p[0] for p in prices}  # Map timestamps to prices

            # Sort trades by timestamp
            trades.sort(key=lambda x: x[4])  # Sort by timestamp

            # Track previous price for comparison
            prev_price = None

            for price, quantity, buyer_id, seller_id, timestamp in trades:
                display_ts = timestamp.astimezone(self.display_tz).strftime(self.datetime_format)
                price_str = f"{price:.8f}"
                quantity_str = f"{quantity:.8f}"

                # Determine price movement tag
                price_tag = ''
                if prev_price is not None:
                    if price > prev_price:
                        price_tag = 'price_up'
                    elif price < prev_price:
                        price_tag = 'price_down'
                prev_price = price

                # Insert the trade with appropriate tags
                item = self.trade_tree.insert('', tk.END, values=(
                    price_str, quantity_str, buyer_id, seller_id, display_ts
                ), tags=(price_tag,))

                # Set agent colors
                self.trade_tree.item(item, tags=(price_tag,))
                self.trade_tree.tag_configure(f'buyer_{buyer_id}', foreground=self._get_agent_color(buyer_id))
                self.trade_tree.tag_configure(f'seller_{seller_id}', foreground=self._get_agent_color(seller_id))
                self.trade_tree.item(item, tags=(price_tag, f'buyer_{buyer_id}', f'seller_{seller_id}'))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while fetching trade data: {e}")

    def _analyze_price_impact(self):
        """Analyzes price impact of trades during the selected period"""
        asset = self.impact_asset_combo.get()
        start_str = self.impact_start_entry.get()
        end_str = self.impact_end_entry.get()
        window_minutes = int(self.window_size.get())

        if not asset:
            messagebox.showwarning("Input Missing", "Please select or enter an asset.")
            return

        start_time = self._parse_datetime(start_str)
        if start_str and start_time is None:
            return

        end_time = self._parse_datetime(end_str)
        if end_str and end_time is None:
            return

        try:
            # Get trade history
            trades = db_utils.get_trade_history(asset, start_time, end_time)
            if not trades:
                messagebox.showinfo("No Results", f"No trades found for '{asset}' within the specified criteria.")
                return

            # Get price history
            prices = db_utils.get_price_history(asset, start_time, end_time)
            if not prices:
                messagebox.showinfo("No Results", f"No price history found for '{asset}' within the specified criteria.")
                return

            # Convert to numpy arrays for easier processing
            trade_times = np.array([t[4].timestamp() for t in trades])
            trade_prices = np.array([t[0] for t in trades])
            trade_quantities = np.array([t[1] for t in trades])
            trade_buyers = np.array([t[2] for t in trades])
            trade_sellers = np.array([t[3] for t in trades])

            price_times = np.array([p[1].timestamp() for p in prices])
            price_values = np.array([p[0] for p in prices])

            # Calculate price impact for each trade
            agent_impacts = {}
            window_seconds = window_minutes * 60

            for i, (time, price, quantity, buyer, seller) in enumerate(zip(trade_times, trade_prices, trade_quantities, trade_buyers, trade_sellers)):
                # Find price before and after the trade
                before_idx = np.searchsorted(price_times, time - window_seconds, side='left')
                after_idx = np.searchsorted(price_times, time + window_seconds, side='right')
                
                if before_idx < len(price_values) and after_idx < len(price_values):
                    price_before = price_values[before_idx]
                    price_after = price_values[after_idx]
                    
                    # Calculate price impact
                    impact = (price_after - price_before) / price_before * 100
                    
                    # Update buyer's impact
                    if buyer not in agent_impacts:
                        agent_impacts[buyer] = {'impacts': [], 'volume': 0}
                    agent_impacts[buyer]['impacts'].append(impact)
                    agent_impacts[buyer]['volume'] += quantity
                    
                    # Update seller's impact
                    if seller not in agent_impacts:
                        agent_impacts[seller] = {'impacts': [], 'volume': 0}
                    agent_impacts[seller]['impacts'].append(-impact)  # Negative impact for sellers
                    agent_impacts[seller]['volume'] += quantity

            # Clear previous results
            for item in self.impact_tree.get_children():
                self.impact_tree.delete(item)

            # Update the table with results
            for agent, data in agent_impacts.items():
                if data['impacts']:
                    avg_impact = np.mean(data['impacts'])
                    max_impact = np.max(np.abs(data['impacts']))
                    self.impact_tree.insert('', tk.END, values=(
                        agent,
                        f"{data['volume']:.2f}",
                        f"{avg_impact:.2f}%",
                        f"{max_impact:.2f}%"
                    ))

            # Create bar chart
            self.impact_figure.clear()
            ax = self.impact_figure.add_subplot(111)
            
            agents = list(agent_impacts.keys())
            avg_impacts = [np.mean(data['impacts']) for data in agent_impacts.values()]
            
            bars = ax.bar(agents, avg_impacts)
            ax.set_title(f'Average Price Impact by Agent ({asset})')
            ax.set_ylabel('Price Impact (%)')
            ax.set_xlabel('Agent')
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom')
            
            self.impact_figure.tight_layout()
            self.impact_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while analyzing price impact: {e}")

if __name__ == "__main__":
    app = DbExplorerApp()
    app.mainloop()
    # Optional: Gracefully shutdown the db manager's background thread if needed
    # This might be important if the GUI is the *only* thing using db_utils
    print("Shutting down database manager...")
    db_utils.shutdown_db()
    print("Shutdown complete.")