import tkinter as tk
from tkinter import ttk
from market_simulator.gui.charts import ChartFrame
from market_simulator.gui.news_feed import NewsFeedFrame
from market_simulator.utils.market_utils import assets, last_prices, reset_markets, price_history, initial_prices
from market_simulator.utils.news_generator import generate_news_thread, generate_chat_thread
import tkinter.messagebox as messagebox
import webbrowser


class MainWindow:
    """
    Lean Tkinter control window.  Heavy visualisations (leaderboard, economy
    dashboard, sparklines, action feed) live in the web frontend at port 8000.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Market Simulator — Control Panel")
        self.root.geometry("1100x620")

        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        for c in (0, 1, 2, 3):
            self.frame.columnconfigure(c, weight=1)
        self.frame.columnconfigure(4, weight=0)
        for r in (1, 2, 3):
            self.frame.rowconfigure(r, weight=1)

        # ── Toolbar ────────────────────────────────────────────────────────────
        self.asset_var = tk.StringVar()
        self.asset_dropdown = ttk.Combobox(
            self.frame, textvariable=self.asset_var, values=assets, width=12
        )
        self.asset_dropdown.set(assets[0])
        self.asset_dropdown.grid(row=0, column=0, sticky="w", padx=4, pady=4)

        self.chart_length_label = ttk.Label(self.frame, text="Length")
        self.chart_length_label.grid(row=0, column=1, sticky="w")

        self.number_var = tk.IntVar(value=200)
        self.number_spinbox = tk.Spinbox(
            self.frame, from_=10, to=2000, textvariable=self.number_var, width=6
        )
        self.number_spinbox.grid(row=0, column=1, sticky="e")

        self.gen_news_button = ttk.Button(
            self.frame, text="Generate News", command=generate_news_thread
        )
        self.gen_news_button.grid(row=0, column=2, sticky="w", padx=4)

        self.gen_chat_button = ttk.Button(
            self.frame, text="Generate Chat", command=generate_chat_thread
        )
        self.gen_chat_button.grid(row=0, column=2, sticky="e")

        self.web_button = ttk.Button(
            self.frame, text="Open Web UI ↗",
            command=lambda: webbrowser.open("http://localhost:8000")
        )
        self.web_button.grid(row=0, column=3, sticky="w", padx=4)

        self.wipe_button = ttk.Button(
            self.frame, text="Wipe DB", command=self.wipe_database
        )
        self.wipe_button.grid(row=0, column=4, sticky="e", padx=4)

        # ── Chart (single asset, throttled in main.py) ────────────────────────
        self.chart_frame = ChartFrame(self.frame)
        self.chart_frame.grid(row=1, column=0, rowspan=3, columnspan=3,
                               sticky="nsew", padx=2, pady=2)

        # ── News feed ─────────────────────────────────────────────────────────
        self.news_feed_frame = NewsFeedFrame(self.frame)
        self.news_feed_frame.grid(row=1, column=3, rowspan=2, columnspan=1,
                                   sticky="nsew", padx=2, pady=2)

        # ── Price table ───────────────────────────────────────────────────────
        self._build_price_table()

        self.asset_dropdown.bind("<<ComboboxSelected>>", self._on_asset_change)

    def _build_price_table(self):
        self.price_frame = ttk.Frame(self.frame)
        self.price_frame.grid(row=3, column=3, columnspan=1,
                               sticky="nsew", padx=2, pady=2)

        self.price_tree = ttk.Treeview(
            self.price_frame,
            columns=("Asset", "Price", "Chg%"),
            show="headings",
            height=10,
        )
        self.price_tree.heading("Asset", text="Asset")
        self.price_tree.heading("Price", text="Price")
        self.price_tree.heading("Chg%",  text="Chg %")
        self.price_tree.column("Asset", width=100, anchor="center")
        self.price_tree.column("Price", width=80,  anchor="e")
        self.price_tree.column("Chg%",  width=60,  anchor="e")
        self.price_tree.tag_configure("up",   foreground="#33aa33")
        self.price_tree.tag_configure("down", foreground="#cc3333")
        self.price_tree.pack(fill=tk.BOTH, expand=True)

    # ── Update callbacks ─────────────────────────────────────────────────────

    def update_prices(self):
        for item in self.price_tree.get_children():
            self.price_tree.delete(item)
        for asset in assets:
            price = last_prices[asset]
            init  = initial_prices[asset]
            chg   = (price - init) / init * 100 if init != 0 else 0.0
            tag   = "up" if chg >= 0 else "down"
            self.price_tree.insert(
                "", "end",
                values=(asset, f"{price:.2f}", f"{chg:+.1f}%"),
                tags=(tag,),
            )

    def update_sentiments(self, _retail_trader):
        """No-op — sentiment data is now shown in the web UI."""
        pass

    def update(self):
        """Redraw the matplotlib chart. Called at throttled rate from main.py."""
        self.chart_frame.update_chart(self.asset_var.get(), self.number_var.get())

    def _on_asset_change(self, _event):
        self.chart_frame.update_chart(self.asset_var.get(), self.number_var.get())
        self.news_feed_frame.clear_chat()

    def wipe_database(self):
        if messagebox.askyesno(
            "Confirm",
            "Wipe the price history database and reset all markets?",
        ):
            from market_simulator.utils.db_utils import wipe_db
            from market_simulator.utils.market_utils import reset_markets

            wipe_db()
            reset_markets()
            for asset in price_history:
                price_history[asset].clear()
                last_prices[asset] = initial_prices[asset]

            messagebox.showinfo("Done", "Database wiped and markets reset.")
