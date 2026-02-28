import tkinter as tk
from tkinter import ttk
from market_simulator.utils.market_utils import markov_model
from market_simulator.models.marketState import (
    MarketTrend, InflationState, InterestRateState,
    UnemploymentState, EconomicGrowthState, Sector
)

# Maps each enum state to a background color for its indicator label
_STATE_COLORS = {
    MarketTrend.BEAR:              "#cc3333",
    MarketTrend.NEUTRAL:           "#888888",
    MarketTrend.BULL:              "#33aa33",
    InflationState.LOW:            "#33aa33",
    InflationState.MODERATE:       "#cc8800",
    InflationState.HIGH:           "#cc3333",
    InterestRateState.LOW:         "#33aa33",
    InterestRateState.MODERATE:    "#cc8800",
    InterestRateState.HIGH:        "#cc3333",
    UnemploymentState.LOW:         "#33aa33",
    UnemploymentState.MODERATE:    "#cc8800",
    UnemploymentState.HIGH:        "#cc3333",
    EconomicGrowthState.RECESSION: "#cc3333",
    EconomicGrowthState.SLOW:      "#cc6600",
    EconomicGrowthState.MODERATE:  "#cc8800",
    EconomicGrowthState.RAPID:     "#33aa33",
}

_DIMENSIONS = [
    ("Market Trend",   "market_trend"),
    ("Inflation",      "inflation"),
    ("Interest Rate",  "interest_rate"),
    ("Unemployment",   "unemployment"),
    ("Growth",         "economic_growth"),
]

# Map Sector enum to asset name string shown in the market
_SECTOR_ASSET_MAP = {
    Sector.TECHNOLOGY: "TECHNOLOGY",
    Sector.HEALTHCARE: "HEALTHCARE",
    Sector.FINANCIAL:  "FINANCIAL",
    Sector.ENERGY:     "ENERGY",
    Sector.CONSUMER:   "CONSUMER",
    Sector.INDUSTRIAL: "INDUSTRIAL",
}


class EconomyDashboard(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, relief="groove", borderwidth=1)

        ttk.Label(self, text="Economy State", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(4, 2)
        )

        self._dim_labels = {}
        for i, (display_name, key) in enumerate(_DIMENSIONS, start=1):
            ttk.Label(self, text=display_name + ":", anchor="w").grid(
                row=i, column=0, sticky="w", padx=(6, 2)
            )
            val = tk.Label(
                self, text="—", width=11, anchor="center",
                font=("Arial", 9, "bold"), bg="#555555", fg="white", relief="flat"
            )
            val.grid(row=i, column=1, sticky="ew", padx=(0, 6), pady=1)
            self._dim_labels[key] = val

        separator_row = len(_DIMENSIONS) + 1
        ttk.Separator(self, orient="horizontal").grid(
            row=separator_row, column=0, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(self, text="Sector Performance", font=("Arial", 9, "bold")).grid(
            row=separator_row + 1, column=0, columnspan=2
        )

        self._sector_labels = {}
        for j, (sector, asset_name) in enumerate(sorted(_SECTOR_ASSET_MAP.items(), key=lambda x: x[1])):
            row = separator_row + 2 + j
            ttk.Label(self, text=asset_name[:10] + ":", anchor="w").grid(
                row=row, column=0, sticky="w", padx=(6, 2)
            )
            pct_lbl = tk.Label(self, text="+0.00%", width=8, anchor="e",
                               font=("Courier", 9))
            pct_lbl.grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=1)
            self._sector_labels[sector] = pct_lbl

        self.after(1000, self._update)

    def _update(self):
        try:
            state = markov_model.get_economy_state()

            for key, label in self._dim_labels.items():
                val = state[key]
                color = _STATE_COLORS.get(val, "#555555")
                label.config(text=val.name, bg=color)

            for sector, label in self._sector_labels.items():
                perf = state["sector_performance"].get(sector, 0.0)
                pct_str = f"{perf * 100:+.2f}%"
                color = "#33aa33" if perf > 0 else "#cc3333" if perf < 0 else "#888888"
                label.config(text=pct_str, fg=color)
        except Exception:
            pass

        self.after(1000, self._update)
