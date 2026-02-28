import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from market_simulator.utils.market_utils import price_history, assets
from market_simulator.config import INITIAL_PRICES

_COLS = 5
_SPARKLINE_POINTS = 200


class SparklineGrid(ttk.Frame):
    """2×N grid of mini price sparklines for every asset, auto-updating every 500 ms."""

    def __init__(self, parent):
        super().__init__(parent)

        num_assets = len(assets)
        rows = (num_assets + _COLS - 1) // _COLS

        self._fig = plt.Figure(
            figsize=(10, 4),
            facecolor="#1a1a2e",
            tight_layout={"pad": 0.4, "h_pad": 0.8, "w_pad": 0.4},
        )
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        gs = GridSpec(rows, _COLS, figure=self._fig)
        self._axes = {}

        for i, asset in enumerate(assets):
            row, col = divmod(i, _COLS)
            ax = self._fig.add_subplot(gs[row, col])
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#333355")
            self._axes[asset] = ax

        self.after(500, self._update)

    def _update(self):
        try:
            for asset, ax in self._axes.items():
                ax.clear()
                ax.set_facecolor("#1a1a2e")
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_color("#333355")

                try:
                    prices = price_history[asset].getLastNPrices(_SPARKLINE_POINTS)
                    if len(prices) < 2:
                        prices = [INITIAL_PRICES[asset], INITIAL_PRICES[asset]]

                    start_price   = prices[0]
                    current_price = prices[-1]
                    pct = (current_price - start_price) / start_price * 100 if start_price != 0 else 0.0

                    color = "#33dd55" if pct >= 0 else "#ff4444"
                    ax.plot(prices, color=color, linewidth=1.0)
                    ax.fill_between(range(len(prices)), prices,
                                    alpha=0.15, color=color)
                    ax.set_title(
                        f"{asset}  {current_price:.2f}  ({pct:+.1f}%)",
                        color="white", fontsize=7, pad=2,
                    )
                except Exception:
                    ax.set_title(asset, color="#888888", fontsize=7)

            self._canvas.draw_idle()
        except Exception:
            pass

        self.after(500, self._update)
