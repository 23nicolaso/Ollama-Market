import tkinter as tk
from tkinter import ttk
from market_simulator.utils.market_utils import agent_registry


class LeaderboardFrame(ttk.Frame):
    """Real-time P&L leaderboard showing all registered funds sorted by portfolio value."""

    def __init__(self, parent):
        super().__init__(parent, relief="groove", borderwidth=1)

        ttk.Label(self, text="P&L Leaderboard", font=("Arial", 10, "bold")).pack(pady=(4, 2))

        cols = ("rank", "fund", "value", "pnl", "pct")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", height=11)
        self._tree.heading("rank",  text="#")
        self._tree.heading("fund",  text="Fund")
        self._tree.heading("value", text="Portfolio")
        self._tree.heading("pnl",   text="P&L")
        self._tree.heading("pct",   text="P&L %")

        self._tree.column("rank",  width=28,  anchor="center", stretch=False)
        self._tree.column("fund",  width=130, anchor="w")
        self._tree.column("value", width=105, anchor="e")
        self._tree.column("pnl",   width=90,  anchor="e")
        self._tree.column("pct",   width=65,  anchor="e")

        self._tree.tag_configure("profit",  foreground="#33aa33")
        self._tree.tag_configure("loss",    foreground="#cc3333")
        self._tree.tag_configure("neutral", foreground="#aaaaaa")

        sb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

        self.after(1000, self._update)

    def _update(self):
        try:
            entries = []
            for name, (agent, initial_cash) in agent_registry.items():
                try:
                    value = agent.account.getValue()
                    pnl   = value - initial_cash
                    pct   = (pnl / initial_cash * 100) if initial_cash > 0 else 0.0
                    entries.append((name, value, pnl, pct))
                except Exception:
                    pass

            entries.sort(key=lambda x: x[1], reverse=True)

            for item in self._tree.get_children():
                self._tree.delete(item)

            for rank, (name, value, pnl, pct) in enumerate(entries, start=1):
                tag = "profit" if pnl > 0 else "loss" if pnl < 0 else "neutral"
                self._tree.insert(
                    "", "end",
                    values=(
                        rank,
                        name,
                        f"${value:>12,.0f}",
                        f"${pnl:>+10,.0f}",
                        f"{pct:>+.1f}%",
                    ),
                    tags=(tag,),
                )
        except Exception:
            pass

        self.after(1000, self._update)
