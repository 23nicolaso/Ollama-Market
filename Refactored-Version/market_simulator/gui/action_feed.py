import tkinter as tk
from tkinter import ttk
from market_simulator.utils.market_utils import action_queue

_MAX_LINES = 150
_BG = "#1a1a2e"


class ActionFeedFrame(ttk.Frame):
    """Scrolling live log of agent trading decisions, color-coded by direction."""

    def __init__(self, parent):
        super().__init__(parent, relief="groove", borderwidth=1)

        ttk.Label(self, text="Agent Action Feed", font=("Arial", 10, "bold")).pack(pady=(4, 2))

        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(container, orient="vertical")
        self._text = tk.Text(
            container,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=_BG,
            fg="#dddddd",
            font=("Courier", 8),
            height=10,
            width=42,
            yscrollcommand=sb.set,
            relief="flat",
            borderwidth=0,
        )
        sb.config(command=self._text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._text.tag_configure("buy",      foreground="#33dd55")
        self._text.tag_configure("sell",     foreground="#ff5555")
        self._text.tag_configure("risk_on",  foreground="#55aaff")
        self._text.tag_configure("risk_off", foreground="#ffaa33")
        self._text.tag_configure("hft",      foreground="#cc88ff")
        self._text.tag_configure("default",  foreground="#cccccc")

        self.after(200, self._poll)

    def _poll(self):
        try:
            while not action_queue.empty():
                msg = action_queue.get_nowait()
                self._append(msg)
        except Exception:
            pass
        self.after(200, self._poll)

    def _append(self, msg: str):
        self._text.config(state=tk.NORMAL)

        low = msg.lower()
        if "hft" in low:
            tag = "hft"
        elif "risk on" in low:
            tag = "risk_on"
        elif "risk off" in low:
            tag = "risk_off"
        elif "buy" in low:
            tag = "buy"
        elif "sell" in low:
            tag = "sell"
        else:
            tag = "default"

        self._text.insert(tk.END, msg + "\n", tag)

        # Trim to _MAX_LINES to prevent unbounded growth
        line_count = int(self._text.index("end-1c").split(".")[0])
        if line_count > _MAX_LINES:
            self._text.delete("1.0", f"{line_count - _MAX_LINES}.0")

        self._text.see(tk.END)
        self._text.config(state=tk.DISABLED)
