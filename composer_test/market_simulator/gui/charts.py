import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from market_simulator.utils.market_utils import price_history

class ChartFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create a figure for the chart
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_chart(self, asset, num_points=500):
        self.ax.clear()
        try:
            self.ax.plot(price_history[asset][-num_points:])
        except:
            self.ax.plot(price_history[asset][-200:])
        self.ax.set_title(f"{asset} Price History")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.canvas.draw() 