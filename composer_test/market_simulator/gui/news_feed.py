import tkinter as tk
from tkinter import ttk
import queue
from market_simulator.utils.market_utils import news_queue, chat_queue

class NewsFeedFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create news feed
        self.news_feed = tk.Text(self, height=40, width=40)
        self.news_feed.grid(row=0, column=0, rowspan=3, columnspan=1)
        self.news_feed.insert(tk.END, "News Feed:\n")
        self.news_feed.see(tk.END)

        # Create chat window
        self.chat_window = tk.Text(self, height=40, width=40)
        self.chat_window.grid(row=0, column=1, rowspan=3, columnspan=1)
        self.chat_window.insert(tk.END, "Chat Window:\n")
        self.chat_window.see(tk.END)

        # Start update loops
        self._update_news_feed()
        self._update_chat_window()

    def _update_news_feed(self):
        """Internal method to update news feed"""
        try:
            while True:
                headline = news_queue.get_nowait()
                self.news_feed.insert(tk.END, f"{headline}\n\n")
                self.news_feed.see(tk.END)
        except queue.Empty:
            pass
        finally:
            # Schedule next update using instance method
            self.after(100, self._update_news_feed)

    def _update_chat_window(self):
        """Internal method to update chat window"""
        try:
            while True:
                message = chat_queue.get_nowait()
                self.chat_window.insert(tk.END, f"{message}\n\n")
                self.chat_window.see(tk.END)
        except queue.Empty:
            pass
        finally:
            # Schedule next update using instance method
            self.after(100, self._update_chat_window)

    def add_news(self, headline):
        """Add news to the queue"""
        news_queue.put(headline)

    def add_chat(self, message):
        """Add chat message to the queue"""
        chat_queue.put(message)

    def clear_chat(self):
        """Clear the chat window"""
        self.chat_window.delete(1.0, tk.END)
        self.chat_window.insert(tk.END, "Chat Window:\n")
        self.chat_window.see(tk.END) 