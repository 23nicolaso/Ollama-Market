import tkinter as tk
from tkinter import ttk
import queue
from market_simulator.utils.market_utils import news_queue, chat_queue
from market_simulator.web.server import emit_news, emit_chat  # Import the emit function

class NewsFeedFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create news input frame
        self.news_input_frame = ttk.Frame(self)
        self.news_input_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Create news input entry
        self.news_input = ttk.Entry(self.news_input_frame, width=60)
        self.news_input.pack(side=tk.LEFT, padx=5)
        
        # Create submit button
        self.submit_button = ttk.Button(self.news_input_frame, text="Submit News", command=self._submit_news)
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        # Create news feed
        self.news_feed = tk.Text(self, height=38, width=40)  # Reduced height to accommodate input
        self.news_feed.grid(row=1, column=0, rowspan=3, columnspan=1)
        self.news_feed.insert(tk.END, "News Feed:\n")
        self.news_feed.see(tk.END)

        # Create chat window
        self.chat_window = tk.Text(self, height=38, width=40)  # Reduced height to match news feed
        self.chat_window.grid(row=1, column=1, rowspan=3, columnspan=1)
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
                # Emit the news to web clients
                emit_news(headline)
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
                # Emit the chat message to web clients
                emit_chat(message)
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

    def _submit_news(self):
        """Handle news submission"""
        news_text = self.news_input.get().strip()
        if news_text:
            # Call generate_news with custom headline
            from market_simulator.utils.news_generator import generate_news_thread
            generate_news_thread(news_text)
            # Clear the input field
            self.news_input.delete(0, tk.END) 