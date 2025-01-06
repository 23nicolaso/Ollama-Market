from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from flask import Flask
from flask_socketio import SocketIO
import threading

app = Flask(__name__, static_folder=os.path.dirname(__file__))
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return app.send_static_file('index.html')

def emit_news(headline):
    """Emit news headline to all connected clients"""
    socketio.emit('news', {'headline': headline})

def emit_chat(chat):
    """Emit new chat messages to all connected clients"""
    socketio.emit('chat', {'chat': chat})

def run(port=8000):
    print(f'Starting web server on port {port}...')
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    run() 