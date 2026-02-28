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

@socketio.on('generate_news')
def handle_generate_news():
    from market_simulator.utils.news_generator import generate_news_thread
    generate_news_thread()

def emit_news(headline):
    socketio.emit('news', {'headline': headline})

def emit_chat(chat):
    socketio.emit('chat', {'chat': chat})

def emit_action(msg):
    socketio.emit('action_update', {'msg': msg})

def emit_leaderboard(data):
    socketio.emit('leaderboard_update', data)

def emit_economy(data):
    socketio.emit('economy_update', data)

def run(port=8000):
    print(f'Starting web server on port {port}...')
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    run() 