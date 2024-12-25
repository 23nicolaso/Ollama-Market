from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import time
from market_simulator.utils.market_utils import price_history, assets

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/assets')
def get_assets():
    """Return list of available assets"""
    return jsonify(assets)

@app.route('/price_history/<asset>')
def get_price_history(asset):
    """Return price history for a specific asset"""
    if asset in price_history:
        return jsonify({
            'asset': asset,
            'prices': price_history[asset]
        })
    return jsonify({'error': 'Asset not found'}), 404

def emit_price_updates():
    """Emit price updates through websocket"""
    while True:
        for asset in assets:
            if asset in price_history and price_history[asset]:
                socketio.emit('price_update', {
                    'asset': asset,
                    'price': price_history[asset][-1],
                    'history': price_history[asset][-100:]  # Send last 100 prices
                })
        time.sleep(0.1)  # Update every 0.1 seconds

def start_server(host='0.0.0.0', port=5000):
    """Start the price server"""
    # Start the price update thread
    update_thread = threading.Thread(target=emit_price_updates)
    update_thread.daemon = True
    update_thread.start()
    
    # Start the Flask-SocketIO server
    socketio.run(app, host=host, port=port)

if __name__ == '__main__':
    start_server() 