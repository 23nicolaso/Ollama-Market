from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, Handler)
    print(f'Starting web server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run() 