"""Tiny web server for the Mininet server node. GET / -> 200. POST /login -> always 401 (failed login).
Every request is logged as: <unix_time> <client_ip> <path> <status>  so the collector can count auth failures."""
import sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/server_log.txt"

class H(BaseHTTPRequestHandler):
    def _log(self, code):
        with open(LOG, "a") as f: f.write(f"{time.time()} {self.client_address[0]} {self.path} {code}\n")
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); self._log(200)
    def do_POST(self):
        self.send_response(401); self.end_headers(); self._log(401)
    def log_message(self, *a): pass

ThreadingHTTPServer(("0.0.0.0", 80), H).serve_forever()
