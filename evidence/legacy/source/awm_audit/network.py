"""HTTPS downloads with an opt-in FQDN SNI compatibility mode.

Some networks reset TLS for the short SNI spelling of Hugging Face. The
trailing-dot spelling still verifies BOTH the CA chain and the canonical
hostname. No insecure TLS option or proxy is used.
"""
import http.client
import os
import socket
import ssl
import urllib.request


class FQDNConnection(http.client.HTTPSConnection):
    def connect(self):
        from urllib3.util.ssl_match_hostname import match_hostname

        # This compatibility transport is restricted to the public HF origins.
        if not (self.host.endswith(".huggingface.co") or self.host == "huggingface.co"
                or self.host.endswith(".hf.co") or self.host == "hf.co"):
            return super().connect()
        context = ssl.create_default_context()
        context.check_hostname = False
        raw = socket.create_connection((self.host, self.port), self.timeout)
        try:
            self.sock = context.wrap_socket(raw, server_hostname=self.host + ".")
            match_hostname(self.sock.getpeercert(), self.host)
        except BaseException:
            raw.close()
            if self.sock:
                self.sock.close()
            raise


class FQDNHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(FQDNConnection, req)


def open_remote(url, timeout=120):
    if os.environ.get("AWM_AUDIT_TLS_FQDN") == "1":
        return urllib.request.build_opener(FQDNHandler()).open(url, timeout=timeout)
    return urllib.request.urlopen(url, timeout=timeout)
