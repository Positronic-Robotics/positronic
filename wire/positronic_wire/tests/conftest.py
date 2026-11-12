"""Fixtures the wire tests share."""

import datetime
import socket
import ssl
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from cryptography.x509.oid import NameOID


class Certified(NamedTuple):
    """A self-signed certificate for one host name, and a server context that presents it."""

    certificate: bytes
    key: bytes
    # Presents the certificate, and records in ``names_asked`` the server name each client sends.
    server: ssl.SSLContext
    names_asked: list[str | None]


@pytest.fixture
def certified_for(tmp_path: Path) -> Callable[[str], Certified]:
    """Makes a certificate valid for one host name alone, PEM encoded, valid from yesterday."""

    def make(host: str) -> Certified:
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
        day = datetime.timedelta(days=1)
        now = datetime.datetime.now(datetime.UTC)
        certificate = (
            x509
            .CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - day)
            .not_valid_after(now + day)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        ).public_bytes(Encoding.PEM)
        private = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        chain, key_file = tmp_path / f'{host}.pem', tmp_path / f'{host}.key'
        chain.write_bytes(certificate)
        key_file.write_bytes(private)
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.load_cert_chain(chain, key_file)
        names_asked: list[str | None] = []
        server.sni_callback = lambda _sock, server_name, _context: names_asked.append(server_name)
        return Certified(certificate, private, server, names_asked)

    return make


@pytest.fixture
def tls_on_loopback() -> Iterator[Callable[[ssl.SSLContext, Callable[[ssl.SSLSocket], None]], int]]:
    """Serves TLS with a server context on a loopback port, and runs a handler on each session, each on its own thread.

    Returns the port. A client that refuses the certificate ends its session in the handshake, and the handler then
    does not run.
    """
    listeners: list[socket.socket] = []

    def start(server: ssl.SSLContext, handle: Callable[[ssl.SSLSocket], None]) -> int:
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(8)
        listeners.append(listener)

        def run(conn: socket.socket) -> None:
            try:
                with server.wrap_socket(conn, server_side=True) as session:
                    handle(session)
            except (ssl.SSLError, ConnectionError):
                conn.close()

        def accept_each() -> None:
            while True:
                try:
                    conn, _ = listener.accept()
                except OSError:
                    return  # the fixture closed the listener
                threading.Thread(target=run, args=(conn,), daemon=True).start()

        threading.Thread(target=accept_each, daemon=True).start()
        return listener.getsockname()[1]

    yield start
    for listener in listeners:
        listener.close()
