"""End-to-end probe for Streamlit's file upload endpoint.

Starts `streamlit run upload_probe_app.py` on a local port, completes the
WebSocket handshake to obtain a session id, then PUTs a small multipart body
to /_stcore/upload_file/<session_id>/<file_id> the same way the browser does.

Usage:
    python tools/upload_probe.py [--port 8765] [--max-upload 500] [--max-message 500]

Exit code 0 means the upload endpoint accepted the file (HTTP 204).
"""

from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import tornado
import websockets

STREAMLIT_APP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "upload_probe_app.py"
)


def _strip_proxy_env() -> None:
    """Localhost probe must not go through any configured proxy."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        os.environ.pop(key, None)


def multipart_body(filename: str, data: bytes, boundary: str) -> tuple[bytes, str]:
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


async def get_session_id(ws_uri: str) -> str:
    async with websockets.connect(
        ws_uri,
        subprotocols=["streamlit", "PLACEHOLDER_AUTH_TOKEN"],
        proxy=None,
    ) as ws:
        print("websocket connected; subprotocol:", ws.subprotocol, flush=True)
        # Streamlit 1.51 only starts the first script run after the browser
        # sends a rerun_script BackMsg; new_session arrives after that.
        from streamlit.proto.BackMsg_pb2 import BackMsg

        back = BackMsg()
        back.rerun_script.page_script_hash = "upload_probe_app.py"
        await ws.send(back.SerializeToString())
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            if isinstance(raw, bytes):
                from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

                msg = ForwardMsg()
                msg.ParseFromString(raw)
                if msg.HasField("new_session"):
                    print("got new_session", flush=True)
                    return msg.new_session.initialize.session_id


def put_upload(base: str, session_id: str) -> tuple[int, str]:
    # Mimic the browser: fetch the index page so the _streamlit_xsrf cookie
    # is set, then include both the cookie and X-Xsrftoken header.
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.open(base + "/", timeout=10).read()
    xsrf = next(
        (c.value for c in jar if c.name == "_streamlit_xsrf"), ""
    )

    file_id = str(uuid.uuid4())
    boundary = "----phosprobe" + uuid.uuid4().hex
    body, ctype = multipart_body("probe.jpg", b"\xff\xd8\xff\xe0small-bytes", boundary)
    req = urllib.request.Request(
        f"{base}/_stcore/upload_file/{session_id}/{file_id}",
        data=body,
        method="PUT",
        headers={
            "Content-Type": ctype,
            "Cookie": f"_streamlit_xsrf={xsrf}",
            "X-Xsrftoken": xsrf,
        },
    )
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:500]


async def main() -> int:
    _strip_proxy_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-upload", type=int, default=500)
    ap.add_argument("--max-message", type=int, default=500)
    args = ap.parse_args()

    port = args.port
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        STREAMLIT_APP,
        "--server.port",
        str(port),
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--server.maxUploadSize",
        str(args.max_upload),
        "--server.maxMessageSize",
        str(args.max_message),
    ]
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        env.pop(key, None)
    env["HOME"] = env.get(
        "PHOS_PROBE_HOME", os.path.join("/private/tmp", f"phos-probe-{os.getpid()}")
    )
    os.makedirs(env["HOME"], exist_ok=True)
    log_path = os.path.join(env["HOME"], "streamlit.log")
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            if proc.poll() is not None:
                print("server exited early:")
                print(open(log_path, encoding="utf-8").read())
                return 1
            try:
                urllib.request.urlopen(base + "/_stcore/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("server did not become healthy")
            print(open(log_path, encoding="utf-8").read())
            return 1

        session_id = await get_session_id(f"ws://127.0.0.1:{port}/_stcore/stream")
        status, body = put_upload(base, session_id)
        print(
            f"tornado={tornado.version} maxUpload={args.max_upload} "
            f"maxMessage={args.max_message} -> HTTP {status}",
            flush=True,
        )
        if status != 204:
            print("response body:", body, flush=True)
        return 0 if status == 204 else 2
    finally:
        log_file.close()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
