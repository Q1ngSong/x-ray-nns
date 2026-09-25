"""Local viewer server: browse recorded runs with the current page and record new CLIP runs from it."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import email.parser
import email.policy
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

from xray.exporters.html import render_page
from xray.exporters.scene import build_scene
from xray.ir import InferenceTrace

MAX_BODY = 20 * 1024 * 1024
MAX_PROMPTS = 8
MAX_PROMPT_CHARS = 300
_CACHED_RUNS = 4
_START_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>x-ray · new run</title>
<style>body{margin:40px auto;max-width:520px;font:13px/1.5 -apple-system,BlinkMacSystemFont,sans-serif;color:#213442}
form{display:grid;gap:10px}input{font:inherit;padding:6px 8px;border:1px solid #d9e2ea;border-radius:8px}
.error{padding:9px 12px;background:#f8eee7;border-radius:8px;color:#7a5a45}</style></head><body>
<h1>x-ray</h1><p>No recorded runs yet. Pick an image and prompts to record the first one.</p>__ERROR__
<form method="post" action="/runs/new" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<input name="prompt" maxlength="300" placeholder="a photo of a cat" required>
<input name="prompt" maxlength="300" placeholder="a photo of a dog">
<button type="submit">Run CLIP</button></form></body></html>"""


class RunLibrary:
    """Run library: find trace bundles under the runs root and cache their trace and 3D scene. [主线]"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._labels: dict[str, tuple[float, str]] = {}
        self._loaded: OrderedDict[str, tuple[float, InferenceTrace, dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def bundle(self, relative: str) -> Path | None:
        """Bundle lookup: resolve a run path such as ``clip/live/20260923-172501`` inside the root. [主线]

        Args:
            relative: Slash-separated run path from a URL or form; paths leaving the root are refused.
        Returns:
            Path | None — bundle directory holding ``trace.json``, or ``None``.
        """
        path = (self.root / relative).resolve()
        inside = path == self.root or self.root in path.parents
        return path if inside and (path / "trace.json").is_file() else None

    def runs(self) -> list[dict[str, str]]:
        """Run list: every bundle up to four levels below the root, newest ``trace.json`` first. [主线]

        Returns:
            list[dict[str, str]] — ``path`` (relative, slash-separated) and ``label`` (time, path, first prompt).
        """
        found = []
        for depth in range(1, 5):
            for trace_file in self.root.glob("/".join(["*"] * depth) + "/trace.json"):
                relative, modified = trace_file.parent.relative_to(self.root).as_posix(), trace_file.stat().st_mtime
                found.append((modified, relative, self._label(relative, trace_file, modified)))
        found.sort(reverse=True)
        return [{"path": relative, "label": label} for _, relative, label in found]

    def _label(self, relative: str, trace_file: Path, modified: float) -> str:
        """Run label: modification time, path and first prompt, cached per ``trace.json`` version. [主线]

        Args:
            relative: Run path shown in the label.
            trace_file: Manifest read once per modification time for its ``inputs.text``.
            modified: ``trace_file`` modification time, the cache key.
        Returns:
            str — e.g. ``09-23 17:25 · clip/live/20260923-172501 · a photo of a cat``.
        """
        cached = self._labels.get(relative)
        if cached and cached[0] == modified:
            return cached[1]
        try:
            prompts = json.loads(trace_file.read_text(encoding="utf-8")).get("inputs", {}).get("text") or []
        except (OSError, ValueError):  # A bundle being rewritten still lists, just without its prompt.
            prompts = []
        prompt = str(prompts[0])[:40] if prompts else ""
        label = " · ".join(filter(None, (f"{datetime.fromtimestamp(modified):%m-%d %H:%M}", relative, prompt)))
        self._labels[relative] = (modified, label)
        return label

    def load(self, relative: str) -> tuple[InferenceTrace, dict[str, Any]] | None:
        """Run data: the trace and its 3D scene, rebuilt only when ``trace.json`` changes. [主线]

        Args:
            relative: Run path; unknown or escaping paths return ``None``.
        Returns:
            tuple[InferenceTrace, dict[str, Any]] | None — trace and ``build_scene`` output.
        """
        bundle = self.bundle(relative)
        if bundle is None:
            return None
        key, modified = bundle.relative_to(self.root).as_posix(), (bundle / "trace.json").stat().st_mtime
        # Page requests arrive on separate threads; one lock keeps the small LRU consistent.
        with self._cache_lock:
            cached = self._loaded.get(key)
            if cached is None or cached[0] != modified:
                trace = InferenceTrace.load_json(bundle / "trace.json")
                cached = (modified, trace, build_scene(trace, bundle))
                self._loaded[key] = cached
            self._loaded.move_to_end(key)
            while len(self._loaded) > _CACHED_RUNS:
                self._loaded.popitem(last=False)
        return cached[1], cached[2]

    def input_image(self, relative: str) -> Path | None:
        """Reused input: the image recorded by an existing run, for submissions without an upload. [主线]

        Args:
            relative: Run whose ``inputs.image.asset_path`` is reused.
        Returns:
            Path | None — image file inside that bundle, or ``None`` when the run has none.
        """
        loaded, bundle = self.load(relative), self.bundle(relative)
        image = loaded[0].inputs.get("image") if loaded else None
        asset = image.get("asset_path") if isinstance(image, dict) else None
        if not asset or bundle is None:
            return None
        path = (bundle / asset).resolve()
        return path if bundle in path.parents and path.is_file() else None

    def new_run_directory(self) -> Path:
        """New run: ``clip/live/<YYYYmmdd-HHMMSS>`` under the root, suffixed when that second is taken. [主线]

        Returns:
            Path — directory that does not exist yet; the recorder creates it.
        """
        base = self.root / "clip" / "live" / f"{datetime.now():%Y%m%d-%H%M%S}"
        candidate, index = base, 2
        while candidate.exists():
            candidate, index = base.with_name(f"{base.name}-{index}"), index + 1
        return candidate


class RunServer(ThreadingHTTPServer):
    """Viewer server bound to 127.0.0.1; one CLIP run at a time, pages keep serving meanwhile. [主线]"""

    daemon_threads = True

    def __init__(self, port: int, library: RunLibrary, model_path: Path, device: str, record: Callable[..., Any]) -> None:
        super().__init__(("127.0.0.1", port), RunHandler)
        self.library, self.model_path, self.device, self.record = library, model_path, device, record
        self.run_lock = threading.Lock()

    def origins(self) -> set[str]:
        """Allowed origins: the two local host names on the bound port. [基础设施]

        Returns:
            set[str] — ``http://127.0.0.1:<port>`` and ``http://localhost:<port>``.
        """
        port = self.server_address[1]
        return {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


class RunHandler(BaseHTTPRequestHandler):
    """Routes: ``/`` → newest run, ``/runs/<run>/`` → page, ``/runs/<run>/<file>`` → file, ``POST /runs/new``. [主线]"""

    server: RunServer

    def do_GET(self) -> None:
        """Serve a run page, a bundle file, or the start page. [主线]"""
        if not self._local_host():
            return
        parts = urlsplit(self.path)
        path, error = unquote(parts.path), parse_qs(parts.query).get("error", [None])[0]
        if path == "/":
            runs = self.server.library.runs()
            if runs:
                self._redirect(_run_url(runs[0]["path"]))
            else:
                notice = f'<p class="error">{html.escape(error)}</p>' if error else ""
                self._send(HTTPStatus.OK, _START_PAGE.replace("__ERROR__", notice).encode("utf-8"), "text/html; charset=utf-8")
            return
        if not path.startswith("/runs/"):
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        relative = path[len("/runs/"):]
        if relative.endswith("/") or relative == "":
            self._page(relative.rstrip("/"), error)
        elif self.server.library.bundle(relative):
            self._redirect(_run_url(relative))
        else:
            self._file(relative)

    def do_POST(self) -> None:
        """Record a new CLIP run from the page's form, then open it or return with an error. [主线]"""
        if not self._local_host():
            return
        if not self._same_origin():
            self._send(HTTPStatus.FORBIDDEN, b"cross-origin submission refused", "text/plain")
            return
        if urlsplit(self.path).path != "/runs/new":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        library = self.server.library
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            self.close_connection = True
            self._redirect("/?error=" + quote("The upload must be smaller than 20 MB."))
            return
        fields, files = _multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
        base = (fields.get("base") or [""])[0]
        back = (_run_url(base) if library.bundle(base) else "/") + "?error="
        prompts = [prompt.strip() for prompt in fields.get("prompt", []) if prompt.strip()]
        problem = ("Add at least one prompt." if not prompts
                   else f"Use at most {MAX_PROMPTS} prompts." if len(prompts) > MAX_PROMPTS
                   else f"Keep each prompt under {MAX_PROMPT_CHARS} characters." if max(map(len, prompts)) > MAX_PROMPT_CHARS
                   else None)
        if problem:
            self._redirect(back + quote(problem))
            return
        with tempfile.TemporaryDirectory() as scratch:
            upload = files.get("image")
            image = _saved_image(upload[1], Path(scratch)) if upload and upload[1] else library.input_image(base)
            if image is None:
                self._redirect(back + quote("Choose an image that Pillow can open (PNG, JPEG, WebP, …)."))
                return
            with self.server.run_lock:
                output = library.new_run_directory()
                try:
                    self.server.record(self.server.model_path, image, prompts, output, device=self.server.device, trace_id=output.name)
                except Exception as error:  # Any recorder failure returns to the page; the half-written run is removed.
                    shutil.rmtree(output, ignore_errors=True)
                    self._redirect(back + quote(f"The CLIP run failed: {error}"))
                    return
        self._redirect(_run_url(output.relative_to(library.root).as_posix()))

    def _local_host(self) -> bool:
        """Host check: refuse requests addressed to any other name (DNS rebinding). [基础设施]

        Returns:
            bool — True when ``Host`` is ``127.0.0.1:<port>`` or ``localhost:<port>``; otherwise 403 is sent.
        """
        if "http://" + (self.headers.get("Host") or "") in self.server.origins():
            return True
        self._send(HTTPStatus.FORBIDDEN, b"unexpected host", "text/plain")
        return False

    def _same_origin(self) -> bool:
        """Origin check: accept form posts only from this server's own pages (CSRF). [基础设施]

        Returns:
            bool — True when ``Origin`` (or, without it, ``Referer``) belongs to this server.
        """
        origin = self.headers.get("Origin")
        if origin:
            return origin in self.server.origins()
        referer = urlsplit(self.headers.get("Referer") or "")
        return f"{referer.scheme}://{referer.netloc}" in self.server.origins()

    def _page(self, relative: str, error: str | None) -> None:
        """Render one run with the current viewer, the run list and an optional error. [主线]

        Args:
            relative: Run path taken from the URL.
            error: Message from a failed submission, shown at the top of the page.
        """
        loaded = self.server.library.load(relative)
        if loaded is None:
            self._send(HTTPStatus.NOT_FOUND, b"run not found", "text/plain")
            return
        runs = {"current": self.server.library.bundle(relative).relative_to(self.server.library.root).as_posix(),
                "runs": self.server.library.runs(), "error": error}
        self._send(HTTPStatus.OK, render_page(*loaded, runs).encode("utf-8"), "text/html; charset=utf-8")

    def _file(self, relative: str) -> None:
        """Serve a file inside the runs root, such as a run's input image. [主线]

        Args:
            relative: File path below the root; anything outside it or missing is 404.
        """
        root = self.server.library.root
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        self._send(HTTPStatus.OK, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def _redirect(self, location: str) -> None:
        """Send a 303 so the browser follows with GET (also after a form post). [基础设施]

        Args:
            location: Absolute path on this server.
        """
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        """Send a complete response without caching, since runs change under the same URL. [基础设施]

        Args:
            status: HTTP status.
            body: Response bytes.
            content_type: MIME type header value.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _run_url(relative: str) -> str:
    """Page URL of a run, percent-encoded but keeping its slashes. [基础设施]

    Args:
        relative: Run path below the runs root.
    Returns:
        str — ``/runs/<path>/``.
    """
    return "/runs/" + quote(relative) + "/"


def _multipart(content_type: str, body: bytes) -> tuple[dict[str, list[str]], dict[str, tuple[str, bytes]]]:
    """Form fields: split a ``multipart/form-data`` body with the standard-library MIME parser. [主线]

    Args:
        content_type: Request ``Content-Type`` carrying the boundary; other types yield no fields.
        body: Raw request body.
    Returns:
        tuple — text fields (name → values in order) and files (name → (filename, bytes)).
    """
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields: dict[str, list[str]] = {}
    files: dict[str, tuple[str, bytes]] = {}
    if not message.is_multipart():
        return fields, files
    for part in message.iter_parts():
        name, filename = part.get_param("name", header="content-disposition"), part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if not name:
            continue
        if filename is not None:
            files[name] = (filename, payload)
        else:
            fields.setdefault(name, []).append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
    return fields, files


def _saved_image(data: bytes, folder: Path) -> Path | None:
    """Uploaded image: keep it only if Pillow can read it, named with the detected format. [主线]

    Args:
        data: Uploaded bytes.
        folder: Scratch directory that outlives the CLIP run.
    Returns:
        Path | None — ``upload.<format>`` inside ``folder``, or ``None`` for unreadable data.
    """
    from PIL import Image, UnidentifiedImageError

    path = folder / "upload"
    path.write_bytes(data)
    try:
        with Image.open(path) as image:
            image.verify()
            fmt = (image.format or "png").lower()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None
    return path.rename(folder / f"upload.{'jpg' if fmt == 'jpeg' else fmt}")


def serve(model_path: Path, runs_root: Path, *, port: int = 8765, device: str = "auto") -> None:
    """Start the local viewer server until interrupted. [主线]

    Args:
        model_path: Local Hugging Face CLIP directory used for new runs.
        runs_root: Directory whose bundles are listed; new runs go to ``<runs_root>/clip/live/``.
        port: Local port; ``0`` picks a free one.
        device: ``auto``, ``cpu`` or ``mps``, passed to ``run_clip``.
    """
    from xray.recorder.clip import run_clip

    server = RunServer(port, RunLibrary(runs_root), Path(model_path), device, run_clip)
    print(f"x-ray viewer on http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
