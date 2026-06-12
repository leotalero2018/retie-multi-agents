"""HTTP client for NotebookLM MCP Server (v2.0.0, Streamable HTTP transport).

Protocol: JSON-RPC 2.0 over HTTP POST to /mcp
Session:  MCP session ID captured from response headers (Mcp-Session-Id).
          The server allows only ONE active transport at a time, so the session ID
          is persisted to disk and reused across Python process restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Cache file next to this module — survives process restarts
_SESSION_CACHE = Path(__file__).parent / ".nlm_session_cache.json"


class NotebookLMError(Exception):
    pass


class NotebookLMCache:
    """In-process TTL cache for NotebookLM answers.

    Keyed by SHA-256(question + notebook_id) so collisions across different
    notebooks are impossible even when the same question is asked.
    Optionally stores the question embedding to allow SEMANTIC lookups:
    a re-worded question similar enough to a cached one reuses its answer
    instead of paying ~30s of browser automation.
    Thread-safe for read-heavy workloads (GIL protects dict ops).
    """

    def __init__(self, ttl: int = 3600, max_entries: int = 256) -> None:
        self._ttl = ttl
        self._max = max_entries
        # key -> (answer, sources, ts, embedding|None, notebook_id)
        self._store: Dict[str, Tuple[str, List[Dict[str, Any]], float, Optional[List[float]], Optional[str]]] = {}

    def _key(self, question: str, notebook_id: Optional[str]) -> str:
        raw = f"{question.strip().lower()}|{notebook_id or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) >= self._ttl

    def get(
        self, question: str, notebook_id: Optional[str]
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        k = self._key(question, notebook_id)
        entry = self._store.get(k)
        if entry is None:
            return None
        if self._expired(entry[2]):
            del self._store[k]
            return None
        return entry[0], entry[1]

    def get_semantic(
        self,
        embedding: List[float],
        notebook_id: Optional[str],
        min_sim: float = 0.93,
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """Busca la entrada más similar por coseno entre las cacheadas con embedding."""
        if not embedding or min_sim <= 0:
            return None
        import math
        nq = math.sqrt(sum(x * x for x in embedding)) or 1.0
        best: Optional[Tuple[float, str]] = None
        for k, (answer, _src, ts, emb, nb) in list(self._store.items()):
            if emb is None or nb != notebook_id:
                continue
            if self._expired(ts):
                self._store.pop(k, None)
                continue
            dot = sum(a * b for a, b in zip(embedding, emb))
            ne = math.sqrt(sum(x * x for x in emb)) or 1.0
            sim = dot / (nq * ne)
            if sim >= min_sim and (best is None or sim > best[0]):
                best = (sim, k)
        if best is None:
            return None
        entry = self._store[best[1]]
        return entry[0], entry[1]

    def set(
        self,
        question: str,
        notebook_id: Optional[str],
        answer: str,
        sources: List[Dict[str, Any]],
        embedding: Optional[List[float]] = None,
    ) -> None:
        if not answer:
            return
        if len(self._store) >= self._max:
            # Expulsa la entrada más vieja (cache pequeño, escaneo lineal OK)
            oldest = min(self._store.items(), key=lambda kv: kv[1][2])[0]
            self._store.pop(oldest, None)
        k = self._key(question, notebook_id)
        self._store[k] = (answer, sources, time.time(), embedding, notebook_id)


class NotebookLMClient:
    """Synchronous JSON-RPC client for the NotebookLM MCP HTTP server.

    The notebooklm-mcp server only supports one active transport per instance.
    This client persists the MCP session ID to disk so subsequent Python processes
    can reuse the same session without calling initialize() again (which would fail
    with 500 "Already connected to a transport").
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        notebook_id: Optional[str] = None,
        timeout: float = 120.0,
        page_timeout_ms: int = 15000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.notebook_id = notebook_id
        self._timeout = timeout
        self._page_timeout_ms = page_timeout_ms
        self._mcp_session: Optional[str] = None   # MCP protocol session
        self._nlm_session: Optional[str] = None   # NotebookLM conversational session
        self._request_id = 0

        # Try to restore a cached session ID from a previous run
        self._load_session_cache()

    # ── session cache ─────────────────────────────────────────────────────────

    def _load_session_cache(self) -> None:
        try:
            if _SESSION_CACHE.exists():
                data = json.loads(_SESSION_CACHE.read_text(encoding="utf-8"))
                self._mcp_session = data.get("mcp_session")
        except Exception:
            pass

    def _save_session_cache(self) -> None:
        try:
            _SESSION_CACHE.write_text(
                json.dumps({"mcp_session": self._mcp_session}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _clear_session_cache(self) -> None:
        try:
            if _SESSION_CACHE.exists():
                _SESSION_CACHE.unlink()
        except Exception:
            pass

    # ── internal helpers ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._mcp_session:
            h["Mcp-Session-Id"] = self._mcp_session
        return h

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = httpx.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:400]
            if "already connected" in body.lower() or "no transport" in body.lower():
                # Server has a stale/missing transport — our cached session ID is no
                # longer valid. Clear cache so next _ensure_session() calls initialize().
                self._mcp_session = None
                self._nlm_session = None
                self._clear_session_cache()
            raise NotebookLMError(
                f"HTTP {exc.response.status_code} from MCP server: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise NotebookLMError(f"HTTP error calling MCP server: {exc}") from exc

        # Capture and persist MCP session ID from response headers
        sid = resp.headers.get("Mcp-Session-Id")
        if sid and sid != self._mcp_session:
            self._mcp_session = sid
            self._save_session_cache()

        ct = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(resp.text)
        try:
            return resp.json()
        except Exception as exc:
            raise NotebookLMError(
                f"Could not parse MCP response as JSON: {exc}"
            ) from exc

    def _parse_sse(self, text: str) -> Dict[str, Any]:
        """Extract the last complete JSON-RPC message from an SSE stream."""
        result: Dict[str, Any] = {}
        for line in text.splitlines():
            if line.startswith("data: "):
                data = line[6:].strip()
                if data and data != "[DONE]":
                    try:
                        msg = json.loads(data)
                        if "result" in msg or "error" in msg:
                            result = msg
                    except json.JSONDecodeError:
                        pass
        return result

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        return self._post(payload)

    # ── session lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Open a new MCP protocol session.

        Called automatically when no cached session exists.
        Fails with 500 if the server already has an active transport — in that
        case the caller should retry with the existing session ID.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "retie-agent", "version": "1.0"},
            },
        }
        self._post(payload)
        return self._mcp_session is not None

    def _ensure_session(self) -> None:
        """Ensure we have a session ID, initializing only when none exists.

        Deliberately avoids probing the session on every call: the notebooklm-mcp
        server only allows one active transport at a time, so calling initialize()
        when a session is already live returns HTTP 500 "Already connected". The
        retry logic in ask_question handles stale/expired sessions instead.
        """
        if not self._mcp_session:
            self.initialize()

    def reset_conversation(self) -> None:
        """Clear the NotebookLM conversational context without closing MCP."""
        if self._nlm_session:
            try:
                self._call_tool("reset_session", {"session_id": self._nlm_session})
            except Exception:
                pass
            self._nlm_session = None

    # ── public tools ──────────────────────────────────────────────────────────

    def get_health(self) -> Dict[str, Any]:
        self._ensure_session()
        return self._call_tool("get_health", {})

    def is_authenticated(self) -> bool:
        """Returns True if the MCP server has a valid Google session."""
        try:
            resp = self.get_health()
            content = resp.get("result", {}).get("content", [{}])
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                        return bool(parsed.get("authenticated"))
                    except json.JSONDecodeError:
                        return "authenticated" in text.lower()
        except Exception:
            pass
        return False

    def ask_question(
        self,
        question: str,
        source_format: str = "footnotes",
        notebook_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Query NotebookLM and return (answer_text, sources).

        Reuses the conversational session for multi-turn context. On a session
        error OR a page timeout (an overlay intercepting the textarea), retries
        once with a FRESH conversational session — a clean tab has no residual
        overlay, so the retry usually succeeds.
        """
        self._ensure_session()
        nb_id = notebook_id or self.notebook_id

        def _build_args(include_session: bool) -> Dict[str, Any]:
            a: Dict[str, Any] = {"question": question, "source_format": source_format}
            if nb_id:
                a["notebook_id"] = nb_id
            if self._page_timeout_ms:
                # Fail a stuck click fast instead of hanging the server's 30s default.
                a["browser_options"] = {"timeout_ms": self._page_timeout_ms}
            if include_session and self._nlm_session:
                a["session_id"] = self._nlm_session
            return a

        def _attempt(include_session: bool) -> Tuple[str, List[Dict[str, Any]]]:
            resp = self._call_tool("ask_question", _build_args(include_session))
            return self._extract_answer(resp)

        try:
            return _attempt(include_session=True)
        except NotebookLMError as exc:
            msg = str(exc).lower()
            is_session_error = any(
                k in msg for k in ("session", "already connected", "no transport", "transport")
            )
            # Un 500 genérico suele ser un transporte/pestaña en mal estado del
            # lado del servidor; un reintento con sesión fresca lo resuelve.
            is_server_error = "http 500" in msg or "internal server error" in msg
            is_timeout = any(
                k in msg for k in ("timeout", "page.click", "intercept")
            )
            if not (is_session_error or is_timeout or is_server_error):
                raise

            logger.warning(
                "NotebookLM: retrying with fresh session after: %s",
                str(exc).splitlines()[0],
            )
            # A stale MCP transport needs a brand-new protocol session…
            if is_session_error or is_server_error:
                self._mcp_session = None
                self._clear_session_cache()
            # …and either way, drop the conversational session so the retry opens
            # a clean tab without the overlay that intercepted the click.
            self._nlm_session = None
            self._ensure_session()
            return _attempt(include_session=False)

    @staticmethod
    def _error_detail(content: Any) -> str:
        """Extract a short human-readable error message from MCP error content."""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return str(item.get("text", "")).splitlines()[0][:200]
        return str(content)[:200]

    def _extract_answer(
        self, resp: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if "error" in resp:
            raise NotebookLMError(f"MCP error: {resp['error']}")

        result = resp.get("result", {})

        # MCP tool-level failure (e.g. browser timeout) is flagged with isError
        if result.get("isError"):
            detail = self._error_detail(result.get("content", []))
            raise NotebookLMError(f"NotebookLM query failed: {detail}")

        # Persist NotebookLM conversational session ID
        prov = result.get("_provenance", {})
        if prov.get("session_id"):
            self._nlm_session = prov["session_id"]

        content = result.get("content", [])
        raw_text = ""
        sources: List[Dict[str, Any]] = []

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    raw_text = item.get("text", "")
                elif item.get("type") == "resource":
                    sources.append(item)
        elif isinstance(content, str):
            raw_text = content

        # The MCP server wraps tool output in {"success":true/false,"data"/"error":...}
        # Try to unwrap that envelope to get the real answer text.
        answer = raw_text
        try:
            envelope = json.loads(raw_text)
            if isinstance(envelope, dict):
                # Explicit failure envelope → surface as an exception.
                # Keep only the first line: the server appends Playwright's full
                # "Call log:" dump, which is noise in our logs.
                if envelope.get("success") is False:
                    err = envelope.get("error", "unknown error")
                    first_line = str(err).splitlines()[0][:200] if err else "unknown error"
                    raise NotebookLMError(f"NotebookLM query failed: {first_line}")
                if envelope.get("success"):
                    data = envelope.get("data", {})
                    # ask_question → data.answer
                    if "answer" in data:
                        answer = data["answer"]
                        # Extract structured sources if present
                        if not sources and "sources" in data:
                            sources = data["sources"]
                        # Persist session from data envelope
                        if data.get("session_id") and not self._nlm_session:
                            self._nlm_session = data["session_id"]
        except json.JSONDecodeError:
            pass  # raw_text is plain text, use as-is

        # Last-resort fallbacks
        if not answer:
            answer = result.get("answer", result.get("text", ""))
        if not sources:
            sources = result.get("sources", [])

        return answer.strip(), sources
