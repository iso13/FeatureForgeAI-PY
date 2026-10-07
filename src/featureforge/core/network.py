"""Browser network capture for the Step Intelligence Layer.

TS equivalent: ``attachNetworkCapture`` in ``src/core/intelligence/layerValidator.ts``.

Two fixes over the TS version:
* listeners are attached once per page; the TS ``BeforeStep`` hook attached a new
  pair on every step, so listeners (and duplicate entries) piled up;
* responses are matched to their request object, not by URL + method, so repeated
  calls to the same endpoint no longer all land on the first entry.
Response bodies are only read when a step fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from featureforge.core.intelligence.models import NetworkRequest


@dataclass
class _Entry:
    request: Any
    url: str
    method: str
    request_body: Any = None
    response: Any = None


class NetworkCapture:
    def __init__(self, page: Any) -> None:
        self._entries: list[_Entry] = []
        self._by_request: dict[int, _Entry] = {}
        self._mark = 0
        page.on("request", self._on_request)
        page.on("response", self._on_response)

    def _on_request(self, request: Any) -> None:
        try:
            body = request.post_data_json
        except Exception:
            body = None
        entry = _Entry(request=request, url=request.url, method=request.method, request_body=body)
        self._entries.append(entry)
        self._by_request[id(request)] = entry

    def _on_response(self, response: Any) -> None:
        entry = self._by_request.get(id(response.request))
        if entry is not None:
            entry.response = response

    def mark(self) -> None:
        """Start a new step window."""
        self._mark = len(self._entries)

    def since_mark(self, *, read_bodies: bool = True) -> list[NetworkRequest]:
        """Requests made during the current step."""
        return self._export(self._entries[self._mark :], read_bodies)

    def all(self, *, read_bodies: bool = True) -> list[NetworkRequest]:
        """Every request in the scenario (failures often start in an earlier step)."""
        return self._export(self._entries, read_bodies)

    def _export(self, entries: list[_Entry], read_bodies: bool) -> list[NetworkRequest]:
        captured: list[NetworkRequest] = []
        for entry in entries:
            status: int | None = None
            body: Any = None
            if entry.response is not None:
                status = entry.response.status
                if read_bodies:
                    try:
                        body = entry.response.json()
                    except Exception:
                        body = None
            captured.append(
                NetworkRequest(
                    url=entry.url,
                    method=entry.method,
                    status=status,
                    request_body=entry.request_body,
                    response_body=body,
                )
            )
        return captured
