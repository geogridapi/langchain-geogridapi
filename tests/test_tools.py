from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from geogridapi import GeoGridAPI

from langchain_geogridapi import (
    GeoGridAPIToolkit,
    GeoGridFindBusiness,
    GeoGridGetHistory,
    GeoGridRunGrid,
    GeoGridRunGrids,
)


class FakeServer:
    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        return httpx.Response(reply.get("status", 200), json=reply.get("body"))

    def client(self) -> GeoGridAPI:
        return GeoGridAPI("gga_live_x", http_client=httpx.Client(transport=httpx.MockTransport(self.handler)))


COMPLETED_GRID = {
    "id": "g1",
    "object": "grid",
    "status": "completed",
    "place": {"place_id": "ChIJ1", "name": "Joe's Pizza", "address": "7 Carmine St", "lat": 40.7, "lng": -74.0},
    "keyword": "pizza",
    "shape": "square",
    "size": 3,
    "points_total": 9,
    "points_charged": 9,
    "summary": {"average_rank": 2.1, "average_rank_with_misses": 2.1, "top3_points": 8, "found_points": 9},
    "ranks": [[1, 2, 3], [1, 1, 2], [4, 3, 2]],
    "points": [{"row": 0, "col": 0, "rank": 1, "top": []}],
    "competitors": [{"place_id": f"c{i}", "name": f"C{i}", "top3_points": 9 - i} for i in range(8)],
    "previous": {"grid_id": "g0", "created_at": "2026-10-01", "changes": {"average_rank": -0.5}, "ranks": [[2]]},
}


def test_run_grid_waits_and_returns_a_compact_view(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("geogridapi._client.time.sleep", lambda _: None)
    server = FakeServer({"status": 202, "body": {"id": "g1", "status": "pending"}}, {"body": COMPLETED_GRID})
    tool = GeoGridRunGrid(client=server.client())

    result = json.loads(tool.invoke({"query": "Joe's Pizza, New York", "keyword": "pizza", "size": 3}))

    assert json.loads(server.requests[0].content) == {"query": "Joe's Pizza, New York", "keyword": "pizza", "size": 3}
    assert result["status"] == "completed"
    assert result["ranks"] == COMPLETED_GRID["ranks"]
    assert result["credits_charged"] == 9
    assert len(result["competitors_ahead"]) == 5
    assert result["since_previous_grid"]["changes"] == {"average_rank": -0.5}
    assert "points" not in result  # per-point details are left out


def test_run_grid_returns_the_grid_id_when_wait_is_zero() -> None:
    server = FakeServer({"status": 202, "body": {"id": "g1", "status": "pending"}})
    result = json.loads(
        GeoGridRunGrid(client=server.client()).invoke({"place_id": "ChIJ1", "keyword": "pizza", "wait_seconds": 0})
    )
    assert result["grid_id"] == "g1"
    assert "geogrid_get_grid" in result["next_step"]


def test_validation_errors_are_returned_to_the_agent() -> None:
    tool = GeoGridRunGrid(client=FakeServer().client())
    message = tool.invoke({"keyword": "pizza"})
    assert "exactly one of place_id or query" in message
    assert "Invalid arguments" in message


def test_api_errors_become_readable_tool_messages() -> None:
    server = FakeServer(
        {
            "status": 402,
            "body": {"error": {"code": "insufficient_points", "message": "Not enough points on this account."}},
        }
    )
    message = GeoGridRunGrid(client=server.client()).invoke({"place_id": "ChIJ1", "keyword": "pizza"})
    assert "Not enough GeoGridAPI credits" in message
    assert "geogridapi.io" in message


def test_find_business() -> None:
    server = FakeServer({"body": {"object": "list", "data": [{"place_id": "ChIJ1", "name": "Joe's Pizza"}]}})
    result = json.loads(GeoGridFindBusiness(client=server.client()).invoke({"q": "Joe's Pizza New York"}))
    assert result == {"places": [{"place_id": "ChIJ1", "name": "Joe's Pizza"}]}


def test_history_needs_a_map() -> None:
    assert "Provide grid_id, or place_id and keyword" in GeoGridGetHistory(client=FakeServer().client()).invoke({})


def test_run_grids_sends_items_and_format(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(
        {
            "status": 202,
            "body": {
                "id": "b1",
                "status": "completed",
                "items": [{"index": 0, "status": "completed", "grid_id": "g1"}],
            },
        }
    )
    result = json.loads(
        GeoGridRunGrids(client=server.client()).invoke(
            {"size": 5, "items": [{"place_id": "ChIJ1", "keyword": "pizza"}, {"place_id": "ChIJ1", "keyword": "pasta"}]}
        )
    )
    assert json.loads(server.requests[0].content) == {
        "size": 5,
        "items": [{"place_id": "ChIJ1", "keyword": "pizza"}, {"place_id": "ChIJ1", "keyword": "pasta"}],
    }
    assert result["batch_id"] == "b1"


def test_toolkit_read_only_leaves_out_spending_tools() -> None:
    client = FakeServer().client()
    all_names = {tool.name for tool in GeoGridAPIToolkit(client=client).get_tools()}
    read_only = {tool.name for tool in GeoGridAPIToolkit(client=client, read_only=True).get_tools()}

    assert len(all_names) == 13
    assert "geogrid_run_grid" in all_names
    assert "geogrid_run_grid" not in read_only
    assert "geogrid_delete_schedule" not in read_only
    assert "geogrid_get_balance" in read_only


def test_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEOGRIDAPI_API_KEY", "gga_test_env")
    assert GeoGridFindBusiness().gga.is_test_key


def test_errors_while_waiting_keep_the_grid_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("geogridapi._client.time.sleep", lambda _: None)
    server = FakeServer(
        {"status": 202, "body": {"id": "g-123", "status": "pending"}},
        {"status": 404, "body": {"error": {"code": "not_found", "message": "Not found."}}},
    )

    result = json.loads(GeoGridRunGrid(client=server.client()).invoke({"place_id": "ChIJ1", "keyword": "pizza"}))
    assert result["grid_id"] == "g-123"
    assert "do not run it again" in result["next_step"]


def test_failed_grid_shows_the_reason() -> None:
    failed = {
        "id": "g1",
        "status": "failed",
        "error": {"code": "place_not_found", "message": "No Google Business Profile matches this place."},
    }
    server = FakeServer({"status": 202, "body": failed})
    result = json.loads(GeoGridRunGrid(client=server.client()).invoke({"query": "Nowhere", "keyword": "pizza"}))
    assert result["error"]["code"] == "place_not_found"
    assert "refunded" in result["note"]
