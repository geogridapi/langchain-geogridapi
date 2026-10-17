"""Compact, token-friendly views of API objects for an LLM."""

from __future__ import annotations

import json
from typing import Any

MAX_COMPETITORS = 5


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _place(place: Any) -> dict[str, Any] | None:
    if not isinstance(place, dict):
        return None
    return {key: place.get(key) for key in ("place_id", "name", "address")}


def grid_view(grid: dict[str, Any]) -> dict[str, Any]:
    """A grid without per-point details: ranks matrix, summary, top competitors and change since last time."""
    status = grid.get("status")
    view: dict[str, Any] = {
        "grid_id": grid.get("id"),
        "status": status,
        "mode": grid.get("mode"),
        "place": _place(grid.get("place")),
        "keyword": grid.get("keyword"),
        "shape": grid.get("shape"),
        "size": grid.get("size"),
        "density": grid.get("density"),
        "radius_km": grid.get("radius_km"),
        "points_total": grid.get("points_total"),
        "credits_charged": grid.get("points_charged"),
        "created_at": grid.get("created_at"),
    }
    if status in ("pending", "running"):
        view["next_step"] = "Still running (20 s to 2 min in total): call geogrid_get_grid with this grid_id shortly."
        return view
    if status == "failed":
        view["error"] = grid.get("error")  # {"code", "message"}
        view["note"] = "The grid failed and its credits were refunded."
        return view
    view["summary"] = grid.get("summary")
    view["ranks"] = grid.get("ranks")
    view["ranks_legend"] = "Row 0 = north, column 0 = west. Lower is better; null = not in the top 20."
    competitors = grid.get("competitors") or []
    view["competitors_ahead"] = competitors[:MAX_COMPETITORS]
    previous = grid.get("previous")
    if isinstance(previous, dict):
        view["since_previous_grid"] = {
            "previous_grid_id": previous.get("grid_id"),
            "previous_created_at": previous.get("created_at"),
            "changes": previous.get("changes"),
            "points": previous.get("points"),
        }
    view["completed_at"] = grid.get("completed_at")
    return view


def grid_list_view(page: dict[str, Any]) -> dict[str, Any]:
    grids = [
        {
            "grid_id": grid.get("id"),
            "status": grid.get("status"),
            "place": _place(grid.get("place")),
            "keyword": grid.get("keyword"),
            "shape": grid.get("shape"),
            "size": grid.get("size"),
            "density": grid.get("density"),
            "average_rank": (grid.get("summary") or {}).get("average_rank"),
            "created_at": grid.get("created_at"),
        }
        for grid in page.get("data", [])
    ]
    return {"grids": grids, "has_more": page.get("has_more"), "next_before": page.get("next_before")}


def batch_view(batch: dict[str, Any]) -> dict[str, Any]:
    items = [
        {
            "index": item.get("index"),
            "keyword": item.get("keyword"),
            "place": _place(item.get("place")) or item.get("query"),
            "status": item.get("status"),
            "grid_id": item.get("grid_id"),
            "summary": item.get("summary"),
            "error": item.get("error"),
        }
        for item in batch.get("items", [])
    ]
    view = {
        "batch_id": batch.get("id"),
        "status": batch.get("status"),
        "counts": batch.get("counts"),
        "credits_estimated": batch.get("credits_estimated"),
        "credits_charged": batch.get("credits_charged"),
        "items": items,
    }
    if batch.get("status") != "completed":
        view["next_step"] = "Grids are still running: call geogrid_get_batch with this batch_id later."
    return view
