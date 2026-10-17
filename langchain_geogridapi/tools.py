"""GeoGridAPI tools for LangChain agents (same tools as the GeoGridAPI MCP server)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from geogridapi import GeoGridAPI, GeoGridAPIError, InsufficientCreditsError, WaitTimeoutError
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from ._format import batch_view, grid_list_view, grid_view, to_json

TOP_UP_URL = "https://geogridapi.io/en/console"
GRID_POLL_SECONDS = 3.0
BATCH_POLL_SECONDS = 5.0


# --- Input schemas ---------------------------------------------------------------------------


class GridTargetInput(BaseModel):
    place_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=256,
        description="Google place id of the listing (ChIJ...), from geogrid_find_business. Give place_id or query.",
    )
    query: str | None = Field(
        default=None,
        min_length=2,
        max_length=200,
        description='Listing name and city, e.g. "Joe\'s Pizza, New York"; the first match is measured. '
        "Give place_id or query.",
    )
    keyword: str = Field(min_length=1, max_length=120, description="Keyword searched on Google Maps, e.g. 'pizza'.")

    @model_validator(mode="after")
    def _one_listing(self) -> GridTargetInput:
        if bool(self.place_id) == bool(self.query):
            raise ValueError("Provide exactly one of place_id or query.")
        return self


class GridFormatInput(BaseModel):
    shape: Literal["square", "circle", "diamond", "hexagon", "cross", "ellipse", "rectangle"] | None = Field(
        default=None, description="Grid shape (default square)."
    )
    size: Literal[3, 5, 7, 9] | None = Field(
        default=None, description="Square only (default 7): side of the grid. Cost: size x size credits."
    )
    density: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Other shapes only (default medium): 13 to 49 points."
    )
    radius_km: float | None = Field(
        default=None, ge=0.5, le=30, description="Distance from center to edge in km (0.5 to 30; shapes from 1)."
    )
    language: str | None = Field(default=None, pattern=r"^[a-zA-Z]{2}$", description="Search language, 2 letters.")
    country: str | None = Field(default=None, pattern=r"^[a-zA-Z]{2}$", description="Search country, ISO 2 letters.")

    def format_params(self) -> dict[str, Any]:
        return self.model_dump(include=set(GridFormatInput.model_fields), exclude_none=True)


class RunGridInput(GridTargetInput, GridFormatInput):
    wait_seconds: int = Field(
        default=120, ge=0, le=300, description="How long to wait for the result before returning the grid_id."
    )


class EstimateGridInput(GridTargetInput, GridFormatInput):
    pass


class FindBusinessInput(BaseModel):
    q: str = Field(min_length=2, max_length=200, description='Listing name and city, e.g. "Joe\'s Pizza, New York".')
    language: str | None = Field(default=None, pattern=r"^[a-zA-Z]{2}$", description="Result language, 2 letters.")


class GridIdInput(BaseModel):
    grid_id: str = Field(description="Grid id returned by geogrid_run_grid or geogrid_list_grids.")


class ListGridsInput(BaseModel):
    place_id: str | None = Field(default=None, description="Only the grids of this listing.")
    keyword: str | None = Field(default=None, description="Only the grids of this exact keyword (case-insensitive).")
    limit: int = Field(default=10, ge=1, le=100, description="Number of grids, newest first.")
    before: str | None = Field(default=None, description="next_before of the previous page.")


class HistoryInput(BaseModel):
    grid_id: str | None = Field(default=None, description="Any grid of the map to follow.")
    place_id: str | None = Field(default=None, description="With keyword: the listing to follow.")
    keyword: str | None = Field(default=None, description="With place_id: the keyword to follow.")
    limit: int | None = Field(default=None, ge=1, le=100, description="Number of grids (default 50).")

    @model_validator(mode="after")
    def _one_map(self) -> HistoryInput:
        if not self.grid_id and not (self.place_id and self.keyword):
            raise ValueError("Provide grid_id, or place_id and keyword.")
        return self


class RunGridsInput(GridFormatInput):
    items: list[GridTargetInput] = Field(
        min_length=1,
        max_length=50,
        description="1 to 50 grids, each with place_id or query and a keyword: one listing with several keywords, "
        "or several listings with one keyword.",
    )
    wait_seconds: int = Field(
        default=60, ge=0, le=600, description="How long to wait for every grid before returning the batch_id."
    )


class BatchIdInput(BaseModel):
    batch_id: str = Field(description="Batch id returned by geogrid_run_grids.")


class ScheduleGridInput(BaseModel):
    grid_id: str = Field(description="An existing completed grid to repeat.")
    frequency: Literal["weekly", "monthly"]
    weekday: int | None = Field(default=None, ge=1, le=7, description="Weekly: 1 = Monday ... 7 = Sunday.")
    month_day: int | None = Field(default=None, ge=1, le=28, description="Monthly: day of the month, 1 to 28.")


class UpdateScheduleInput(BaseModel):
    schedule_id: str
    status: Literal["active", "paused"] = Field(description="paused to pause, active to resume.")


class ScheduleIdInput(BaseModel):
    schedule_id: str


class NoInput(BaseModel):
    pass


# --- Base tool -------------------------------------------------------------------------------


def _validation_message(error: ValidationError | Any) -> str:
    try:
        issues = error.errors()
    except AttributeError:
        return f"Invalid arguments: {error}"
    parts = [f"{'.'.join(str(p) for p in issue.get('loc', ())) or 'input'}: {issue.get('msg')}" for issue in issues]
    return "Invalid arguments: " + "; ".join(parts)


def _explain(error: GeoGridAPIError) -> str:
    if isinstance(error, InsufficientCreditsError):
        return f"Not enough GeoGridAPI credits for this request ({error.message}). Top up at {TOP_UP_URL}."
    return f"GeoGridAPI error {error.code}: {error.message}"


def _still_running(error: GeoGridAPIError, tool: str, id_field: str) -> str:
    reason = (
        "it is still running" if isinstance(error, WaitTimeoutError) else f"it could not be read yet ({error.code})"
    )
    return f"Created and paid for, but {reason}. Call {tool} with this {id_field} later; do not run it again."


class GeoGridBaseTool(BaseTool):
    """Shared client and error handling. Pass ``client`` or ``api_key``, or set ``GEOGRIDAPI_API_KEY``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: GeoGridAPI | None = Field(default=None, exclude=True)
    api_key: SecretStr | None = Field(default=None, exclude=True)
    handle_tool_error: bool = True
    handle_validation_error: Callable[[Any], str] = _validation_message

    @model_validator(mode="after")
    def _build_client(self) -> GeoGridBaseTool:
        if self.client is None:
            key = self.api_key.get_secret_value() if self.api_key else None
            self.client = GeoGridAPI(api_key=key)
        return self

    @property
    def gga(self) -> GeoGridAPI:
        assert self.client is not None
        return self.client

    def _call(self, action: Callable[[], Any]) -> str:
        try:
            return to_json(action())
        except GeoGridAPIError as error:
            raise ToolException(_explain(error)) from error


# --- Tools -----------------------------------------------------------------------------------


class GeoGridFindBusiness(GeoGridBaseTool):
    name: str = "geogrid_find_business"
    description: str = (
        "Search Google Maps listings (Google Business Profiles) by name and city. "
        "Returns place_id values to use with geogrid_run_grid. Free."
    )
    args_schema: type[BaseModel] = FindBusinessInput

    def _run(self, q: str, language: str | None = None, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return self._call(lambda: {"places": self.gga.places.search(q, language=language)["data"]})


class GeoGridEstimateGrid(GeoGridBaseTool):
    name: str = "geogrid_estimate_grid"
    description: str = (
        "Exact number of points and credits a grid would cost before running it (points at sea are removed), "
        "and the balance after. Same parameters as geogrid_run_grid. Free: nothing is reserved or charged."
    )
    args_schema: type[BaseModel] = EstimateGridInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None, **kwargs: Any) -> str:
        params = EstimateGridInput(**kwargs).model_dump(exclude_none=True)

        def estimate() -> dict[str, Any]:
            result = self.gga.grids.estimate(**params)
            keys = ("points", "points_at_sea", "exact", "credits", "balance", "balance_after", "enough")
            return {key: result.get(key) for key in keys}

        return self._call(estimate)


class GeoGridRunGrid(GeoGridBaseTool):
    name: str = "geogrid_run_grid"
    description: str = (
        "Measure where a Google Business Profile ranks on Google Maps for a keyword, on every point of a grid "
        "around it. Shapes: square (size 3, 5, 7 or 9) or circle, diamond, hexagon, cross, ellipse, rectangle "
        "(density low, medium or high, up to 49 points). Costs one credit per point (square 3x3 = 9, 7x7 = 49, "
        "9x9 = 81; points at sea are skipped and not charged), refunded if the grid fails; call "
        "geogrid_estimate_grid first when the budget matters. Returns the ranks matrix (north at the top, lower is "
        "better, null = not in the top 20), the summary, the competitors ranking ahead and the change since the "
        "previous grid. Test keys (gga_test_) return free mock grids."
    )
    args_schema: type[BaseModel] = RunGridInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None, **kwargs: Any) -> str:
        args = RunGridInput(**kwargs)
        params = args.model_dump(exclude={"wait_seconds"}, exclude_none=True)

        def run() -> dict[str, Any]:
            grid = self.gga.grids.create(**params)
            if grid.get("status") in ("pending", "running") and args.wait_seconds > 0:
                try:
                    grid = self.gga.grids.wait(grid["id"], poll_interval=GRID_POLL_SECONDS, timeout=args.wait_seconds)
                except GeoGridAPIError as error:
                    # The grid exists and is paid for: never let the agent lose its id and run it again.
                    return {**grid_view(dict(grid)), "next_step": _still_running(error, "geogrid_get_grid", "grid_id")}
            return grid_view(dict(grid))

        return self._call(run)


class GeoGridGetGrid(GeoGridBaseTool):
    name: str = "geogrid_get_grid"
    description: str = (
        "Status and result of a grid: ranks matrix (north at the top), summary, competitors most often in the "
        "top 3 and change since the previous grid of the same listing. Free."
    )
    args_schema: type[BaseModel] = GridIdInput

    def _run(self, grid_id: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return self._call(lambda: grid_view(dict(self.gga.grids.get(grid_id))))


class GeoGridListGrids(GeoGridBaseTool):
    name: str = "geogrid_list_grids"
    description: str = (
        "The account's grids, newest first, optionally for one listing (place_id) or one exact keyword. "
        "Returns grid_id values to use with geogrid_get_grid, geogrid_get_history or geogrid_schedule_grid. Free."
    )
    args_schema: type[BaseModel] = ListGridsInput

    def _run(
        self,
        place_id: str | None = None,
        keyword: str | None = None,
        limit: int = 10,
        before: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self._call(
            lambda: grid_list_view(
                dict(self.gga.grids.list(limit=limit, before=before, place_id=place_id, keyword=keyword))
            )
        )


class GeoGridGetHistory(GeoGridBaseTool):
    name: str = "geogrid_get_history"
    description: str = (
        "How a listing's ranking evolves for a keyword: every completed grid of the same map (same format, radius "
        "and center), oldest first, with the change between the first and the last (negative average_rank = "
        "better). Give grid_id, or place_id and keyword. Free."
    )
    args_schema: type[BaseModel] = HistoryInput

    def _run(
        self,
        grid_id: str | None = None,
        place_id: str | None = None,
        keyword: str | None = None,
        limit: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def history() -> dict[str, Any]:
            result = self.gga.history.get(grid_id=grid_id, place_id=place_id, keyword=keyword, limit=limit)
            return {key: result.get(key) for key in ("place", "keyword", "shape", "size", "density", "data", "change")}

        return self._call(history)


class GeoGridGetBalance(GeoGridBaseTool):
    name: str = "geogrid_get_balance"
    description: str = "Credits left on the GeoGridAPI account (1 credit = 1 grid point). Free."
    args_schema: type[BaseModel] = NoInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return self._call(lambda: {"credits": self.gga.balance.get()["points"]})


class GeoGridRunGrids(GeoGridBaseTool):
    name: str = "geogrid_run_grids"
    description: str = (
        "Run up to 50 grids in one call, all with the same format: one listing with several keywords, or several "
        "listings with one keyword. Each item needs place_id or query, and a keyword. Costs one credit per point "
        "of each grid (refunded if a grid fails); refused upfront when the balance cannot cover them all. "
        "Waits up to wait_seconds, then returns the batch so far: call geogrid_get_batch for the rest."
    )
    args_schema: type[BaseModel] = RunGridsInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None, **kwargs: Any) -> str:
        args = RunGridsInput(**kwargs)
        items: list[Any] = [item.model_dump(exclude_none=True) for item in args.items]

        def run() -> dict[str, Any]:
            batch = self.gga.batches.create(items=items, **args.format_params())
            if batch.get("status") != "completed" and args.wait_seconds > 0:
                try:
                    batch = self.gga.batches.wait(
                        batch["id"], poll_interval=BATCH_POLL_SECONDS, timeout=args.wait_seconds
                    )
                except GeoGridAPIError as error:
                    return {
                        **batch_view(dict(batch)),
                        "next_step": _still_running(error, "geogrid_get_batch", "batch_id"),
                    }
            return batch_view(dict(batch))

        return self._call(run)


class GeoGridGetBatch(GeoGridBaseTool):
    name: str = "geogrid_get_batch"
    description: str = (
        "Progress of a batch started with geogrid_run_grids: each grid waiting, running, completed with its "
        "average rank, or failed and why. Free."
    )
    args_schema: type[BaseModel] = BatchIdInput

    def _run(self, batch_id: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return self._call(lambda: batch_view(dict(self.gga.batches.get(batch_id))))


class GeoGridScheduleGrid(GeoGridBaseTool):
    name: str = "geogrid_schedule_grid"
    description: str = (
        "Rerun an existing grid (same listing, keyword, format and radius) every week (weekday 1 = Monday ... "
        "7 = Sunday) or every month (month_day 1 to 28), around 7 am Paris time. Each run costs the grid's "
        "credits; the schedule stops and the account gets an email when the balance runs out."
    )
    args_schema: type[BaseModel] = ScheduleGridInput

    def _run(
        self,
        grid_id: str,
        frequency: Literal["weekly", "monthly"],
        weekday: int | None = None,
        month_day: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self._call(
            lambda: self.gga.schedules.create(
                grid_id=grid_id, frequency=frequency, weekday=weekday, month_day=month_day
            )
        )


class GeoGridListSchedules(GeoGridBaseTool):
    name: str = "geogrid_list_schedules"
    description: str = (
        "The account's repeated grids: rhythm, status (active, paused, or stopped and why), next run and credits "
        "per month. Free."
    )
    args_schema: type[BaseModel] = NoInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return self._call(lambda: {"schedules": self.gga.schedules.list()["data"]})


class GeoGridUpdateSchedule(GeoGridBaseTool):
    name: str = "geogrid_update_schedule"
    description: str = (
        "Pause a repeated grid (status paused) or resume it (status active; a schedule stopped for lack of "
        "credits restarts too)."
    )
    args_schema: type[BaseModel] = UpdateScheduleInput

    def _run(
        self,
        schedule_id: str,
        status: Literal["active", "paused"],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self._call(lambda: self.gga.schedules.update(schedule_id, status=status))


class GeoGridDeleteSchedule(GeoGridBaseTool):
    name: str = "geogrid_delete_schedule"
    description: str = "Stop repeating a grid for good. Grids already run are kept."
    args_schema: type[BaseModel] = ScheduleIdInput

    def _run(self, schedule_id: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        return self._call(lambda: self.gga.schedules.delete(schedule_id))


READ_ONLY_TOOLS: tuple[type[GeoGridBaseTool], ...] = (
    GeoGridFindBusiness,
    GeoGridEstimateGrid,
    GeoGridGetGrid,
    GeoGridListGrids,
    GeoGridGetHistory,
    GeoGridGetBalance,
    GeoGridGetBatch,
    GeoGridListSchedules,
)
"""Free tools that change nothing."""

WRITE_TOOLS: tuple[type[GeoGridBaseTool], ...] = (
    GeoGridRunGrid,
    GeoGridRunGrids,
    GeoGridScheduleGrid,
    GeoGridUpdateSchedule,
    GeoGridDeleteSchedule,
)
"""Tools that spend credits or change schedules."""
