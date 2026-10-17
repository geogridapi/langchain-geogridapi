"""LangChain standard unit tests for every tool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_geogridapi import GeoGridFindBusiness, GeoGridGetGrid, GeoGridRunGrid, GeoGridRunGrids

KEY = {"api_key": "gga_test_standard"}


class TestFindBusiness(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return GeoGridFindBusiness

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return KEY

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"q": "Joe's Pizza, New York"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"GEOGRIDAPI_API_KEY": "gga_test_env"}, {}, {}


class TestRunGrid(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return GeoGridRunGrid

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return KEY

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "Joe's Pizza, New York", "keyword": "pizza", "size": 7}


class TestRunGrids(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return GeoGridRunGrids

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return KEY

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"size": 5, "items": [{"place_id": "ChIJ1", "keyword": "pizza"}]}


class TestGetGrid(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return GeoGridGetGrid

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return KEY

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"grid_id": "3f2c9b1e-6a7d-4c1e-9f0a-2b8d7e6c5a41"}
