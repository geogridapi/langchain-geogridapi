"""All GeoGridAPI tools in one toolkit."""

from __future__ import annotations

from geogridapi import GeoGridAPI
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field, SecretStr, model_validator

from .tools import READ_ONLY_TOOLS, WRITE_TOOLS


class GeoGridAPIToolkit(BaseToolkit):
    """GeoGridAPI tools for an agent.

    Pass ``client`` or ``api_key``, or set ``GEOGRIDAPI_API_KEY``. With ``read_only=True`` the agent only gets
    the free tools that change nothing (search, estimate, read grids, history, balance).

    >>> from langchain_geogridapi import GeoGridAPIToolkit
    >>> tools = GeoGridAPIToolkit().get_tools()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: GeoGridAPI | None = Field(default=None, exclude=True)
    api_key: SecretStr | None = Field(default=None, exclude=True)
    read_only: bool = False

    @model_validator(mode="after")
    def _build_client(self) -> GeoGridAPIToolkit:
        if self.client is None:
            self.client = GeoGridAPI(api_key=self.api_key.get_secret_value() if self.api_key else None)
        return self

    def get_tools(self) -> list[BaseTool]:
        classes = READ_ONLY_TOOLS if self.read_only else READ_ONLY_TOOLS + WRITE_TOOLS
        # Every concrete tool class sets its own name and description.
        return [tool_class(client=self.client) for tool_class in classes]  # type: ignore[call-arg]
