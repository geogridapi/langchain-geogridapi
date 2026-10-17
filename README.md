# langchain-geogridapi

[![PyPI](https://img.shields.io/pypi/v/langchain-geogridapi.svg)](https://pypi.org/project/langchain-geogridapi/)

[LangChain](https://www.langchain.com/) tools for [GeoGridAPI](https://geogridapi.io): let an AI agent measure where a Google Business Profile ranks on Google Maps, point by point around the business (geo-grid rank tracking for local SEO), see who ranks ahead and follow the trend over time.

The same 13 tools as the GeoGridAPI MCP server, for LangChain and LangGraph agents.

## Install

```bash
pip install langchain-geogridapi
```

Create an account on [geogridapi.io](https://geogridapi.io) (500 free credits), create an API key in the console and set it:

```bash
export GEOGRIDAPI_API_KEY="gga_live_..."  # or a gga_test_ key: free mock grids
```

## Use with an agent

The example uses LangChain's agent and an Anthropic model: `pip install langchain langchain-anthropic` and set `ANTHROPIC_API_KEY` (any chat model with tool calling works).

```python
from langchain.agents import create_agent
from langchain_geogridapi import GeoGridAPIToolkit

tools = GeoGridAPIToolkit().get_tools()
agent = create_agent("anthropic:claude-sonnet-5-5", tools)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How does Joe's Pizza in New York rank for 'pizza' around the shop? Who ranks ahead?",
            }
        ]
    }
)
print(result["messages"][-1].content)
```

The agent finds the listing, can estimate the cost, runs the grid and answers with the average rank, the rank at every point and the competitors ahead.

`GeoGridAPIToolkit(read_only=True)` gives only the free tools that change nothing (search, estimate, read grids, history, balance): the agent can explore past grids without spending credits.

## Tools

| Tool | What it does | Cost |
|---|---|---|
| `geogrid_find_business` | Finds a Google Business Profile by name and city, with its place_id | Free |
| `geogrid_estimate_grid` | Exact points and credits of a grid before running it | Free |
| `geogrid_run_grid` | Measures a listing for a keyword on a grid (square 3×3 to 9×9, or circle, diamond, hexagon, cross, ellipse, rectangle) and waits for the result | 1 credit per point |
| `geogrid_get_grid` | Ranks matrix, summary, competitors ahead and change since the previous grid | Free |
| `geogrid_list_grids` | Past grids, by listing or keyword | Free |
| `geogrid_get_history` | Ranking trend of the same map over time | Free |
| `geogrid_get_balance` | Credits left | Free |
| `geogrid_run_grids` | Up to 50 grids in one call | 1 credit per point |
| `geogrid_get_batch` | Progress of a batch | Free |
| `geogrid_schedule_grid` | Repeats a grid every week or month | Credits of each run |
| `geogrid_list_schedules`, `geogrid_update_schedule`, `geogrid_delete_schedule` | List, pause, resume or stop repeated grids | Free |

Tools return compact JSON (no per-point coordinates) to save tokens. When a grid outlives `wait_seconds`, the tool returns its `grid_id` with a `next_step`: the agent reads it later with `geogrid_get_grid` instead of paying for it twice. Ranks are positions in the Google Maps results at each grid point: lower is better, `null` means not in the top 20. A failed grid is refunded. API errors (not enough credits, unknown listing…) come back to the agent as a readable message instead of an exception.

### One tool at a time

```python
from geogridapi import GeoGridAPI
from langchain_geogridapi import GeoGridFindBusiness, GeoGridRunGrid

client = GeoGridAPI(api_key="gga_live_...")
run_grid = GeoGridRunGrid(client=client)

print(run_grid.invoke({"query": "Joe's Pizza, New York", "keyword": "pizza", "size": 5}))
```

Every tool also accepts `api_key="..."`; by default they read `GEOGRIDAPI_API_KEY`.

## Also available

- [`geogridapi`](https://pypi.org/project/geogridapi/): the Python SDK these tools are built on (sync and async)
- **MCP server** for Claude, ChatGPT, Cursor and any MCP client: `https://api.geogridapi.io/mcp`
- **Node.js**: `npm install geogridapi`
- [API reference](https://geogridapi.io/en/docs) · [OpenAPI specification](https://geogridapi.io/openapi.json)

## License

[MIT](LICENSE)
