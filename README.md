# Dataverse MCP Server Evaluation Framework

Automated evaluation harness that tests a **Dataverse MCP server** by running an LLM agent against it and grading the agent's behavior. It validates that the MCP server correctly exposes Dataverse operations (CRUD tables, records, skills, search, queries) and that an AI agent can successfully use those tools.

## How It Works

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────┐
│  Scenarios  │────▶│  LLM Agent    │────▶│  Dataverse MCP   │
│  (YAML)     │     │  (GPT-4.1)    │     │  Server          │
└─────────────┘     └───────────────┘     └──────────────────┘
                           │                        │
                    ┌──────▼──────┐          (actual API calls)
                    │  Evaluator  │
                    │  (4 checks) │
                    └─────────────┘
```

### 3-Stage Lifecycle

- **Setup** — Creates resources (table, record, skill)
- **Verify** — Validates the agent can query/search/describe those resources
- **Teardown** — Cleans up everything

### Evaluation (4 evaluators)

1. **tool_call_check** — Did the agent call the expected tool? (supports acceptable alternatives with tiered scoring)
2. **tool_param_check** — Were the right parameters passed?
3. **success_criteria** — LLM-graded check against natural language criteria
4. **response_content** — Concrete string matching against actual tool responses (not LLM summaries)

### Key Design Decisions

- **Cross-tenant auth** — MCP server can be in a different tenant; uses MSAL device code flow with auto-refreshing token provider
- **Propagation retries** — Dataverse has eventual consistency; the harness retries (try-first, wait 30s between retries, up to 5 attempts) rather than hardcoding long sleeps
- **Dependency graph** — Scenarios declare dependencies; unmet deps are auto-resolved across stages, with cycle detection at load time
- **Error-resilient scoring** — Skips agent self-corrections (first tool call fails, retry succeeds); only scores successful responses
- **Single-file HTML report** — Embeds JSON data for easy sharing; includes visual charts + downloadable raw data

## Quick Start

### Prerequisites
- Python 3.10+
- Azure OpenAI resource with API key (via Azure AI Foundry)
- Access to the Dataverse MCP server (may be in a separate tenant)

### Setup

```bash
# Clone and install
cd dataverse-mcp-eval
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint (e.g. `https://your-resource.openai.azure.com/openai/v1`) |
| `AZURE_OPENAI_API_KEY` | API key for Azure OpenAI |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (default: `gpt-4.1`) |
| `MCP_SERVER_URL` | Dataverse MCP server URL (e.g. `https://myorg.crm.dynamics.com/api/mcp`) |
| `MCP_AUTH_TOKEN` | Bearer token for MCP (optional — if empty, device code flow prompts sign-in) |
| `MCP_TENANT_ID` | Target tenant ID for device code auth (optional) |
| `AZURE_AI_CONNECTION_STRING` | Azure AI Foundry project connection string (optional) |

### Running Evaluations

```bash
# Run all stages with automatic propagation wait between stages
python run.py all --wait 30

# Or run stages individually
python run.py setup      # Creates tables, records, skills
python run.py verify     # Validates resources via search/list/query
python run.py teardown   # Cleans up all resources

# Verbose output
python run.py --verbose all --wait 30
```

## Test Scenarios

| # | Scenario | Stage | Description |
|---|----------|-------|-------------|
| 1 | Create table | setup | Create an EvalProjects table with Name, Status, DueDate |
| 2 | Create record | setup | Create a record in the EvalProjects table |
| 3 | Create skill | setup | Create a joke-telling skill (EvalJokeTeller) |
| 4 | Fetch record | verify | Query the created record via SQL |
| 5 | List all tables | verify | List all tables using describe |
| 6 | Search (search tool) | verify | Search for account tables |
| 7 | Search records | verify | Find specific records by query |
| 8 | Create table + search | verify | Verify created table appears in describe |
| 9 | List all skills | verify | List all skills via describe |
| 10 | Follow a skill | verify | Look up skill and follow its instructions |
| 11 | Delete record | teardown | Find and delete the test record |
| 12 | Delete table | teardown | Delete the EvalProjects table |
| 13 | Delete skill | teardown | Delete the EvalJokeTeller skill |

## Results

Results are saved to `results/` (gitignored) including:
- `report_<timestamp>.html` — Self-contained HTML report with embedded JSON (single file to share)
- `results_<timestamp>.json` — Raw evaluation results with scores and traces
- `state.json` — Resource status persisted between stages
