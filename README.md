# Dataverse MCP Server Evaluation Framework

Automated evaluation framework for testing the Dataverse MCP server using Azure AI Foundry evaluators and an LLM agent powered by GitHub Models.

## Quick Start

### Prerequisites
- Python 3.10+
- GitHub Enterprise token (for GitHub Models API)
- Azure AI Foundry project (for evaluation)
- Access to the Dataverse MCP server

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
| `GITHUB_TOKEN` | GitHub PAT with Models API access |
| `MCP_SERVER_URL` | Dataverse MCP server URL |
| `MCP_AUTH_TOKEN` | Bearer token for MCP server authentication |
| `AZURE_AI_CONNECTION_STRING` | Azure AI Foundry project connection string |
| `GITHUB_MODELS_MODEL` | Model to use (default: `gpt-4o`) |

### Running Evaluations

The framework supports staged execution to handle Dataverse propagation delays (~15 min for some operations).

```bash
# Run all stages with automatic wait
python run.py all --wait 900

# Or run stages individually
python run.py setup      # Creates tables, records, skills
# ... wait ~15 minutes ...
python run.py verify     # Validates resources via search/list
python run.py teardown   # Cleans up all resources
```

### Running with pytest

```bash
# Run all tests
pytest

# Run specific stage
pytest -m stage_setup
pytest -m stage_verify
pytest -m stage_teardown

# Run specific scenario group
pytest tests/test_crud.py
pytest tests/test_search.py
pytest tests/test_skills.py
```

## Architecture

```
LLM Agent (GPT-4o via GitHub Models)
    │
    ├── Sends prompts from scenarios.yaml
    ├── Receives tool call requests
    │
    ▼
MCP Client (HTTP)
    │
    ├── Discovers tools via tools/list
    ├── Invokes tools on Dataverse MCP server
    ├── Captures full traces
    │
    ▼
Azure AI Foundry Evaluators
    │
    ├── ToolCallAccuracyEvaluator
    ├── TaskCompletionEvaluator
    ├── IntentResolutionEvaluator
    └── Custom MCP evaluators
```

## Test Scenarios

| # | Scenario | Stage | Description |
|---|----------|-------|-------------|
| 1 | Create table | setup | Create a Projects table with Name, Status, DueDate |
| 2 | Create record | setup | Create a record in the Projects table |
| 3 | Fetch record | verify | Fetch details of the created record |
| 4 | Delete record | teardown | Delete the record |
| 5 | Delete table | teardown | Delete the Projects table |
| 6 | List all tables | verify | List all tables in Dataverse |
| 7 | Search (search tool) | verify | Search for account tables |
| 8 | Search records (search_data) | verify | Find specific records by name |
| 9 | Create table + search | setup+verify | Create table then verify it appears in search |
| 10 | Create skill | setup | Create a joke-telling skill |
| 11 | List all skills | verify | List all skills in Dataverse |
| 12 | Follow a skill | verify | Verify agent follows skill instructions |
| 13 | Delete skill | teardown | Delete the created skill |

## Results

Test results are saved to `results/` (gitignored) including:
- `state.json` — Resource IDs persisted between stages
- `results_<timestamp>.json` — Full evaluation results with scores and traces
