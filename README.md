# MisterRisker

A LangGraph-based Slack messaging agent that sends messages to Slack and receives responses.

## Features

- 📤 Send messages to Slack channels
- 📥 Wait for and receive responses
- 🔄 Interactive conversation mode
- 🏗️ Built with LangGraph for stateful workflow management
- ✅ Comprehensive test suite (TDD approach)

## Installation

```bash
# Clone the repository
git clone https://github.com/jhancock1975/misterrisker.git
cd misterrisker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Slack credentials:
   - `SLACK_BOT_TOKEN`: Your Slack bot OAuth token (starts with `xoxb-`)
   - `SLACK_CHANNEL_ID`: The channel ID to send messages to
   - `SLACK_TIMEOUT`: (optional) Response timeout in seconds

### Getting Slack Credentials

1. Go to [Slack API Apps](https://api.slack.com/apps)
2. Create a new app or select an existing one
3. Under "OAuth & Permissions", add these bot token scopes:
   - `chat:write` - Send messages
   - `channels:history` - Read messages from public channels
   - `groups:history` - Read messages from private channels (if needed)
4. Install the app to your workspace
5. Copy the Bot User OAuth Token

## Usage

### Send a single message and wait for response:

```bash
python main.py "Hello from LangGraph!"
```

### Send without waiting for response:

```bash
python main.py --no-wait "Fire and forget message"
```

### Interactive conversation mode:

```bash
python main.py --interactive
```

## Project Structure

```
misterrisker/
├── main.py                 # Application entry point
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── slack_agent.py  # SlackAgent class
│   └── workflows/
│       ├── __init__.py
│       └── slack_workflow.py  # LangGraph workflow
├── tests/
│   ├── __init__.py
│   ├── test_slack_agent.py     # SlackAgent tests
│   └── test_slack_workflow.py  # Workflow tests
├── resources/
│   └── schwab-api.md       # Schwab API documentation
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
└── .env.example            # Example environment file
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_slack_agent.py -v
```

## Architecture

This project uses **LangGraph** to orchestrate the Slack messaging workflow:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Entry Point    │────▶│  Send Message    │────▶│ Receive Message │
│  (Conditional)  │     │  Node            │     │ Node            │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                    ┌─────────┐
                                                    │   END   │
                                                    └─────────┘
```

- **SlackAgent**: Handles direct communication with Slack API
- **SlackWorkflow**: Orchestrates the send/receive flow using LangGraph
- **SlackState**: TypedDict managing conversation state

## Development

This project was built using Test-Driven Development (TDD):

1. Tests were written first in `tests/`
2. Implementation was created to pass the tests
3. Refactoring maintains test coverage

## License

MIT
