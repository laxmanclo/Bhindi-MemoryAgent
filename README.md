# bhindi-memoryOS

A memory management system for AI agents, enabling contextual memory storage and retrieval through a Python API with a Bhindi agent integration.

> **Note:** This project is inspired by and adapts code from [BAI-LAB's MemoryOS](https://github.com/BAI-LAB/MemoryOS) under the Apache License 2.0.

## Components

The system consists of two main components:

1. **MemoryOS Python API** (`MemoryOS/memoryos-pypi/`): Core memory functionality, including:
   - Short-term memory management
   - Mid-term memory with heat threshold
   - Long-term knowledge storage
   - Memory retrieval services

2. **Bhindi Agent** (`MemoryOS/bhindi-agent/`): Node.js-based agent providing memory tools:
   - Query memory to retrieve contextual responses
   - Create and manage chat sessions
   - Add memory interactions
   - Communicate with the MemoryOS API

3. **Ngrok** (`ngrok/`): Used for exposing the local API to the internet for integration with external services.

## Deployment

### Prerequisites

- Windows machine
- PowerShell
- Node.js and npm installed
- Python installed with pip
- ngrok account (for external access)

### Quick Start

1. Start both services using the provided script:
   ```
   .\start-services.bat
   ```
   This will:
   - Start the MemoryOS API on http://localhost:5000
   - Start the Bhindi agent on http://localhost:3000

2. Test the integration:
   ```
   .\test-memory.ps1
   ```
   This performs basic memory operations to verify functionality.

### External Access Setup

To expose your local MemoryOS API for external access:

1. Start ngrok tunnel:
   ```
   .\ngrok\ngrok.exe http 5000
   ```

2. Update the Bhindi agent with the ngrok URL:
   - Create/modify `.env` file in `MemoryOS/bhindi-agent/` with:
     ```
     MEMORYOS_API_URL=https://your-ngrok-url-here
     PORT=3000
     ```
   - Restart the Bhindi agent

## API Endpoints

The MemoryOS API provides several endpoints:

- `/tools`: Lists available memory tools
- `/tools/<tool_name>`: Executes specific memory operations
- `/resource`: Provides contextual information about the memory system
- `/chats`: Gets a list of available chat sessions
- `/new_chat`: Creates a new chat session
- `/switch_chat`: Switches to an existing chat session
- `/chat`: Processes chat messages and retrieves memory-enhanced responses
- `/agent_query`: Endpoint for agents to query MemoryOS
- `/agent_memory`: Endpoint for agents to add memories
- `/health`: Health check endpoint

## Bhindi Agent Tools

The Bhindi agent provides these memory tools:

- `queryMemory`: Retrieve contextual information from memory
- `createNewChat`: Create a new chat session
- `switchToChat`: Switch to an existing chat session
- `addMemory`: Store new memory interactions
- `getChatList`: Retrieve a list of available chat sessions

## Troubleshooting

Common issues and solutions:

1. **Service Connection Issues**:
   - Verify both services are running (check terminal windows)
   - Check that ports 5000 and 3000 are available

2. **Memory Not Stored or Retrieved**:
   - Check log output in terminal windows
   - Verify environment variables are set correctly

3. **API Response Format Issues**:
   - The fixed memory service handles different response formats
   - If format errors persist, check actual API endpoint responses

4. **Ngrok Issues**:
   - If tunnel expires, restart ngrok and update URLs
   - Update the URL in the Bhindi agent's .env file

## Configuration

### MemoryOS API Configuration

Key configuration options in `app.py`:

- `USER_ID`: The user ID for memory storage
- `ASSISTANT_ID`: The assistant ID for memory context
- `OPENROUTER_MODEL`: The AI model to use for responses
- `DATA_STORAGE_PATH`: Path for local memory storage

### Bhindi Agent Configuration

Key configuration in `.env`:

- `MEMORYOS_API_URL`: URL of the MemoryOS API
- `PORT`: Port to run the Bhindi agent on (default: 3000)

## Adding as a Bhindi Agent

To add MemoryOS as a Bhindi agent in your project:

1. Clone this repository to your local machine
2. Start the MemoryOS API and Bhindi agent using `.\start-services.bat`
3. Configure your Bhindi agent settings:
   - Update your agent configuration to include the MemoryOS tools
   - Set the endpoint to `http://localhost:3000` (or your ngrok URL for external access)
4. Use the provided memory tools in your agent prompts:
   ```
   - queryMemory: Retrieve contextual information
   - createNewChat: Start a new chat session
   - switchToChat: Change to a different chat
   - addMemory: Store new information
   - getChatList: List available chat sessions
   ```

5. Test the integration by making a sample query to ensure proper functionality

## GitHub Repository Setup

To set up this project as a GitHub repository named "bhindi-memoryOS":

1. Create a new repository on GitHub:
   - Go to [GitHub](https://github.com) and sign in
   - Click the "+" icon in the top-right corner and select "New repository"
   - Enter "bhindi-memoryOS" as the repository name
   - Add an optional description
   - Choose public or private visibility
   - Click "Create repository"

2. Push your local repository to GitHub:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/bhindi-memoryOS.git
   git push -u origin main
   ```
   (Replace `YOUR-USERNAME` with your GitHub username)

## License

This project is based on [BAI-LAB's MemoryOS](https://github.com/BAI-LAB/MemoryOS) and is distributed under the Apache License 2.0.

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
```

See the LICENSE file for full details.
