import os
import json
import uuid
import time
import asyncio
import threading
from typing import Optional, Dict, List, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

# Import memoryos components
from memoryos import Memoryos

# Load environment variables
load_dotenv()

# === Configuration ===
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "anthropic/claude-sonnet-4"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
DATA_STORAGE_PATH = "memoryos_data"
CHAT_HISTORY_FILE = os.path.join(DATA_STORAGE_PATH, "chat_histories.json")

# Initialize FastAPI app
app = FastAPI(
    title="MemoryOS API",
    description="A high-performance API for MemoryOS memory management system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Data Models ===
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = USER_ID

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

class MemoryRequest(BaseModel):
    user_input: str
    agent_response: str
    user_id: Optional[str] = USER_ID

class ChatIdRequest(BaseModel):
    chat_id: str

class ChatNameRequest(BaseModel):
    chat_name: Optional[str] = "New Chat"

class SuccessResponse(BaseModel):
    success: bool = True
    responseType: str
    data: Dict[str, Any]

class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]

# === Connection Manager ===
class PineconeConnectionManager:
    """Manages Pinecone connections across multiple users to optimize performance"""
    
    def __init__(self):
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_cache = {}  # Cache Pinecone indexes to avoid reconnection
        self.vector_cache = {}  # Cache common vectors to reduce Pinecone queries
        self.cache_lock = threading.Lock()
        self.query_times = []  # Track query performance
        
    def get_index(self, index_name):
        """Get a cached Pinecone index or create a new connection"""
        if index_name in self.index_cache:
            return self.index_cache[index_name]
            
        # Import here to avoid circular imports
        from pinecone import Pinecone
        
        # Create new connection
        try:
            pinecone_client = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            index = pinecone_client.Index(index_name)
            
            # Cache the index for reuse
            self.index_cache[index_name] = index
            return index
        except Exception as e:
            print(f"Error connecting to Pinecone index {index_name}: {e}")
            return None
    
    def cache_vector(self, query, vector, namespace):
        """Cache commonly used vectors to reduce embedding generation"""
        cache_key = f"{query}:{namespace}"
        with self.cache_lock:
            self.vector_cache[cache_key] = (vector, time.time())
            
            # Clean old cache entries (older than 1 hour)
            current_time = time.time()
            keys_to_remove = []
            for k, (_, timestamp) in self.vector_cache.items():
                if current_time - timestamp > 3600:  # 1 hour
                    keys_to_remove.append(k)
            
            for k in keys_to_remove:
                del self.vector_cache[k]
    
    def get_cached_vector(self, query, namespace):
        """Get a cached vector if available"""
        cache_key = f"{query}:{namespace}"
        with self.cache_lock:
            if cache_key in self.vector_cache:
                vector, _ = self.vector_cache[cache_key]
                return vector
        return None
    
    def record_query_time(self, duration):
        """Record query time for performance monitoring"""
        self.query_times.append(duration)
        if len(self.query_times) > 100:
            self.query_times.pop(0)
    
    def get_avg_query_time(self):
        """Get average query time for performance monitoring"""
        if not self.query_times:
            return 0
        return sum(self.query_times) / len(self.query_times)

# === Global Variables ===
pinecone_manager = PineconeConnectionManager()
memo_instances = {}  # Dict to store multiple Memoryos instances (one per user)
chat_histories = {}
current_chat_id = None
query_executor = ThreadPoolExecutor(max_workers=10)

# === Helper Functions ===
def load_chat_histories():
    global chat_histories, current_chat_id
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            data = json.load(f)
            chat_histories = data.get("chats", {})
            current_chat_id = data.get("current_chat_id")
            print(f"Loaded {len(chat_histories)} chat histories.")
    else:
        chat_histories = {}
        current_chat_id = None
        print("No existing chat histories found.")

def save_chat_histories():
    # Ensure the directory exists
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
    
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump({"chats": chat_histories, "current_chat_id": current_chat_id}, f, indent=4)
    print("Chat histories saved.")

def get_or_initialize_memoryos(user_id=USER_ID):
    """
    Get or initialize a MemoryOS instance for a specific user.
    Uses shared connection manager for better performance across users.
    """
    global memo_instances
    
    # Check if this user already has an instance
    if user_id in memo_instances:
        print(f"Using existing MemoryOS instance for user {user_id}")
        return memo_instances[user_id]
    
    print(f"Initializing MemoryOS for user {user_id}...")
    try:
        # Create a new instance
        user_memo = Memoryos(
            user_id=user_id,
            openai_api_key=OPENROUTER_API_KEY, 
            openai_base_url="https://openrouter.ai/api/v1", 
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=OPENROUTER_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,  
            mid_term_heat_threshold=1000,  
            retrieval_queue_capacity=10,
            long_term_knowledge_capacity=100,
            mid_term_similarity_threshold=0.6,
            embedding_model_name="all-MiniLM-L6-v2",
            retrieval_timeout=10  # Add timeout to prevent hanging
        )
        
        # Store in the instances dictionary
        memo_instances[user_id] = user_memo
        print(f"MemoryOS initialized successfully for user {user_id}!")
        return user_memo
    except Exception as e:
        print(f"Error initializing MemoryOS for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize MemoryOS: {str(e)}")

# Initialize resources on startup
@app.on_event("startup")
async def startup_event():
    # Create the data storage directory if it doesn't exist
    if not os.path.exists(DATA_STORAGE_PATH):
        os.makedirs(DATA_STORAGE_PATH)
    
    # Load chat histories
    load_chat_histories()
    
    # Initialize default MemoryOS instance
    get_or_initialize_memoryos(USER_ID)

# === API Endpoints ===

# Tool API endpoints for agent framework
@app.get("/tools", response_model=Dict[str, List[Dict[str, Any]]])
async def get_tools():
    """List all available tools for the agent framework"""
    return {
        "tools": [
            {
                "name": "query_memory",
                "description": "Query the MemoryOS system to retrieve relevant information and get contextual responses",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query or question to ask the memory system"
                        }
                    },
                    "required": ["query"]
                },
                "confirmationRequired": False
            },
            {
                "name": "create_new_chat",
                "description": "Create a new chat session in the memory system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chat_name": {
                            "type": "string",
                            "description": "Optional name for the new chat session",
                            "default": "New Chat"
                        }
                    },
                    "required": []
                },
                "confirmationRequired": False
            },
            {
                "name": "switch_to_chat",
                "description": "Switch to an existing chat session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chat_id": {
                            "type": "string",
                            "description": "The ID of the chat session to switch to"
                        }
                    },
                    "required": ["chat_id"]
                },
                "confirmationRequired": False
            },
            {
                "name": "add_memory",
                "description": "Store a new memory interaction in the MemoryOS system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "The user's input/message to store"
                        },
                        "agent_response": {
                            "type": "string",
                            "description": "The agent's response to store"
                        }
                    },
                    "required": ["user_input", "agent_response"]
                },
                "confirmationRequired": False
            },
            {
                "name": "get_chat_list",
                "description": "Get a list of all available chat sessions",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "confirmationRequired": False
            }
        ]
    }

@app.post("/tools/query_memory", response_model=Union[SuccessResponse, ErrorResponse])
async def query_memory(request: QueryRequest, background_tasks: BackgroundTasks):
    """Query the MemoryOS system to retrieve relevant information"""
    start_time = time.time()
    
    try:
        # Get or initialize the MemoryOS instance for this user
        user_memo = get_or_initialize_memoryos(request.user_id)
        
        # Get previous conversation context
        previous_conversation = ""
        if current_chat_id and current_chat_id in chat_histories:
            chat_history = chat_histories[current_chat_id]
            if chat_history:
                previous_conversation = "\n".join([
                    f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
                ])
        
        # Pass previous conversation as meta_data
        meta_data = {
            "previous_conversation": previous_conversation or "",
            "query_start_time": start_time
        }
        
        # Process query in background task
        def process_query():
            try:
                response = user_memo.get_response(query=request.query, user_conversation_meta_data=meta_data)
                query_time = time.time() - start_time
                pinecone_manager.record_query_time(query_time)
                print(f"Query processed in {query_time:.2f}s. Avg: {pinecone_manager.get_avg_query_time():.2f}s")
                return response
            except Exception as e:
                print(f"Error processing query: {e}")
                raise e
        
        # For sync execution
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(query_executor, process_query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "responseType": "text",
            "data": {
                "text": response,
                "processing_time": processing_time,
                "avg_query_time": pinecone_manager.get_avg_query_time()
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": "Failed to process query",
                "code": 500,
                "details": str(e)
            }
        }

@app.post("/tools/create_new_chat", response_model=Union[SuccessResponse, ErrorResponse])
async def create_new_chat(request: ChatNameRequest):
    """Create a new chat session"""
    global current_chat_id
    
    try:
        new_id = str(uuid.uuid4())
        chat_histories[new_id] = []
        current_chat_id = new_id
        
        # Save chat histories in background
        save_chat_histories()
        
        return {
            "success": True,
            "responseType": "text",
            "data": {
                "text": f"Created new chat session with ID: {new_id}",
                "chat_id": new_id
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": "Failed to create new chat",
                "code": 500,
                "details": str(e)
            }
        }

@app.post("/tools/switch_to_chat", response_model=Union[SuccessResponse, ErrorResponse])
async def switch_to_chat(request: ChatIdRequest):
    """Switch to an existing chat session"""
    global current_chat_id
    
    try:
        if request.chat_id not in chat_histories:
            return {
                "success": False,
                "error": {
                    "message": "Chat ID not found",
                    "code": 404,
                    "details": f"Chat with ID {request.chat_id} does not exist"
                }
            }
        
        current_chat_id = request.chat_id
        save_chat_histories()
        
        return {
            "success": True,
            "responseType": "text",
            "data": {
                "text": f"Successfully switched to chat {request.chat_id}"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": "Failed to switch chat",
                "code": 500,
                "details": str(e)
            }
        }

@app.post("/tools/add_memory", response_model=Union[SuccessResponse, ErrorResponse])
async def add_memory(request: MemoryRequest, background_tasks: BackgroundTasks):
    """Store a new memory interaction in the MemoryOS system"""
    global current_chat_id
    
    try:
        # Get or initialize the MemoryOS instance for this user
        user_memo = get_or_initialize_memoryos(request.user_id)
        
        # Get previous conversation context
        previous_conversation = ""
        if current_chat_id and current_chat_id in chat_histories:
            chat_history = chat_histories[current_chat_id]
            if chat_history:
                previous_conversation = "\n".join([
                    f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
                ])
        
        # Add to chat history
        if current_chat_id:
            if current_chat_id not in chat_histories:
                chat_histories[current_chat_id] = []
            chat_histories[current_chat_id].append({"sender": "user", "message": request.user_input})
            chat_histories[current_chat_id].append({"sender": "agent", "message": request.agent_response})
            background_tasks.add_task(save_chat_histories)
        
        # Pass previous conversation as meta_data
        meta_data = {"previous_conversation": previous_conversation or ""}
        
        # Add memory in background
        def add_memory_task():
            user_memo.add_memory(
                user_input=request.user_input,
                agent_response=request.agent_response,
                meta_data=meta_data
            )
        
        # Use thread pool to avoid blocking
        background_tasks.add_task(add_memory_task)
        
        return {
            "success": True,
            "responseType": "text",
            "data": {
                "text": "Memory added successfully"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": "Failed to add memory",
                "code": 500,
                "details": str(e)
            }
        }

@app.post("/tools/get_chat_list", response_model=Union[SuccessResponse, ErrorResponse])
async def get_chat_list():
    """Get a list of all available chat sessions"""
    try:
        return {
            "success": True,
            "responseType": "text",
            "data": {
                "text": f"Available chats: {list(chat_histories.keys())}, Current chat: {current_chat_id}",
                "chats": list(chat_histories.keys()),
                "current_chat_id": current_chat_id
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": "Failed to get chat list",
                "code": 500,
                "details": str(e)
            }
        }

@app.post("/tools/{tool_name}", response_model=Union[SuccessResponse, ErrorResponse])
async def execute_tool(tool_name: str, request: Request, background_tasks: BackgroundTasks):
    """General endpoint for tool execution (fallback for custom tools)"""
    try:
        data = await request.json()
        
        return {
            "success": False,
            "error": {
                "message": f"Tool '{tool_name}' not found",
                "code": 404,
                "details": f"Available tools: query_memory, create_new_chat, switch_to_chat, add_memory, get_chat_list"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": f"Error executing tool {tool_name}",
                "code": 500,
                "details": str(e)
            }
        }

# Resource endpoint for agent framework
@app.post("/resource", response_model=Union[SuccessResponse, ErrorResponse])
async def get_resource():
    """Provide contextual information about the memory system"""
    try:
        # Get default MemoryOS instance
        user_memo = get_or_initialize_memoryos(USER_ID)
        
        resource_info = {
            "agent_type": "MemoryOS Agent (FastAPI)",
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "current_chat_id": current_chat_id,
            "total_chats": len(chat_histories),
            "active_users": len(memo_instances),
            "model": OPENROUTER_MODEL,
            "avg_query_time": f"{pinecone_manager.get_avg_query_time():.2f}s",
            "capabilities": [
                "Memory storage and retrieval",
                "Chat session management",
                "Contextual responses",
                "Long-term knowledge retention",
                "Multi-user support",
                "Concurrent request processing"
            ]
        }
        
        return {
            "success": True,
            "responseType": "mixed",
            "data": resource_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": "Failed to get resource info",
                "code": 500,
                "details": str(e)
            }
        }

# Regular endpoints for web interface
@app.get("/chats")
async def get_chats():
    """Get all chat histories"""
    return {"chats": chat_histories, "currentChatId": current_chat_id}

@app.post("/new_chat")
async def new_chat():
    """Create a new chat session"""
    global current_chat_id
    new_id = str(uuid.uuid4())
    chat_histories[new_id] = []
    current_chat_id = new_id
    save_chat_histories()
    return {"chat_id": new_id}

@app.post("/switch_chat")
async def switch_chat(request: ChatIdRequest):
    """Switch to an existing chat session"""
    global current_chat_id
    
    if request.chat_id in chat_histories:
        current_chat_id = request.chat_id
        save_chat_histories()
        return {"status": "success", "message": f"Switched to chat {request.chat_id}"}
    
    raise HTTPException(status_code=404, detail="Chat ID not found")

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Process a chat message and get response"""
    global current_chat_id
    
    if not request.message:
        raise HTTPException(status_code=400, detail="Please enter a message")
    
    # Get or create memo instance
    user_memo = get_or_initialize_memoryos(USER_ID)
    
    # Handle chat ID
    if request.chat_id and request.chat_id != current_chat_id:
        if request.chat_id in chat_histories:
            current_chat_id = request.chat_id
        else:
            chat_histories[request.chat_id] = []
            current_chat_id = request.chat_id
        background_tasks.add_task(save_chat_histories)
    
    if current_chat_id not in chat_histories:
        chat_histories[current_chat_id] = []
    
    # Add user message to chat history
    chat_histories[current_chat_id].append({
        "sender": "user", 
        "message": request.message, 
        "timestamp": os.path.getmtime(CHAT_HISTORY_FILE) if os.path.exists(CHAT_HISTORY_FILE) else time.time()
    })
    
    # Get previous conversation context
    previous_conversation = ""
    if current_chat_id in chat_histories:
        chat_history = chat_histories[current_chat_id]
        if chat_history:
            previous_conversation = "\n".join([
                f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
            ])
    
    # Pass previous conversation as meta_data
    meta_data = {"previous_conversation": previous_conversation or ""}
    
    try:
        # Process in executor to avoid blocking
        def get_response_task():
            return user_memo.get_response(query=request.message, user_conversation_meta_data=meta_data)
        
        loop = asyncio.get_event_loop()
        agent_response = await loop.run_in_executor(query_executor, get_response_task)
        
        # Add agent response to chat history
        chat_histories[current_chat_id].append({
            "sender": "agent", 
            "message": agent_response, 
            "timestamp": time.time()
        })
        
        # Save chat histories
        background_tasks.add_task(save_chat_histories)
        
        return {"response": agent_response}
    except Exception as e:
        # Log error and return
        print(f"Error getting response: {e}")
        
        # Add error message to chat history
        chat_histories[current_chat_id].append({
            "sender": "agent", 
            "message": f"An error occurred: {e}", 
            "timestamp": time.time()
        })
        
        background_tasks.add_task(save_chat_histories)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/agent_query")
async def agent_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Endpoint for agents to query MemoryOS"""
    start_time = time.time()
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Please provide a 'query' in the request body")
    
    # Get or initialize user-specific memo instance
    user_memo = get_or_initialize_memoryos(request.user_id)
    
    # Get previous conversation context
    previous_conversation = ""
    if current_chat_id and current_chat_id in chat_histories:
        chat_history = chat_histories[current_chat_id]
        if chat_history:
            previous_conversation = "\n".join([
                f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
            ])
    
    # Meta data with timing information
    meta_data = {
        "previous_conversation": previous_conversation or "",
        "query_start_time": start_time
    }
    
    # Process query in background
    def process_query():
        try:
            response = user_memo.get_response(query=request.query, user_conversation_meta_data=meta_data)
            query_time = time.time() - start_time
            pinecone_manager.record_query_time(query_time)
            print(f"Query processed in {query_time:.2f}s. Avg: {pinecone_manager.get_avg_query_time():.2f}s")
            return response
        except Exception as e:
            print(f"Error processing query: {e}")
            raise e
    
    try:
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        agent_response = await loop.run_in_executor(query_executor, process_query)
        
        # Calculate processing time
        query_time = time.time() - start_time
        
        return {
            "response": agent_response,
            "processing_time": query_time,
            "avg_query_time": pinecone_manager.get_avg_query_time()
        }
    except Exception as e:
        print(f"Error or timeout processing agent query: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "response": f"An error occurred while processing agent query: {str(e)}",
                "processing_time": time.time() - start_time
            }
        )

@app.post("/agent_memory")
async def agent_memory(request: MemoryRequest, background_tasks: BackgroundTasks):
    """Endpoint for agents to add memories to MemoryOS"""
    if not request.user_input or not request.agent_response:
        raise HTTPException(
            status_code=400, 
            detail="Please provide 'user_input' and 'agent_response' in the request body"
        )
    
    # Get or initialize MemoryOS instance
    user_memo = get_or_initialize_memoryos(request.user_id)
    
    # Get previous conversation context
    previous_conversation = ""
    if current_chat_id and current_chat_id in chat_histories:
        chat_history = chat_histories[current_chat_id]
        if chat_history:
            previous_conversation = "\n".join([
                f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
            ])
    
    # Meta data
    meta_data = {"previous_conversation": previous_conversation or ""}
    
    # Add memory in background
    def add_memory_task():
        try:
            user_memo.add_memory(
                user_input=request.user_input,
                agent_response=request.agent_response,
                meta_data=meta_data
            )
        except Exception as e:
            print(f"Error adding memory: {e}")
    
    background_tasks.add_task(add_memory_task)
    
    return {"status": "success", "message": "Memory added successfully."}

@app.get("/health")
async def health_check():
    """Health check endpoint with performance metrics"""
    return {
        "status": "healthy",
        "version": "FastAPI 1.0.0",
        "memo_initialized": len(memo_instances) > 0,
        "active_users": len(memo_instances),
        "chat_histories": len(chat_histories),
        "avg_query_time": pinecone_manager.get_avg_query_time(),
        "pinecone_indexes": list(pinecone_manager.index_cache.keys()),
        "vector_cache_size": len(pinecone_manager.vector_cache),
        "system_time": time.time()
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Ensure data directory exists
    if not os.path.exists(DATA_STORAGE_PATH):
        os.makedirs(DATA_STORAGE_PATH)
    
    # Run with uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
