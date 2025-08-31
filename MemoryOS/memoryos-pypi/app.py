import os
import json
import uuid
import time
import threading
from flask import Flask, render_template, request, jsonify
from memoryos import Memoryos  # Direct import when running as script
from openai import OpenAI  # Using OpenAI-compatible client for OpenRouter
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env file

# Add connection pooling and caching for better multi-user performance
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

app = Flask(__name__)

# --- Configuration ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
# It's highly recommended to use environment variables for API keys in production
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") # Get from environment variable
OPENROUTER_MODEL = "anthropic/claude-sonnet-4" # Claude Sonnet 4
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") # Get from environment variable
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1") # Using us-east-1 as default region (AWS)
DATA_STORAGE_PATH = "memoryos_data" # Local directory for memory storage
CHAT_HISTORY_FILE = os.path.join(DATA_STORAGE_PATH, "chat_histories.json")

# Initialize OpenAI client for OpenRouter (OpenRouter is compatible with OpenAI API)
openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# Global executor for concurrent user request processing
query_executor = ThreadPoolExecutor(max_workers=10)

# User cache and connection management
memo_instances = {}  # Dict to store multiple Memoryos instances (one per user)
chat_histories = {}
current_chat_id = None

# Initialize connection manager for multi-user support
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

# Create global connection manager
pinecone_manager = PineconeConnectionManager()

# Initialize MemoryOS (will be done per user)
memo = None  # Default instance for backward compatibility

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
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump({"chats": chat_histories, "current_chat_id": current_chat_id}, f, indent=4)
    print("Chat histories saved.")

def initialize_memoryos(user_id=USER_ID):
    """
    Initialize a MemoryOS instance for a specific user.
    Uses shared connection manager for better performance across users.
    """
    global memo, memo_instances
    
    # Check if this user already has an instance
    if user_id in memo_instances:
        print(f"Using existing MemoryOS instance for user {user_id}")
        memo = memo_instances[user_id]
        return memo
    
    print(f"Initializing MemoryOS for user {user_id}...")
    try:
        # Pinecone API key and environment are read from environment variables inside Memoryos
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
        
        # Store in the instances dictionary and set as default memo
        memo_instances[user_id] = user_memo
        if user_id == USER_ID:  # Set as default for backward compatibility
            memo = user_memo
            
        print(f"MemoryOS initialized successfully for user {user_id}!")
        return user_memo
    except Exception as e:
        print(f"Error initializing MemoryOS for user {user_id}: {e}")
        return None

@app.before_request
def before_first_request():
    initialize_memoryos(USER_ID)  # Initialize default instance
    load_chat_histories()

# ============================================
# NEW AGENT FRAMEWORK ENDPOINTS (ADD THESE)
# ============================================

@app.route('/tools', methods=['GET'])
def get_tools():
    """
    Required endpoint: Lists all available tools for the agent framework
    """
    return jsonify({
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
    })

@app.route('/tools/<tool_name>', methods=['POST'])
def execute_tool(tool_name):
    """
    Required endpoint: Executes a specific tool with provided parameters
    """
    global current_chat_id
    try:
        data = request.json or {}
        
        if tool_name == "query_memory":
            # Use your existing agent_query logic
            query = data.get('query')
            if not query:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "Query parameter is required",
                        "code": 400,
                        "details": "Please provide a 'query' parameter"
                    }
                })
            
            if memo is None:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "MemoryOS is not initialized",
                        "code": 500,
                        "details": "Check server logs for initialization errors"
                    }
                })
            
            # Get current chat history for previous_conversation context
            previous_conversation = ""
            if current_chat_id and current_chat_id in chat_histories:
                chat_history = chat_histories[current_chat_id]
                if chat_history:
                    previous_conversation = "\n".join([
                        f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
                    ])
            
            # Pass empty string if no previous conversation exists
            meta_data = {"previous_conversation": previous_conversation or ""}
            
            agent_response = memo.get_response(query=query, user_conversation_meta_data=meta_data)
            return jsonify({
                "success": True,
                "responseType": "text",
                "data": {
                    "text": agent_response
                }
            })
            
        elif tool_name == "create_new_chat":
            # Use your existing new_chat logic
            new_id = str(uuid.uuid4())
            chat_histories[new_id] = []
            current_chat_id = new_id
            save_chat_histories()
            
            return jsonify({
                "success": True,
                "responseType": "text",
                "data": {
                    "text": f"Created new chat session with ID: {new_id}"
                }
            })
            
        elif tool_name == "switch_to_chat":
            # Use your existing switch_chat logic
            chat_id = data.get('chat_id')
            if not chat_id:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "chat_id parameter is required",
                        "code": 400,
                        "details": "Please provide a 'chat_id' parameter"
                    }
                })
            
            if chat_id in chat_histories:
                current_chat_id = chat_id
                save_chat_histories()
                return jsonify({
                    "success": True,
                    "responseType": "text",
                    "data": {
                        "text": f"Successfully switched to chat {chat_id}"
                    }
                })
            else:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "Chat ID not found",
                        "code": 404,
                        "details": f"Chat with ID {chat_id} does not exist"
                    }
                })
                
        elif tool_name == "add_memory":
            # Use your existing agent_memory logic
            user_input = data.get('user_input')
            agent_response = data.get('agent_response')
            
            if not user_input or not agent_response:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "Both user_input and agent_response are required",
                        "code": 400,
                        "details": "Please provide both 'user_input' and 'agent_response' parameters"
                    }
                })
            
            if memo is None:
                return jsonify({
                    "success": False,
                    "error": {
                        "message": "MemoryOS is not initialized",
                        "code": 500,
                        "details": "Check server logs for initialization errors"
                    }
                })
            
            # Get current chat history for previous_conversation context
            previous_conversation = ""
            if current_chat_id and current_chat_id in chat_histories:
                chat_history = chat_histories[current_chat_id]
                if chat_history:
                    previous_conversation = "\n".join([
                        f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
                    ])
            
            # Add to chat history as well
            if current_chat_id:
                if current_chat_id not in chat_histories:
                    chat_histories[current_chat_id] = []
                chat_histories[current_chat_id].append({"sender": "user", "message": user_input})
                chat_histories[current_chat_id].append({"sender": "agent", "message": agent_response})
                save_chat_histories()
            
            # Pass empty string if no previous conversation exists
            meta_data = {"previous_conversation": previous_conversation or ""}
            
            memo.add_memory(user_input=user_input, agent_response=agent_response, meta_data=meta_data)
            return jsonify({
                "success": True,
                "responseType": "text",
                "data": {
                    "text": "Memory added successfully"
                }
            })
            
        elif tool_name == "get_chat_list":
            # Use your existing get_chats logic
            return jsonify({
                "success": True,
                "responseType": "text",
                "data": {
                    "text": f"Available chats: {list(chat_histories.keys())}, Current chat: {current_chat_id}"
                }
            })
            
        else:
            return jsonify({
                "success": False,
                "error": {
                    "message": f"Tool '{tool_name}' not found",
                    "code": 404,
                    "details": f"Available tools: query_memory, create_new_chat, switch_to_chat, add_memory, get_chat_list"
                }
            })
            
    except Exception as e:
        print(f"Error executing tool {tool_name}: {e}")
        return jsonify({
            "success": False,
            "error": {
                "message": f"Internal server error while executing {tool_name}",
                "code": 500,
                "details": str(e)
            }
        })

# Optional endpoint for providing context about the agent
@app.route('/resource', methods=['POST'])
def get_resource():
    """
    Optional endpoint: Provides contextual information about the memory system
    """
    try:
        if memo is None:
            return jsonify({
                "success": False,
                "error": {
                    "message": "MemoryOS is not initialized",
                    "code": 500,
                    "details": "Check server logs for initialization errors"
                }
            })
        
        resource_info = {
            "agent_type": "MemoryOS Agent",
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "current_chat_id": current_chat_id,
            "total_chats": len(chat_histories),
            "model": OPENROUTER_MODEL,
            "capabilities": [
                "Memory storage and retrieval",
                "Chat session management",
                "Contextual responses",
                "Long-term knowledge retention"
            ]
        }
        
        return jsonify({
            "success": True,
            "responseType": "mixed",
            "data": resource_info
        })
        
    except Exception as e:
        print(f"Error getting resource info: {e}")
        return jsonify({
            "success": False,
            "error": {
                "message": "Internal server error while getting resource info",
                "code": 500,
                "details": str(e)
            }
        })

# ============================================
# YOUR EXISTING ENDPOINTS (KEEP ALL OF THESE)
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chats', methods=['GET'])
def get_chats():
    return jsonify({"chats": chat_histories, "currentChatId": current_chat_id})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    global current_chat_id
    new_id = str(uuid.uuid4())
    chat_histories[new_id] = []
    current_chat_id = new_id
    save_chat_histories()
    return jsonify({"chat_id": new_id})

@app.route('/switch_chat', methods=['POST'])
def switch_chat():
    global current_chat_id
    data = request.json
    chat_id = data.get('chat_id')
    if chat_id in chat_histories:
        current_chat_id = chat_id
        save_chat_histories()
        return jsonify({"status": "success", "message": f"Switched to chat {chat_id}"})
    return jsonify({"status": "error", "message": "Chat ID not found"}), 404

@app.route('/chat', methods=['POST'])
def chat():
    global current_chat_id
    user_message = request.json.get('message')
    chat_id_from_frontend = request.json.get('chat_id')

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    if memo is None:
        return jsonify({"response": "MemoryOS is not initialized. Check server logs for errors."}), 500

    # Ensure current_chat_id is consistent with frontend or set if not already
    if chat_id_from_frontend and chat_id_from_frontend != current_chat_id:
        if chat_id_from_frontend in chat_histories:
            current_chat_id = chat_id_from_frontend
        else:
            # If frontend sends a new chat_id not known to backend, create it
            chat_histories[chat_id_from_frontend] = []
            current_chat_id = chat_id_from_frontend
        save_chat_histories()

    if current_chat_id not in chat_histories:
        chat_histories[current_chat_id] = [] # Initialize if somehow missing

    # Add user message to current chat history
    chat_histories[current_chat_id].append({"sender": "user", "message": user_message, "timestamp": os.path.getmtime(CHAT_HISTORY_FILE) if os.path.exists(CHAT_HISTORY_FILE) else 0})

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
        # Get response from MemoryOS
        agent_response = memo.get_response(query=user_message, user_conversation_meta_data=meta_data)
        # Add agent response to current chat history
        chat_histories[current_chat_id].append({"sender": "agent", "message": agent_response, "timestamp": os.path.getmtime(CHAT_HISTORY_FILE) if os.path.exists(CHAT_HISTORY_FILE) else 0})
        save_chat_histories() # Save after each message exchange
        return jsonify({"response": agent_response})
    except Exception as e:
        print(f"Error getting response from MemoryOS: {e}")
        # Add error message to chat history
        chat_histories[current_chat_id].append({"sender": "agent", "message": f"An error occurred: {e}", "timestamp": os.path.getmtime(CHAT_HISTORY_FILE) if os.path.exists(CHAT_HISTORY_FILE) else 0})
        save_chat_histories()
        return jsonify({"response": f"An error occurred: {e}"}), 500

@app.route('/agent_query', methods=['POST'])
def agent_query():
    """
    Endpoint for agents to query MemoryOS.
    Expects JSON with 'query' field.
    
    Optimized for high throughput and parallel processing.
    """
    start_time = time.time()
    agent_query_text = request.json.get('query')
    user_id = request.json.get('user_id', USER_ID)  # Support custom user IDs

    if not agent_query_text:
        return jsonify({"response": "Please provide a 'query' in the request body."}), 400

    # Get or initialize user-specific memo instance
    user_memo = memo_instances.get(user_id)
    if not user_memo:
        user_memo = initialize_memoryos(user_id)
        if not user_memo:
            return jsonify({"response": "MemoryOS initialization failed. Check server logs for errors."}), 500

    # Get previous conversation context from current chat
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
        "query_start_time": start_time  # Track when query started for timing
    }

    # Process query in thread pool to handle multiple concurrent users
    def process_query():
        try:
            return user_memo.get_response(query=agent_query_text, user_conversation_meta_data=meta_data)
        except Exception as e:
            print(f"Error processing agent query: {e}")
            return f"An error occurred while processing agent query: {e}"

    try:
        # Submit query to thread pool
        future = query_executor.submit(process_query)
        agent_response = future.result(timeout=30)  # 30-second timeout
        
        # Record performance metrics
        query_time = time.time() - start_time
        pinecone_manager.record_query_time(query_time)
        
        print(f"Query processed in {query_time:.2f}s. Avg query time: {pinecone_manager.get_avg_query_time():.2f}s")
        
        return jsonify({
            "response": agent_response,
            "processing_time": query_time,
            "avg_query_time": pinecone_manager.get_avg_query_time()
        })
    except Exception as e:
        print(f"Error or timeout processing agent query: {e}")
        return jsonify({
            "response": f"An error occurred while processing agent query: {e}",
            "processing_time": time.time() - start_time
        }), 500

@app.route('/agent_memory', methods=['POST'])
def agent_memory():
    """
    Endpoint for agents to add memories to MemoryOS.
    Expects JSON with 'user_input' and 'agent_response' fields.
    """
    user_input = request.json.get('user_input')
    agent_response = request.json.get('agent_response')

    if not user_input or not agent_response:
        return jsonify({"response": "Please provide 'user_input' and 'agent_response' in the request body."}), 400

    if memo is None:
        return jsonify({"response": "MemoryOS is not initialized. Check server logs for errors."}), 500

    # Get previous conversation context from current chat
    previous_conversation = ""
    if current_chat_id and current_chat_id in chat_histories:
        chat_history = chat_histories[current_chat_id]
        if chat_history:
            previous_conversation = "\n".join([
                f"User: {msg.get('message', '')} " for msg in chat_history if msg.get('sender') == 'user'
            ])
    
    # Pass previous conversation as meta_data
    meta_data = {"previous_conversation": previous_conversation or ""}

    try:
        memo.add_memory(user_input=user_input, agent_response=agent_response, meta_data=meta_data)
        return jsonify({"status": "success", "message": "Memory added successfully."})
    except Exception as e:
        print(f"Error adding agent memory: {e}")
        return jsonify({"status": "error", "message": f"An error occurred while adding memory: {e}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Enhanced health check endpoint with performance metrics
    """
    return jsonify({
        "status": "healthy",
        "memo_initialized": memo is not None,
        "active_users": len(memo_instances),
        "chat_histories": len(chat_histories),
        "avg_query_time": pinecone_manager.get_avg_query_time(),
        "pinecone_indexes": list(pinecone_manager.index_cache.keys()),
        "vector_cache_size": len(pinecone_manager.vector_cache),
        "system_time": time.time()
    })

if __name__ == '__main__':
    # Create the data storage directory if it doesn't exist
    if not os.path.exists(DATA_STORAGE_PATH):
        os.makedirs(DATA_STORAGE_PATH)
    app.run(debug=True, port=5000)
