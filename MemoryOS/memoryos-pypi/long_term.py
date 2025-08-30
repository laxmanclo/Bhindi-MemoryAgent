import json
from collections import deque
from pinecone import Pinecone # Main Pinecone client
import uuid # For generating unique IDs for Pinecone vectors
import time # For rate limiting if needed
# PodSpec and ServerlessSpec are now typically accessed as attributes of the Pinecone client.

# Using direct imports when running as a script
from utils import get_timestamp, get_embedding, normalize_vector, ensure_directory_exists

class LongTermMemory:
    def __init__(self, file_path, knowledge_capacity=100, embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None,
                 pinecone_api_key: str = None, pinecone_environment: str = None, pinecone_index_name: str = "memoryos-knowledge"):
        self.file_path = file_path # Used for user profiles
        ensure_directory_exists(self.file_path)
        self.knowledge_capacity = knowledge_capacity # Soft limit for Pinecone
        self.user_profiles = {} # {user_id: {data: "profile_string", "last_updated": "timestamp"}}
        
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs if embedding_model_kwargs is not None else {}

        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_client = None
        self.pinecone_index = None
        self.dimension = None # Will be set after first embedding or Pinecone index description

        self.load() # Load user profiles from local file

        # Initialize Pinecone if API key and environment are provided
        if self.pinecone_api_key and self.pinecone_environment:
            try:
                self.pinecone_client = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
                print(f"LongTermMemory: Initialized Pinecone client for environment: {self.pinecone_environment}")
                self._get_pinecone_index() # Try to connect to or create index
                if not self.pinecone_index:
                    raise Exception("Failed to initialize Pinecone index")
            except Exception as e:
                print(f"LongTermMemory: Error initializing Pinecone: {e}. Pinecone is required for operation.")
                raise e
        else:
            error_msg = "Pinecone API key and environment are required. Cannot operate without Pinecone."
            print(f"LongTermMemory: {error_msg}")
            raise ValueError(error_msg)
        
        # No local storage - Pinecone only

    def _get_pinecone_index(self):
        if not self.pinecone_client:
            return None
        
        try:
            # First try to directly connect to the index assuming it exists
            try:
                print(f"Connecting to existing Pinecone index '{self.pinecone_index_name}'...")
                # Get index description to confirm it exists and get dimension
                index_description = self.pinecone_client.describe_index(self.pinecone_index_name)
                self.dimension = index_description.dimension
                print(f"Successfully connected to Pinecone index '{self.pinecone_index_name}' with dimension {self.dimension}.")
                self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
                return self.pinecone_index
            except Exception as e:
                if "NOT_FOUND" in str(e):
                    # Index doesn't exist, create it
                    print(f"Pinecone index '{self.pinecone_index_name}' not found. Creating new index...")
                    # Determine dimension from embedding model
                    sample_embedding = get_embedding("test", model_name=self.embedding_model_name, **self.embedding_model_kwargs)
                    self.dimension = len(sample_embedding)
                    
                    # Create the index with AWS parameters (matching manually created index)
                    from pinecone import ServerlessSpec
                    self.pinecone_client.create_index(
                        name=self.pinecone_index_name,
                        dimension=self.dimension,
                        metric='cosine',
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                    print(f"Pinecone index '{self.pinecone_index_name}' created with dimension {self.dimension}.")
                    self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
                    return self.pinecone_index
                else:
                    # Some other error occurred
                    raise e
        except Exception as e:
            print(f"Error connecting to or creating Pinecone index '{self.pinecone_index_name}': {e}")
            self.pinecone_index = None
            return None

    def update_user_profile(self, user_id, new_data, merge=True):
        if merge and user_id in self.user_profiles and self.user_profiles[user_id].get("data"):
            current_data = self.user_profiles[user_id]["data"]
            if isinstance(current_data, str) and isinstance(new_data, str):
                updated_data = f"{current_data}\n\n--- Updated on {get_timestamp()} ---\n{new_data}"
            else:
                updated_data = new_data
        else:
            updated_data = new_data
        
        self.user_profiles[user_id] = {
            "data": updated_data,
            "last_updated": get_timestamp()
        }
        print(f"LongTermMemory: Updated user profile for {user_id} (merge={merge}).")
        self.save() # User profiles are still saved locally

    def get_raw_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {}).get("data", "None")

    def get_user_profile_data(self, user_id):
        return self.user_profiles.get(user_id, {})

    def add_knowledge_entry(self, knowledge_text, namespace: str, type_name="knowledge"):
        if not knowledge_text or knowledge_text.strip().lower() in ["", "none", "- none", "- none."]:
            print(f"LongTermMemory: Empty {type_name} received, not saving.")
            return
        
        vec = get_embedding(
            knowledge_text, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        vec = normalize_vector(vec).tolist()

        entry_id = str(uuid.uuid4())
        metadata = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "type": type_name,
            "namespace": namespace # Store namespace in metadata for filtering if needed
        }

        if self.pinecone_index:
            try:
                self.pinecone_index.upsert(vectors=[{"id": entry_id, "values": vec, "metadata": metadata}], namespace=namespace)
                print(f"LongTermMemory: Added {type_name} to Pinecone in namespace '{namespace}'.")
            except Exception as e:
                print(f"LongTermMemory: Error adding {type_name} to Pinecone: {e}. Pinecone is required.")
                raise e  # Re-raise the exception as Pinecone is required
        else:
            # Pinecone index should always be initialized at this point due to checks in __init__
            error_msg = "Pinecone index is not initialized but is required for operation"
            print(f"LongTermMemory: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"LongTermMemory: Added {type_name} to Pinecone.")

    def add_user_knowledge(self, knowledge_text):
        self.add_knowledge_entry(knowledge_text, namespace="user-knowledge", type_name="user knowledge")

    def add_assistant_knowledge(self, knowledge_text):
        self.add_knowledge_entry(knowledge_text, namespace="assistant-knowledge", type_name="assistant knowledge")

    def get_user_knowledge(self):
        # This method could be reimplemented to query Pinecone directly
        print("Warning: get_user_knowledge is not implemented for Pinecone-only mode")
        return []

    def get_assistant_knowledge(self):
        # This method could be reimplemented to query Pinecone directly
        print("Warning: get_assistant_knowledge is not implemented for Pinecone-only mode")
        return []

    def _search_knowledge(self, query, namespace: str, threshold=0.1, top_k=5):
        query_vec = get_embedding(
            query, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec).tolist()

        if self.pinecone_index:
            try:
                # Pinecone search
                response = self.pinecone_index.query(
                    vector=query_vec,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )
                results = []
                for match in response.matches:
                    if match.score >= threshold:
                        results.append(match.metadata)
                print(f"LongTermMemory: Searched Pinecone in namespace '{namespace}' for '{query[:30]}...'. Found {len(results)} matches.")
                return results
            except Exception as e:
                print(f"LongTermMemory: Error searching Pinecone in namespace '{namespace}': {e}. Pinecone is required.")
                raise e  # Re-raise the exception as Pinecone is required
        else:
            # Pinecone index should always be initialized at this point due to checks in __init__
            error_msg = "Pinecone index is not initialized but is required for operation"
            print(f"LongTermMemory: {error_msg}")
            raise ValueError(error_msg)

    # Removed local storage methods - using Pinecone only

    def search_user_knowledge(self, query, threshold=0.1, top_k=5):
        return self._search_knowledge(query, namespace="user-knowledge", threshold=threshold, top_k=top_k)

    def search_assistant_knowledge(self, query, threshold=0.1, top_k=5):
        return self._search_knowledge(query, namespace="assistant-knowledge", threshold=threshold, top_k=top_k)

    def save(self):
        # Only save user profiles locally, all knowledge is in Pinecone
        data = {
            "user_profiles": self.user_profiles
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving LongTermMemory (user profiles) to {self.file_path}: {e}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.user_profiles = data.get("user_profiles", {})
            print(f"LongTermMemory: Loaded user profiles from {self.file_path}.")
        except FileNotFoundError:
            print(f"LongTermMemory: No user profile file found at {self.file_path}. Initializing new profiles.")
        except json.JSONDecodeError:
            print(f"LongTermMemory: Error decoding JSON from {self.file_path}. Initializing new profiles.")
        except Exception as e:
             print(f"LongTermMemory: An unexpected error occurred during user profile load from {self.file_path}: {e}. Initializing new profiles.")
