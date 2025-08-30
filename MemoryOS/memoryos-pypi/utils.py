import os
import time
import json
import re
import uuid
from datetime import datetime
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import importlib.util
try:
    from typing import Callable, List, Dict, Any, Optional, Tuple, Union
except ImportError:
    pass

def ensure_directory_exists(file_path):
    """
    Ensures that the directory for the given file path exists.
    If it doesn't, creates the directory.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_timestamp():
    """
    Get current timestamp in ISO format.
    """
    return datetime.now().isoformat()

def compute_time_decay(older_timestamp_str, newer_timestamp_str, tau_hours=24):
    """
    Compute time decay factor between two timestamps.
    
    Args:
        older_timestamp_str: ISO format timestamp string for older time
        newer_timestamp_str: ISO format timestamp string for newer time
        tau_hours: Time constant in hours for exponential decay
        
    Returns:
        Float decay factor between 0 and 1
    """
    if not older_timestamp_str or not newer_timestamp_str:
        return 1.0  # Default to no decay if timestamps not available
    
    try:
        t1 = datetime.fromisoformat(older_timestamp_str)
        t2 = datetime.fromisoformat(newer_timestamp_str)
        
        dt = t2 - t1
        hours = dt.total_seconds() / 3600.0
        
        # Cap the decay at a minimum to avoid zero values
        decay = max(0.01, np.exp(-hours / tau_hours))
        return float(decay)
    except (ValueError, TypeError) as e:
        print(f"Error computing time decay: {e}. Using default value.")
        return 1.0  # Default to no decay on error

def generate_id(prefix=""):
    """
    Generate a unique ID with optional prefix.
    """
    return f"{prefix}_{str(uuid.uuid4())}"

# Embedding model and vector operations
_models = {}

def get_embedding(text, model_name="all-MiniLM-L6-v2", **kwargs):
    """
    Get embedding for text using Sentence Transformers.
    Caches models for efficiency.
    
    Args:
        text: Text to embed
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Numpy array of embeddings
    """
    # For empty text, return zero vector (proper length determined by model)
    if not text or text.strip() == "":
        # Need to create model to know embedding size, unfortunately
        if model_name not in _models:
            try:
                print(f"Loading model: {model_name}...")
                _models[model_name] = SentenceTransformer(model_name)
                print(f"-> Using SentenceTransformer with init kwargs: {kwargs}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise
                
        # Get embedding size from model
        zero_vec = np.zeros(_models[model_name].get_sentence_embedding_dimension())
        return zero_vec
    
    # Get or create model
    if model_name not in _models:
        try:
            print(f"Loading model: {model_name}...")
            _models[model_name] = SentenceTransformer(model_name)
            print(f"-> Using SentenceTransformer with init kwargs: {kwargs}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    # Get embedding
    try:
        model = _models[model_name]
        print(f"-> Encoding with SentenceTransformer using kwargs: {kwargs}")
        embedding = model.encode(text, **kwargs)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def normalize_vector(vec):
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Vector to normalize (numpy array)
        
    Returns:
        Normalized vector as numpy array
    """
    if isinstance(vec, list):
        vec = np.array(vec, dtype=np.float32)
        
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1, vec2: Vectors to compare
        
    Returns:
        Cosine similarity score (float between -1 and 1)
    """
    if isinstance(vec1, list):
        vec1 = np.array(vec1, dtype=np.float32)
    if isinstance(vec2, list):
        vec2 = np.array(vec2, dtype=np.float32)
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

class OpenAIClient:
    """
    Client for OpenAI API, with support for optional base URL override.
    """
    def __init__(self, api_key, base_url=None):
        """
        Initialize OpenAI client with API key and optional base URL.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API endpoint (for API-compatible services)
        """
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
            
    def chat_completion(self, model, messages, temperature=0.7, max_tokens=None, stream=False, **kwargs):
        """
        Get chat completion from OpenAI.
        
        Args:
            model: Model to use
            messages: List of message dicts with role and content
            temperature: Temperature for sampling
            max_tokens: Maximum tokens in response
            stream: Whether to stream responses
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Completion text, or iterator if streaming
        """
        print(f"Calling OpenAI API. Model: {model}")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                # For streaming, return an iterator of text chunks
                def iter_text():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return iter_text()
            else:
                # For non-streaming, return the full text
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return a fallback message on error
            return "I'm sorry, but I'm having trouble generating a response right now. Please try again later."

def run_parallel_tasks(task_functions, max_workers=None):
    """
    Run multiple task functions in parallel and return their results.
    
    Args:
        task_functions: List of callable functions with no arguments
        max_workers: Maximum number of worker threads (default: number of tasks)
        
    Returns:
        List of results from the tasks, in the same order as the input functions
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = len(task_functions)
    
    results = [None] * len(task_functions)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, task_fn in enumerate(task_functions):
            future = executor.submit(task_fn)
            futures[future] = i
            
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error in task {idx}: {e}")
                results[idx] = None
                
    return results

def gpt_user_profile_analysis(pages, client, model="gpt-4o", existing_user_profile=""):
    """Generate a user profile analysis using GPT."""
    # Get formatted conversation data
    conversation_data = []
    for page in pages:
        conversation_data.append({
            "user_input": page.get("user_input", ""),
            "agent_response": page.get("agent_response", ""),
            "timestamp": page.get("timestamp", "")
        })
    
    # Create system and user prompts
    system_prompt = """You are MemoryOS's user profile analyzer. Your task is to analyze conversations and 
update the user profile with new information. Based on the conversation history provided, create 
a comprehensive and updated user profile that summarizes everything we know about the user."""

    # Prepare the conversation for the prompt
    conversations_str = ""
    for i, conv in enumerate(conversation_data):
        conversations_str += f"\n--- Conversation {i+1} ---\n"
        conversations_str += f"User: {conv['user_input']}\n"
        conversations_str += f"Assistant: {conv['agent_response']}\n"
        conversations_str += f"Time: {conv['timestamp']}\n"
    
    user_prompt = f"""
## Previous User Profile
{existing_user_profile or "No existing profile information."}

## Recent Conversations
{conversations_str}

Based on both the previous profile information and these recent conversations, create a comprehensive updated user profile. 
Include any new personal information, preferences, interests, technical knowledge, behaviors, or communication patterns you observe.
If there's conflicting information, prioritize the most recent data.

Organize your response as a well-structured profile. Do not include temporary information (like current questions being asked).
Focus on persistent user characteristics, preferences, background, and interests that would be valuable to remember long-term.
"""

    # Call the LLM
    try:
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        return response
    except Exception as e:
        print(f"Error in profile analysis: {e}")
        return existing_user_profile  # Return existing profile on error

def gpt_knowledge_extraction(pages, client, model="gpt-4o"):
    """Extract knowledge from conversations using GPT."""
    # Get formatted conversation data
    conversation_data = []
    for page in pages:
        conversation_data.append({
            "user_input": page.get("user_input", ""),
            "agent_response": page.get("agent_response", ""),
            "timestamp": page.get("timestamp", "")
        })
    
    # Create system and user prompts
    system_prompt = """You are MemoryOS's knowledge extraction system. Your task is to extract important information from 
conversations that should be stored in the long-term memory system. You will extract two types of knowledge:

1. PRIVATE USER KNOWLEDGE: Important facts about the user that should be remembered (preferences, background, needs, etc.)
2. ASSISTANT KNOWLEDGE: Important facts and instructions the assistant should remember for future interactions

Focus on extracting specific, factual information that would be valuable to remember long-term."""

    # Prepare the conversation for the prompt
    conversations_str = ""
    for i, conv in enumerate(conversation_data):
        conversations_str += f"\n--- Conversation {i+1} ---\n"
        conversations_str += f"User: {conv['user_input']}\n"
        conversations_str += f"Assistant: {conv['agent_response']}\n"
        conversations_str += f"Time: {conv['timestamp']}\n"
    
    user_prompt = f"""
## Conversations to Analyze
{conversations_str}

Extract key knowledge from these conversations in two categories:

1. PRIVATE USER KNOWLEDGE: Facts about the user worth remembering (preferences, background, needs, etc.)
2. ASSISTANT KNOWLEDGE: Information the assistant should remember for future interactions

For each category, extract 3-5 specific, factual statements (one per line). Each statement should be:
- Self-contained and specific (avoid vague statements)
- Factual rather than inferential when possible
- Valuable for future interactions
- Written as complete sentences

If no meaningful information exists for a category, write "None" for that category.

Format your response using these exact headings:
PRIVATE USER KNOWLEDGE:
[bullet points here]

ASSISTANT KNOWLEDGE:
[bullet points here]
"""

    # Call the LLM
    try:
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        # Extract the sections from the response
        private_knowledge = "None"
        assistant_knowledge = "None"
        
        if "PRIVATE USER KNOWLEDGE:" in response and "ASSISTANT KNOWLEDGE:" in response:
            parts = response.split("ASSISTANT KNOWLEDGE:")
            if len(parts) >= 2:
                private_part = parts[0].split("PRIVATE USER KNOWLEDGE:")
                if len(private_part) >= 2:
                    private_knowledge = private_part[1].strip()
                assistant_knowledge = parts[1].strip()
                
        # Return structured data
        return {
            "private": private_knowledge,
            "assistant_knowledge": assistant_knowledge
        }
    except Exception as e:
        print(f"Error in knowledge extraction: {e}")
        return {
            "private": "None",
            "assistant_knowledge": "None"
        }  # Return empty results on error

def gpt_generate_multi_summary(conversation_text, client, model="gpt-4o", max_themes=3):
    """Generate multiple summaries for different themes in the conversation."""
    system_prompt = """You are MemoryOS's multi-topic summarizer. Your task is to analyze a conversation and identify different themes or topics discussed. 
For each theme, you'll create a concise summary and extract keywords."""

    user_prompt = f"""
## Conversation to Analyze
{conversation_text}

Analyze this conversation and identify up to {max_themes} distinct themes or topics discussed. For each theme:
1. Create a descriptive theme name (2-5 words)
2. Write a concise summary (2-4 sentences) capturing the key points related to that theme
3. Extract 3-5 relevant keywords that characterize the theme

If the conversation is very focused on a single topic, you may identify fewer themes.

Format your response as a JSON object with this structure:
{{
  "summaries": [
    {{
      "theme": "Theme Name 1",
      "content": "Concise summary of this theme...",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }},
    {{
      "theme": "Theme Name 2",
      "content": "Concise summary of this theme...",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ]
}}
"""

    try:
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        try:
            # Parse JSON response
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # If response isn't valid JSON, extract JSON-like content if possible
            json_pattern = r"({[\s\S]*})"
            matches = re.search(json_pattern, response)
            if matches:
                try:
                    result = json.loads(matches.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create a basic structure with one summary
            print("Warning: Failed to parse multi-summary JSON. Using fallback single summary.")
            return {
                "summaries": [
                    {
                        "theme": "General Conversation",
                        "content": "Summary of the recent conversation.",
                        "keywords": ["conversation", "general", "interaction"]
                    }
                ]
            }
    except Exception as e:
        print(f"Error generating multi-summary: {e}")
        # Return a basic fallback structure
        return {
            "summaries": [
                {
                    "theme": "General Conversation",
                    "content": "Summary of the recent conversation.",
                    "keywords": ["conversation", "general", "interaction"]
                }
            ]
        }

def check_conversation_continuity(previous_page, current_page, client, model="gpt-4o-mini"):
    """Determine if two conversation pages are continuous using LLM."""
    if not previous_page or not current_page:
        return False
    
    prev_text = f"User: {previous_page.get('user_input', '')}\nAssistant: {previous_page.get('agent_response', '')}"
    curr_text = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    
    system_prompt = """You are MemoryOS's continuity analyzer. Your task is to determine if two conversation snippets are continuous 
or part of the same logical thread of conversation."""

    user_prompt = f"""
## Previous Conversation
{prev_text}

## Current Conversation
{curr_text}

Determine if the current conversation is a direct continuation or closely related to the previous conversation.
Consider:
1. Topic similarity
2. References to entities or ideas from the previous conversation
3. Logical flow and context dependencies

Answer with just "YES" if they are continuous/related or "NO" if they appear to be separate topics or unrelated.
"""

    try:
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=5
        )
        
        return "YES" in response.upper()
    except Exception as e:
        print(f"Error checking conversation continuity: {e}")
        return False  # Default to not continuous on error

def generate_page_meta_info(previous_meta_info, current_page, client, model="gpt-4o-mini"):
    """Generate meta-information for a page, considering previous meta-info if available."""
    # If there's no previous meta_info, we need to create a new one from scratch
    curr_text = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    
    if not previous_meta_info:
        system_prompt = """You are MemoryOS's meta-information generator. Your task is to create a brief 
description of a conversation to help identify its context later."""

        user_prompt = f"""
## Conversation
{curr_text}

Create a brief (15-25 words) description that captures the essence of this conversation. 
The description should help identify what this conversation is about when retrieving it from memory.
Focus on the topic, any specific entities mentioned, and the general purpose of the exchange.
"""

        try:
            response = client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=50
            )
            
            return response.strip()
        except Exception as e:
            print(f"Error generating page meta info: {e}")
            return "Conversation about an unspecified topic."  # Fallback
    
    # If there is previous meta_info, we should update it based on the new content
    system_prompt = """You are MemoryOS's meta-information updater. Your task is to update an existing 
conversation description based on new content in the conversation."""

    user_prompt = f"""
## Current Description
{previous_meta_info}

## New Conversation Content
{curr_text}

Update the current description to incorporate this new conversation content. The description should:
1. Be brief (15-25 words)
2. Capture the evolving nature of the conversation 
3. Highlight any new topics or shifts in focus

If the new content is just a continuation of the same topic, you can make minor refinements to the existing description.
"""

    try:
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=50
        )
        
        return response.strip()
    except Exception as e:
        print(f"Error updating page meta info: {e}")
        return previous_meta_info  # On error, keep the previous meta_info
