/**
 * MemoryOS Service (Fixed Version)
 * 
 * This service handles communication with the MemoryOS API to perform memory operations
 * such as querying memory, adding memories, and managing chat sessions.
 */
import { config } from 'dotenv';

// Load environment variables
config();

// The base URL for the MemoryOS API (adjust as needed)
const MEMORYOS_API_URL = process.env.MEMORYOS_API_URL || 'http://localhost:5000';

/**
 * Query the MemoryOS system to retrieve contextual responses
 * @param query The query or question to ask the memory system
 * @returns The response from MemoryOS
 */
export const queryMemory = async (query: string) => {
  try {
    // Always use the underscore version of the endpoint
    let response = await fetch(`${MEMORYOS_API_URL}/tools/query_memory`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    // Fallback to camelCase if underscore version fails
    if (!response.ok) {
      response = await fetch(`${MEMORYOS_API_URL}/tools/queryMemory`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to query memory');
    }

    const data = await response.json();
    console.log('Query memory response:', JSON.stringify(data));
    return {
      success: true,
      responseType: 'text',
      data: {
        operation: `Memory Query: "${query}"`,
        result: data.data?.text || data.response || JSON.stringify(data),
        message: data.data?.text || data.response || JSON.stringify(data),
        tool_type: 'memory'
      }
    };
  } catch (error) {
    console.error('Error querying memory:', error);
    return {
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error querying memory',
        code: 500,
        details: 'Check if the MemoryOS server is running and accessible'
      }
    };
  }
};

/**
 * Create a new chat session in MemoryOS
 * @param chatName Optional name for the new chat session
 * @returns Information about the newly created chat
 */
export const createNewChat = async (chatName: string = 'New Chat') => {
  try {
    // Always use the underscore version of the endpoint
    let response = await fetch(`${MEMORYOS_API_URL}/tools/create_new_chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ chat_name: chatName }),
    });

    // Fallback to camelCase if underscore version fails
    if (!response.ok) {
      response = await fetch(`${MEMORYOS_API_URL}/tools/createNewChat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ chat_name: chatName }),
      });
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to create new chat');
    }

    const data = await response.json();
    console.log('Create new chat response:', JSON.stringify(data));
    return {
      success: true,
      responseType: 'text',
      data: {
        operation: `Create New Chat: "${chatName}"`,
        result: data.data?.text || data.chat_id || JSON.stringify(data),
        message: data.data?.text || `Chat created with ID: ${data.chat_id}` || JSON.stringify(data),
        tool_type: 'memory'
      }
    };
  } catch (error) {
    console.error('Error creating new chat:', error);
    return {
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error creating chat',
        code: 500,
        details: 'Check if the MemoryOS server is running and accessible'
      }
    };
  }
};

/**
 * Switch to an existing chat session
 * @param chatId The ID of the chat to switch to
 * @returns Confirmation of chat switch
 */
export const switchToChat = async (chatId: string) => {
  try {
    // Always use the underscore version of the endpoint
    let response = await fetch(`${MEMORYOS_API_URL}/tools/switch_to_chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ chat_id: chatId }),
    });

    // Fallback to camelCase if underscore version fails
    if (!response.ok) {
      response = await fetch(`${MEMORYOS_API_URL}/tools/switchToChat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ chat_id: chatId }),
      });
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to switch chat');
    }

    const data = await response.json();
    console.log('Switch chat response:', JSON.stringify(data));
    return {
      success: true,
      responseType: 'text',
      data: {
        operation: `Switch to Chat: "${chatId}"`,
        result: data.data?.text || data.status || JSON.stringify(data),
        message: data.data?.text || data.message || JSON.stringify(data),
        tool_type: 'memory'
      }
    };
  } catch (error) {
    console.error('Error switching chat:', error);
    return {
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error switching chat',
        code: 500,
        details: 'Check if the MemoryOS server is running and accessible'
      }
    };
  }
};

/**
 * Add a new memory interaction to MemoryOS
 * @param userInput The user's message to store
 * @param agentResponse The agent's response to store
 * @returns Confirmation of memory addition
 */
export const addMemory = async (userInput: string, agentResponse: string) => {
  try {
    // Always use the underscore version of the endpoint
    let response = await fetch(`${MEMORYOS_API_URL}/tools/add_memory`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_input: userInput, agent_response: agentResponse }),
    });

    // Fallback to camelCase if underscore version fails
    if (!response.ok) {
      response = await fetch(`${MEMORYOS_API_URL}/tools/addMemory`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput, agent_response: agentResponse }),
      });
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to add memory');
    }

    const data = await response.json();
    console.log('Add memory response:', JSON.stringify(data));
    return {
      success: true,
      responseType: 'text',
      data: {
        operation: 'Add Memory',
        result: data.data?.text || data.status || JSON.stringify(data),
        message: data.data?.text || data.message || "Memory added successfully",
        tool_type: 'memory'
      }
    };
  } catch (error) {
    console.error('Error adding memory:', error);
    return {
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error adding memory',
        code: 500,
        details: 'Check if the MemoryOS server is running and accessible'
      }
    };
  }
};

/**
 * Get a list of all available chat sessions
 * @returns List of available chats
 */
export const getChatList = async () => {
  try {
    // Always use the underscore version of the endpoint
    let response = await fetch(`${MEMORYOS_API_URL}/tools/get_chat_list`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });

    // Fallback to camelCase if underscore version fails
    if (!response.ok) {
      response = await fetch(`${MEMORYOS_API_URL}/tools/getChatList`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to get chat list');
    }

    const data = await response.json();
    console.log('Get chat list response:', JSON.stringify(data));
    return {
      success: true,
      responseType: 'text',
      data: {
        operation: 'Get Chat List',
        result: data.data?.text || JSON.stringify(data.chats) || JSON.stringify(data),
        message: data.data?.text || `Available chats: ${JSON.stringify(data.chats)}` || JSON.stringify(data),
        tool_type: 'memory'
      }
    };
  } catch (error) {
    console.error('Error getting chat list:', error);
    return {
      success: false,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error getting chat list',
        code: 500,
        details: 'Check if the MemoryOS server is running and accessible'
      }
    };
  }
};
