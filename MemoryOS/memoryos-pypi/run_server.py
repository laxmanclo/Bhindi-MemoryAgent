#!/usr/bin/env python3
"""
MemoryOS Server Launcher

This script launches either the Flask or FastAPI version of the MemoryOS server.
The FastAPI version offers better performance, concurrency, and scalability.
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import webbrowser

def signal_handler(sig, frame):
    print("Shutting down server...")
    sys.exit(0)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import fastapi
        import uvicorn
        import pinecone
        import torch
        import sentence_transformers
        print("All required dependencies are installed.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False

def launch_flask(host, port, debug=False):
    """Launch the Flask version of the server"""
    print(f"Starting Flask server on http://{host}:{port}")
    from app import app
    app.run(host=host, port=port, debug=debug)

def launch_fastapi(host, port, reload=False):
    """Launch the FastAPI version of the server"""
    print(f"Starting FastAPI server on http://{host}:{port}")
    import uvicorn
    uvicorn.run("fastapi_app:app", host=host, port=port, reload=reload)

def main():
    parser = argparse.ArgumentParser(description="Launch MemoryOS server")
    parser.add_argument(
        "--server", 
        choices=["flask", "fastapi"], 
        default="fastapi",
        help="Server implementation to use (default: fastapi)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host IP address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000 for FastAPI, 5000 for Flask)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--open-browser", 
        action="store_true", 
        help="Open web browser after server starts"
    )
    
    args = parser.parse_args()
    
    # Set default ports if not specified
    if args.port is None:
        if args.server == "flask":
            args.port = 5000
        else:
            args.port = 8000

    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check for dependencies
    check_dependencies()
    
    # Create data directory if it doesn't exist
    os.makedirs("memoryos_data", exist_ok=True)
    
    # Launch the selected server
    if args.server == "flask":
        print("⚠️ Warning: Flask server is slower and less scalable. Consider using FastAPI for production.")
        
        if args.open_browser:
            webbrowser.open(f"http://localhost:{args.port}")
            
        launch_flask(args.host, args.port, args.debug)
    else:
        print("✅ Using FastAPI server for optimal performance and scalability.")
        
        if args.open_browser:
            # Give the server a moment to start before opening browser
            def delayed_browser_open():
                time.sleep(1)
                webbrowser.open(f"http://localhost:{args.port}")
            
            import threading
            threading.Thread(target=delayed_browser_open).start()
            
        launch_fastapi(args.host, args.port, args.debug)

if __name__ == "__main__":
    main()
