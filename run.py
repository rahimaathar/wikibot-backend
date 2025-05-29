import uvicorn
import sys
import os
from pathlib import Path

# Get the absolute path to the backend directory
backend_dir = Path(__file__).parent.absolute()
app_dir = backend_dir / "app"

# Add the backend directory to Python path
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    try:
        print(f"Starting server from directory: {backend_dir}")
        print(f"Python path: {sys.path}")
        
        # Get port from environment variable or use default
        port = int(os.getenv("PORT", "8000"))
        
        # Use import string format for uvicorn.run
        uvicorn.run(
            "app.main:app",  # Use import string format
            host="0.0.0.0",
            port=port,
            reload=False,  # Disable reload in production
            log_level="info"  # Use info level for production
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 