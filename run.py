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
        
        uvicorn.run(
            "app.main:app",  # Use import string instead of app instance
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(app_dir)],
            log_level="debug"  # Changed to debug for more detailed logs
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 