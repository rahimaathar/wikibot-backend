import uvicorn
import os
from app.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="debug"
    ) 