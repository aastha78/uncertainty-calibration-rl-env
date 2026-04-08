"""Server entry point for OpenEnv multi-mode deployment."""

import os
import uvicorn

# Import the app from the root server module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server import app


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
