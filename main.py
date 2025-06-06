from app.main import app

# This file serves as the entry point for the application
# The actual FastAPI app is defined in app/main.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 