  from fastapi import FastAPI
  from main import app # Import your FastAPI application

  if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)
