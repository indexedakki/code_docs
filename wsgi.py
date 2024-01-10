  from fastapi import FastAPI
  from main import main # Import your FastAPI application

  if __name__ == "__main__":
      import uvicorn
      uvicorn.run(main, host="0.0.0.0", port=8000)
