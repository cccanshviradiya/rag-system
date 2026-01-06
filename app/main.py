from fastapi import FastAPI
from app.api import router
from app.db import create_table

app = FastAPI(title="Mini RAG System")

@app.on_event("startup")
def startup_event():
    create_table()

app.include_router(router)
