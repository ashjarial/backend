from dotenv import load_dotenv
load_dotenv()
from urllib.parse import urlparse

import os
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
import chatbot
from chatbot import ask_about_ashish, RateLimitError
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# from llm import ask_about_ashish, RateLimitError

SCHEMA = """
CREATE TABLE IF NOT EXISTS visitors (
    id        BIGSERIAL PRIMARY KEY,
    name      TEXT NOT NULL,
    questions TEXT NOT NULL DEFAULT ''
);
"""

pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    password = os.getenv("DB_PASSWORD")
    if password:
        pool = await asyncpg.create_pool(
            host="aws-0-ap-northeast-1.pooler.supabase.com",
            port=5432,
            user="postgres.agnuhtwiatrqzqowfqir",
            password=password,
            database="postgres",
            min_size=1, max_size=3, ssl="require", statement_cache_size=0,
        )
        async with pool.acquire() as con:
            await con.execute(SCHEMA)
    yield
    if pool:
        await pool.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://portfolio-sigma-jet-46.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class VisitorRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    name: str = Field("", max_length=60)
    visitor_id: Optional[int] = None   # the row number from /visitor


def clean(q: str) -> str:
    # commas separate questions in the column, so swap any inside a question
    return " ".join(q.replace(",", ";").split())


async def append_question(visitor_id: Optional[int], name: str, question: str) -> None:
    if not pool or not name:
        return
    q = clean(question)
    try:
        async with pool.acquire() as con:
            updated = None
            if visitor_id:
                updated = await con.fetchval(
                    """
                    UPDATE visitors
                    SET questions = CASE WHEN questions = '' THEN $2
                                         ELSE questions || ', ' || $2 END
                    WHERE id = $1
                    RETURNING id
                    """,
                    visitor_id, q,
                )
            if updated is None:
                # fallback: row missing (e.g. /visitor failed), so create it now
                await con.execute(
                    "INSERT INTO visitors (name, questions) VALUES ($1, $2)", name, q
                )
    except Exception as e:
        print("append_question failed:", e, flush=True)


@app.get("/health")
async def health():
    if pool:
        async with pool.acquire() as con:
            await con.fetchval("SELECT 1")
    return {"ok": True}


@app.post("/visitor")
async def register_visitor(req: VisitorRequest):
    """Called when the visitor submits their name: creates the row, returns its number."""
    if not pool:
        return {"id": None}
    try:
        async with pool.acquire() as con:
            new_id = await con.fetchval(
                "INSERT INTO visitors (name) VALUES ($1) RETURNING id", req.name.strip()
            )
        return {"id": new_id}
    except Exception as e:
        print("register_visitor failed:", e, flush=True)
        return {"id": None}


@app.post("/chat")
async def chat(req: ChatRequest, background: BackgroundTasks):
    try:
        answer = await ask_about_ashish(req.question)
    except RateLimitError:
        answer = "I'm getting a lot of questions right now. Please try again in a minute."
    except Exception as e:
        print("LLM error:", e, flush=True)
        answer = "Sorry, I could not answer that right now. Please try again in a moment."

    background.add_task(append_question, req.visitor_id, req.name.strip(), req.question)
    return {"answer": answer}


@app.get("/admin/logs")
async def get_logs(x_admin_token: str = Header(default="")):
    if not os.getenv("ADMIN_TOKEN") or x_admin_token != os.getenv("ADMIN_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not pool:
        return []
    async with pool.acquire() as con:
        rows = await con.fetch("SELECT id, name, questions FROM visitors ORDER BY id DESC LIMIT 500")
    return [dict(r) for r in rows]



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)