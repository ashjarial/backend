import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot import ask_about_ashish

app = FastAPI(title="Ashish AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/")
def home():
    return {"message": "Ashish AI is running!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        answer = await ask_about_ashish(request.question)
        return {"answer": answer}
    except Exception as error:
        traceback.print_exc()  # prints the real error in the uvicorn terminal
        raise HTTPException(
            status_code=503,
            detail=f"AI service error: {error}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)