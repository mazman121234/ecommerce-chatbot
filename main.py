from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open("public/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: HTML file not found</h1>"
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        message = request.message.strip()
        
        if not message or len(message) == 0:
            return ChatResponse(response="Hi! How can I help you today?")
        
        if len(message) > 1000:
            return ChatResponse(response="Message too long. Please keep it under 1000 characters.")
        
        messages = [
            {"role": "system", "content": """You are a helpful shopping assistant for an e-commerce store. Your job is to have natural conversations with customers about shopping.

KEY RULE: Accept and respond naturally to ANYTHING the customer types. Never reject input. Never say "I can only help with shopping questions."

- If they type shopping-related stuff (products, prices, shipping, returns) - answer helpfully
- If they type a name (john, sarah, john smith) - acknowledge it as their name/account
- If they type a number (4, 10, 500) - treat it as quantity or order amount
- If they type random words or gibberish - acknowledge it and stay helpful
- If they ask unrelated stuff (math, sports, jokes) - respond naturally but gently redirect to shopping

NEVER restart conversations. NEVER show a default greeting mid-conversation. Always continue what they started.

Be conversational, friendly, and flexible. Work with whatever they give you."""}
        ]
        
        if request.history:
            messages.extend(request.history)
        
        messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.9
        )
        
        bot_response = response.choices[0].message.content
        
        if not bot_response or len(bot_response.strip()) == 0:
            return ChatResponse(response="I'm here to help! What would you like?")
        
        return ChatResponse(response=bot_response.strip())
    
    except ValueError as e:
        return ChatResponse(response="Let me help you with that!")
    except Exception as e:
        return ChatResponse(response="Sorry, I'm temporarily unavailable. Please try again in a moment.")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}