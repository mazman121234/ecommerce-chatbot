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

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open("public/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>HTML file not found</h1>"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful e-commerce customer service chatbot. Your role is to assist customers with:
- Product information, features, and specifications
- Product recommendations based on their needs
- Availability and stock status
- Pricing, discounts, and promotions
- Shipping times and delivery information
- Order tracking and status
- Returns, refunds, and exchanges
- Payment methods and checkout issues
- Size, color, and variant options
- Account and login help
- General shopping questions

If someone asks a question that is NOT related to shopping, products, or our store, politely respond with:
"I'm here to help with questions about our products, orders, shipping, and returns. How can I assist you with your shopping?"

Be friendly, helpful, and professional. Provide specific product details when asked. For complex issues, offer to connect them with a human representative."""},
                {"role": "user", "content": request.message}
            ],
            max_tokens=500
        )
        
        bot_response = response.choices[0].message.content
        
        return ChatResponse(response=bot_response)
    
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}