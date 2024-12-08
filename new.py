from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
openai_client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("API_KEY", "lm-studio")
)


def call_llm(question: str) -> str:
    """
    Call the LLM to answer a legal question.
    """
    prompt = f"""
    You are a highly skilled lawyer specializing in legal matters. Answer the user's question in a clear, concise manner, simplifying complex legal terms.

    Question:
    {question}

    Answer:
    """
    try:
        response = openai_client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error while querying the LLM: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def get_ui() -> HTMLResponse:
    """
    Render a sleek UI for user questions.
    """
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Legal Chatbot</title>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
                    color: #fff;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 15px;
                    padding: 30px;
                    width: 80%;
                    max-width: 800px;
                    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
                    text-align: center;
                }
                h1 {
                    font-size: 2.5rem;
                    margin-bottom: 20px;
                }
                textarea {
                    width: 100%;
                    height: 100px;
                    margin: 10px 0;
                    border-radius: 10px;
                    padding: 10px;
                    border: none;
                    font-size: 1rem;
                    outline: none;
                }
                button {
                    background: linear-gradient(to right, #56ab2f, #a8e063);
                    border: none;
                    border-radius: 20px;
                    padding: 10px 20px;
                    color: white;
                    font-size: 1.2rem;
                    cursor: pointer;
                    transition: background 0.3s ease-in-out;
                }
                button:hover {
                    background: linear-gradient(to right, #a8e063, #56ab2f);
                }
                .result {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    margin-top: 20px;
                    padding: 20px;
                    font-size: 1rem;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Legal Chatbot</h1>
                <p>Ask a legal question and get instant answers from a virtual lawyer.</p>
                <form action="/ask" method="post">
                    <textarea name="question" placeholder="Type your legal question here..."></textarea>
                    <button type="submit">Ask</button>
                </form>
                <div id="results">
                    <h2>Your Results Will Appear Here</h2>
                </div>
            </div>
        </body>
        </html>
    """)


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(question: str = Form(...)) -> HTMLResponse:
    """
    Handle user questions and respond using the LLM.
    """
    if not question.strip():
        return HTMLResponse(content="""
            <div class="container">
                <h1>Error:</h1>
                <p>Question cannot be empty. Please enter a valid question.</p>
                <a href="/" style="color: #56ab2f;">Back</a>
            </div>
        """)

    try:
        answer = call_llm(question)
        return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Legal Chatbot</title>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        margin: 0;
                        padding: 0;
                        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
                        color: #fff;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                    }}
                    .container {{
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 15px;
                        padding: 30px;
                        width: 80%;
                        max-width: 800px;
                        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
                        text-align: center;
                    }}
                    h1 {{
                        font-size: 2.5rem;
                        margin-bottom: 20px;
                    }}
                    p {{
                        font-size: 1.2rem;
                    }}
                    a {{
                        color: #56ab2f;
                        text-decoration: none;
                        font-size: 1rem;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Your Question:</h1>
                    <p>{question}</p>
                    <h1>Answer:</h1>
                    <p>{answer}</p>
                    <a href="/">Ask Another Question</a>
                </div>
            </body>
            </html>
        """)
    except Exception as e:
        return HTMLResponse(content=f"""
            <div class="container">
                <h1>Error:</h1>
                <p>{str(e)}</p>
                <a href="/" style="color: #56ab2f;">Back</a>
            </div>
        """)
