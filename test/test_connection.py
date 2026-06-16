from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load the GROQ_API_KEY from your .env file into environment variables
load_dotenv()

# Connect to a free, open-source model hosted by Groq
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL"),
)

response = llm.invoke("Say hello and tell me what model you are in one sentence.")
print("Test LLM Response:")
print(response.content)