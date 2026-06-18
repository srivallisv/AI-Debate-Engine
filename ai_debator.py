import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 1. SETUP THE LLM & API KEY
# Replace 'YOUR_API_KEY' with your actual Gemini API Key, or leave it if you set an environment variable.
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

# Initialize Gemini 1.5 Flash (fast, smart, and free tier friendly)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2) # Lower temperature = more consistent judging

# 2. SETUP THE KNOWLEDGE BASE (RAG)
# Let's mock a small database of "facts" the agents can research.
facts_dataset = [
    "Fact: A 2024 study showed that working from home increased employee happiness by 22% but reduced spontaneous collaboration by 14%.",
    "Fact: Global corporate real estate expenses dropped by $15 billion globally due to remote work shifts between 2021 and 2025.",
    "Fact: Cybersecurity breaches in remote-first companies rose by 35% compared to traditional office environments.",
    "Fact: Urban restaurants and local businesses experienced an average 18% decline in lunchtime revenue due to empty city offices."
]

# Convert strings into LangChain Document objects
documents = [Document(page_content=text) for text in facts_dataset]

# Initialize Embeddings and Vector Store (stored in memory for this demo)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma.from_documents(documents, embeddings)

def retrieve_facts(query: str) -> str:
    """Queries the vector database for facts relevant to the debate topic."""
    docs = vector_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

# 3. DEFINE THE DEBATE LOGIC
def run_debate(motion: str):
    print(f"🎬 DEBATE MOTION: {motion}\n" + "="*50 + "\n")
    
    # Step A: RAG - Fetch evidence for both sides
    evidence = retrieve_facts(motion)
    
    # Step B: Pro Agent speaks
    pro_prompt = f"""
    You are an expert debater. You are strictly **PRO** (in favor of) the motion: '{motion}'.
    Use the following real-world data points to back up your argument if relevant:
    {evidence}
    
    Write a concise, compelling opening statement (max 150 words). Focus heavily on logic and data.
    """
    pro_argument = llm.invoke(pro_prompt).content
    print(f"🎙️ [PRO AGENT]:\n{pro_argument}\n" + "-"*40)

    # Step C: Con Agent speaks
    con_prompt = f"""
    You are an expert debater. You are strictly **CON** (against) the motion: '{motion}'.
    You just heard the Pro Agent say: '{pro_argument}'
    
    Use the following real-world data points to back up your counter-argument if relevant:
    {evidence}
    
    Write a concise, compelling rebuttal (max 150 words). Directly counter their points using logic and data.
    """
    con_argument = llm.invoke(con_prompt).content
    print(f"🎙️ [CON AGENT]:\n{con_argument}\n" + "-"*40)

    # Step D: The Judge evaluates
    judge_prompt = f"""
    You are an unbiased, highly critical Debate Judge. 
    Evaluate the following debate on the motion: '{motion}'.
    
    PRO Argument: {pro_argument}
    CON Argument: {con_argument}
    
    Provide your verdict in this exact format:
    ### 📊 JUDGE SCORECARD
    * **Pro Score:** [0-10] / 10
    * **Con Score:** [0-10] / 10
    * **Winner:** [Pro / Con / Tie]
    * **Reasoning:** [1-2 sentences explaining why based on logic and use of data]
    """
    verdict = judge_llm.invoke(judge_prompt).content
    print(f"{verdict}\n" + "="*50)

# 4. EXECUTE THE DEBATE
if __name__ == "__main__":
    debate_topic = "Remote work is ultimately bad for the economy."
    run_debate(debate_topic)