import os
from typing import TypedDict, Annotated
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- 1. SETUP AND DATA LOADING ---

# Set your OpenAI API key
# Make sure to replace "YOUR_OPENAI_API_KEY" with your actual key
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Load the guidelines document
loader = TextLoader('./injury_guidelines.txt')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a Chroma vector store and retriever
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Define the dictionary of ICD-10 codes
ICD10_CODES = {
    "S93.401A": "Sprain of unspecified ligament of right ankle, initial encounter",
    "S93.402A": "Sprain of unspecified ligament of left ankle, initial encounter",
    "S52.501A": "Unspecified displaced fracture of the radial styloid process of right radius, initial encounter for closed fracture",
    "S52.502A": "Unspecified displaced fracture of the radial styloid process of left radius, initial encounter for closed fracture",
    "S06.0X0A": "Concussion without loss of consciousness, initial encounter",
    "S06.0X1A": "Concussion with loss of consciousness of 30 minutes or less, initial encounter"
}

# --- 2. AGENT AND GRAPH DEFINITION ---

@tool
def guideline_retriever_tool(injury_description: str):
    """
    Searches and retrieves relevant treatment guidelines for a given injury description.
    """
    print(f"---TOOL CALLED: Retrieving guidelines for '{injury_description}'---")
    return retriever.invoke(injury_description)

# Define the state for our graph, which will be a list of messages
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Define the Researcher Agent
# This agent decides if a tool should be used to get more information.
research_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
research_agent_llm = research_llm.bind_tools([guideline_retriever_tool])

def research_agent_node(state: AgentState):
    """Invokes the researcher LLM to decide on an action."""
    print("---NODE: RESEARCH AGENT---")
    response = research_agent_llm.invoke(state['messages'])
    return {"messages": [response]}

# Define the Coder Agent
# This agent analyzes all information and provides the final answer.
coder_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

def coder_agent_node(state: AgentState):
    """Invokes the coder LLM to generate the final response."""
    print("---NODE: CODER AGENT---")
    
    # The state contains the full conversation history, including the tool's output.
    # We create a new prompt for the coder agent to synthesize this information.
    prompt = f"""You are an expert medical coder. Your task is to analyze the user's injury description and the retrieved treatment guidelines to suggest the top 3 most relevant ICD-10 codes.

    The conversation history, including the retrieved guidelines, is as follows:
    {state['messages']}

    Based on all of this information, analyze the treatment and suggest the top 3 most relevant ICD-10 codes from the following list. Respond with only the codes and their descriptions.

    Available ICD-10 Codes:
    {ICD10_CODES}
    """
    response = coder_llm.invoke(prompt)
    return {"messages": [response]}

# Define the pre-built ToolNode that will execute our tools
tool_node = ToolNode([guideline_retriever_tool])

# --- 3. GRAPH CONSTRUCTION ---

# Define the conditional logic for routing
def should_continue(state: AgentState) -> str:
    """Determines the next step after the researcher agent has run."""
    print("---EDGE: Checking for tool calls---")
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        # If the LLM made a tool call, we route to the tool executor
        return "call_tool"
    # Otherwise, we end the process
    return "end"

# Create the StateGraph
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("coder_agent", coder_agent_node)

# Define the edges and the workflow's entry point
workflow.set_entry_point("research_agent")

# Add a conditional edge: after the researcher runs, check if a tool was called
workflow.add_conditional_edges(
    "research_agent",
    should_continue,
    {
        "call_tool": "tool_executor",
        "end": END
    }
)

# Define the edge from the tool executor to the coder agent
workflow.add_edge("tool_executor", "coder_agent")

# The coder agent's response marks the end of the workflow
workflow.add_edge("coder_agent", END)

# Compile the graph into a runnable application
app = workflow.compile()


# --- 4. RUNNING THE WORKFLOW ---

# Define the initial input for the workflow
inputs = {"messages": [HumanMessage(content="The patient has a sprained ankle and we need to follow the RICE protocol.")]}

print("---STARTING WORKFLOW---")
# Invoke the workflow and stream the output
for output in app.stream(inputs, {"recursion_limit": 100}):
    # The 'output' dictionary contains the state of the graph at each step
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

# The final result is in the last message of the final state
final_result = app.invoke(inputs, {"recursion_limit": 100})
final_answer = final_result['messages'][-1]

print("---FINAL SUGGESTED ICD-10 CODES---")
print(final_answer.content)
