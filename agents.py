import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Make sure to set your OpenAI API Key ---
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Load the guidelines
loader = TextLoader('./injury_guidelines.txt')
documents = loader.load()

# 2. Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Setup the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# 5. ICD-10 Codes Dictionary
ICD10_CODES = {
    "S93.401A": "Sprain of unspecified ligament of right ankle, initial encounter",
    "S93.402A": "Sprain of unspecified ligament of left ankle, initial encounter",
    "S52.501A": "Unspecified displaced fracture of the radial styloid process of right radius, initial encounter for closed fracture",
    "S52.502A": "Unspecified displaced fracture of the radial styloid process of left radius, initial encounter for closed fracture",
    "S06.0X0A": "Concussion without loss of consciousness, initial encounter",
    "S06.0X1A": "Concussion with loss of consciousness of 30 minutes or less, initial encounter"
}


from typing import TypedDict, Annotated, List
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# Define the tool for retrieving guidelines
retriever = vectorstore.as_retriever()
@tool
def guideline_retriever_tool(injury_description: str):
    """
    Searches and retrieves relevant treatment guidelines for a given injury description.
    """
    return retriever.invoke(injury_description)

# Define the state for our graph. It will be a list of messages.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- AGENT 1: The Researcher ---
# This agent will decide to use the guideline retriever tool
tool_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
research_agent_llm = tool_llm.bind_tools([guideline_retriever_tool])

def research_agent_node(state: AgentState):
    print("---CALLING RESEARCHER AGENT---")
    # Get the last message
    last_message = state['messages'][-1]
    # Invoke the LLM with the tool
    response = research_agent_llm.invoke(state['messages'])
    # The response is an AI message with tool calls
    return {"messages": [response]}


# --- AGENT 2: The Coder ---
# This agent will analyze the retrieved info and suggest codes.
# It does not need a tool; it will receive the information in the prompt.
coder_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

def coder_agent_node(state: AgentState):
    print("---CALLING CODER AGENT---")
    # The state now contains the user query, the tool call, and the tool response.
    # We build a new prompt with all this context.
    prompt = f"""You are an expert medical coder. Your task is to analyze the user's injury description and the retrieved treatment guidelines to suggest the top 3 most relevant ICD-10 codes.

    Here is the user's request and the relevant guidelines that were found:
    {state['messages']}

    Based on all of this information, analyze the treatment and suggest the top 3 ICD-10 codes from the following list. Provide only the codes and their descriptions, nothing else.

    Available ICD-10 Codes:
    {ICD10_CODES}
    """
    response = coder_llm.invoke(prompt)
    return {"messages": [response]}


# --- Tool Execution Node ---
# This is a pre-built node that executes any tool calls it finds.
tool_node = ToolNode([guideline_retriever_tool])


# The conditional edge will decide whether to call the tool or finish.
def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    # If the last message is an AIMessage with tool calls, we call the tool_node.
    if last_message.tool_calls:
        return "call_tool"
    # Otherwise, we are done.
    return "end"

# Create the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("coder_agent", coder_agent_node)


# Define the edges
workflow.set_entry_point("research_agent")

# Conditional routing: after the researcher runs, check if a tool was called.
workflow.add_conditional_edges(
    "research_agent",
    should_continue,
    {
        "call_tool": "tool_executor",
        "end": "end" # This path is unlikely in our case but good practice
    }
)

# After the tool is executed, the result is passed to the coder agent
workflow.add_edge("tool_executor", "coder_agent")

# After the coder agent runs, the process ends
workflow.add_edge("coder_agent", END)

# Compile the graph
app = workflow.compile()```

#### Step 4: Run the Workflow (Key Changes)

The invocation now requires the input to be in the `messages` format.

```python
# The input is now a list of messages.
inputs = {"messages": [HumanMessage(content="The patient has a sprained ankle and we need to follow the RICE protocol.")]}

# Invoke the workflow
result = app.invoke(inputs, {"recursion_limit": 100})

# Print the final result, which is the last message from the coder agent
print("\n---FINAL SUGGESTED ICD-10 CODES---")
print(result['messages'][-1].content)
