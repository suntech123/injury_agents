ICD10_CODES = {
    "S93.401A": "Sprain of unspecified ligament of right ankle, initial encounter",
    "S93.402A": "Sprain of unspecified ligament of left ankle, initial encounter",
    "S52.501A": "Unspecified displaced fracture of the radial styloid process of right radius, initial encounter for closed fracture",
    "S52.502A": "Unspecified displaced fracture of the radial styloid process of left radius, initial encounter for closed fracture",
    "S06.0X0A": "Concussion without loss of consciousness, initial encounter",
    "S06.0X1A": "Concussion with loss of consciousness of 30 minutes or less, initial encounter"
}

##Step 2: Building the ChromaDB Vector Store
##We will now load the guidelines, split them into manageable chunks, and store them in a ChromaDB vector store.

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Load the guidelines
loader = TextLoader('./injury_guidelines.txt')
documents = loader.load()

# Split the document into chunks. [2, 3]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Set up the vector store. [12, 13, 15]
vectorstore = Chroma.from_documents(docs, embeddings)


##Step 3: Defining the Graph State and Tools
##The state will be a dictionary that is passed between the nodes of our graph. We also define the tools our agents can use.


from typing import TypedDict, Annotated, List
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# The dictionary of ICD-10 codes
ICD10_CODES = {
    "S93.401A": "Sprain of unspecified ligament of right ankle, initial encounter",
    "S93.402A": "Sprain of unspecified ligament of left ankle, initial encounter",
    "S52.501A": "Unspecified displaced fracture of the radial styloid process of right radius, initial encounter for closed fracture",
    "S52.502A": "Unspecified displaced fracture of the radial styloid process of left radius, initial encounter for closed fracture",
    "S06.0X0A": "Concussion without loss of consciousness, initial encounter",
    "S06.0X1A": "Concussion with loss of consciousness of 30 minutes or less, initial encounter"
}

# Define the tools
retriever = vectorstore.as_retriever()

@tool
def guideline_retriever_tool(injury_description: str):
    """Retrieves relevant treatment guidelines for a given injury."""
    return retriever.invoke(injury_description)

@tool
def icd_code_lookup_tool(query: str):
    """Provides a list of all available ICD-10 codes and their descriptions."""
    return ICD10_CODES

# Define the state for our graph
class AgentState(TypedDict):
    injury_description: str
    retrieved_guidelines: List[str]
    analyzed_treatment: str
    suggested_codes: dict



# Step 4: Creating the Agents (Nodes)
# Now, we will define the functions that represent our agents.

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Guideline Analysis Agent
class GuidelineAnalysis(BaseModel):
    """Analyzed treatment information from the guidelines."""
    treatment_summary: str = Field(description="A summary of the treatment protocols from the retrieved guidelines.")

structured_llm = llm.with_structured_output(GuidelineAnalysis)

def guideline_analysis_agent(state: AgentState):
    print("---ANALYZING GUIDELINES---")
    prompt = f"""You are a medical expert. Analyze the following retrieved treatment guidelines and provide a concise summary of the treatment.
    Guidelines: {state['retrieved_guidelines']}"""
    response = structured_llm.invoke(prompt)
    return {"analyzed_treatment": response.treatment_summary}

# ICD-10 Code Selection Agent
class ICDCodeSelection(BaseModel):
    """The top 3 most relevant ICD-10 codes."""
    top_3_codes: List[str] = Field(description="A list of the top 3 ICD-10 codes.")

structured_llm_icd = llm.with_structured_output(ICDCodeSelection)

def icd_code_selection_agent(state: AgentState):
    print("---SELECTING ICD-10 CODES---")
    prompt = f"""Based on the following treatment analysis, select the top 3 most relevant ICD-10 codes from the provided list.
    Treatment Analysis: {state['analyzed_treatment']}
    ICD-10 Codes: {ICD10_CODES}"""
    response = structured_llm_icd.invoke(prompt)
    suggested_codes = {code: ICD10_CODES[code] for code in response.top_3_codes if code in ICD10_CODES}
    return {"suggested_codes": suggested_codes}


# Step 5: Constructing the Graph
# Now we will wire everything together in a StateGraph.


from langgraph.prebuilt import ToolNode

# Define the tool nodes
guideline_tool_node = ToolNode([guideline_retriever_tool])
icd_tool_node = ToolNode([icd_code_lookup_tool])

# Create the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("guideline_retriever", guideline_tool_node)
workflow.add_node("guideline_analyzer", guideline_analysis_agent)
workflow.add_node("icd_selector", icd_code_selection_agent)

# Define the edges
workflow.set_entry_point("guideline_retriever")
workflow.add_edge("guideline_retriever", "guideline_analyzer")
workflow.add_edge("guideline_analyzer", "icd_selector")
workflow.add_edge("icd_selector", END)

# Compile the graph
app = workflow.compile()



# Step 6: Running the Agentic Workflow
# Finally, we can run our multi-agent system with an initial injury description.

# Invoke the workflow
inputs = {"injury_description": "The patient has a sprained ankle and we need to follow the RICE protocol."}
result = app.invoke(inputs, {"recursion_limit": 100})

# Print the final result
print("\n---SUGGESTED ICD-10 CODES---")
for code, description in result['suggested_codes'].items():
    print(f"- {code}: {description}")




















