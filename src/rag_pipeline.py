from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import yaml

llm = ChatOpenAI(model="gpt-4o-mini")

# Load saved books embeddings 
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="interview_books",
    embedding_function=embeddings,
    persist_directory="data/embedings/chroma_langchain_books_db",  
)

# Load prompts used in this RAG
def load_prompts(config_path="data/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)["prompts"]

PROMPTS = load_prompts()


# Initiate LangChain computational graph
graph_builder = StateGraph(MessagesState)


# Custom document tool to retrieve k most simillar docs to a given query
@tool(response_format="content_and_artifact")
def document_tool(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    array = [doc.metadata for doc in retrieved_docs]
    return  retrieved_docs, array



# Wikipedia API Wrapper Tool
wikipedia_tool = WikipediaAPIWrapper(load_all_available_meta= True)
@tool(response_format="content_and_artifact")
def wikipedia_data(query: str):
    """Retrieve a summary from Wikipedia for a given query."""
    article_link = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"

    summary = wikipedia_tool.run(query)
    
    return summary, article_link

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    
    prompt = PROMPTS["query_or_respond_prompt"]
    
    messages_with_prompt = [{"role": "system", "content": prompt}] + state["messages"]

    llm_with_tools = llm.bind_tools([document_tool,wikipedia_data])
    
    # Invoke the LLM with the current message context
    response = llm_with_tools.invoke(messages_with_prompt)
    
    # MessagesState appends responses to state instead of overwriting
    return {"messages": [response]}


tools = ToolNode([document_tool,wikipedia_data])


# Step 3: Generate a response using the retrieved content.
# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    
    # Extract the latest tool messages
    tool_messages = [message for message in reversed(state["messages"]) if message.type == "tool"][::-1]

    # Extract content and artifacts
    docs_content = "\n\n".join(message.content for message in tool_messages if hasattr(message, "content"))
    system_message_content = f"{PROMPTS['generate_prompt'].format()}\n\n{docs_content}"

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    
    
    artifacts = [message.artifact for message in tool_messages if hasattr(message, "artifact")]
    if len(artifacts) > 1:
        merged_dict = {key: value for d in  artifacts[1] for key, value in d.items()}
        response.content += "\n\n Reference Book is {}, page = {}".format(merged_dict["source"],merged_dict["page"])
        

    return {"messages": [response]}


# Ensure that we only transition to the 'generate' step if relevant content is found
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

# Entry point for the graph is `query_or_respond`
graph_builder.set_entry_point("query_or_respond")

# Define the conditional edges based on whether tools were used or not
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()
