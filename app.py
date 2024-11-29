import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph
from langchain.document_loaders import PyPDFLoader
import streamlit as st


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="interview_books",
    embedding_function=embeddings,
    persist_directory="embedings/chroma_langchain_books_db",  
)


pdf_directory = "pdfs"
all_documents = []

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = filename
        all_documents.extend(documents) 



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_documents)
_ = vector_store.add_documents(documents=all_splits)



graph_builder = StateGraph(MessagesState)

# Wikipedia API Wrapper Tool
wikipedia_tool = WikipediaAPIWrapper(load_all_available_meta= True)


"function that defines agent tool, returns retrieved subdocs and meta info such as book name and page"
@tool(response_format="content_and_artifact")
def document_tool(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


@tool(response_format="content_and_artifact")
def wikipedia_data(query: str):
    """Retrieve a summary from Wikipedia for a given query."""
    
    summary = wikipedia_tool.run(query)
    
    
    
    return summary


"define graph nodes by functions each graph nodes executes"

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    
    prompt = """
    If the user's message requires information available in the documents, use the `document_tool` to retrieve relevant content. 
    If no relevant content is found in the documents or if additional context is needed, use the `wikipedia_data` tool to fetch a Wikipedia summary.
    Provide a detailed response to the user's question, referencing both:
    1. The content retrieved from documents (book chapters, articles, etc.).
    2. A relevant Wikipedia article, including a link for further reading.

    If both tools provide useful information, combine them to give the user a complete answer and reference both the document content and the Wikipedia article.
    """
    
    messages_with_prompt = [{"role": "system", "content": prompt}] + state["messages"]

    llm_with_tools = llm.bind_tools([document_tool,wikipedia_data])
    
    # Invoke the LLM with the current message context
    response = llm_with_tools.invoke(messages_with_prompt)
    
    # MessagesState appends responses to state instead of overwriting
    return {"messages": [response]}


tools = ToolNode([document_tool,wikipedia_data])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer based on retrieved content from both tools."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Separate content from Wikipedia and document tools

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
   
    # System message content
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run LLM
    response = llm.invoke(prompt)

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


def generate_response(input_text):
    """Stream responses from the graph execution."""
    st.info("Generating response...")  # Feedback for user
    response_container = st.empty()  # Placeholder for live updates
    
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_text}]},
        stream_mode="values",
    ):
        # Update the placeholder with the latest message
        response_container.info(step["messages"][-1].pretty_print())


# Streamlit App
st.title("Interactive Interview Q&A")

with st.form("input_form"):
    # Text area for user input
    input_message = st.text_area(
        "Enter your question or topic:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        generate_response(input_message)




