import streamlit as st
from rag_pipeline import graph

# Capture user input
input_message = st.chat_input("Ask a question:")

# Ensure the input is not empty
if input_message:
    # Display user input as a message
    st.chat_message("user").write(input_message)

    # Process the input with the RAG pipeline (streaming mode)
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
    ):
        assistant_message = step["messages"][-1]["content"]
        
        # Display the assistant's message in a chat bubble
        with st.chat_message("assistant"):
            st.write(assistant_message)

