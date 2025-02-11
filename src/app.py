import streamlit as st
from rag_pipeline import graph

# Title and layout
st.title("Chat with RAG Pipeline")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capture user input
input_message = st.chat_input("Ask a question:")

if input_message and input_message.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input_message})
    with st.chat_message("user"):
        st.write(input_message)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        with st.spinner("Thinking..."):
            try:
                for step in graph.stream(
                    {"messages": st.session_state.messages},
                    stream_mode="values",
                ):
                    last_message = step["messages"][-1]
                    assistant_message = last_message.content if hasattr(last_message, "content") else str(last_message)
                    response_container.write(assistant_message)
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Debug info: {step}")  # Log the step causing the error
