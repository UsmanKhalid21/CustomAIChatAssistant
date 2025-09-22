import streamlit as st
from Agent import agent_executor, memory
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chat Assistant", page_icon="🤖", layout="centered")

st.title("🤖 AI Chat Assistant")
st.markdown("Chat with your custom agent powered by **LangChain + Groq**.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message here..."):
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Add to memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    try:
        # Call your agent
        response = agent_executor.invoke({"input": user_input})
        output = response.get("output", "")

        # Show assistant response
        st.chat_message("assistant").markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})

        # Save to memory
        memory.chat_memory.add_message(AIMessage(content=output))

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.chat_message("assistant").markdown(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
