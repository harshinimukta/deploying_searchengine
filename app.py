#https://onedrive.live.com/view.aspx?resid=A8BDCD2EBBB8891B%211219&id=documents&wd=target%28New%20Section%201.one%7C9E02F9E6-35B5-4A5B-905F-D2B38B3AD4A6%2FSearch%20engine%20based%20on%20tools%20and%20agents%7CF313F94B-9C7A-47DA-B5E7-55E418D564DD%2F%29
"""
khdsdgfjchkchkgk,cgajfd
kjdhegdghgck,hckgegf
djgdadjgdjgjdl

"""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
#Imports a callback handler that allows the agent's thoughts and actions to be displayed in real-time in the Streamlit app
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
##DuckDuckGo has an API, but because it is simple and doesn't require much configuration, 
# the DuckDuckGoSearchRun tool can directly handle interactions with it. 
# The other services (Arxiv and Wikipedia) have more complex APIs, requiring additional layers (wrappers) to manage settings and customize requests.
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]
#Iterates through the messages stored in the session state and displays them in the chat interface, showing both user and assistant messages.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

