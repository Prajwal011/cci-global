
import streamlit as st
import pickle
import os
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import openai
import langchain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_core.output_parsers import StrOutputParser
import os

import streamlit as st

os.environ['GROQ_API_KEY'] = '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'
KB = 'cci_kb'
# Initialize Groq LLM
# # # llm = ChatGroq(
# # #     model_name="llama-3.1-8b-instant",
# # #     temperature=0.7,
# # #     max_tokens = 300
# # # )

st.title("CCI global Bot")
chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {content}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {content}")
# User input
description = st.chat_input("ask me anything")


llm = ChatGroq(
    model_name="qwen-qwq-32b",  
    # model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens = 1000
)


# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ('system',"""
    [System Instruction]
    You are a friendly, professional cci global assistant. Your purpose is to answer user questions using facts from our knowledge base and provide concise, formal responses.

    [Chat History]
    {chat_history}

    [Output Instructions]
    1. If question is not related to cci global,continue casual chat.
    2. If cci global question, answer from context: {context}. If no context, say "Insufficient info."
    3. Always prioritize chat history.
    4. Keep your answers well within 300 words always
    5. If user asks to contact cci global ask them to mail at cci@global.com and don't share any other cci global contact also ask for their email,phone and name after checking in context if previously provided or not
    """),
    # ("system","""
    # Act as a course guidance expert. Provide information on courses, offer career choice guidance by asking up to 3 questions to determine suitable course among Data Analyst, Machine Learning Engineer, or Software Engineer, and consider chat history to provide contextual answers
    # """),
    # ("system", """Helpful bot for
    #  1. Carrer guidance and helps identify the field or courses in ai also for questions related to courses refer context:{context}
    #  2. If asked about carrer guidance bot will ask series of 3 questions to decide whether they are fit for data analyst, machine learning engineer or Software engineer
    #  Also to frame questions refer chat_history:{chat_history}
    #  Keep your answers short and concise and provide sufficient knowledge of courses
    # """),
    ("user", "{input}")
])

parser = StrOutputParser()

# Create the chain that guarantees JSON output
chain = prompt | llm | parser

def parse_product(description,context,chat_history) -> dict:
    result = chain.invoke({"input": description,'context':context,'chat_history':chat_history})
    return result


# Example usage
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
loaded_vector_store = FAISS.load_local(KB, embeddings, allow_dangerous_deserialization=True)
# with open(KB, "wb") as f:
#         pickle.dump(loaded_vector_store, f)
# if os.path.exists(KB):
#   with open(KB, "rb") as f:
#       loaded_vector_store = pickle.load(f)
retriever = loaded_vector_store.as_retriever(search_kwargs={"k": 5})
if description:
  # try :
    st.session_state.messages.append({"role": "user", "content": description})
    chat_history.append(['user',description])
    # retriever.search_kwargs = config
    relevant_docs= retriever.get_relevant_documents(query = description)

    # print("Total Relevant Documents Found: {}".format(len(relevant_docs)))
    # for doc in relevant_docs:
    #     print("====================================")
    #     print(doc.page_content)
    context =  [doc for doc in relevant_docs]
    context = "\n\n".join(["{}\nSource URL: {}".format(doc.page_content, doc.metadata.get("url", "URL not available")) for doc in relevant_docs])
    hist_str = "\n".join(['user: '+i['content'] if id%2==0 else 'assistant: '+i['content'] for id,i in enumerate(st.session_state.messages[-9:]) ])
    print(hist_str)
    # hist_str = '\n'.join([ i['role']+':'+i[1] for i in chat_history ])
    # context = get_context_scraped(query=description, retriever=retriever, config=kb_params)

    ans = parse_product(description,context,hist_str)
    # ans = 'poiuytrewqgyhujkjhgfd'
    ans = ans[ans.find("</think>")+len("</think>"):]

    st.session_state.messages.append({"role": "assistant", "content": ans})

    # st.write(f"**Model:** {ans}")
    st.chat_message("user").markdown(description)

    with st.chat_message("assistant"):

      st.markdown(ans)

    print(ans)

    chat_history.append(["system",ans])
    # st.write(st.session_state.messages)

  # except:
  #   pass
  #     llm = ChatGroq(
  #         model_name="llama3-8b-8192",
  #         temperature=0.7,
  #         max_tokens = 300
  #     )
