import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from load_docs import load_docs
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message #pip istall streamli_chat

load_dotenv(find_dotenv())

openai.api_key =os.getenv("OPENAI_API_KEY")

#===========Using OPEAI CHAT API====================
llm_model ="gpt-3.5-turbo"

llm =ChatOpenAI(temperature=0.0, model =llm_model)

#load the files
# pdf_loader = PyPDFLoader('./docs/nyasha_cv.pdf')
documents = load_docs()

chat_history = []

#set up qa chain
# chain = load_qa_chain(llm, verbose=True)
# query = "Where Albet  attend his school?"
# response = chain.run(input_documents = documents, question =query)

# print(response)

#Now we split the data into chunks
text_splitter =CharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap =10
)
docs =text_splitter.split_documents(documents)

#create our vector db chromadb

vectordb = Chroma.from_documents(
    documents =docs,
    embedding= OpenAIEmbeddings(),
    persist_directory = './data'
)
vectordb.persist()

qa_chain = ConversationalRetrievalChain.from_llm(llm,vectordb.as_retriever(search_kwargs= {'k': 6}),
                                                 return_source_documents= True,
                                                 verbose=False
                                                 )
#==============Streamlit front end =================================================

st.title("Docs QA Bot using Langchain")
st.header("Ask anything about your documents.....💁‍♂️")

if 'generated' not in st.session_state:
    st.session_state['generated'] =[]

if 'past' not in st.session_state:
    st.session_state['past']= []

def get_query():
    input_text = st.chat_input("Ask a question about your documents.....")
    return input_text

# retrive user input
user_input =get_query()
if user_input:
    result =qa_chain({'question': user_input, 'chat_history': chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user =True, key= str(i)+'_user')