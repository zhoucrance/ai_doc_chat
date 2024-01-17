import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv(find_dotenv())

openai.api_key =os.getenv("OPENAI_API_KEY")

#===========Using OPEAI CHAT API====================
llm_model ="gpt-3.5-turbo"

llm =ChatOpenAI(temperature=0.0, model =llm_model)

#load the pdf file
pdf_loader = PyPDFLoader('./docs/nyasha_cv.pdf')
documents = pdf_loader.load()

#set up qa chain
# chain = load_qa_chain(llm, verbose=True)
# query = "Where Albet  attend his school?"
# response = chain.run(input_documents = documents, question =query)

# print(response)

#Now we split the data into chunks
text_splitter =CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap =200
)
docs =text_splitter.split_documents(documents)

#create our vector db chromadb

vectordb = Chroma.from_documents(
    documents =docs,
    embedding= OpenAIEmbeddings(),
    persist_directory = './data'
)
vectordb.persist()

# Use RetrivalQA chain to get info from the vector store

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vectordb.as_retriever(search_kwargs = {'k':3}),
    return_source_documents =True
)

result = qa_chain("Who is CV about")
#result = qa_chain({'query': 'Who is the Cv about?'}) # the otherway)
print(result['result'])