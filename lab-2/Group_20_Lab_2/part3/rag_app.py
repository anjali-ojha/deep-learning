import streamlit as st
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import PyPDF2
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from operator import itemgetter

model = Ollama(model='llama3')
embeddings = OllamaEmbeddings(model='llama3')

template = """
Answer the question based on the context below. If you can't answer the question, reply 'I don't know'.

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

st.title('PDF-based Question Answering App')

def process_file():
    uploaded_file = st.session_state['file_uploader']
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        documents = [Document(page_content=page.extract_text()) for page in pdf_reader.pages if page.extract_text()]
        if documents:
            st.info('Creating the embeddings vector store.')
            vectorstore = FAISS.from_documents(documents, embedding=embeddings)
            st.session_state['vectorstore'] = vectorstore
            st.session_state['embeddings_ready'] = True
            st.success('Embeddings generated and stored. Now you can ask questions.')
        else:
            st.error('No text could be extracted from the uploaded PDF.')
            st.session_state['embeddings_ready'] = False

uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], on_change=process_file, key="file_uploader")
user_question = st.text_input("Enter your question:")

if 'embeddings_ready' in st.session_state and st.session_state['embeddings_ready']:
    st.success('Embeddings are ready. You can continue asking questions.')
else:
    st.info('Please upload a PDF file to start.')


if st.button('Answer Question'):
    if user_question and 'vectorstore' in st.session_state and st.session_state['embeddings_ready']:
        # Chain components together using the stored retriever
        retriever = st.session_state['vectorstore'].as_retriever()

        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | model
            | parser
        )

        # Invoke the chain to get an answer
        st.info('Generating answer')
        answer = chain.invoke({'question': user_question})
        st.write("Answer:", answer)
    else:
        st.warning("Please upload a PDF file and enter a question or ensure embeddings have been generated.")