import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chunk_size = 1000
chunk_overlap = 200
model_name = "BAAI/bge-m3"
top_k = 5
URI = "./data/milvus_example.db"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preparing_data_and_vectorize(document_dir):
    if not os.path.exists(document_dir):
        print(f"{document_dir} does not exist")
        return

    document = []
    pdf_file = PyPDFLoader(file_path=document_dir)
    document.extend(pdf_file.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(document)
    print(f"splits: {len(splits)}, documents: {len(document)}")

    model_kwargs = {"device": device}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="milvus",
        connection_args={"uri": URI,},
        drop_old = False,
    )

    return vectorstore

def retrieve_document(question, k, vectorstore):
    docs = vectorstore.similarity_search(question, k)
    chunks = []
    for doc in docs:
        src = doc.metadata.get("source", "<unknown>")
        text = doc.page_content.strip().replace("\n", "")
        chunks.append(f"[source: {src}]\n{text}")
    return chunks

if __name__ == "__main__":
    document_dir = "data/The_Illustrated_Transformer.pdf"
    vectorstore= preparing_data_and_vectorize(document_dir)

    template = """You are a meticulous Q&A assistant. 
    You can only use the following pieces of retrieved context to answer the question. 
    Don't fabricate information. If you don't know the answer, just say that you don't know. 
    Question: {question} 
    Context: {context} 
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    print(prompt)

    llm = ChatOllama(
        model="llama3:8b",
        temperature=0,
    )
    chain = prompt | llm | StrOutputParser()
    while True:
        input_text = input("Enter question: ")
        if input_text == "":
            break

        context = retrieve_document(input_text, top_k, vectorstore)
        if not context:
            print("The retrieved result is empty")
            continue
        context_str = "\n\n".join(context)
        print("chunks: ", context_str)

        answer = chain.invoke({"question": input_text, "context": context_str})
        print(answer)








