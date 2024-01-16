import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema.document import Document
from langchain import VectorDBQA, OpenAI
import pinecone
 #ncodncvodn
import requests
import re
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.post("/")
async def read_root(data: str):
    def fetch_github_files_recursive(github_link, file_regex=None):
        # Extract owner, repo, and branch from the GitHub link
        # Example link: https://github.com/owner/repo/tree/branch
        parts = github_link.rstrip('/').split('/')
        owner, repo, _, branch = parts[-4:]

        # GitHub API URL for fetching repository contents
        api_url = f'https://api.github.com/repos/{owner}/{repo}/contents'

        # Recursive function to fetch files from all folders
        def fetch_files_recursive(folder_path=''):
            response = requests.get(f'{api_url}/{folder_path}', params={'ref': branch})

            if response.status_code == 200:
                    files = response.json()
                    fetched_files = []

                    for file in files:
                        # If it's a directory, recursively fetch files from it
                        if file['type'] == 'dir':
                            fetched_files.extend(fetch_files_recursive(file['path']))
                        else:
                            # Check if the file matches the provided regex pattern
                            if file_regex and any(re.match(pattern, file['name']) for pattern in file_regex):
                                    #continue
        
                                    # Fetch file content
                                    content_response = requests.get(file['download_url'])
                                    if content_response.status_code == 200:
                                        fetched_files.append({
                                            'name': file['name'],
                                            'content': content_response.text
                                        })
            
                        return fetched_files

            else:
                print(f"Failed to fetch GitHub files. Status code: {response.status_code}")
                return []

        # Start recursive fetching from the root folder
        return fetch_files_recursive()
        
        
        
    def initialize_device():
        import torch
        print(torch.cuda.is_available())
        print(torch.version.cuda)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def configure_model():
        # config = BitsAndBytesConfig(
        #     load_in_8bit_fp32_cpu_offload=True,
        #     device_map={}
        # )
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", config=config)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        # local_model_path = "D:/code_docs/models--mistralai--Mistral-7B-Instruct-v0.1"
        # model = AutoModelForCausalLM.from_pretrained(local_model_path, load_in_4bit=True, device_map='auto')
        # tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        return model, tokenizer

    def create_text_generation_pipeline(model, tokenizer):
        text_generation_pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        return text_generation_pipeline

    def create_huggingface_pipeline(pipeline):
        mistral_llm = HuggingFacePipeline(pipeline=pipeline)
        return mistral_llm

    def setup_openai_embeddings():
        embeddings = OpenAIEmbeddings(deployment="EMBEDDER")
        return embeddings

    def initialize_pinecone():
        pinecone.init(api_key="01618612-2c53-4b56-9517-4c675001f3a6", environment="gcp-starter")

    def create_document(text):
        document = Document(page_content=text)
        return document

    def split_documents(documents):
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    def setup_pinecone_index(embeddings, texts, index_name):
        docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
        return docsearch

    def initialize_conversational_retrieval_chain(llm, retriever):
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
        return chain

    def perform_question_answering(chain, query, chat_history):
        result = chain({"question": query, "chat_history": chat_history})
        return result['answer']

    def main():
        # Example usage:
        link = data.get('data')
        github_link = link +'/tree/main'
        # github_link = 'https://github.com/indexedakki/Code-Documentation/tree/main'
        file_regex = [r'.*\.sql', r'.*\.py'] # Specify a regex pattern if needed
        files = fetch_github_files_recursive(github_link, file_regex)

        print(type(files))
        if files:
            for file in files:
                print(f"File Name: {file['name']}")
                print(f"File Content:\n{file['content']}\n{'='*40}")

        device = initialize_device()
        model, tokenizer = configure_model()
        #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        text_generation_pipeline = create_text_generation_pipeline(model, tokenizer)
        mistral_llm = create_huggingface_pipeline(text_generation_pipeline)

        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2022-12-01"
        os.environ["OPENAI_API_BASE"] = "https://trail-outcome.openai.azure.com/"
        os.environ["OPENAI_API_KEY"] = "b59f23e204b3426c9dbe1f6741b80acb"

        loader = TextLoader("/Workspace/Users/admin@m365x99442612.onmicrosoft.com/Code Documentation POC/Akash/documentation.txt")
        document1 = loader.load()

        string_variable = files[0]['content']
        document = create_document(string_variable)

        texts = split_documents([document])

        embeddings = setup_openai_embeddings()
        initialize_pinecone()
        docsearch = setup_pinecone_index(embeddings, texts, index_name="code-documentation-index")

        chain = initialize_conversational_retrieval_chain(llm=mistral_llm, retriever=docsearch.as_retriever())

        chat_history = []
        query = "what the above code does"
        result = perform_question_answering(chain, query, chat_history)

        # return {"message": "Script executed successfully"}
        print(result)
    main()
