from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import Ollama

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Step 1 - setup OCI Generative AI llm
# use default authN method API-key
llm = Ollama(model='llama2')

# Step 2 - Create a Prompt
base_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot who explains in steps."
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

load_dotenv()
embeddings = OllamaEmbeddings()

# Define your documents
loader = PyPDFLoader('results.pdf')
docs = loader.load_and_split()

# Initialize vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Use the vector store as needed
retriever = vectorstore.as_retriever()

# Step 3 - Create a memory to remember our chat with the llm
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

# Step 4 - Function to include retrieved documents in the prompt
def get_retrieved_documents(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return docs_text

# Step 5 - Modify prompt to include retrieved documents
def create_conversation_chain(question):
    retrieved_docs = get_retrieved_documents(question)
    full_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot who explains in steps."
            ),
            SystemMessagePromptTemplate.from_template(
                f"Here are some relevant documents to help you answer:\n{retrieved_docs}"
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    conversation = LLMChain(llm=llm, prompt=full_prompt, verbose=True, memory=memory)
    return conversation

# Example usage
question = "What are the main findings in the document?"
conversation = create_conversation_chain(question)
response = conversation.run(question)
print(response)
