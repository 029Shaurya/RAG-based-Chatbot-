In this project I have created a RAG stands for "Retrieval-Augmented Generation (RAG) based Question-Answer Chatbot which can answer all your queries based on the documents provided in the backend related to particular documents like company's policies, FAQ related to some online store, online course related queries, etc
It utilises Llama2 model by meta that is run locally on the system that enhances privacy of the information present in the documents.
The pdf document provided is broken into chunks and  then these chunks are stored in the form of embeddings in a vector store like Chroma or FAISS which are designed for fast retrival of document chunks closely related to user query (based on cosine similarity).

The retrived chunks along with user query and past memory is given to LLM that generates the desired response.
I have also used LangChain Memory to enhance the response of LLM.

The user friendly interface is created using Streamlit.

Significance:
1. Cost efficient: No need to fine tune the model again and again for frequently changing data such as Company's policies, course content, results, etc.
2. Privacy: As the model is run locally, the data is remained on the system. Hence privacy is maintained.

NOTE: A samole pdf is used in this project. We can upload any pdf file.
It is adviced to run this on GPU, but can also be run on CPU as well.

Command to run the project: streamlit run app.py



