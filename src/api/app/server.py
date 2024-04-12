from . import prompts 
from fastapi.responses import RedirectResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from os import getenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever, Document
from operator import itemgetter
from typing import List
from langchain_core.runnables import chain
from langserve import add_routes

app = FastAPI(
    title="Financial Report Q&A API",
    version="0.1.0",
    description="This API is used to answer questions about 10-Q & 10-Ks that publiclly traded companies file with the SEC.  The API uses the RAG model to answer the questions"
)

# Add CORS Middleware in order for applications to call this 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Azure Search Configurations
ai_search_endpoint: str = getenv("AZURE_AI_SEARCH_ENDPOINT")
ai_search_key: str = getenv("AZURE_AI_SEARCH_KEY")
ai_search_index_name: str = getenv("AZURE_AI_SEARCH_INDEX_NAME")

# Azure Open AI Configurations
open_ai_api_version: str = getenv("AZURE_OPENAI_API_VERSION")
open_ai_deployment_name: str = getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
open_ai_model_version: str = getenv("AZURE_OPENAI_MODEL_VERSION")
open_ai_embeddings_deployment_name: str = getenv("AZURE_AI_EMBEDDING_DEPLOYMENT_NAME")
open_ai_api_key: str = getenv("AZURE_OPENAI_API_KEY")

# Azure Open AI Model Configuration
model: AzureChatOpenAI = AzureChatOpenAI(
    openai_api_version=open_ai_api_version,
    azure_deployment=open_ai_deployment_name,
    model=open_ai_model_version,
    temperature=0.1,
    api_key=open_ai_api_key
)

# Azure Open AI Embeddings Configuration
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=open_ai_embeddings_deployment_name
)

# Azure AI Search Configurations
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=ai_search_endpoint,
    azure_search_key=ai_search_key,
    index_name=ai_search_index_name,
    embedding_function=embeddings.embed_query
)

# Azure AI Search Vector Store Retriever
vector_store_retreiver: AzureSearchVectorStoreRetriever = AzureSearchVectorStoreRetriever(
    vectorstore=vector_store,
    search_type="hybrid",
    k=3
)

# format all the documents in the list as a string that will be used as the
# context to answer the question
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# From the original question, rephrase the question used to answer the question
# regardless of the number of companies you want to know for.   
rephrase_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.rephrase_prompt),
        #MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# For the rephrased question for each company that was identified, how should you answer
# this is the prompt to help with answering the question
qna_prompt = ChatPromptTemplate.from_template(prompts.qna_prompt)

# This is the chain that is used to take the original question and break it apart into multiple questions
# based on the number of companies in the original question. 
rephraser_chain = (
    {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
    | rephrase_question_prompt
    | model
    | JsonOutputParser()
)

#  This is the chain that is used to take each individual question and answer it with the context.  This chain is also
#  used to summarize the answers and answer the original question
qna_chain = (
    { "question": itemgetter("question"), "context": itemgetter("context")}
    | qna_prompt
    | model
    | StrOutputParser()
)

def process_questions(question) -> str:
    documents_as_string = format_docs(vector_store_retreiver.invoke(input=question["question"], filters=question["filter"] ))
    answer = qna_chain.invoke({"question": question["question"], "context": documents_as_string })
    return answer 

@chain
def answer_financial_question_chain(question: str): 
    all_answers = []
    rephrased_output = rephraser_chain.invoke({"input": question, "chat_history": ""})
    runnable = RunnableLambda(process_questions)
    all_answers = runnable.batch(rephrased_output["output"])
    final_answer = qna_chain.invoke({"question": rephrased_output["rephrasedQuestion"], "context": "\n\n".join(all_answers)})
    return final_answer

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, answer_financial_question_chain, path="/financialreports")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
