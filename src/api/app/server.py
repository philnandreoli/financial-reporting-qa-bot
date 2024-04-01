from os import getenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from typing import List
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever, Document
from langchain_core.runnables import chain
from operator import itemgetter
from langchain.globals import set_debug

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
    temperature=1,
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
    k=15
)

parse_question_for_query_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant that's an expert in taking a user's question and doing some steps to get it ready for use in a RAG pattern.
    Here are the steps that you need to follow:
    1.  Find the company names that are in the user's question and determine their stock symbol.  
    2.  Determine what financial report that they are looking for and the valid values are 10-K, 10-Q or not specified.   If the financial report is not specified, do not return anything. The financial report type will be put in a field called form_type
    3.  Determine if they are looking for the most recent financial report.  If they are looking for the most recent report, then return true else return false.
    4.  Determine what quarter they are looking for and then return the quarter number. 
    5.  Determine what year are they are looking for and then return the year. 
    6.  You should craft the filter that will be used in the search.  
            Example 1:
            search.in(stock_symbol, 'MSFT,NVDA', ',') and latest eq true
            Example 2:
            search.in(stock_symbol, 'MSFT,NVDA', ',') and latest eq false and year eq 2020 and cy_quarter eq 'Q2' and form_type eq '10-Q'
            Example 3:
            stock_symbol eq 'MSFT' and latest eq true 
            Example 4:
            stock_symbol eq 'MSFT' and latest eq false and cy_quarter eq 'Q2' and form_type eq '10-Q'
    7.  You should only return a JSON object and nothing else that has a property called filter and another property called question.  The filter property will have the output from step 6 and the question property will have the question that the user asked.  The json object should look like the followng:

    Question: {user_question}
    """)

    
answer_question_with_context = ChatPromptTemplate.from_template(
    """
    You are an AI assistant that will answer the user's question fron the context that is provided to you.  Please include the source and the page number as a citation for the answers given.   
    If you do not know the answer to the user's question, then return a message that says that you do not know the answer.

    Context: {context}

    Question: {user_question}
    """)

# Define the output parsers needed for the code
filter_output_parser: JsonOutputParser = JsonOutputParser()
answer_output_parser: StrOutputParser = StrOutputParser()

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# Create a custom chain that does the following
# 1. Takes the user question and parses it to get the filter
# 2. Queries the vector store to get the relavant content from the question
# 3. Summarizes the content to answer the users question
# 4. Returns the reponse to the user
@chain
def custom_chain(question: str) -> str:
    prompt1 = parse_question_for_query_prompt.invoke({"user_question": question})
    filter = model.invoke(prompt1)
    filter_output = JsonOutputParser().invoke(filter)
    context = format_docs(vector_store_retreiver.invoke(input=question, filters=filter_output["filter"] ))
    chain2 : StrOutputParser = (
        { "user_question": itemgetter("user_question"), "context": itemgetter("context")}
        | answer_question_with_context
        | model
        | answer_output_parser
    )
    return chain2.invoke({"user_question": question, "context": context })

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, custom_chain, path="/financial-report")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
