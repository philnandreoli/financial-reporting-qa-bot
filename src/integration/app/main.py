from os import getenv, path, mkdir
from datetime import datetime
from fastapi import FastAPI, HTTPException 
import pytz
import html
import requests
import re
import base64
from typing import List, Dict
from requests.exceptions import RequestException
import pandas as pd
import pdfkit
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat, DocumentTable
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmMetric,
    SearchField
)

from prepdocslib.pdfparser import DocumentAnalysisParser 
from prepdocslib.page import Page
from glob import glob


HEADERS = { "User-Agent": f"{getenv('COMPANY_NAME')} {getenv('COMPANY_EMAIL')}"}

"""
Here are the steps to download a financial report from the SEC website and the store it in Azure AI Search:

1.  Download the SEC Reports
2.  Check to make sure the index has been created and has the necessary fields
3.  Make a call to Azure Document Intelligence for the document
4.  Process the document
"""



# Get the list of stock symbols from the SEC Website and return the central index key for the specific stock symbol
def get_central_index_key(stock_symbol: str) -> dict[any, any]:
    cik_lookup_json = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS).json()
    cik_lookup = dict([(val['ticker'], val) for key, val in cik_lookup_json.items()])
    return cik_lookup[stock_symbol]

def download_financial_report(
        cik_number: str, 
        stock_symbol: str, 
        storage_account_name: str,
        container_name: str,
        blob_service_client: BlobServiceClient,
        start_date: str = None, 
        end_date: str = None) -> List:
    results = []
    
    try:
        # Get the list of filings for the specific stock symbol
        filings = requests.get(f"https://data.sec.gov/submissions/CIK{cik_number:0>10}.json", headers=HEADERS).json()
        # Convert the recent filings to a pandas dataframe
        recent_filings = pd.DataFrame(filings['filings']['recent'])
        
        recent_filings = recent_filings[(recent_filings['form'] == "10-Q") | (recent_filings['form'] == "10-K")]


        # If a report date is provided, filter for dates greater than or equal to the report date
        if start_date and end_date is None:
            recent_filings = recent_filings[(recent_filings['reportDate'] == start_date)]
        elif start_date and end_date:
            recent_filings = recent_filings[(recent_filings['reportDate'] >= start_date) & (recent_filings['reportDate'] <= end_date)]
        
        # Create the URL for each report
        recent_filings['url'] = recent_filings.apply(lambda x: f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{x['accessionNumber'].replace('-', '')}/{x['primaryDocument']}", axis=1)
        # Create the local path for each report
        recent_filings['localPath'] = recent_filings.apply(lambda x: f"{getenv('DOWNLOAD_ROOT_PATH')}/{stock_symbol}/{stock_symbol}-{x['form']}-{x['reportDate']}.pdf", axis=1)
        recent_filings['blobFullPath'] = recent_filings.apply(lambda x: f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{stock_symbol}/{stock_symbol}-{x['form']}-{x['reportDate']}.pdf", axis=1)
        recent_filings['blobContainerPath'] = recent_filings.apply(lambda x: f"{stock_symbol}/{stock_symbol}-{x['form']}-{x['reportDate']}.pdf", axis=1)
        recent_filings['id'] = recent_filings.apply(lambda x: filename_to_id(f"{stock_symbol}-{x['form']}-{x['reportDate']}.pdf"), axis=1)
        recent_filings['fileName'] = recent_filings.apply(lambda x: f"{stock_symbol}-{x['form']}-{x['reportDate']}.pdf", axis=1)

        # sort the latest recent filings data frame by report date
        recent_filings = recent_filings.sort_values(by="reportDate", ascending=False)
        # create a new field in the dataframe called latest and set the first row to True all the other rows should be false
        recent_filings['latest'] = False
        recent_filings.loc[recent_filings.index[0], 'latest'] = True

        for index, row in recent_filings.iterrows():
            # Download the report
            result = {}
            pdfkit.from_url(
                url=row['url'], 
                output_path=row['localPath']
            )
            upload_to_blob_storage(
                blob_service_client=blob_service_client,
                container_name=container_name,
                file_path=row['localPath'],
                blob_path=row['blobContainerPath']
            )
            result['url'] = row['url']
            result['localPath'] = row['localPath']
            result['filingDate'] = row['filingDate']
            result['form'] = row['form']
            result['reportDate']= row['reportDate']
            result['blobFullPath'] = row['blobFullPath']
            result['blobContainerPath'] = row['blobContainerPath']
            result['id'] = row["id"]
            result['fileName'] = row['fileName']
            result['latest'] = row['latest']
            results.append(result)

    except RequestException as e:
        print(f"An error occurred while trying to download the financial report: {e}")
        return results
    
    return results

def create_ai_search_index(ai_search_endpoint: str, credential: AzureKeyCredential, index_name: str, vector_search_dimensions: int = 3072) -> bool:
    
    status = True

    search_index_client = SearchIndexClient(endpoint=ai_search_endpoint, credential=credential) 

    fields = [
        SearchableField(
            name="id",
            type="Edm.String",
            key=True,
            retrievable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            searchable=False 
        ),
        SearchableField(
            name="content",
            type="Edm.String",
            retrievable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            searchable=True
        ),
        SearchableField(
            name="page",
            type="Edm.String",
            retrievable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            searchable=True
        ),
        SearchableField(
            name="url",
            type="Edm.String",
            retreivable=True,
            searchable=True,
            filterable=False,
            sortable=False,
            facetable=False
        ),
        SearchableField(
            name="filename",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=False,
            facetable=False,
            searchable=True 
        ),
        SearchableField(
            name="company_name",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=True,
            facetable=True,
            searchable=True
        ),
        SearchField(
            name="content_vector",
            type="Edm.Collection(Edm.Single)",
            searchable=True,
            vector_search_dimensions=vector_search_dimensions,
            vector_search_profile_name="hnsw_config_profile"
        ),
        SearchableField(
            name="stock_symbol",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=True,
            facetable=True,
            searchable=True
        ),
        SearchableField(
            name="form_type",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=True,
            facetable=True,
            searchable=True
        ),
        SearchField(
            name="year",
            type="Edm.Int32",
            filterable=True,
            sortable=True,
            facetable=True
        ),
        SearchableField(
            name="report_date",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=True,
            facetable=True,
            searchable=True
        ),
        SearchableField(
            name="filing_date",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=True,
            facetable=True,
            searchable=True
        ),
        SearchableField(
            name="cy_quarter",
            type="Edm.String",
            retrievable=True,
            filterable=True,
            sortable=True,
            facetable=True,
            searchable=True
        ),
        SearchField(
            name="latest",
            type="Edm.Boolean",
            filterable=True,
            sortable=True,
            facetable=True
        ),
        SearchField(
            name="report_date_utc",
            type="Edm.DateTimeOffset",
            filterable=True,
            sortable=True,
            facetable=True
        ),
        SearchField(
            name="filing_date_utc",
            type="Edm.DateTimeOffset",
            filterable=True,
            sortable=True,
            facetable=True
        )
    ]

    index = SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=None,
                        content_fields=[
                            SemanticField(field_name="content"),
                            SemanticField(field_name="stock_symbol"),
                            SemanticField(field_name="form_type"),
                            SemanticField(field_name="report_date"),
                            SemanticField(field_name="filing_date"),
                            SemanticField(field_name="cy_quarter"),
                            SemanticField(field_name="company_name")
                        ]
                    )
                )
            ]
        ),
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw_config",
                    parameters=HnswParameters(
                        m=4,  
                        ef_construction=400,  
                        ef_search=500,  
                        metric=VectorSearchAlgorithmMetric.COSINE, 
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw_config_profile",
                    algorithm_configuration_name="hnsw_config"
                )
            ]
        )
    )
    
    try:
        search_index_client.create_or_update_index(index, allow_index_downtime=True) 
    except Exception as e:
        print(f"An error occurred while trying to create the AI Search Index: {e}")
        status = False

    return status
        
def parse_documents(openai_embedding: AzureOpenAIEmbeddings, filePath: str, verbose: bool = False):
    document_intelligence_client = DocumentIntelligenceClient(endpoint=getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"), credential=AzureKeyCredential(getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")))
    with open(filePath, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    analyze_request=f,
                    content_type="application/octet-stream")

        document_intelligence_results = poller.result()
        offset = 0
        parsed_docs = []
        for page_num, page in enumerate(document_intelligence_results.pages):
            doc ={}
            tables_on_page = [
                table
                for table in (document_intelligence_results.tables or [])
                if table.bounding_regions and table.bounding_regions[0].page_number == page_num + 1
            ]

            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1] * page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    for i in range(span.length):
                        idx = span.offset - page_offset + i 
                        if idx >= 0 and idx < page_length:
                            table_chars[idx] = table_id
        
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += document_intelligence_results.content[page_offset + idx]
                elif table_id not in added_tables:
                    page_text += DocumentAnalysisParser.table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)
            


            doc["page_num"] = page_num
            doc["offset"] = offset
            doc["page_text"] = page_text
            doc["page_embedding"] = openai_embedding.embed_query(page_text)
            parsed_docs.append(doc)
            offset += len(page_text)
    return parsed_docs

def table_to_html(table: DocumentTable):
    table_html = "<table>"
    rows = [
        sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index)
        for i in range(table.row_count)
    ]

    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span is not None and cell.column_span > 1:
                cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span is not None and cell.row_span > 1:
                cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html

def upload_to_blob_storage(blob_service_client: BlobServiceClient, container_name: str, file_path: str, blob_path: str):
    container_client = blob_service_client.get_container_client(container=container_name)
    with open(file_path, "rb") as f:
        blob_client = container_client.upload_blob(name=blob_path, data=f, overwrite=True)

def filename_to_id(filename: str):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode("utf-8")).decode("ascii")
    return f"file-{filename_ascii}-{filename_hash}"

def create_directory(stock_symbol: str):
    if not path.exists(f"{getenv('DOWNLOAD_ROOT_PATH')}/{stock_symbol}"):
        mkdir(f"{getenv('DOWNLOAD_ROOT_PATH')}/{stock_symbol}")

start_date = "2023-01-01"
end_date = "2024-03-15"

timezone = pytz.timezone("America/Chicago")

stock_symbol = "NKE"

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=getenv("AZURE_OPENAI_API_KEY"),
    api_version=getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
)

blob_service_client = BlobServiceClient(
    account_url=f"https://{getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net/",
    credential=getenv("AZURE_STORAGE_ACCOUNT_KEY")
)

azure_search_client = SearchClient(
    endpoint=getenv("AZURE_AI_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(getenv("AZURE_AI_SEARCH_KEY")),
    index_name=getenv("AZURE_AI_SEARCH_INDEX_NAME")
)

cik_number = get_central_index_key(stock_symbol)
create_directory(stock_symbol)
documents = download_financial_report(
    cik_number=cik_number["cik_str"], 
    stock_symbol=cik_number["ticker"],
    storage_account_name=getenv("AZURE_STORAGE_ACCOUNT_NAME"),
    container_name=getenv("AZURE_STORAGE_CONTAINER_NAME"),
    blob_service_client=blob_service_client,
    start_date=start_date,
    end_date=end_date
)

# create the ai search index that will be used to store the content and vectors
search_index_created = create_ai_search_index(
    ai_search_endpoint=getenv("AZURE_AI_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(getenv("AZURE_AI_SEARCH_KEY")),
    index_name=getenv("AZURE_AI_SEARCH_INDEX_NAME")
)

if search_index_created == True:
    for document in documents:
        pages = parse_documents(openai_embedding=embeddings, filePath=document['localPath'])
        AI_SEARCH_DOCUMENTS=[]
        for page in pages:
            ai_search_page = {
                "id": f"{document['id']}-page-{page['page_num']}",
                "content": page["page_text"],
                "page": f"{document['fileName']}-{page['page_num']}",
                "url": document["blobFullPath"],
                "filename": document["blobContainerPath"],
                "content_vector": page["page_embedding"],
                "stock_symbol": stock_symbol,
                "form_type": document["form"],
                "year": int(document["reportDate"].split("-")[0]),
                "report_date": document["reportDate"],
                "filing_date": document["filingDate"],
                "cy_quarter": f"Q{int(document['reportDate'].split('-')[1])/4 + 1}",
                "latest": document["latest"],
                "company_name": cik_number["title"],
                "report_date_utc": timezone.localize(datetime.strptime(document["reportDate"], "%Y-%m-%d")).astimezone(pytz.utc),
                "filing_date_utc": timezone.localize(datetime.strptime(document["filingDate"], "%Y-%m-%d")).astimezone(pytz.utc)
            }
            AI_SEARCH_DOCUMENTS.append(ai_search_page)
        
        azure_search_client.merge_or_upload_documents(AI_SEARCH_DOCUMENTS)