# Financial Report Q&A Chat Bot

This repository contains the source code and documentation for the Financial Report Q&A Chat Bot. The chat bot is designed to answer questions related to financial reporting.

## Installation

To install and run the chat bot, follow these steps:

1. Clone the repository.
2. Add .env files to the src/api and the src/chat directores.  
    1. The env file in the src/api folder should have the following entries:
        AZURE_OPENAI_API_VERSION=  
        AZURE_OPENAI_ENDPOINT=  
        AZURE_OPENAI_API_KEY=  
        AZURE_OPENAI_DEPLOYMENT_NAME=  
        AZURE_AI_SEARCH_ENDPOINT=  
        AZURE_AI_SEARCH_KEY=  
        AZURE_AI_SEARCH_INDEX_NAME=  
        AZURE_AI_EMBEDDING_DEPLOYMENT_NAME=  
        AZURE_OPENAI_MODEL_VERSION=  
        LANGCHAIN_TRACING_V2=  
        LANGCHAIN_ENDPOINT=  
        LANGCHAIN_API_KEY=  
        LANGCHAIN_PROJECT=  
        DEBUG=  
        ENABLE_LANGCHAIN_PLAYGROUND=  
    2.  The env file in the src/chat folder should have teh following entries:  
        QNA_API_ENDPOINT=  
3. Configure the chat bot with the necessary credentials.
4. To run locally, use docker compose file in the root directory to spin up the necessary containers.  

## Usage

Once the chat bot is up and running, users can interact with it by asking questions related to financial reporting. The chat bot will provide answers based on its knowledge base.

## Contributing

Contributions are welcome! If you would like to contribute to the development of the Financial Report Q&A Chat Bot, please follow the guidelines outlined in the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

## License

This project is licensed under the [MIT License](./LICENSE).
