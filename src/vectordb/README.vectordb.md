# Package sstructure
 - vectordb/ 
    - embedding/ 
        - logic to embed data, for now embedding locally with huggingface model. 
    - logic to interface with the vector database (pinecone)
 - inference/
    - interface to together AI 
 - rag/
    - Not sure I will keep this folder, 
    - but it should tie together the vectordb and the inference module.


## Next goals
 - insert test data into the vector database
    - visindavefurinn
 - Need some kind of local table to store the text chunks and their ids along with metadata.
 



### RAG Module Specification

This module implements a Retrieval-Augmented Generation (RAG) system supporting multiple languages. Each language is organized into a separate index, with sources partitioned using namespaces within the corresponding index. The design supports efficient retrieval of embeddings based on both language and source.

#### Structure:
- **Indexes**: One index per language (e.g., `en-index`, `is-index`, etc.).
    - Using an index per language allows us to use a language-specific embedding models if needed. 
    - And by going with the serverless approach, it does not introduce much more complexity to the system.
- **Namespaces**: Each index contains multiple namespaces representing different sources (e.g., `wikipedia`, `visindavefurinn`, etc.).

#### Tech Stack:
- **Vector Database**: Pinecone (Serverless)
- **Model Architecture**: Language-specific embedding models (TBD)

This structure ensures flexibility and scalability as the system grows to support more languages and sources.
