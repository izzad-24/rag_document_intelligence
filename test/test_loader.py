from langchain_community.document_loaders import PyPDFLoader, TextLoader

# the file to ingest
FILE = "data/pdf/sample.pdf"

# load the file using the appropriate loader based on the file extension
print("Ingesting file:", FILE)
if FILE.endswith(".pdf"):
    loader = PyPDFLoader(FILE)
else:    
    loader = TextLoader(FILE)

# load the documents from the file 
documents = loader.load() 

# inspect the loaded documents
print(f"Loaded Documents: {len(documents)}")
print("-" * 50)

# print the content and metadata of each document
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")  # print the first 200 characters
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)