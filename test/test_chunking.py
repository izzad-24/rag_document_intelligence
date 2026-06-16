from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE = "data/pdf/sample.pdf"

# Step 1: Load the document using the appropriate loader based on the file extension
loader = PyPDFLoader(FILE)
documents = loader.load()
print(f"Loaded Documents: {len(documents)}")
print("-" * 50)

# Step 2: Split the loaded documents into smaller chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max characters per chunk
    chunk_overlap=100,     # characters shared between adjacent chunks
    length_function=len,   # how to measure size (character count)
    separators=["\n\n", "\n", ". ", " ", ""]
    # ^ tries to split here in order — paragraphs first, then lines,
    #   then sentences, then words, then characters as last resort
)

# Step 3: Create chunks from the loaded documents
chunks = splitter.split_documents(documents)
print(f"Total Chunks Created: {len(chunks)}")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print("-" * 50)

# Step 4: Inspect the created chunks
for i in [0, 1, 2]:  # print the first 3 chunks for inspection
    print(f"Chunk {i+1}:")
    print(f"Length: {len(chunks[i].page_content)} characters")
    print(f"Content: {chunks[i].page_content[:200]}...")  # print the first 200 characters of the chunk
    print(f"Metadata: {chunks[i].metadata}")
    print("-" * 50)

# Step 5: Check for overlap between chunks to ensure the chunking is working as intended
print("\n" + "=" * 50)
print("OVERLAP CHECK — end of Chunk 0 vs start of Chunk 1:")
print(f"\n  Last 120 chars of Chunk 0:\n  '{chunks[0].page_content[-120:]}'")
print(f"\n  First 120 chars of Chunk 1:\n  '{chunks[1].page_content[:120]}'")
print("\n" + "=" * 50)
print("\nChunking test completed successfully!")

