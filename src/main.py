import os
import argparse
import fitz  # PyMuPDF, imported as fitz
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --- Constants ---
# It's a good practice to use a pre-trained model suitable for sentence embeddings.
# 'all-MiniLM-L6-v2' is a popular choice for its balance of speed and performance.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Core Functions ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        str: The concatenated text content from all pages of the PDF.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist at the given path.
        Exception: For other errors during PDF processing.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: The file '{pdf_path}' was not found.")
        
    print(f"Extracting text from '{pdf_path}'...")
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc, start=1):
            full_text += page.get_text()
        doc.close()
        print(f"Text extraction from {len(doc)} pages successful.")
        return full_text
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        raise

def generate_embedding(text: str, model: SentenceTransformer) -> list[float]:
    """
    Generates a vector embedding for the given text using a sentence-transformer model.

    Args:
        text (str): The input text to be vectorized.
        model (SentenceTransformer): The initialized sentence-transformer model.

    Returns:
        list[float]: The generated vector embedding.
    """
    print(f"Generating embedding...")
    # The model.encode() method returns a NumPy array.
    # Pinecone's client expects a list of floats, so we convert it.
    embedding = model.encode(text, convert_to_numpy=True)
    print("Embedding generation successful.")
    return embedding.tolist()

def upsert_vector_to_pinecone(index: 'pinecone.Index', vector_id: str, vector: list[float], metadata: dict):
    """
    Upserts (updates or inserts) a vector into the specified Pinecone index.

    Args:
        index (pinecone.Index): The initialized Pinecone index object.
        vector_id (str): A unique identifier for the vector.
        vector (list[float]): The vector embedding.
        metadata (dict): A dictionary of metadata to associate with the vector.
    """
    print(f"Upserting vector with ID '{vector_id}' to Pinecone index...")
    try:
        index.upsert(
            vectors=[
                {
                    'id': vector_id,
                    'values': vector,
                    'metadata': metadata
                }
            ]
        )
        print("Successfully upserted vector to Pinecone.")
    except Exception as e:
        print(f"An error occurred during Pinecone upsert: {e}")
        raise

def main():
    """
    Main function to orchestrate the PDF to vector conversion and database upload.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Converts a PDF document to a vector and upserts it to a Pinecone index."
    )
    parser.add_argument(
        "--pdf-file",
        type=str,
        required=True,
        help="Path to the PDF file to be vectorized."
    )
    parser.add_argument(
        "--vector-id",
        type=str,
        required=False,
        help="Optional unique ID for the vector. If not provided, the PDF filename will be used."
    )
    args = parser.parse_args()

    pdf_path = args.pdf_file
    # Use the provided vector ID or default to the base name of the PDF file.
    vector_id = args.vector_id or os.path.basename(pdf_path)

    # --- Environment Variable Loading for Pinecone ---
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    if not pinecone_index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

    # --- Main Workflow ---
    try:
        # 1. Initialize clients and models
        print("Initializing SentenceTransformer model...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if the target index exists.
        if pinecone_index_name not in pc.list_indexes().names():
            raise NameError(
                f"Pinecone index '{pinecone_index_name}' not found. "
                f"Please create it first via the Pinecone console or API."
            )

        index = pc.Index(pinecone_index_name)
        print(f"Successfully connected to Pinecone index '{pinecone_index_name}'.")

        # 2. Extract text from the PDF
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text.strip():
            print("Warning: No text could be extracted from the PDF. Exiting.")
            return

        # 3. Generate the vector embedding
        vector_embedding = generate_embedding(document_text, model)

        # 4. Prepare metadata for context
        metadata = {
            "source_file": os.path.basename(pdf_path),
            "original_path": pdf_path,
            # Store a snippet for quick reference in the vector database
            "text_snippet": document_text[:1000] 
        }

        # 5. Upsert the vector to Pinecone
        upsert_vector_to_pinecone(index, vector_id, vector_embedding, metadata)

        print("\n--- Process Complete ---")
        print(f"PDF '{pdf_path}' has been successfully vectorized and stored.")
        print(f"Vector ID: {vector_id}")
        print("------------------------")

    except (FileNotFoundError, ValueError, NameError) as e:
        print(f"\nConfiguration or File Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()