import hashlib
import re
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pydantic
from bs4 import BeautifulSoup


class Chunk(pydantic.BaseModel):
    id: str
    language: str
    pinecone_index_name: str
    pinecone_namespace: str
    chunk_text: str
    chunk_index: int
    source_title: str
    source_url: str
    vector: Optional[List[float]] = None


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from the provided text using BeautifulSoup.

    Args:
        text (str): The text containing HTML content.

    Returns:
        str: The sanitized text without HTML tags.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def generate_source_id(source_title: str, source_url: str) -> str:
    """Generate a unique ID for a source document."""
    return hashlib.md5(f"{source_title}{source_url}".encode()).hexdigest()


def normalize_whitespace(text: str) -> str:
    """
    Clean up excessive whitespace (triple or more spaces/tabs/newlines) from the text.
    Preserve meaningful paragraph breaks and reduce unnecessary gaps while keeping tabs intact.
    """
    # Step 1: Replace 3 or more consecutive newlines with two newlines (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 2: Reduce 3 or more consecutive spaces to a single space
    text = re.sub(r" {3,}", " ", text)

    # Step 3: Reduce 3 or more consecutive tabs to a single tab
    text = re.sub(r"\t{3,}", "\t", text)

    # Step 4: Normalize multiple spaces or tabs around newlines (remove spaces/tabs before/after line breaks)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    return text


def _combine_short_chunks(chunks: List[str], min_words_per_chunk: int) -> List[str]:
    combined_chunks = []
    temp_chunk = ""

    for chunk in chunks:
        word_count = len(chunk.split())

        if word_count < min_words_per_chunk:
            if combined_chunks:
                # Merge with the last chunk in combined_chunks
                combined_chunks[-1] += " " + chunk
            else:
                # If there is no previous chunk, store in temp_chunk
                temp_chunk += " " + chunk
        else:
            if temp_chunk:
                # Merge the temp_chunk with the current chunk
                combined_chunks.append(temp_chunk.strip() + " " + chunk)
                temp_chunk = ""
            else:
                combined_chunks.append(chunk)

    # If there's any remaining temp_chunk, append it to the last chunk
    if temp_chunk:
        if combined_chunks:
            combined_chunks[-1] += " " + temp_chunk.strip()
        else:
            combined_chunks.append(temp_chunk.strip())

    return combined_chunks


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    discard_chunk_n_words_cutoff: Optional[int] = None,
    clean_whitespace: bool = True,
    clean_html: bool = True,
    min_words_per_chunk: Optional[int] = None,
) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.

    discard_chunk_n_words_cutoff: If set, discard chunks with less than this number of words.

    min_words_per_chunk: If set, combine chunks with less than this number of words.
    """

    if clean_html:
        text = remove_html_tags(text)

    if clean_whitespace:
        text = normalize_whitespace(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_text(text)
    if discard_chunk_n_words_cutoff is not None:
        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.split()) >= discard_chunk_n_words_cutoff
        ]

    if min_words_per_chunk:
        chunks = _combine_short_chunks(chunks, min_words_per_chunk)

    return chunks


def create_chunks(
    text: str,
    language: str,
    pinecone_index_name: str,
    pinecone_namespace: str,
    source_title: str,
    source_url: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """Create chunks from text with associated metadata."""
    text_chunks = split_text(text, chunk_size, chunk_overlap)
    source_id = generate_source_id(source_title, source_url)

    chunks = []
    for i, chunk in enumerate(text_chunks):
        chunks.append(
            Chunk(
                id=f"{source_id}#{i}",
                language=language,
                pinecone_index_name=pinecone_index_name,
                pinecone_namespace=pinecone_namespace,
                chunk_text=chunk,
                chunk_index=i,
                source_title=source_title,
                source_url=source_url,
            )
        )
    return chunks
