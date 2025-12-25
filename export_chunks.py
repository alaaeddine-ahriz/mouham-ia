#!/usr/bin/env python
"""Export all chunks from Pinecone to a JSON file for manual inspection."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from legal_rag.pinecone_store import (
    get_pinecone_client,
    get_settings,
    NAMESPACE_LAW_CODES,
    NAMESPACE_USER_CONTRACTS,
)


def export_chunks(namespace: str, output_file: str, batch_size: int = 100):
    """Export all chunks from a namespace to JSON."""
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    
    # Get index stats to know how many vectors
    stats = index.describe_index_stats()
    ns_stats = stats.namespaces.get(namespace)
    if not ns_stats:
        print(f"No vectors found in namespace '{namespace}'")
        return
    
    total_vectors = ns_stats.vector_count
    print(f"Found {total_vectors} vectors in '{namespace}'")
    
    # Query in batches using zero vector (gets random samples)
    # Note: Pinecone doesn't support listing all vectors directly,
    # so we use a high top_k with zero vector
    zero_vec = [0.0] * settings.embedding_dimensions
    
    results = index.query(
        vector=zero_vec,
        namespace=namespace,
        top_k=min(total_vectors, 10000),  # Pinecone max is 10000
        include_metadata=True,
    )
    
    # Convert to list of dicts
    chunks = []
    for match in results.matches:
        meta = match.metadata
        
        # Parse JSON fields
        section_path = meta.get("section_path", "[]")
        if isinstance(section_path, str):
            try:
                section_path = json.loads(section_path)
            except:
                section_path = []
        
        highlights = meta.get("highlights", "[]")
        if isinstance(highlights, str):
            try:
                highlights = json.loads(highlights)
            except:
                highlights = []
        
        # Clean text (remove newlines for readability)
        text = meta.get("text", "")
        text_clean = " ".join(text.split())
        
        chunk = {
            "id": match.id,
            "score": match.score,
            "article_id": meta.get("article_id", meta.get("section_header", "")),
            "article_number": meta.get("article_number", ""),
            "source": meta.get("source", ""),
            "doc_id": meta.get("doc_id", ""),
            "category": meta.get("category", "general"),
            "page_start": meta.get("page_start", meta.get("page_number", 0)),
            "page_end": meta.get("page_end", meta.get("page_number", 0)),
            "section_path": section_path,
            "sub_chunk_index": meta.get("sub_chunk_index", 0),
            "total_sub_chunks": meta.get("total_sub_chunks", 1),
            "highlights_count": len(highlights),
            "text": text_clean,
            "text_length": len(text),
        }
        chunks.append(chunk)
    
    # Sort by source, then article_id
    chunks.sort(key=lambda x: (x["source"], x["article_id"], x["sub_chunk_index"]))
    
    # Write to file
    output = {
        "namespace": namespace,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Exported {len(chunks)} chunks to {output_file}")


def main():
    import csv
    
    # Export law codes to JSON
    export_chunks(NAMESPACE_LAW_CODES, "chunks_law_codes.json")
    
    # Also export as CSV
    with open("chunks_law_codes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    csv_file = "chunks_law_codes.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        if data["chunks"]:
            writer = csv.DictWriter(f, fieldnames=data["chunks"][0].keys())
            writer.writeheader()
            writer.writerows(data["chunks"])
    
    print(f"Also exported to {csv_file}")
    
    # Optionally export contracts too
    # export_chunks(NAMESPACE_USER_CONTRACTS, "chunks_contracts.json")


if __name__ == "__main__":
    main()
