"""RAG (Retrieval Augmented Generation) Service using ChromaDB.

This service manages a vector database of trading resources and provides
semantic search capabilities for enhancing trading recommendations.

Resources indexed:
- The Intelligent Investor principles
- Trading Strategies and Market Psychology
- API documentation (Coinbase, Schwab)
- Statistical learning concepts
"""

import os
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("mister_risker.rag")


class RAGService:
    """Service for Retrieval Augmented Generation using ChromaDB.
    
    This service:
    1. Chunks and embeds markdown documents from resources/
    2. Stores embeddings in a persistent ChromaDB collection
    3. Retrieves relevant passages based on semantic similarity
    4. Provides context for trading recommendations
    
    Attributes:
        client: ChromaDB client
        collection: The main document collection
        resources_dir: Path to resources directory
    """
    
    # Chunk settings
    CHUNK_SIZE = 1000  # characters per chunk
    CHUNK_OVERLAP = 200  # overlap between chunks
    
    # Collection name
    COLLECTION_NAME = "mister_risker_resources"
    
    def __init__(
        self,
        persist_directory: str = ".chromadb",
        resources_dir: str = "resources",
        auto_index: bool = True
    ):
        """Initialize the RAG service.
        
        Args:
            persist_directory: Where to store the ChromaDB data
            resources_dir: Directory containing markdown resources
            auto_index: Whether to index resources on init if collection is empty
        """
        self.resources_dir = Path(resources_dir)
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        # Using default embedding function (sentence-transformers all-MiniLM-L6-v2)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Mister Risker trading resources"}
        )
        
        logger.info(f"RAGService initialized. Collection has {self.collection.count()} documents")
        
        # Auto-index if collection is empty
        if auto_index and self.collection.count() == 0:
            logger.info("Collection is empty, indexing resources...")
            self.index_resources()
    
    def _chunk_text(self, text: str, source: str) -> list[dict]:
        """Split text into overlapping chunks.
        
        Args:
            text: The full text to chunk
            source: Source filename for metadata
            
        Returns:
            List of dicts with 'text', 'source', 'chunk_index'
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.CHUNK_SIZE
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars of chunk
                search_start = max(end - 100, start)
                last_period = text.rfind('. ', search_start, end)
                last_newline = text.rfind('\n\n', search_start, end)
                
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_index": chunk_index
                })
                chunk_index += 1
            
            start = end - self.CHUNK_OVERLAP
            if start < 0:
                start = end
        
        return chunks
    
    def _extract_section_context(self, text: str, position: int) -> str:
        """Extract the nearest heading/section for context.
        
        Args:
            text: Full document text
            position: Position in text
            
        Returns:
            Section heading or empty string
        """
        # Look backwards for markdown heading
        search_text = text[:position]
        lines = search_text.split('\n')
        
        for line in reversed(lines):
            if line.startswith('#'):
                return line.strip()
        
        return ""
    
    def index_resources(self, force_reindex: bool = False) -> int:
        """Index all markdown files in the resources directory.
        
        Args:
            force_reindex: If True, delete existing and re-index all
            
        Returns:
            Number of chunks indexed
        """
        if force_reindex:
            logger.info("Force reindex: deleting existing collection")
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Mister Risker trading resources"}
            )
        
        if not self.resources_dir.exists():
            logger.warning(f"Resources directory not found: {self.resources_dir}")
            return 0
        
        total_chunks = 0
        
        # Find all markdown files
        md_files = list(self.resources_dir.glob("*.md"))
        org_files = list(self.resources_dir.glob("*.org"))
        all_files = md_files + org_files
        
        logger.info(f"Found {len(all_files)} resource files to index")
        
        for file_path in all_files:
            try:
                logger.info(f"Indexing: {file_path.name}")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Chunk the document
                chunks = self._chunk_text(content, file_path.name)
                
                if not chunks:
                    continue
                
                # Prepare for ChromaDB
                ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
                documents = [c["text"] for c in chunks]
                metadatas = [
                    {
                        "source": c["source"],
                        "chunk_index": c["chunk_index"],
                        "file_type": file_path.suffix
                    }
                    for c in chunks
                ]
                
                # Add to collection in batches
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch_end = min(i + batch_size, len(chunks))
                    self.collection.add(
                        ids=ids[i:batch_end],
                        documents=documents[i:batch_end],
                        metadatas=metadatas[i:batch_end]
                    )
                
                logger.info(f"  Indexed {len(chunks)} chunks from {file_path.name}")
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
        
        logger.info(f"Indexing complete. Total chunks: {total_chunks}")
        return total_chunks
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        source_filter: Optional[list[str]] = None
    ) -> list[dict]:
        """Query the vector database for relevant passages.
        
        Args:
            query_text: The query to search for
            n_results: Number of results to return
            source_filter: Optional list of source filenames to filter by
            
        Returns:
            List of dicts with 'text', 'source', 'score'
        """
        if self.collection.count() == 0:
            logger.warning("Collection is empty, no results to return")
            return []
        
        # Build where filter if specified
        where = None
        if source_filter:
            where = {"source": {"$in": source_filter}}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted.append({
                        "text": doc,
                        "source": results["metadatas"][0][i]["source"],
                        "chunk_index": results["metadatas"][0][i]["chunk_index"],
                        "distance": results["distances"][0][i] if results["distances"] else None
                    })
            
            logger.debug(f"Query '{query_text[:50]}...' returned {len(formatted)} results")
            return formatted
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return []
    
    def get_trading_context(
        self,
        query: str,
        include_intelligent_investor: bool = True,
        include_trading_psychology: bool = True,
        include_stats: bool = False,
        n_results: int = 3
    ) -> str:
        """Get relevant trading context for a query.
        
        This method retrieves relevant passages from trading resources
        to augment trading recommendations.
        
        Args:
            query: The user's query or trading context
            include_intelligent_investor: Include "The Intelligent Investor" content
            include_trading_psychology: Include trading psychology content
            include_stats: Include statistical learning content
            n_results: Results per source category
            
        Returns:
            Formatted context string for LLM prompt
        """
        context_parts = []
        
        # Build source filter based on preferences
        sources = []
        if include_intelligent_investor:
            sources.append("intelligent-investor.md")
        if include_trading_psychology:
            sources.append("Trading Strategies and Market Psychology.md")
        if include_stats:
            sources.append("ISLR_First_Printing.md")
        
        # Query with source filter if specified
        source_filter = sources if sources else None
        results = self.query(query, n_results=n_results * len(sources) if sources else n_results, source_filter=source_filter)
        
        if not results:
            return ""
        
        # Group by source
        by_source = {}
        for r in results:
            src = r["source"]
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(r)
        
        # Format context
        for source, passages in by_source.items():
            source_name = source.replace(".md", "").replace("-", " ").title()
            context_parts.append(f"\n### From {source_name}:\n")
            
            for p in passages[:n_results]:
                # Truncate very long passages
                text = p["text"][:800] + "..." if len(p["text"]) > 800 else p["text"]
                context_parts.append(f"- {text}\n")
        
        return "".join(context_parts)
    
    def get_stats(self) -> dict:
        """Get statistics about the indexed resources.
        
        Returns:
            Dict with collection stats
        """
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.COLLECTION_NAME,
            "persist_directory": self.persist_directory
        }
