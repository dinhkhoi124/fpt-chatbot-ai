"""ChromaDB retriever implementation"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retriever.base import BaseRetriever
from app.embeddings.gemini_embedder import GeminiEmbedder
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaRetriever(BaseRetriever):
    """
    ChromaDB-based document retriever
    
    Features:
    - Local vector database with ChromaDB
    - Cosine similarity search
    - Persistent storage
    - Metadata filtering support
    """
    
    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        db_path: str = None,
        collection_name: str = None
    ):
        """
        Initialize ChromaDB retriever
        
        Args:
            embedder: Embedding model (default: GeminiEmbedder)
            db_path: Path to ChromaDB directory (default from settings)
            collection_name: Collection name (default from settings)
        """
        self.embedder = embedder or GeminiEmbedder()
        self.db_path = str(db_path or settings.CHROMA_DB_PATH)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded ChromaDB collection: {self.collection_name}")
            logger.info(f"Collection count: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to load collection '{self.collection_name}': {e}")
            raise ValueError(
                f"Collection '{self.collection_name}' not found in {self.db_path}. "
                "Please ensure the vector database is properly initialized."
            )
    
    def search(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from settings)
            score_threshold: Minimum similarity score (default from settings)
            
        Returns:
            List of documents with content and metadata
            Format: [{"content": str, "metadata": dict, "score": float}, ...]
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        top_k = top_k or settings.TOP_K
        score_threshold = score_threshold or settings.SCORE_THRESHOLD
        
        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self.embedder.embed_text(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents = []
            if results and results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    # Lower distance = higher similarity
                    # Convert to 0-1 scale where 1 is most similar
                    score = 1.0 / (1.0 + distance)
                    
                    # Apply score threshold
                    if score >= score_threshold:
                        documents.append({
                            "content": doc,
                            "metadata": metadata or {},
                            "score": score
                        })
                        logger.debug(f"Document {i+1}: score={score:.3f}, length={len(doc)}")
            
            logger.info(f"Retrieved {len(documents)} documents (threshold: {score_threshold})")
            return documents
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics
        
        Returns:
            Dict with collection stats
        """
        try:
            return {
                "collection_name": self.collection_name,
                "document_count": self.collection.count(),
                "db_path": self.db_path,
                "embedding_dimension": self.embedder.get_dimension(),
                "embedding_model": self.embedder.model
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "error": str(e)
            }
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> None:
        """
        Add new documents to the collection
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts (optional)
            ids: List of document IDs (optional, will auto-generate if not provided)
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedder.embed_batch(documents)
            
            # Generate IDs if not provided
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Prepare metadatas
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
