"""ChromaDB retriever implementation"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retriever.base import BaseRetriever
from app.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
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
        embedder = None,
        db_path: str = None,
        collection_name: str = None
    ):
        """
        Initialize ChromaDB retriever
        
        Args:
            embedder: Embedding model (default: auto-select from config)
            db_path: Path to ChromaDB directory (default from settings)
            collection_name: Collection name (default from settings)
        """
        # Auto-select embedder based on config if not provided
        if embedder is None:
            embedding_type = getattr(settings, 'EMBEDDING_TYPE', 'sentence-transformers')
            if embedding_type == 'sentence-transformers':
                self.embedder = SentenceTransformerEmbedder()
                logger.info("Using SentenceTransformerEmbedder (matches ai/ logic)")
            else:
                self.embedder = GeminiEmbedder()
                logger.info("Using GeminiEmbedder")
        else:
            self.embedder = embedder
        
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
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity
        
        Matches ai/rag_system.py logic - no threshold filtering, just top_k
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from settings)
            
        Returns:
            List of documents with content and metadata
            Format: [{"content": str, "metadata": dict, "score": float}, ...]
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        top_k = top_k or settings.TOP_K
        
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
            
            # Process results - NO FILTERING, return all top_k results (matches ai/ logic)
            documents = []
            if results and results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # ChromaDB with cosine similarity returns distance in [0, 2] range
                    # where 0 = identical, 2 = opposite
                    # Convert to similarity score: similarity = 1 - (distance / 2)
                    # This gives us a score in [0, 1] where 1 is most similar
                    score = 1.0 - (distance / 2.0)
                    
                    # Always add document - no threshold filtering (matches ai/rag_system.py)
                    documents.append({
                        "content": doc,
                        "metadata": metadata or {},
                        "score": score
                    })
                    logger.debug(f"Document {i+1}: score={score:.3f}, distance={distance:.3f}, length={len(doc)}")
            
            logger.info(f"Retrieved {len(documents)} documents")
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
