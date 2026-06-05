from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class VectorItem(BaseModel):
    id: str
    text: str
    vector: List[float | int]
    metadata: Any


class GetResult(BaseModel):
    ids: Optional[List[List[str]]]
    documents: Optional[List[List[str]]]
    metadatas: Optional[List[List[Any]]]


class SearchResult(GetResult):
    distances: Optional[List[List[float | int]]]


class VectorDBBase(ABC):
    """
    Abstract base class for all vector database backends.

    Implementations of this class provide methods for collection management,
    vector insertion, deletion, similarity search, and metadata filtering.

    Any custom vector database integration must inherit from this class and
    implement all abstract methods.
    """

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check if the collection exists in the vector DB."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector DB."""
        pass

    @abstractmethod
    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        """Insert a list of vector items into a collection."""
        pass

    @abstractmethod
    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        """Insert or update vector items in a collection."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        vectors: List[List[Union[float, int]]],
        filter: Optional[Dict] = None,
        limit: int = 10,
    ) -> Optional[SearchResult]:
        """Search for similar vectors in a collection."""
        pass

    @abstractmethod
    def query(
        self, collection_name: str, filter: Dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        """Query vectors from a collection using metadata filter."""
        pass

    @abstractmethod
    def get(self, collection_name: str) -> Optional[GetResult]:
        """Retrieve all vectors from a collection."""
        pass

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
    ) -> None:
        """Delete vectors by ID or filter from a collection."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the vector database by removing all collections or those matching a condition."""
        pass

    def peek(self, collection_name: str, limit: int = 10) -> Optional[GetResult]:
        """Fetch a small sample from a collection without iterating all items.
        Default: falls back to get(). Override for O(1) performance."""
        return self.get(collection_name)

    def bm25_search(
        self,
        collection_name: str,
        query: str,
        query_properties: List[str],
        limit: int = 15,
    ) -> Optional[SearchResult]:
        """Native BM25 keyword search on pre-built inverted index.
        Default: returns None (caller falls back to in-memory BM25). Override for production."""
        return None
