from abc import ABC, abstractmethod
from typing import List, AsyncGenerator
from ..models.base import Document

class BaseLoader(ABC):
    """Base class for all document loaders."""
    
    @abstractmethod
    async def load(self) -> AsyncGenerator[Document, None]:
        """Load documents from the source.
        
        Yields:
            Document: A document from the source.
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the source.
        
        Returns:
            bool: True if the connection is valid, False otherwise.
        """
        pass 