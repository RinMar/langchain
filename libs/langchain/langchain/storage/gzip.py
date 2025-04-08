import os

from typing import Iterator, Optional
import pickle
import gzip
from langchain_core.stores import ByteStore

from langchain.storage.exceptions import InvalidKeyException

from langchain_core.documents.base import Document


class DocumentGzipStore(ByteStore):
    """BaseStore interface that works on the local file system.

    Examples:
        Create a DocumentGzipStore instance and perform operations on it:

        .. code-block:: python

            from langchain.storage import DocumentGzipStore

            # Instantiate the DocumentGzipStore with the root path
            document_store = DocumentGzipStore("/path/to/root/file_name.gz")

            # Set values for keys
            document_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = document_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            document_store.mdelete(["key1"])

            # Iterate over keys
            for key in document_store.yield_keys():
                print(key)  # noqa: T201

    """
    def __init__(self, file_path: str = 'parent_docs.gz'):
        self.file_path = file_path
        self._store = self._load_store()
        """Implement the BaseStore interface for the local file system.

       Args:
           file_path str: The path of the document store. All keys are stored in a gzip file.
       """

    def _load_store(self) -> dict[str, bytes]:

        if os.path.exists(self.file_path):
            with gzip.open(self.file_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _persist(self) -> None:
        with gzip.open(self.file_path, "wb") as f:
            pickle.dump(self._store, f)

    def mget(self, keys: list[str]) -> list[Optional[Document]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return [pickle.loads(self._store[k]) if k in self._store else None for k in keys]

    def mset(self, key_value_pairs: list[tuple[str, Document]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for k, v in key_value_pairs:
            self._store[k] = pickle.dumps(v)
        self._persist()

    def mdelete(self, keys: list[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for k in keys:
            self._store.pop(k, None)
        self._persist()

    def __getitem__(self, key: str) -> Document:
        return pickle.loads(self._store[key])

    def __setitem__(self, key: str, value: Document) -> None:
        self._store[key] = pickle.dumps(value)
        self._persist()

    def __delitem__(self, key: str) -> None:
        del self._store[key]
        self._persist()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def yield_keys(self, ) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        return self.file_store.yield_keys()
