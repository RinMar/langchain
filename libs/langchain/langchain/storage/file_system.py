import os
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
import pickle
import gzip
from langchain_core.stores import ByteStore

from langchain.storage.exceptions import InvalidKeyException

from langchain_core.documents.base import Document


class LocalFileStore(ByteStore):
    """BaseStore interface that works on the local file system.

    Examples:
        Create a LocalFileStore instance and perform operations on it:

        .. code-block:: python

            from langchain.storage import LocalFileStore

            # Instantiate the LocalFileStore with the root path
            file_store = LocalFileStore("/path/to/root")

            # Set values for keys
            file_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = file_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            file_store.mdelete(["key1"])

            # Iterate over keys
            for key in file_store.yield_keys():
                print(key)  # noqa: T201

    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
        update_atime: bool = False,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
            chmod_file: (optional, defaults to `None`) If specified, sets permissions
                for newly created files, overriding the current `umask` if needed.
            chmod_dir: (optional, defaults to `None`) If specified, sets permissions
                for newly created dirs, overriding the current `umask` if needed.
            update_atime: (optional, defaults to `False`) If `True`, updates the
                filesystem access time (but not the modified time) when a file is read.
                This allows MRU/LRU cache policies to be implemented for filesystems
                where access time updates are disabled.
        """
        self.root_path = Path(root_path).absolute()
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir
        self.update_atime = update_atime

    def _get_full_path(self, key: str) -> Path:
        """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise InvalidKeyException(f"Invalid characters in key: {key}")
        full_path = os.path.abspath(self.root_path / key)
        common_path = os.path.commonpath([str(self.root_path), full_path])
        if common_path != str(self.root_path):
            raise InvalidKeyException(
                f"Invalid key: {key}. Key should be relative to the full path."
                f"{self.root_path} vs. {common_path} and full path of {full_path}"
            )

        return Path(full_path)

    def _mkdir_for_store(self, dir: Path) -> None:
        """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
        if not dir.exists():
            self._mkdir_for_store(dir.parent)
            dir.mkdir(exist_ok=True)
            if self.chmod_dir is not None:
                os.chmod(dir, self.chmod_dir)

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        values: List[Optional[bytes]] = []
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                value = full_path.read_bytes()
                values.append(value)
                if self.update_atime:
                    # update access time only; preserve modified time
                    os.utime(full_path, (time.time(), os.stat(full_path).st_mtime))
            else:
                values.append(None)
        return values

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            full_path = self._get_full_path(key)
            self._mkdir_for_store(full_path.parent)
            full_path.write_bytes(value)
            if self.chmod_file is not None:
                os.chmod(full_path, self.chmod_file)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                full_path.unlink()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        prefix_path = self._get_full_path(prefix) if prefix else self.root_path
        for file in prefix_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(self.root_path)
                yield str(relative_path)

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
        return [pickle.loads(self._store[k]) if k in self._store else None for k in keys]

    def mset(self, key_value_pairs: list[tuple[str, Document]]) -> None:
        for k, v in key_value_pairs:
            self._store[k] = pickle.dumps(v)
        self._persist()

    def mdelete(self, keys: list[str]) -> None:
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
        return self.file_store.yield_keys()
