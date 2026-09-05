"""
File Tracker for Incremental Indexing

Tracks indexed files with UUIDs, modification times, and metadata
to enable incremental indexing and selective deletion.
"""

import json
import uuid
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FileTracker:
    """
    Tracks indexed files with UUIDs for incremental indexing.
    
    Maintains a database of:
    - file_uuid: Unique identifier for each file
    - file_path: Absolute path to the file
    - file_hash: Content hash for change detection
    - mtime: Modification timestamp
    - indexed_at: When the file was indexed
    - collection_name: Which Qdrant collection it belongs to
    """

    def __init__(self, db_path: str = "./data/file_tracker.json"):
        """
        Initialize FileTracker.
        
        Args:
            db_path: Path to the tracking database JSON file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Database structure: {file_path: {uuid, hash, mtime, indexed_at, collection}}
        self.db: Dict[str, Dict] = self._load_db()

        logger.info(f"FileTracker initialized with {len(self.db)} tracked files")

    def _load_db(self) -> Dict[str, Dict]:
        """Load tracking database from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted tracking database, creating new one")
                return {}
        return {}

    def _save_db(self):
        """Save tracking database to file."""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.db, f, indent=2, ensure_ascii=False)
            logger.debug(f"Tracking database saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save tracking database: {e}")

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """
        Compute MD5 hash of file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return ""

    def generate_file_uuid(self, file_path: Path) -> str:
        """
        Generate a unique UUID for a file based on its absolute path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            UUID string
        """
        # Use UUID5 with file's absolute path for deterministic UUIDs
        abs_path = str(file_path.absolute())
        file_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, abs_path))
        return file_uuid

    def is_file_indexed(self, file_path: Path) -> bool:
        """
        Check if a file is already indexed.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is tracked
        """
        return str(file_path.absolute()) in self.db

    def is_file_modified(self, file_path: Path) -> bool:
        """
        Check if a tracked file has been modified since indexing.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is modified or not tracked
        """
        abs_path = str(file_path.absolute())

        if abs_path not in self.db:
            return True  # Not tracked = treat as modified

        try:
            # Check modification time
            current_mtime = file_path.stat().st_mtime
            tracked_mtime = self.db[abs_path].get('mtime', 0)

            if current_mtime > tracked_mtime:
                return True

            # Optionally check content hash for certainty
            current_hash = self._compute_file_hash(file_path)
            tracked_hash = self.db[abs_path].get('file_hash', '')

            return current_hash != tracked_hash

        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return False

    def add_file(
            self,
            file_path: Path,
            collection_name: str,
            file_uuid: Optional[str] = None
    ) -> str:
        """
        Track a newly indexed file.
        
        Args:
            file_path: Path to the file
            collection_name: Qdrant collection name
            file_uuid: Optional pre-generated UUID
            
        Returns:
            The file's UUID
        """
        abs_path = str(file_path.absolute())

        if file_uuid is None:
            file_uuid = self.generate_file_uuid(file_path)

        try:
            file_exists = file_path.exists()
            file_hash = self._compute_file_hash(file_path) if file_exists else ""
            mtime = file_path.stat().st_mtime if file_exists else 0.0
            file_size = file_path.stat().st_size if file_exists else 0

            self.db[abs_path] = {
                'file_uuid': file_uuid,
                'file_name': file_path.name,
                'file_hash': file_hash,
                'mtime': mtime,
                'indexed_at': datetime.now().isoformat(),
                'collection_name': collection_name,
                'file_size': file_size
            }

            self._save_db()
            logger.info(f"Tracked file: {file_path.name} (UUID: {file_uuid})")

        except Exception as e:
            logger.error(f"Failed to track file {file_path}: {e}")

        return file_uuid

    def get_file_uuid(self, file_path: Path) -> Optional[str]:
        """
        Get the UUID for a tracked file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            UUID string or None if not tracked
        """
        abs_path = str(file_path.absolute())
        return self.db.get(abs_path, {}).get('file_uuid')

    def remove_file(self, file_path: Path) -> Optional[str]:
        """
        Remove a file from tracking.
        
        Args:
            file_path: Path to the file
            
        Returns:
            The removed file's UUID or None if not tracked
        """
        abs_path = str(file_path.absolute())

        if abs_path in self.db:
            file_info = self.db.pop(abs_path)
            file_uuid = file_info.get('file_uuid')
            self._save_db()
            logger.info(f"Removed file from tracking: {file_path.name} (UUID: {file_uuid})")
            return file_uuid

        return None

    def get_deleted_files(self, directory: Path) -> List[Dict]:
        """
        Find files that are tracked but no longer exist in the directory.
        
        Args:
            directory: Directory to check
            
        Returns:
            List of file info dicts for deleted files
        """
        deleted = []
        abs_dir = str(directory.absolute())

        for file_path, file_info in list(self.db.items()):
            # Check if file path is under the directory
            if file_path.startswith(abs_dir):
                if not Path(file_path).exists():
                    deleted.append({
                        'file_path': file_path,
                        'file_uuid': file_info.get('file_uuid'),
                        'file_name': file_info.get('file_name'),
                        'collection_name': file_info.get('collection_name')
                    })

        return deleted

    def get_files_to_index(
            self,
            file_paths: List[Path],
            collection_name: str
    ) -> Dict[str, List[Path]]:
        """
        Determine which files need indexing (new or modified).
        
        Args:
            file_paths: List of file paths to check
            collection_name: Target collection name
            
        Returns:
            Dictionary with 'new', 'modified', and 'unchanged' lists
        """
        new_files = []
        modified_files = []
        unchanged_files = []

        for file_path in file_paths:
            if not self.is_file_indexed(file_path):
                new_files.append(file_path)
            elif self.is_file_modified(file_path):
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)

        logger.info(f"Files to index: {len(new_files)} new, {len(modified_files)} modified, "
                    f"{len(unchanged_files)} unchanged")

        return {
            'new': new_files,
            'modified': modified_files,
            'unchanged': unchanged_files
        }

    def get_all_tracked_files(self, collection_name: Optional[str] = None) -> List[Dict]:
        """
        Get all tracked files, optionally filtered by collection.
        
        Args:
            collection_name: Optional collection name to filter by
            
        Returns:
            List of file info dicts
        """
        files = []
        for file_path, file_info in self.db.items():
            if collection_name is None or file_info.get('collection_name') == collection_name:
                files.append({
                    'file_path': file_path,
                    'file_uuid': file_info.get('file_uuid'),
                    'file_name': file_info.get('file_name'),
                    'indexed_at': file_info.get('indexed_at'),
                    'collection_name': file_info.get('collection_name')
                })
        return files

    def clear_collection(self, collection_name: str) -> int:
        """
        Remove all files tracked for a specific collection.
        
        Args:
            collection_name: Collection name to clear
            
        Returns:
            Number of files removed
        """
        to_remove = [
            path for path, info in self.db.items()
            if info.get('collection_name') == collection_name
        ]

        for path in to_remove:
            del self.db[path]

        if to_remove:
            self._save_db()
            logger.info(f"Cleared {len(to_remove)} files from collection '{collection_name}'")

        return len(to_remove)

    def get_stats(self) -> Dict:
        """Get statistics about tracked files."""
        collections = {}
        for file_info in self.db.values():
            coll = file_info.get('collection_name', 'unknown')
            collections[coll] = collections.get(coll, 0) + 1

        return {
            'total_files': len(self.db),
            'collections': collections,
            'db_path': str(self.db_path)
        }
