"""KB lifecycle manager for multi-KB orchestration."""

import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from urllib.parse import unquote, urlparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.api.document_manager import DocumentManager
from src.indexing.embedding_generator import EmbeddingGenerator
from src.indexing.vector_store import VectorStore
from src.config import load_defaults, load_profile_template, load_prompt_text, load_yaml_file
from src.config.schemas import KBConfigSnapshot, PromptPolicy
from src.config.resolver import resolve_profile_template_id, ProfileResolution

from .kb_registry import KBRegistry

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


KB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,63}$")


@dataclass
class KBManagerConfig:
    qdrant_url: str = "http://localhost:6333"
    data_dir: str = "./data"
    config_dir: str = "./src/config"
    collection_prefix: str = "kb_"
    distance_metric: str = "Cosine"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_cache_dir: str = "./data/embeddings_cache"
    indexing_mode: str = "online"
    backfill_ingest_state_on_startup: bool = True
    online_min_text_length: int = 200
    online_min_quality_score: float = 0.4
    online_remove_urls: bool = False
    online_remove_emails: bool = False
    online_use_unstructured_fallback: bool = True
    enable_entity_extraction: bool = True
    ner_backend: str = "spacy"
    ner_model: str = "en_core_web_lg"


class SharedEmbeddingDocumentManager(DocumentManager):
    """DocumentManager variant that reuses a shared EmbeddingGenerator."""

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str,
        file_tracker_db: str,
        embedding_generator: EmbeddingGenerator,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        cache_embeddings: bool = True,
        insertion_batch_size: int = 100,
    ):
        # Manually set fields to avoid loading another embedding model
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.cache_embeddings = cache_embeddings
        self.insertion_batch_size = insertion_batch_size

        logger.info("Initializing SharedEmbeddingDocumentManager for collection: %s", collection_name)

        # Initialize file tracker
        from src.indexing.file_tracker import FileTracker
        self.file_tracker = FileTracker(db_path=file_tracker_db)

        # Initialize chunker
        from src.indexing.recursive_chunker import RecursiveCharacterTextSplitter
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

        # Use shared embedding generator
        self.embedding_generator = embedding_generator

        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_dim=self.embedding_generator.embedding_dim,
            qdrant_url=qdrant_url,
            recreate_collection=False,
        )

        logger.info("SharedEmbeddingDocumentManager initialized successfully")


class KBManager:
    """Handles KB lifecycle and document operations."""

    def __init__(
        self,
        registry: KBRegistry,
        config: KBManagerConfig,
        shared_embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        self.registry = registry
        self.config = config
        self._lock = threading.RLock()
        self._document_managers: Dict[str, DocumentManager] = {}
        self.shared_embedding_generator = shared_embedding_generator

        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = Path(self.config.config_dir)
        self.profile_defaults = load_defaults(self.config_dir)
        self.kb_config_dir = self.data_dir / "kb_configs"
        self.kb_config_dir.mkdir(parents=True, exist_ok=True)
        self.file_tracker_dir = self.data_dir / "file_tracker"
        self.file_tracker_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir = self.data_dir / "conversations"
        self.cache_dir = self.data_dir / "query_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.qdrant = QdrantClient(url=self.config.qdrant_url)

    @staticmethod
    def _validate_kb_id(kb_id: str):
        if not kb_id or not KB_ID_PATTERN.match(kb_id):
            raise ValueError(
                "Invalid kb_id. Use 3-64 chars: letters, numbers, underscore, dash."
            )

    def _collection_name(self, kb_id: str) -> str:
        if kb_id.startswith(self.config.collection_prefix):
            return kb_id
        return f"{self.config.collection_prefix}{kb_id}"

    def _generate_kb_id(self) -> str:
        prefix = self.config.collection_prefix.rstrip("_")
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _collection_exists(self, collection_name: str) -> bool:
        collections = self.qdrant.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def _ensure_collection(self, collection_name: str):
        collections = self.qdrant.get_collections().collections
        if any(c.name == collection_name for c in collections):
            return

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance = distance_map.get(self.config.distance_metric, Distance.COSINE)

        logger.info("Creating Qdrant collection: %s", collection_name)
        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.config.embedding_dim, distance=distance),
        )

    def _get_document_manager(self, kb_id: str) -> DocumentManager:
        with self._lock:
            if kb_id in self._document_managers:
                return self._document_managers[kb_id]

            kb = self.registry.get_kb(kb_id)
            if not kb:
                raise ValueError(f"KB not found: {kb_id}")

            file_tracker_db = str(self.file_tracker_dir / f"{kb_id}.json")

            manager = DocumentManager(
                collection_name=kb["collection_name"],
                qdrant_url=self.config.qdrant_url,
                file_tracker_db=file_tracker_db,
                embedding_model=self.config.embedding_model,
                embedding_cache_dir=self.config.embedding_cache_dir,
                embedding_generator=self.shared_embedding_generator,
                indexing_mode=self.config.indexing_mode,
                backfill_ingest_state_on_startup=self.config.backfill_ingest_state_on_startup,
                online_min_text_length=self.config.online_min_text_length,
                online_min_quality_score=self.config.online_min_quality_score,
                online_remove_urls=self.config.online_remove_urls,
                online_remove_emails=self.config.online_remove_emails,
                online_use_unstructured_fallback=self.config.online_use_unstructured_fallback,
                enable_entity_extraction=self.config.enable_entity_extraction,
                ner_backend=self.config.ner_backend,
                ner_model=self.config.ner_model,
            )

            self._document_managers[kb_id] = manager
            return manager

    def _resolve_profile_template(
        self,
        kb_id: str,
        collection_name: str,
        display_name: str,
        description: Optional[str],
        forced_template_id: Optional[str] = None,
    ) -> tuple:
        if forced_template_id:
            resolution = ProfileResolution(template_id=forced_template_id, reason="forced")
            template_id = forced_template_id
        else:
            resolution = resolve_profile_template_id(
                kb_id=kb_id,
                collection_name=collection_name,
                display_name=display_name,
                description=description,
                config_dir=self.config_dir,
            )
            template_id = resolution.template_id
        profile_path = self.config_dir / "profiles" / f"{template_id}.yaml"
        if not profile_path.exists():
            logger.warning("Profile template not found for %s; falling back to default", template_id)
            template_id = "default"
            profile_path = self.config_dir / "profiles" / "default.yaml"
        profile_template = load_profile_template(profile_path, self.profile_defaults)
        return profile_template, resolution

    def _resolve_prompt_template_text(
        self,
        prompt_policy: PromptPolicy,
        *,
        profile_template_id: Optional[str] = None,
        kb_id: Optional[str] = None,
    ) -> tuple:
        if prompt_policy.template or prompt_policy.template_name:
            template_text, template_source = load_prompt_text(
                template=prompt_policy.template,
                template_name=prompt_policy.template_name,
                config_dir=self.config_dir,
            )
            return template_text or "", template_source

        for candidate_id in (profile_template_id, kb_id):
            if not candidate_id:
                continue
            candidate_path = self.config_dir / "prompts" / f"{candidate_id}_system_prompt.j2"
            if candidate_path.exists():
                return candidate_path.read_text(encoding="utf-8"), str(candidate_path)

        template_text, template_source = load_prompt_text(
            template_name="default",
            config_dir=self.config_dir,
        )
        return template_text or "", template_source

    def _write_snapshot(self, snapshot: KBConfigSnapshot) -> Path:
        if yaml is None:
            raise RuntimeError("PyYAML is required for KB config snapshot persistence.")
        snapshot_path = self.kb_config_dir / f"{snapshot.kb_id}.yaml"
        tmp_path = snapshot_path.with_suffix(".tmp")
        data = snapshot.model_dump() if hasattr(snapshot, "model_dump") else snapshot.dict()
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
        tmp_path.replace(snapshot_path)
        return snapshot_path

    def _snapshot_registry_path(self, snapshot_path: Path) -> str:
        try:
            return str(snapshot_path.relative_to(self.data_dir))
        except ValueError:
            return str(snapshot_path)

    @staticmethod
    def _now_version() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _plain_dict(value: Any, *, exclude_none: bool = True) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=exclude_none)
        if hasattr(value, "dict"):
            return value.dict(exclude_none=exclude_none)
        if isinstance(value, dict):
            return {
                str(key): item
                for key, item in value.items()
                if not exclude_none or item is not None
            }
        return {}

    @staticmethod
    def _policy_to_dict(policy: Any) -> Dict[str, Any]:
        if hasattr(policy, "model_dump"):
            return policy.model_dump()
        if hasattr(policy, "dict"):
            return policy.dict()
        return dict(policy or {})

    def _apply_policy_overrides(
        self,
        policy: Any,
        policy_type: Type[Any],
        overrides: Optional[Dict[str, Any]],
    ) -> Any:
        data = self._policy_to_dict(policy)
        for key, value in (overrides or {}).items():
            if value is not None and key in data:
                data[key] = value
        return policy_type(**data)

    def _load_existing_snapshot(self, kb_id: str) -> Optional[KBConfigSnapshot]:
        kb = self.registry.get_kb(kb_id)
        snapshot_path: Optional[Path] = None
        if kb and kb.get("config_path"):
            snapshot_path = Path(str(kb["config_path"]))
            if not snapshot_path.is_absolute():
                candidate = (self.data_dir.parent / snapshot_path).resolve()
                if candidate.exists():
                    snapshot_path = candidate
                else:
                    snapshot_path = (self.data_dir / snapshot_path).resolve()
        else:
            snapshot_path = self.kb_config_dir / f"{kb_id}.yaml"

        if not snapshot_path or not snapshot_path.exists():
            return None
        try:
            return KBConfigSnapshot(**load_yaml_file(snapshot_path))
        except Exception:
            logger.warning("Failed to load existing KB snapshot for %s", kb_id, exc_info=True)
            return None

    def _compose_runtime_prompt_template(self, system_prompt: str) -> tuple[str, str]:
        """Embed tenant instructions into the default RAG prompt template."""
        base_template, base_source = load_prompt_text(
            template_name="default",
            config_dir=self.config_dir,
        )
        tenant_block = (
            "### Tenant Instructions (CRITICAL)\n"
            f"{system_prompt.strip()}\n\n"
        )
        marker = "### Knowledge & Grounding Rules (CRITICAL)"
        if marker in base_template:
            return base_template.replace(marker, f"{tenant_block}{marker}", 1), base_source
        return f"{tenant_block}{base_template}", base_source

    def _create_kb_config_snapshot(
        self,
        kb_id: str,
        collection_name: str,
        display_name: str,
        description: Optional[str],
        forced_template_id: Optional[str] = None,
    ) -> tuple:
        if yaml is None:
            raise RuntimeError("PyYAML is required for KB config snapshot persistence.")

        profile_template, resolution = self._resolve_profile_template(
            kb_id=kb_id,
            collection_name=collection_name,
            display_name=display_name,
            description=description,
            forced_template_id=forced_template_id,
        )

        template_text, template_source = self._resolve_prompt_template_text(
            profile_template.prompt,
            profile_template_id=profile_template.profile_template_id,
            kb_id=kb_id,
        )
        prompt_policy = PromptPolicy(
            template=template_text,
            template_name=profile_template.prompt.template_name or profile_template.prompt.template,
        )

        raw_fast_mode = profile_template.routing.structured_query_fast_mode
        resolved_fast_mode = raw_fast_mode if isinstance(raw_fast_mode, bool) else None

        snapshot = KBConfigSnapshot(
            kb_id=kb_id,
            collection_name=collection_name,
            profile_template_id=profile_template.profile_template_id,
            profile_template_version=profile_template.profile_template_version,
            prompt=prompt_policy,
            generation=profile_template.generation,
            routing=profile_template.routing,
            grounding=profile_template.grounding,
            query_handler=profile_template.query_handler,
            resolved_structured_query_fast_mode=resolved_fast_mode,
            notes={
                "profile_resolution_reason": resolution.reason,
                "prompt_template_source": template_source,
            },
        )

        snapshot_path = self._write_snapshot(snapshot)
        return snapshot_path, snapshot

    def create_kb(
        self,
        kb_id: Optional[str],
        display_name: str,
        description: Optional[str] = None,
        existing_collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        if kb_id:
            self._validate_kb_id(kb_id)

        with self._lock:
            if kb_id and self.registry.exists(kb_id):
                raise ValueError(f"KB already exists: {kb_id}")

            if existing_collection:
                collection_name = existing_collection
                if not kb_id:
                    kb_id = self._generate_kb_id()
                    # Ensure generated KB ID is unique in registry
                    attempts = 0
                    while self.registry.exists(kb_id):
                        attempts += 1
                        if attempts > 5:
                            raise ValueError("Failed to generate unique kb_id. Try again.")
                        kb_id = self._generate_kb_id()
                # Validate collection exists
                try:
                    self.qdrant.get_collection(collection_name)
                except Exception as exc:
                    raise ValueError(f"Existing collection not found: {collection_name}") from exc
            else:
                if not kb_id:
                    kb_id = self._generate_kb_id()
                collection_name = self._collection_name(kb_id)

                # Ensure unique collection name if auto-generated
                attempts = 0
                while self._collection_exists(collection_name):
                    attempts += 1
                    if attempts > 5:
                        raise ValueError("Failed to generate unique kb_id. Try again.")
                    kb_id = self._generate_kb_id()
                    collection_name = self._collection_name(kb_id)

                self._ensure_collection(collection_name)

            snapshot_path, snapshot = self._create_kb_config_snapshot(
                kb_id=kb_id,
                collection_name=collection_name,
                display_name=display_name,
                description=description,
            )

            try:
                kb_meta = self.registry.register_kb(
                    kb_id=kb_id,
                    display_name=display_name,
                    collection_name=collection_name,
                    description=description,
                    embedding_model=self.config.embedding_model,
                    embedding_dim=self.config.embedding_dim,
                    config_path=self._snapshot_registry_path(snapshot_path),
                    profile_template_id=snapshot.profile_template_id,
                    profile_template_version=snapshot.profile_template_version,
                )
            except Exception:
                try:
                    snapshot_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to cleanup snapshot for KB %s after registry failure", kb_id)
                raise

            return kb_meta

    def create_or_get_kb(
        self,
        kb_id: str,
        display_name: str,
        description: Optional[str] = None,
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        """Create a KB for tenant provisioning, or reuse/update the existing one."""
        self._validate_kb_id(kb_id)

        with self._lock:
            if replace_existing and self.registry.exists(kb_id):
                self.delete_kb(kb_id)

            existing = self.registry.get_kb(kb_id)
            if existing:
                updated = self.registry.update_kb(
                    kb_id,
                    display_name=display_name,
                    description=description,
                )
                return updated or existing

            return self.create_kb(
                kb_id=kb_id,
                display_name=display_name,
                description=description,
                existing_collection=None,
            )

    def write_runtime_config_snapshot(
        self,
        *,
        kb_id: str,
        tenant_id: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        ai_config: Optional[Dict[str, Any]] = None,
        source: str = "ownify_provisioning",
    ) -> Dict[str, Any]:
        """Persist an Ownify runtime snapshot without writing repo config files."""
        with self._lock:
            kb = self.registry.get_kb(kb_id)
            if not kb:
                raise ValueError(f"KB not found: {kb_id}")

            profile_template, _ = self._resolve_profile_template(
                kb_id=kb_id,
                collection_name=kb["collection_name"],
                display_name=display_name or kb.get("display_name", kb_id),
                description=description if description is not None else kb.get("description"),
                forced_template_id="default",
            )

            existing_snapshot = self._load_existing_snapshot(kb_id)

            base_generation = existing_snapshot.generation if existing_snapshot else profile_template.generation
            base_routing = existing_snapshot.routing if existing_snapshot else profile_template.routing
            base_grounding = existing_snapshot.grounding if existing_snapshot else profile_template.grounding
            base_query_handler = (
                existing_snapshot.query_handler if existing_snapshot else profile_template.query_handler
            )

            ai_config_data = self._plain_dict(ai_config)
            generation = self._apply_policy_overrides(
                base_generation,
                type(base_generation),
                self._plain_dict(ai_config_data.get("generation")),
            )
            routing = self._apply_policy_overrides(
                base_routing,
                type(base_routing),
                self._plain_dict(ai_config_data.get("routing")),
            )
            grounding = self._apply_policy_overrides(
                base_grounding,
                type(base_grounding),
                self._plain_dict(ai_config_data.get("grounding")),
            )

            query_handler_overrides = self._plain_dict(ai_config_data.get("query_handler"))
            if ai_config_data.get("canned_responses") is not None:
                query_handler_overrides["canned_responses"] = ai_config_data.get("canned_responses")
            query_handler = self._apply_policy_overrides(
                base_query_handler,
                type(base_query_handler),
                query_handler_overrides,
            )

            if system_prompt is not None:
                prompt_template, base_prompt_source = self._compose_runtime_prompt_template(system_prompt)
                prompt_policy = PromptPolicy(template=prompt_template, template_name=None)
                prompt_source = f"inline_tenant_instructions:{base_prompt_source}"
            elif existing_snapshot and existing_snapshot.prompt.template:
                prompt_policy = existing_snapshot.prompt
                prompt_source = str(existing_snapshot.notes.get("prompt_template_source", "existing_snapshot"))
            else:
                template_text, prompt_source = self._resolve_prompt_template_text(
                    profile_template.prompt,
                    profile_template_id=profile_template.profile_template_id,
                    kb_id=kb_id,
                )
                prompt_policy = PromptPolicy(
                    template=template_text,
                    template_name=profile_template.prompt.template_name or profile_template.prompt.template,
                )

            config_version = self._now_version()
            raw_fast_mode = routing.structured_query_fast_mode
            resolved_fast_mode = raw_fast_mode if isinstance(raw_fast_mode, bool) else None
            notes = {}
            if existing_snapshot:
                notes.update(existing_snapshot.notes)
            notes.update(
                {
                    "tenant_id": tenant_id,
                    "source": source,
                    "config_version": config_version,
                    "prompt_template_source": prompt_source,
                }
            )

            snapshot = KBConfigSnapshot(
                kb_id=kb_id,
                collection_name=kb["collection_name"],
                profile_template_id="ownify_runtime",
                profile_template_version=config_version,
                resolved_at=config_version,
                prompt=prompt_policy,
                generation=generation,
                routing=routing,
                grounding=grounding,
                query_handler=query_handler,
                resolved_structured_query_fast_mode=resolved_fast_mode,
                notes=notes,
            )

            snapshot_path = self._write_snapshot(snapshot)
            updated = self.registry.update_kb(
                kb_id,
                display_name=display_name,
                description=description,
                config_path=self._snapshot_registry_path(snapshot_path),
                profile_template_id=snapshot.profile_template_id,
                profile_template_version=snapshot.profile_template_version,
            )

            return {
                "status": "success",
                "kb": updated or kb,
                "snapshot_path": str(snapshot_path),
                "config_version": config_version,
            }

    def create_or_update_snapshot(
        self,
        kb_id: str,
        profile_template_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            kb = self.registry.get_kb(kb_id)
            if not kb:
                raise ValueError(f"KB not found: {kb_id}")

            snapshot_path, snapshot = self._create_kb_config_snapshot(
                kb_id=kb_id,
                collection_name=kb["collection_name"],
                display_name=kb.get("display_name", kb_id),
                description=kb.get("description"),
                forced_template_id=profile_template_id,
            )

            updated = self.registry.update_kb(
                kb_id,
                config_path=self._snapshot_registry_path(snapshot_path),
                profile_template_id=snapshot.profile_template_id,
                profile_template_version=snapshot.profile_template_version,
            )

            return {
                "status": "success",
                "kb": updated or kb,
                "snapshot_path": str(snapshot_path),
            }

    def delete_kb(self, kb_id: str) -> Dict[str, Any]:
        with self._lock:
            kb = self.registry.get_kb(kb_id)
            if not kb:
                raise ValueError(f"KB not found: {kb_id}")

            collection_name = kb["collection_name"]

            # Delete collection
            try:
                self.qdrant.delete_collection(collection_name)
            except Exception as exc:
                logger.warning("Failed to delete collection %s: %s", collection_name, exc)

            # Remove registry entry
            self.registry.delete_kb(kb_id)

            # Cleanup trackers & session data
            self._cleanup_kb_storage(kb_id, collection_name)

            # Remove cached document manager
            self._document_managers.pop(kb_id, None)

            return {
                "status": "success",
                "kb_id": kb_id,
                "collection_name": collection_name,
            }

    def _cleanup_kb_storage(self, kb_id: str, collection_name: str):
        # File tracker
        tracker_path = self.file_tracker_dir / f"{kb_id}.json"
        if tracker_path.exists():
            tracker_path.unlink()

        # Conversations
        convo_path = self.conversations_dir / kb_id
        if convo_path.exists():
            shutil.rmtree(convo_path)

        # Query cache
        cache_path = self.cache_dir / f"{kb_id}.json"
        if cache_path.exists():
            cache_path.unlink()

        # KB config snapshot
        snapshot_path = self.kb_config_dir / f"{kb_id}.yaml"
        if snapshot_path.exists():
            snapshot_path.unlink()

        # BM25 cache
        bm25_path = self.data_dir / "bm25_indices" / f"{collection_name}_bm25.pkl"
        if bm25_path.exists():
            bm25_path.unlink()

    def _invalidate_bm25_cache(self, collection_name: str) -> None:
        """Invalidate sparse retrieval cache after collection mutations."""
        bm25_path = self.data_dir / "bm25_indices" / f"{collection_name}_bm25.pkl"
        try:
            if bm25_path.exists():
                bm25_path.unlink()
                logger.info("Invalidated BM25 cache for collection %s", collection_name)
        except Exception:
            logger.warning("Failed to invalidate BM25 cache for %s", collection_name, exc_info=True)

    def get_kb_info(self, kb_id: str) -> Dict[str, Any]:
        kb = self.registry.get_kb(kb_id)
        if not kb:
            raise ValueError(f"KB not found: {kb_id}")

        info: Dict[str, Any] = {}
        try:
            collection_info = self.qdrant.get_collection(kb["collection_name"])
            info = {
                "collection_name": kb["collection_name"],
                "points_count": getattr(collection_info, "points_count", None),
                "vector_size": getattr(collection_info.config.params.vectors, "size", None),
                "distance": getattr(collection_info.config.params.vectors.distance, "name", None),
                "status": getattr(collection_info, "status", None).name if getattr(collection_info, "status", None) else None,
            }
        except Exception as exc:
            logger.warning("Failed to get collection info: %s", exc)

        return {
            "kb": kb,
            "collection": info,
        }

    def list_kbs(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "kbs": self.registry.list_kbs(),
        }

    @staticmethod
    def _infer_file_name_from_url(sas_url: str) -> str:
        parsed = urlparse(sas_url)
        name = Path(unquote(parsed.path)).name
        return name or f"document_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _normalize_document_source(
        manager: DocumentManager,
        *,
        file_id: Optional[str],
        file_name: Optional[str],
        sas_url: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if bool(sas_url) == bool(local_path):
            raise ValueError("Provide exactly one document source: sas_url or local_path.")

        if local_path:
            path = manager._normalize_local_path(local_path)
            resolved_path = str(path.resolve()) if path.exists() else str(path)
            resolved_name = file_name or path.name
            resolved_id = file_id or DocumentManager.generate_source_file_id(
                f"local:{resolved_path}",
                resolved_name,
            )
            return {
                "file_id": resolved_id,
                "file_name": resolved_name,
                "local_path": str(path),
                "sas_url": None,
            }

        resolved_name = file_name or KBManager._infer_file_name_from_url(sas_url or "")
        resolved_id = file_id or DocumentManager.generate_source_file_id(
            f"sas_url:{sas_url}",
            resolved_name,
        )
        return {
            "file_id": resolved_id,
            "file_name": resolved_name,
            "sas_url": sas_url,
            "local_path": None,
        }

    def _index_document_with_manager(
        self,
        manager: DocumentManager,
        *,
        file_id: Optional[str],
        file_name: Optional[str],
        sas_url: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        source = self._normalize_document_source(
            manager,
            file_id=file_id,
            file_name=file_name,
            sas_url=sas_url,
            local_path=local_path,
        )

        if source["local_path"]:
            return manager.index_file_from_local_path(
                file_id=source["file_id"],
                file_name=source["file_name"],
                local_path=source["local_path"],
            )

        return manager.index_file_from_url(
            file_id=source["file_id"],
            file_name=source["file_name"],
            sas_url=source["sas_url"],
        )

    def _after_document_mutation(self, kb_id: str, manager: DocumentManager, changed: bool) -> None:
        if changed:
            kb = self.registry.get_kb(kb_id)
            if kb:
                self._invalidate_bm25_cache(kb["collection_name"])

        try:
            files_info = manager.list_indexed_files()
            self.registry.update_kb(kb_id, doc_count=files_info.get("total_files", 0))
        except Exception:
            logger.warning("Failed to update doc_count for KB %s", kb_id)

    def add_document(
        self,
        kb_id: str,
        file_id: Optional[str] = None,
        file_name: Optional[str] = None,
        sas_url: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        manager = self._get_document_manager(kb_id)
        result = self._index_document_with_manager(
            manager,
            file_id=file_id,
            file_name=file_name,
            sas_url=sas_url,
            local_path=local_path,
        )

        self._after_document_mutation(kb_id, manager, changed=bool(result.get("success")))

        return result

    def add_documents(
        self,
        kb_id: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        directory_path: Optional[str] = None,
        recursive: bool = True,
        fail_fast: bool = False,
    ) -> Dict[str, Any]:
        manager = self._get_document_manager(kb_id)
        requested_documents: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        if directory_path:
            discovery = manager.discover_supported_local_files(
                directory_path=directory_path,
                recursive=recursive,
            )
            if not discovery.get("success"):
                return discovery

            skipped.extend(discovery.get("skipped", []))
            for path in discovery.get("files", []):
                requested_documents.append({
                    "file_id": DocumentManager.generate_source_file_id(
                        f"local:{path.resolve()}",
                        path.name,
                    ),
                    "file_name": path.name,
                    "local_path": str(path),
                })

        for document in documents or []:
            requested_documents.append(dict(document))

        if not requested_documents:
            return {
                "success": False,
                "error_code": "no_documents_to_index",
                "error": "No supported documents were found to index.",
                "directory_path": directory_path,
                "skipped": skipped,
                "supported_formats": sorted(manager.document_processor.supported_extensions),
            }

        indexed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        total_chunks = 0
        total_vectors = 0

        for document in requested_documents:
            try:
                result = self._index_document_with_manager(
                    manager,
                    file_id=document.get("file_id"),
                    file_name=document.get("file_name"),
                    sas_url=document.get("sas_url"),
                    local_path=document.get("local_path"),
                )
            except ValueError as exc:
                result = {
                    "success": False,
                    "error_code": "invalid_document_source",
                    "error": str(exc),
                    "file_id": document.get("file_id"),
                    "file_name": document.get("file_name"),
                    "local_path": document.get("local_path"),
                    "sas_url": document.get("sas_url"),
                }

            if result.get("success"):
                indexed.append(result)
                total_chunks += int(result.get("chunks_created", 0) or 0)
                total_vectors += int(result.get("vectors_inserted", 0) or 0)
            else:
                failed.append(result)
                if fail_fast:
                    break

        self._after_document_mutation(kb_id, manager, changed=bool(indexed))

        return {
            "success": bool(indexed) and not failed,
            "partial_success": bool(indexed) and bool(failed),
            "kb_id": kb_id,
            "submitted_count": len(requested_documents),
            "indexed_count": len(indexed),
            "failed_count": len(failed),
            "skipped_count": len(skipped),
            "chunks_created": total_chunks,
            "vectors_inserted": total_vectors,
            "indexed": indexed,
            "failed": failed,
            "skipped": skipped,
            "directory_path": directory_path,
            "recursive": recursive,
        }

    def remove_document(self, kb_id: str, file_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        manager = self._get_document_manager(kb_id)
        if not file_name:
            file_info = manager.get_file_info(file_id)
            if file_info:
                file_name = file_info.get("file_name")

        result = manager.delete_file(file_id=file_id, file_name=file_name)
        if result.get("success"):
            kb = self.registry.get_kb(kb_id)
            if kb:
                self._invalidate_bm25_cache(kb["collection_name"])

        # Update doc_count in registry
        try:
            files_info = manager.list_indexed_files()
            self.registry.update_kb(kb_id, doc_count=files_info.get("total_files", 0))
        except Exception:
            logger.warning("Failed to update doc_count for KB %s", kb_id)

        return result

    def list_documents(self, kb_id: str) -> Dict[str, Any]:
        manager = self._get_document_manager(kb_id)
        return manager.list_indexed_files()
