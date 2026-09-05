"""Multi-KB pipeline orchestration with shared resources."""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice

from src.pipeline.conversational_rag_pipeline import (
    ConversationalRAGPipeline,
    ConversationalPipelineConfig,
    ConversationalPipelineResult,
)
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.result_fusion import ResultFusion
from src.retrieval.reranker import Reranker
from src.context_handling.context_verifier import ContextVerifier
from src.generation.prompt_builder import RAGPromptBuilder
from src.generation.citation_extractor import CitationExtractor
from src.generation.llm_generator import LLMGenerator, GenerationConfig
from src.indexing.embedding_generator import EmbeddingGenerator
from src.indexing.vector_store import VectorStore

from src.kb_management.kb_registry import KBRegistry
from src.config import KBConfigSnapshot, load_yaml_file

logger = logging.getLogger(__name__)


@dataclass
class MultiKBSettings:
    qdrant_url: str = "http://localhost:6333"
    data_dir: str = "./data"
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7
    llm_top_p: float = 0.8
    llm_repetition_penalty: float = 1.2
    llm_presence_penalty: float = 1.0
    llm_frequency_penalty: float = 0.3
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    use_gpu: bool = False
    rerank_top_k: int = 7
    nli_threshold: float = 0.0
    dedup_threshold: float = 0.85
    normalize_newlines: str = "preserve"
    enable_memory: bool = True
    max_turns: int = 50
    history_in_prompt_turns: int = 10
    auto_save: bool = True
    enable_reformulation: bool = True
    use_llm_reformulation: bool = True
    enable_cache: bool = False
    cache_redis_enabled: bool = False
    cache_redis_ttl_seconds: int = 0
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "synapse"
    redis_socket_timeout_seconds: float = 0.2
    redis_connect_timeout_seconds: float = 0.2
    session_store_ttl_seconds: int = 7 * 24 * 3600
    session_read_through_enabled: bool = True
    allow_general_knowledge_fallback: bool = True
    min_verification_threshold: float = 0.1
    enable_collection_query_anchoring: bool = True
    collection_anchor_terms: List[str] = field(default_factory=list)
    structured_query_fast_mode: Optional[bool] = None
    structured_entity_resolution: bool = True
    structured_natural_response_style: bool = True
    dense_default_filters: Optional[Dict[str, Any]] = None
    max_active_pipelines: int = 2
    llm_backend: str = "local"  # "local" or "api"
    api_model_name: str = "gemini-2.5-flash-lite"


class SharedResources:
    """Holds shared, heavyweight resources (LLM + query embedder + index embedder)."""

    def __init__(self, settings: MultiKBSettings, init_index_embedder: bool = True):
        self.settings = settings
        self.llm_generator = self._init_llm()
        self.query_embedder = self._init_query_embedder()
        self.reranker = self._init_reranker()
        self.context_verifier = self._init_context_verifier()
        self.index_embedder = self._init_index_embedder() if init_index_embedder else None
        self.redis_connection = self._init_redis()
        self.lighter_model, self.lighter_tokenizer = self._init_lighter_llm()

    def _init_lighter_llm(self):
        """No longer pre-loads lighter LLM as intent detection LLM is removed."""
        logger.info("Lighter LLM pre-loading is disabled (intent detection LLM removed)")
        return None, None

    def _init_llm(self) -> LLMGenerator:
        gen_config = GenerationConfig(
            model_name=self.settings.llm_model,
            max_new_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            repetition_penalty=self.settings.llm_repetition_penalty,
            presence_penalty=self.settings.llm_presence_penalty,
            frequency_penalty=self.settings.llm_frequency_penalty,
            device="cuda" if self.settings.use_gpu else "cpu",
            normalize_newlines=self.settings.normalize_newlines,
            llm_backend=self.settings.llm_backend,
            api_model_name=self.settings.api_model_name,
        )
        return LLMGenerator(config=gen_config)

    def _init_query_embedder(self) -> SentenceTransformersTextEmbedder:
        device = ComponentDevice.from_str("cuda" if self.settings.use_gpu else "cpu")
        embedder = SentenceTransformersTextEmbedder(
            model=self.settings.embedding_model,
            device=device,
            normalize_embeddings=True,
        )
        embedder.warm_up()
        return embedder

    def _init_index_embedder(self) -> EmbeddingGenerator:
        cache_dir = str(Path(self.settings.data_dir) / "embeddings_cache")
        return EmbeddingGenerator(
            model_name=self.settings.embedding_model,
            show_progress=False,
            cache_dir=cache_dir,
        )

    def _init_reranker(self) -> Reranker:
        return Reranker(
            top_k=self.settings.rerank_top_k,
            use_gpu=self.settings.use_gpu,
        )

    def _init_context_verifier(self) -> ContextVerifier:
        return ContextVerifier(
            threshold=self.settings.nli_threshold,
            dedup_threshold=self.settings.dedup_threshold,
            use_gpu=self.settings.use_gpu,
            embedding_model=self.settings.embedding_model,
        )

    def _init_redis(self) -> Optional["RedisConnection"]:
        if not self.settings.redis_enabled:
            return None
        
        try:
            from src.concurrency import RedisRuntimeConfig, RedisConnection
            runtime = RedisRuntimeConfig(
                enabled=True,
                url=self.settings.redis_url,
                key_prefix=self.settings.redis_key_prefix,
                socket_timeout_seconds=self.settings.redis_socket_timeout_seconds,
                connect_timeout_seconds=self.settings.redis_connect_timeout_seconds,
            )
            conn = RedisConnection(runtime)
            if conn.init_error:
                logger.warning(f"Redis connection init error: {conn.init_error}")
            return conn
        except Exception as e:
            logger.warning(f"Failed to initialize Redis connection in SharedResources: {e}")
            return None


class SharedEmbedderDenseRetriever(DenseRetriever):
    """DenseRetriever variant that reuses a shared embedder."""

    def __init__(self, *args, shared_embedder: SentenceTransformersTextEmbedder, **kwargs):
        self._shared_embedder = shared_embedder
        super().__init__(*args, **kwargs)

    def _init_embedder(self, use_gpu: bool):  # pylint: disable=unused-argument
        self.embedder = self._shared_embedder
        logger.info("DenseRetriever using shared embedder")


class SharedResourceConversationalPipeline(ConversationalRAGPipeline):
    """Conversational pipeline that reuses shared LLM and query embedder."""

    def __init__(
        self,
        collection_name: str,
        config: ConversationalPipelineConfig,
        shared_resources: SharedResources,
        bm25_cache_dir: str,
    ):
        self._shared_resources = shared_resources
        self._bm25_cache_dir = bm25_cache_dir
        super().__init__(collection_name=collection_name, config=config)

    def _init_retrieval(self):
        logger.info("\nInitializing Retrieval (shared embedder)")
        logger.info("-" * 80)

        if self.config.qdrant_url:
            self.dense_retriever = SharedEmbedderDenseRetriever(
                collection_name=self.collection_name,
                embedding_model=self._shared_resources.settings.embedding_model,
                qdrant_url=self.config.qdrant_url,
                top_k=self.config.dense_top_k,
                use_gpu=self.config.use_gpu,
                default_filters=self.config.dense_default_filters,
                shared_embedder=self._shared_resources.query_embedder,
            )

            self.sparse_retriever = SparseRetriever(
                collection_name=self.collection_name,
                qdrant_url=self.config.qdrant_url,
                top_k=self.config.sparse_top_k,
                index_cache_path=self._bm25_cache_dir,
            )
        elif self.config.storage_path:
            self.dense_retriever = SharedEmbedderDenseRetriever(
                collection_name=self.collection_name,
                embedding_model=self._shared_resources.settings.embedding_model,
                storage_path=self.config.storage_path,
                top_k=self.config.dense_top_k,
                use_gpu=self.config.use_gpu,
                default_filters=self.config.dense_default_filters,
                shared_embedder=self._shared_resources.query_embedder,
            )

            self.sparse_retriever = SparseRetriever(
                collection_name=self.collection_name,
                storage_path=self.config.storage_path,
                top_k=self.config.sparse_top_k,
                index_cache_path=self._bm25_cache_dir,
            )
        else:
            raise ValueError("Must provide either qdrant_url or storage_path")

        self.result_fusion = ResultFusion(
            strategy=self.config.fusion_strategy,
            top_k=self.config.fusion_top_k,
        )

        self.reranker = self._shared_resources.reranker

    def _init_context_handling(self):
        self.context_verifier = self._shared_resources.context_verifier

    def _init_generation(self):
        logger.info("\nInitializing Generation (shared LLM)")
        logger.info("-" * 80)

        self.prompt_builder = RAGPromptBuilder()
        self.llm_generator = self._shared_resources.llm_generator
        self.citation_extractor = CitationExtractor()

        if self.config.enable_query_enhancement:
            from src.query_processing.query_enhancer import QueryEnhancer
            self.query_enhancer = QueryEnhancer(
                existing_model=self._shared_resources.lighter_model,
                existing_tokenizer=self._shared_resources.lighter_tokenizer
            )
            logger.info("QueryEnhancer using shared lighter LLM resources")


class MultiKBPipeline:
    """Routes requests to per-KB pipelines with shared heavyweight resources."""

    def __init__(
        self,
        registry: KBRegistry,
        settings: MultiKBSettings,
        shared_resources: SharedResources,
    ):
        self.registry = registry
        self.settings = settings
        self.shared_resources = shared_resources

        self.data_dir = Path(self.settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir = self.data_dir / "conversations"
        self.cache_dir = self.data_dir / "query_cache"
        self.bm25_cache_dir = self.data_dir / "bm25_indices"
        self.kb_config_dir = self.data_dir / "kb_configs"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_config_dir.mkdir(parents=True, exist_ok=True)

        self._pipelines: "OrderedDict[str, ConversationalRAGPipeline]" = OrderedDict()
        self._lock = threading.RLock()

    def _build_config(self, kb_id: str, collection_name: str) -> ConversationalPipelineConfig:
        memory_directory = str(self.conversations_dir / kb_id)
        cache_file = str(self.cache_dir / f"{kb_id}.json")

        config = ConversationalPipelineConfig(
            use_gpu=self.settings.use_gpu,
            rerank_top_k=self.settings.rerank_top_k,
            nli_threshold=self.settings.nli_threshold,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            repetition_penalty=self.settings.llm_repetition_penalty,
            presence_penalty=self.settings.llm_presence_penalty,
            frequency_penalty=self.settings.llm_frequency_penalty,
            max_tokens=self.settings.llm_max_tokens,
            llm_model=self.settings.llm_model,
            qdrant_url=self.settings.qdrant_url,
            normalize_newlines=self.settings.normalize_newlines,
            enable_memory=self.settings.enable_memory,
            memory_directory=memory_directory,
            max_turns=self.settings.max_turns,
            history_in_prompt_turns=self.settings.history_in_prompt_turns,
            auto_save=self.settings.auto_save,
            enable_reformulation=self.settings.enable_reformulation,
            use_llm_reformulation=self.settings.use_llm_reformulation,
            enable_cache=self.settings.enable_cache,
            cache_file=cache_file,
            cache_redis_enabled=self.settings.cache_redis_enabled,
            cache_redis_ttl_seconds=self.settings.cache_redis_ttl_seconds,
            redis_enabled=self.settings.redis_enabled,
            redis_url=self.settings.redis_url,
            redis_key_prefix=self.settings.redis_key_prefix,
            redis_socket_timeout_seconds=self.settings.redis_socket_timeout_seconds,
            redis_connect_timeout_seconds=self.settings.redis_connect_timeout_seconds,
            session_store_ttl_seconds=self.settings.session_store_ttl_seconds,
            session_read_through_enabled=self.settings.session_read_through_enabled,
            redis_session_namespace=kb_id,
            allow_general_knowledge_fallback=self.settings.allow_general_knowledge_fallback,
            min_verification_threshold=self.settings.min_verification_threshold,
            enable_collection_query_anchoring=self.settings.enable_collection_query_anchoring,
            collection_anchor_terms=self.settings.collection_anchor_terms,
            structured_query_fast_mode=self.settings.structured_query_fast_mode,
            structured_entity_resolution=self.settings.structured_entity_resolution,
            structured_natural_response_style=self.settings.structured_natural_response_style,
            dense_default_filters=self.settings.dense_default_filters,
        )

        snapshot, snapshot_path = self._load_kb_snapshot(kb_id)
        resolved_fast_mode = self._resolve_structured_fast_mode(collection_name, snapshot)

        if snapshot is not None:
            if snapshot.prompt.template:
                config.prompt_template = snapshot.prompt.template
            if snapshot.prompt.template_name:
                config.prompt_template_name = snapshot.prompt.template_name

            config.max_tokens = snapshot.generation.max_tokens
            config.temperature = snapshot.generation.temperature
            config.top_p = snapshot.generation.top_p
            if snapshot.generation.repetition_penalty is not None:
                config.repetition_penalty = snapshot.generation.repetition_penalty
            if snapshot.generation.presence_penalty is not None:
                config.presence_penalty = snapshot.generation.presence_penalty
            if snapshot.generation.frequency_penalty is not None:
                config.frequency_penalty = snapshot.generation.frequency_penalty
            if snapshot.generation.normalize_newlines is not None:
                config.normalize_newlines = snapshot.generation.normalize_newlines
            config.enable_min_tokens_strategy = snapshot.generation.enable_min_tokens_strategy
            config.min_tokens_long_response = snapshot.generation.min_tokens_long_response
            config.long_response_max_tokens = snapshot.generation.long_response_max_tokens

            config.allow_general_knowledge_fallback = snapshot.grounding.allow_general_knowledge_fallback
            config.min_verification_threshold = snapshot.grounding.min_verification_threshold
            config.enable_collection_query_anchoring = snapshot.grounding.enable_collection_query_anchoring
            config.collection_anchor_terms = snapshot.grounding.collection_anchor_terms

            config.structured_entity_resolution = snapshot.routing.structured_entity_resolution
            config.structured_natural_response_style = snapshot.routing.structured_natural_response_style

            config.canned_responses = snapshot.query_handler.canned_responses

        if resolved_fast_mode is not None:
            config.structured_query_fast_mode = resolved_fast_mode

        if snapshot is not None and resolved_fast_mode is not None:
            self._persist_resolved_fast_mode(snapshot, snapshot_path, resolved_fast_mode)

        return config

    def _create_pipeline(self, kb_id: str, collection_name: str) -> ConversationalRAGPipeline:
        config = self._build_config(kb_id, collection_name)
        return SharedResourceConversationalPipeline(
            collection_name=collection_name,
            config=config,
            shared_resources=self.shared_resources,
            bm25_cache_dir=str(self.bm25_cache_dir),
        )

    def _load_kb_snapshot(self, kb_id: str) -> tuple[Optional[KBConfigSnapshot], Optional[Path]]:
        kb = self.registry.get_kb(kb_id)
        snapshot_path: Optional[Path] = None
        if kb and kb.get("config_path"):
            snapshot_path = Path(str(kb["config_path"]))
            if not snapshot_path.is_absolute():
                # Try resolving relative to repo root first, then data_dir.
                repo_root = self.data_dir.parent
                candidate = (repo_root / snapshot_path).resolve()
                if candidate.exists():
                    snapshot_path = candidate
                else:
                    snapshot_path = (self.data_dir / snapshot_path).resolve()
        else:
            snapshot_path = self.kb_config_dir / f"{kb_id}.yaml"

        if snapshot_path and snapshot_path.exists():
            try:
                data = load_yaml_file(snapshot_path)
                return KBConfigSnapshot(**data), snapshot_path
            except Exception:
                logger.warning("Failed to load KB snapshot for %s", kb_id, exc_info=True)
                return None, snapshot_path

        return None, snapshot_path

    def _resolve_structured_fast_mode(
        self,
        collection_name: str,
        snapshot: Optional[KBConfigSnapshot],
    ) -> Optional[bool]:
        if snapshot and snapshot.resolved_structured_query_fast_mode is not None:
            return snapshot.resolved_structured_query_fast_mode

        raw_mode = None
        if snapshot is not None:
            raw_mode = snapshot.routing.structured_query_fast_mode
        elif self.settings.structured_query_fast_mode is not None:
            raw_mode = self.settings.structured_query_fast_mode

        if isinstance(raw_mode, bool):
            return raw_mode

        if isinstance(raw_mode, str) and raw_mode.strip().lower() == "auto":
            keys = ["entity_names", "document_entity_counts", "is_first_chunk"]
            try:
                vector_store = VectorStore(
                    collection_name=collection_name,
                    embedding_dim=self.settings.embedding_dim,
                    qdrant_url=self.settings.qdrant_url,
                    recreate_collection=False,
                )
                missing = any(vector_store.has_missing_payload_key(key) for key in keys)
                return not missing
            except Exception:
                logger.warning("Structured fast-mode probe failed for %s", collection_name, exc_info=True)
                return False

        return None

    def _persist_resolved_fast_mode(
        self,
        snapshot: KBConfigSnapshot,
        snapshot_path: Optional[Path],
        resolved_fast_mode: Optional[bool],
    ) -> None:
        if snapshot_path is None or resolved_fast_mode is None:
            return
        if snapshot.resolved_structured_query_fast_mode is not None:
            return
        snapshot.resolved_structured_query_fast_mode = resolved_fast_mode
        try:
            data = snapshot.model_dump() if hasattr(snapshot, "model_dump") else snapshot.dict()
            import yaml  # type: ignore

            tmp_path = snapshot_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
            tmp_path.replace(snapshot_path)
        except Exception:
            logger.warning("Failed to persist resolved fast mode for %s", snapshot.kb_id, exc_info=True)

    def get_pipeline_for_kb(self, kb_id: str) -> ConversationalRAGPipeline:
        with self._lock:
            if kb_id in self._pipelines:
                pipeline = self._pipelines.pop(kb_id)
                self._pipelines[kb_id] = pipeline
                return pipeline

            kb = self.registry.get_kb(kb_id)
            if not kb:
                raise ValueError(f"KB not found: {kb_id}")

            pipeline = self._create_pipeline(kb_id, kb["collection_name"])
            self._pipelines[kb_id] = pipeline

            if len(self._pipelines) > self.settings.max_active_pipelines:
                evicted_kb, _ = self._pipelines.popitem(last=False)
                logger.info("Evicted pipeline cache for KB: %s", evicted_kb)

            return pipeline

    async def resolve_mcp_context_async(
        self,
        kb_id: str,
        query: str,
        user_id: Optional[str],
        connector: Optional[str],
        session_id: Optional[str] = None,
        google_file_id: Optional[str] = None,
        google_file_name: Optional[str] = None,
        google_calendar_id: Optional[str] = None,
        google_calendar_name: Optional[str] = None,
        gmail_location: Optional[str] = None,
        gmail_category: Optional[str] = None,
        outlook_folder: Optional[str] = None,
        outlook_location: Optional[str] = None,
        outlook_message_id: Optional[str] = None,
        microsoft_file_id: Optional[str] = None,
        microsoft_file_name: Optional[str] = None,
        microsoft_drive_path: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        pipeline = self.get_pipeline_for_kb(kb_id)
        if hasattr(pipeline, "resolve_mcp_context_async"):
            return await pipeline.resolve_mcp_context_async(
                query=query,
                user_id=user_id,
                connector=connector,
                session_id=session_id,
                google_file_id=google_file_id,
                google_file_name=google_file_name,
                google_calendar_id=google_calendar_id,
                google_calendar_name=google_calendar_name,
                gmail_location=gmail_location,
                gmail_category=gmail_category,
                outlook_folder=outlook_folder,
                outlook_location=outlook_location,
                outlook_message_id=outlook_message_id,
                microsoft_file_id=microsoft_file_id,
                microsoft_file_name=microsoft_file_name,
                microsoft_drive_path=microsoft_drive_path,
            )
        return None, None, None

    def evict_kb(self, kb_id: str) -> bool:
        with self._lock:
            return self._pipelines.pop(kb_id, None) is not None

    def create_session(self, kb_id: str) -> Dict[str, Any]:
        pipeline = self.get_pipeline_for_kb(kb_id)
        if pipeline.session_manager is None:
            raise ValueError("Memory is disabled for this server")
        session_id, _ = pipeline.session_manager.get_or_create_session(None)
        try:
            pipeline.save_session(session_id)
        except Exception:
            logger.warning("Failed to persist new session: %s", session_id, exc_info=True)

        return {"status": "success", "session_id": session_id}

    def session_exists(self, kb_id: str, session_id: str) -> bool:
        pipeline = self.get_pipeline_for_kb(kb_id)
        return pipeline.get_session(session_id) is not None

    def query(
        self,
        kb_id: str,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        timeout_seconds: Optional[float] = None,
        web: str = "off",
        kb: str = "on",
        connector: Optional[str] = None,
        mcp_context: Optional[str] = None,
        mcp_service: Optional[str] = None,
        mcp_tool: Optional[str] = None,
        google_file_id: Optional[str] = None,
        google_file_name: Optional[str] = None,
        google_calendar_id: Optional[str] = None,
        google_calendar_name: Optional[str] = None,
        gmail_location: Optional[str] = None,
        gmail_category: Optional[str] = None,
        outlook_folder: Optional[str] = None,
        outlook_location: Optional[str] = None,
        outlook_message_id: Optional[str] = None,
        microsoft_file_id: Optional[str] = None,
        microsoft_file_name: Optional[str] = None,
        microsoft_drive_path: Optional[str] = None,
    ) -> ConversationalPipelineResult:
        pipeline = self.get_pipeline_for_kb(kb_id)
        return pipeline.run(
            query=query,
            session_id=session_id,
            user_id=user_id,
            cancel_event=cancel_event,
            web=web,
            kb=kb,
            connector=connector,
            mcp_context=mcp_context,
            mcp_service=mcp_service,
            mcp_tool=mcp_tool,
            google_file_id=google_file_id,
            google_file_name=google_file_name,
            google_calendar_id=google_calendar_id,
            google_calendar_name=google_calendar_name,
            gmail_location=gmail_location,
            gmail_category=gmail_category,
            outlook_folder=outlook_folder,
            outlook_location=outlook_location,
            outlook_message_id=outlook_message_id,
            microsoft_file_id=microsoft_file_id,
            microsoft_file_name=microsoft_file_name,
            microsoft_drive_path=microsoft_drive_path,
        )

    def get_session_history(self, kb_id: str, session_id: str, n: Optional[int] = None):
        pipeline = self.get_pipeline_for_kb(kb_id)
        return pipeline.get_session_history(session_id, n)

    def delete_session(self, kb_id: str, session_id: str) -> bool:
        pipeline = self.get_pipeline_for_kb(kb_id)
        return pipeline.delete_session(session_id)

    def list_sessions(self, kb_id: str):
        pipeline = self.get_pipeline_for_kb(kb_id)
        return pipeline.list_sessions()

    def stats(self) -> Dict[str, Any]:
        return {
            "active_pipelines": len(self._pipelines),
            "max_active_pipelines": self.settings.max_active_pipelines,
        }
