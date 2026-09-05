"""
Query Router - Step 3 of Query Processing
Routes queries to appropriate pipeline based on classification results
Provides logging and metrics for routing decisions

Supports two backends:
1. Custom (default): Lightweight, zero-dependency routing
2. Haystack: Integration with Haystack's ConditionalRouter
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Literal

# Import classifier
from src.query_processing.query_classifier import QueryClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Haystack import
try:
    from haystack.components.routers import ConditionalRouter as HaystackRouter

    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    logger.debug("Haystack not available - using custom routing only")


@dataclass
class RoutingDecision:
    """
    Routing decision with metadata
    """
    query_id: str
    query_text: str
    route: str  # rag_pipeline, direct_llm, hybrid, reject
    reason: str  # Why this route was chosen
    confidence: float  # Confidence in routing decision
    query_type: str  # Original classification type
    complexity: float  # Query complexity score
    timestamp: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class QueryRouter:
    """
    Routes queries to appropriate pipeline based on classification
    Implements separation of concerns: routing logic separated from classification
    
    Supports two backends:
    - 'custom': Custom priority-based routing (default)
    - 'haystack': Haystack ConditionalRouter integration
    """

    def __init__(
            self,
            backend: Literal["custom", "haystack"] = "custom",
            log_decisions: bool = True,
            log_dir: str = "./logs/routing",
            enable_metrics: bool = True
    ):
        """
        Initialize query router
        
        Args:
            backend: Routing backend ('custom' or 'haystack')
            log_decisions: Whether to log routing decisions
            log_dir: Directory for routing logs
            enable_metrics: Whether to track routing metrics
        """
        self.backend = backend
        self.log_decisions = log_decisions
        self.log_dir = Path(log_dir)
        self.enable_metrics = enable_metrics
        self.haystack_router = None

        # Validate backend
        if backend == "haystack" and not HAYSTACK_AVAILABLE:
            raise ImportError(
                "Haystack backend requested but haystack-ai is not installed. "
                "Install with: pip install haystack-ai"
            )

        # Initialize Haystack router if using haystack backend
        if backend == "haystack":
            self._init_haystack_router()

        # Create log directory
        if self.log_decisions:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.decision_log_file = self.log_dir / "routing_decisions.jsonl"
            logger.info(f"Routing decisions will be logged to: {self.decision_log_file}")

        # Metrics tracking
        self.metrics = {
            "total_routed": 0,
            "routes": {
                "rag_pipeline": 0,
                "direct_llm": 0,
                "hybrid": 0,
                "reject": 0
            },
            "by_query_type": {},
            "by_complexity": {
                "simple": 0,  # < 0.3
                "moderate": 0,  # 0.3-0.6
                "complex": 0  # > 0.6
            }
        }

        logger.info("=" * 80)
        logger.info("QueryRouter initialized")
        logger.info("=" * 80)
        logger.info(f"Backend: {backend}")
        logger.info(f"Logging enabled: {log_decisions}")
        logger.info(f"Metrics enabled: {enable_metrics}")
        logger.info("=" * 80)

    def _init_haystack_router(self) -> None:
        """
        Initialize Haystack ConditionalRouter with routing rules
        """
        logger.info("Initializing Haystack ConditionalRouter...")

        # Define routing rules for Haystack
        # Haystack uses template syntax {{variable}}
        routes = [
            {
                "condition": "{{is_generic and confidence < 0.5}}",
                "output": "{{query}}",
                "output_name": "reject",
                "output_type": str,
            },
            {
                "condition": "{{query_type == 'out_of_scope' and confidence > 0.7}}",
                "output": "{{query}}",
                "output_name": "reject",
                "output_type": str,
            },
            {
                "condition": "{{query_type == 'conversational' and confidence > 0.7}}",
                "output": "{{query}}",
                "output_name": "direct_llm",
                "output_type": str,
            },
            {
                "condition": "{{query_type == 'factual'}}",
                "output": "{{query}}",
                "output_name": "rag_pipeline",
                "output_type": str,
            },
            {
                "condition": "{{query_type == 'hybrid'}}",
                "output": "{{query}}",
                "output_name": "hybrid",
                "output_type": str,
            },
            {
                "condition": "{{query_type == 'reasoning' and complexity > 0.6}}",
                "output": "{{query}}",
                "output_name": "hybrid",
                "output_type": str,
            },
            {
                "condition": "{{query_type == 'reasoning'}}",
                "output": "{{query}}",
                "output_name": "rag_pipeline",
                "output_type": str,
            },
        ]

        try:
            self.haystack_router = HaystackRouter(routes=routes)
            logger.info(" Haystack ConditionalRouter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Haystack router: {e}")
            raise

    def route(
            self,
            classification: Any,  # QueryClassification object
            query_id: Optional[str] = None,
            connector: Optional[str] = None
    ) -> RoutingDecision:
        """
        Route a query based on its classification
        
        Args:
            classification: QueryClassification object from classifier
            query_id: Optional query identifier
            connector: Optional connector name (google, email, calendar, slack, notion)
            
        Returns:
            RoutingDecision object
        """
        # Extract classification details
        query_text = classification.query_text
        query_type = classification.query_type
        confidence = classification.confidence
        complexity = classification.complexity
        is_generic = classification.metadata.get('is_generic', False)

        # Check if a Google Workspace connector is selected
        # If so, route to Google Workspace MCP path instead of RAG
        google_workspace_connectors = ['google', 'email', 'calendar']
        microsoft365_connectors = ['outlook', 'onedrive', 'sharepoint']
        if connector and connector.lower() in google_workspace_connectors:
            route = 'google_workspace_mcp'
            reason = f'Google Workspace connector "{connector}" is selected - routing to Google Workspace MCP'
        elif connector and connector.lower() in microsoft365_connectors:
            route = 'microsoft365_mcp'
            reason = f'Microsoft 365 connector "{connector}" is selected - routing to Microsoft 365 MCP'
        else:
            # Determine routing based on backend
            if self.backend == "haystack":
                route, reason = self._route_with_haystack(
                    query_text=query_text,
                    query_type=query_type,
                    confidence=confidence,
                    complexity=complexity,
                    is_generic=is_generic
                )
            else:  # custom backend
                route, reason = self._determine_route(
                    query_type=query_type,
                    confidence=confidence,
                    complexity=complexity,
                    is_generic=is_generic
                )

        # Generate query ID if not provided
        if not query_id:
            import hashlib
            timestamp = datetime.utcnow().isoformat()
            query_id = hashlib.md5(f"{query_text}_{timestamp}".encode()).hexdigest()[:16]

        # Create routing decision
        decision = RoutingDecision(
            query_id=query_id,
            query_text=query_text,
            route=route,
            reason=reason,
            confidence=confidence,
            query_type=query_type,
            complexity=complexity,
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata={
                "type_scores": classification.type_scores,
                "is_generic": classification.metadata.get('is_generic', False),
                "complexity_metadata": classification.metadata.get('complexity_metadata', {})
            }
        )

        # Update metrics
        if self.enable_metrics:
            self._update_metrics(decision)

        # Log decision
        if self.log_decisions:
            self._log_decision(decision)

        logger.info(f" Routed query to: {route}")
        logger.info(f"   Reason: {reason}")
        logger.info(f"   Query type: {query_type} (confidence: {confidence:.2f})")

        return decision

    def _route_with_haystack(
            self,
            query_text: str,
            query_type: str,
            confidence: float,
            complexity: float,
            is_generic: bool
    ) -> tuple[str, str]:
        """
        Route using Haystack ConditionalRouter
        
        Args:
            query_text: Query text
            query_type: Classified query type
            confidence: Classification confidence
            complexity: Query complexity score
            is_generic: Whether query is too generic
            
        Returns:
            Tuple of (route, reason)
        """
        try:
            # Prepare variables for Haystack router
            variables = {
                "query": query_text,
                "query_type": query_type,
                "confidence": confidence,
                "complexity": complexity,
                "is_generic": is_generic
            }

            # Run Haystack router
            result = self.haystack_router.run(**variables)

            # Extract route from Haystack result
            # Haystack returns dict with output names as keys
            if result:
                # Get the first non-empty output
                for output_name, output_value in result.items():
                    if output_value:
                        route = output_name
                        reason = f"Haystack ConditionalRouter: {query_type} query routed to {route}"
                        return route, reason

            # Fallback if no route matched
            return "rag_pipeline", "Haystack router: default fallback to RAG"

        except Exception as e:
            logger.error(f"Haystack routing failed: {e}")
            logger.warning("Falling back to custom routing")
            # Fallback to custom routing
            return self._determine_route(query_type, confidence, complexity, is_generic)

    @staticmethod
    def _determine_route(
            query_type: str,
            confidence: float,
            complexity: float,
            is_generic: bool
    ) -> tuple[str, str]:
        """
        Determine routing decision and reason
        
        Args:
            query_type: Classified query type
            confidence: Classification confidence
            complexity: Query complexity score
            is_generic: Whether query is too generic
            
        Returns:
            Tuple of (route, reason)
        """
        # Priority 1: Reject generic queries with low confidence
        if is_generic and confidence < 0.5:
            return "reject", "Query too generic/vague with low classification confidence"

        # Priority 2: Out-of-scope with high confidence
        if query_type == "out_of_scope" and confidence > 0.7:
            return "reject", "Query classified as out-of-scope with high confidence"

        # Priority 3: Conversational with high confidence -> Direct LLM
        if query_type == "conversational" and confidence > 0.7:
            return "direct_llm", "Conversational query - no RAG needed"

        # Priority 4: Factual queries -> RAG Pipeline
        if query_type == "factual":
            return "rag_pipeline", "Factual query requiring knowledge retrieval"

        # Priority 5: Hybrid queries -> Hybrid approach
        if query_type == "hybrid":
            return "hybrid", "Hybrid query requiring both RAG and LLM reasoning"

        # Priority 6: Complex reasoning -> Hybrid
        if query_type == "reasoning" and complexity > 0.6:
            return "hybrid", "Complex reasoning query requiring RAG + LLM"

        # Priority 7: Simple reasoning -> RAG
        if query_type == "reasoning":
            return "rag_pipeline", "Reasoning query - try RAG first"

        # Default: Route to RAG pipeline (let retrieval determine relevance)
        return "rag_pipeline", f"Default routing for {query_type} query - trying RAG"

    def _update_metrics(self, decision: RoutingDecision) -> None:
        """
        Update routing metrics
        
        Args:
            decision: RoutingDecision object
        """
        self.metrics["total_routed"] += 1

        # Update route counts
        if decision.route in self.metrics["routes"]:
            self.metrics["routes"][decision.route] += 1

        # Update by query type
        if decision.query_type not in self.metrics["by_query_type"]:
            self.metrics["by_query_type"][decision.query_type] = 0
        self.metrics["by_query_type"][decision.query_type] += 1

        # Update by complexity
        if decision.complexity < 0.3:
            self.metrics["by_complexity"]["simple"] += 1
        elif decision.complexity < 0.6:
            self.metrics["by_complexity"]["moderate"] += 1
        else:
            self.metrics["by_complexity"]["complex"] += 1

    def _log_decision(self, decision: RoutingDecision) -> None:
        """
        Log routing decision to file
        
        Args:
            decision: RoutingDecision object
        """
        try:
            with open(self.decision_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to log routing decision: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current routing metrics
        
        Returns:
            Dictionary with routing metrics
        """
        if not self.enable_metrics:
            return {"metrics_enabled": False}

        # Calculate percentages
        total = self.metrics["total_routed"]
        if total == 0:
            return self.metrics

        metrics_with_percentages = {
            "total_routed": total,
            "routes": {
                route: {
                    "count": count,
                    "percentage": round((count / total) * 100, 2)
                }
                for route, count in self.metrics["routes"].items()
            },
            "by_query_type": self.metrics["by_query_type"],
            "by_complexity": self.metrics["by_complexity"]
        }

        return metrics_with_percentages

    def reset_metrics(self) -> None:
        """Reset routing metrics"""
        self.metrics = {
            "total_routed": 0,
            "routes": {
                "rag_pipeline": 0,
                "direct_llm": 0,
                "hybrid": 0,
                "reject": 0
            },
            "by_query_type": {},
            "by_complexity": {
                "simple": 0,
                "moderate": 0,
                "complex": 0
            }
        }
        logger.info("Routing metrics reset")

    def get_routing_summary(self) -> str:
        """
        Get a human-readable summary of routing metrics
        
        Returns:
            Formatted string with routing summary
        """
        metrics = self.get_metrics()

        if not self.enable_metrics:
            return "Metrics tracking is disabled"

        total = metrics["total_routed"]
        if total == 0:
            return "No queries routed yet"

        summary = f"Routing Summary ({total} queries):\n"
        summary += "=" * 50 + "\n"

        # Routes breakdown
        summary += "\nRoutes:\n"
        for route, data in metrics["routes"].items():
            if isinstance(data, dict):
                count = data["count"]
                pct = data["percentage"]
                summary += f"  {route:15s}: {count:4d} ({pct:5.1f}%)\n"
            else:
                summary += f"  {route:15s}: {data:4d}\n"

        # Complexity breakdown
        summary += "\nBy Complexity:\n"
        for level, count in metrics["by_complexity"].items():
            pct = (count / total) * 100 if total > 0 else 0
            summary += f"  {level:10s}: {count:4d} ({pct:5.1f}%)\n"

        return summary


# Convenience function
def create_query_router(
        backend: Literal["custom", "haystack"] = "custom",
        **kwargs
) -> QueryRouter:
    """
    Convenience function to create a query router
    
    Args:
        backend: Routing backend ('custom' or 'haystack')
        **kwargs: Additional arguments for QueryRouter
        
    Returns:
        QueryRouter instance
    """
    return QueryRouter(backend=backend, **kwargs)
