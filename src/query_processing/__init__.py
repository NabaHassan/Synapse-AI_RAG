"""
Query Processing Module
Handles query intake, validation, classification, routing, enhancement, and reformulation.
"""

from .query_handler import (
    QueryHandler,
    QueryValidator,
    QueryLogger,
    QueryPreprocessor,
    Query,
    QueryMetadata,
    create_query_handler
)

from .query_classifier import (
    QueryClassifier,
    QueryClassification,
    ComplexityAnalyzer,
    GenericQueryDetector,
    create_query_classifier
)

from .query_reformulator import (
    QueryReformulator,
    ReformulatorConfig,
    ReformulationResult,
    create_query_reformulator
)

from .follow_up_detector import (
    FollowUpDetector,
    FollowUpResult,
    create_follow_up_detector
)

# Query Classification and Routing (NEW)
from .query_types import (
    QueryType,
    QueryClassificationResult
)

from .query_classifier_enhanced import (
    QueryClassifierEnhanced
)

from .meta_handler import (
    MetaConversationHandler
)

from .formatting_handler import (
    FormattingRequestHandler
)

from .clarification_handler import (
    ClarificationHandler,
    classify_clarification_request,
)

__all__ = [
    # Query Handler
    "QueryHandler",
    "QueryValidator",
    "QueryLogger",
    "QueryPreprocessor",
    "Query",
    "QueryMetadata",
    "create_query_handler",
    # Query Classifier
    "QueryClassifier",
    "QueryClassification",
    "ComplexityAnalyzer",
    "GenericQueryDetector",
    "create_query_classifier",
    # Query Reformulator
    "QueryReformulator",
    "ReformulatorConfig",
    "ReformulationResult",
    "create_query_reformulator",
    # Follow-Up Detector
    "FollowUpDetector",
    "FollowUpResult",
    "create_follow_up_detector",
    # Query Classification and Routing
    "QueryType",
    "QueryClassificationResult",
    "QueryClassifierEnhanced",
    "MetaConversationHandler",
    "FormattingRequestHandler",
    "ClarificationHandler",
    "classify_clarification_request",
]
