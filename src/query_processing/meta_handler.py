"""
Meta-Conversation Handler for Conversational RAG.

This module handles meta-conversation queries - questions about the
conversation itself that should be answered directly from memory
without performing any retrieval.

Examples:
- "What was my first question?"
- "What did we discuss?"
- "Summarize our conversation"
"""

import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


class MetaConversationHandler:
    """
    Handler for meta-conversation queries.
    
    Answers questions about the conversation itself by reading from
    conversation history. NO retrieval is performed.
    
    This saves resources and provides instant, accurate answers.
    """

    def __init__(self):
        """Initialize the meta-conversation handler."""
        logger.info("MetaConversationHandler initialized")

    def handle(
            self,
            query: str,
            conversation_history: List[Any]
    ) -> str:
        """
        Handle a meta-conversation query.
        
        Args:
            query: The user's meta-question
            conversation_history: List of ConversationTurn objects
            
        Returns:
            Answer string based on conversation history
        """
        query_lower = query.lower().strip()

        # Handle empty history
        if not conversation_history:
            return self._handle_empty_history(query_lower)

        # "What was my first question?"
        if "first question" in query_lower or "initial question" in query_lower:
            return self._handle_first_question(conversation_history)

        # "What was my previous/last question?"
        if "previous question" in query_lower or "last question" in query_lower or "earlier question" in query_lower:
            return self._handle_previous_question(conversation_history)

        # "What did I ask?" (general)
        if "what did i ask" in query_lower or "what did i say" in query_lower:
            return self._handle_what_did_i_ask(conversation_history)

        # "What have we discussed?" or "What did we talk about?"
        if "discussed" in query_lower or "talked about" in query_lower or "covered" in query_lower:
            return self._handle_what_discussed(conversation_history)

        # "Summarize our conversation" or "Recap our chat"
        if "summarize" in query_lower or "recap" in query_lower or "summary" in query_lower:
            return self._handle_summarize(conversation_history)

        # Default: provide general conversation info
        return self._handle_general_meta(conversation_history)

    def _handle_empty_history(self, query_lower: str) -> str:
        """Handle meta-questions when there's no history."""
        if "first" in query_lower or "previous" in query_lower:
            return "This is the first question in our conversation."
        return "We haven't discussed anything yet. This is the start of our conversation."

    def _handle_first_question(self, history: List[Any]) -> str:
        """Handle 'what was my first question?'"""
        first_turn = history[0]
        return f'Your first question was: "{first_turn.query}"'

    def _handle_previous_question(self, history: List[Any]) -> str:
        """Handle 'what was my previous question?'"""
        if len(history) < 2:
            return "There is no previous question. This is your first or second question."

        # Previous question is the second-to-last turn
        previous_turn = history[-2]
        return f'Your previous question was: "{previous_turn.query}"'

    def _handle_what_did_i_ask(self, history: List[Any]) -> str:
        """Handle 'what did I ask?'"""
        if len(history) == 1:
            return f'You asked: "{history[0].query}"'

        # Show last few questions
        recent = history[-3:] if len(history) > 3 else history
        questions = [f"{i + 1}. {turn.query}" for i, turn in enumerate(recent)]

        return "Your recent questions:\n" + "\n".join(questions)

    def _handle_what_discussed(self, history: List[Any]) -> str:
        """Handle 'what have we discussed?'"""
        # Extract main topics from entities
        topics = set()
        for turn in history:
            if hasattr(turn, 'entities_mentioned') and turn.entities_mentioned:
                # Take top 2 entities from each turn
                topics.update(turn.entities_mentioned[:2])

        if not topics:
            # Fallback to listing questions
            questions = [turn.query for turn in history[:5]]
            return "We have discussed:\n" + "\n".join(f"- {q}" for q in questions)

        topics_list = sorted(list(topics))[:10]  # Top 10 topics
        topics_str = ", ".join(topics_list)

        return f"We have discussed the following topics: {topics_str}"

    def _handle_summarize(self, history: List[Any]) -> str:
        """Handle 'summarize our conversation'."""
        summary_lines = ["Conversation summary:"]

        for i, turn in enumerate(history, 1):
            # Use response_summary if available, otherwise truncate answer
            if hasattr(turn, 'response_summary') and turn.response_summary:
                summary = turn.response_summary
            elif hasattr(turn, 'answer'):
                summary = turn.answer[:150] + "..." if len(turn.answer) > 150 else turn.answer
            else:
                summary = "(No answer recorded)"

            summary_lines.append(f"\n{i}. Q: {turn.query}")
            summary_lines.append(f"   A: {summary}")

        return "\n".join(summary_lines)

    def _handle_general_meta(self, history: List[Any]) -> str:
        """Handle general meta-questions."""
        num_turns = len(history)

        # Get main topics
        topics = set()
        for turn in history:
            if hasattr(turn, 'entities_mentioned') and turn.entities_mentioned:
                topics.update(turn.entities_mentioned[:2])

        topics_str = ", ".join(sorted(list(topics))[:5]) if topics else "various topics"

        return (
            f"We've had {num_turns} conversation turn{'s' if num_turns != 1 else ''} "
            f"discussing {topics_str}. "
            f"Your most recent question was: \"{history[-1].query}\""
        )
