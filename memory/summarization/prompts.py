# memory/summarization/prompts.py

from typing import List, Optional
from memory.models import ConversationMessage, ConversationSummary

class SummaryPromptBuilder:
    """
    Pure factory for prompt assembly. 
    Translates Domain Models into instructional strings for the summarization engine.
    """

    SYSTEM_PROMPT = (
        "You are a highly efficient memory compression AI. "
        "Your objective is to compress conversation history into a dense, factual summary. "
        "STRICT RULES:\n"
        "1. Retain all established facts, user preferences, and unresolved questions.\n"
        "2. Retain specific entities (names, tools, code snippets, dates).\n"
        "3. Ignore greetings, pleasantries, repetitions, and filler words.\n"
        "4. Output ONLY the summary text. No conversational preamble."
    )

    def build_user_prompt(
        self, 
        messages_to_summarize: List[ConversationMessage], 
        previous_summary: Optional[ConversationSummary]
    ) -> str:
        """
        Translates raw message list and existing summary state into 
        the final instructional prompt for the LLM.
        """
        
        # 1. Format the transcript
        transcript = "\n".join([
            f"{msg.role.value.upper()}: {msg.content}" 
            for msg in messages_to_summarize
        ])

        # 2. Build the instruction set
        if previous_summary:
            return (
                "You are performing a rolling summary update. \n"
                "Merge the NEW MESSAGES with the PREVIOUS SUMMARY to create a single, updated narrative.\n\n"
                f"=== PREVIOUS SUMMARY ===\n{previous_summary.summary}\n\n"
                f"=== NEW MESSAGES ===\n{transcript}\n\n"
                "Provide the fully updated, comprehensive summary below:"
            )
        else:
            return (
                "Create a dense, factual summary of the following conversation.\n\n"
                f"=== CONVERSATION ===\n{transcript}\n\n"
                "Provide the summary below:"
            )

    def get_system_prompt(self) -> str:
        """Returns the immutable system instruction."""
        return self.SYSTEM_PROMPT