import enum
import logging
from typing import Optional, List

from pydantic import BaseModel, Field
from pykka import ActorRef

from app.core.agents.base import AgentBase
from app.core.llm.adapter import LLMAdapter
from app.core.llm.generator import Generator
from app.core.memory.memory import Memory
from app.core.persona import Persona
from app.core.states.base import StateBase, Transition
from app.states.defaults import text_handler, tools_handler
from app.states.final_answer.state import FinalAnswerState
from app.core.tools.adapter import ToolAdapter

logger = logging.getLogger(__name__)


class States(str, enum.Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVE = "observe"
    FINAL_ANSWER = FinalAnswerState.name


class ReActAgent(AgentBase):
    name = "ReActAgent"
    description = "Reasons and performs actions with tools to solve problems."
    pass


class ObserveState(StateBase):
    name: str = States.OBSERVE

    def build_prompt(self, persona: Persona, memory: Memory, tools: Optional[List[ToolAdapter]]) -> str:
        prompt = f'''{persona.prompt()}

Given a thought and the action from that thought, reflect on what you observe.

The thought:
"""
{memory.data.get("last_thought")}
"""

The action from the thought:
"""
{memory.data.get("last_action")}
"""

Provide a summary of what you observe:'''
        logger.debug(f"Using prompt for action step: {prompt}")
        return prompt

    def handle_transition(self, response: str, memory: Memory, _: Optional[List[ToolAdapter]]) -> Transition:
        return text_handler(response=response, memory=memory, next_state=States.THOUGHT)


class ActionState(StateBase):
    name: str = States.ACTION

    def build_prompt(self, persona: Persona, memory: Memory, tools: Optional[List[ToolAdapter]]) -> str:
        tool_schemas = [tool.schema() for tool in tools]
        prompt = f'''{persona.prompt()}

Action:
- Given a task, solve it to your best ability. 
- You have access to tools which can help you solve the problem, ALWAYS use tools to solve the problem.
- Do not ask for help.
- Don't repeat yourself from previous notes.

Notes:
"""
{memory.scratch_pad.prompt()}
"""

Task:
"""
{memory.data.get("last_thought")}
"""

Here are the schemas for the tools you have access to, pick only one:
"""
{tool_schemas}
"""

Respond with the JSON input for the tool of your choice to best solve the problem.'''
        logger.debug(f"Using prompt for action step: {prompt}")
        return prompt

    def handle_transition(self, response: str, memory: Memory, tools: Optional[List[ToolAdapter]]) -> Transition:
        return tools_handler(response=response, memory=memory, tools=tools, next_state=States.OBSERVE,
                             save_data_key="last_action")


class ThoughtStateType(str, enum.Enum):
    """
    The type of reason for your response.
    """
    ACTION = 'Action'
    ANSWER = 'Answer'


class ThoughtStateChoice(BaseModel):
    """
    The thought of which choice to make next.
    """
    type: ThoughtStateType = Field(..., description="The reason type of choice for your response. Pick Action if you "
                                                    "need to perform work, or Answer if you have a final answer.")
    reason: str = Field(..., description="The reason details for the choice.")


class ThoughtState(StateBase):
    name: str = States.THOUGHT

    @staticmethod
    def _get_thought_component_prompt(last_thought: str) -> str:
        if len(last_thought) > 0:
            return f"Create a thought based on the previous steps you have taken. Use your observations in your notes " \
                   f"to help. Don't give the same thought as last time: {last_thought}\n"
        return "Create a thought based on the message."

    @staticmethod
    def _get_notes_component_prompt(notes: List[str]) -> str:
        if notes:
            return "\nHere are your notes so far in order from oldest to newest, use these to help create your next " \
                   "though. If you know the answer from your notes, say so. Don't repeat yourself from previous " \
                   "notes.\n\nNotes:\n" + "\n".join(notes) + "\n"
        return ""

    @staticmethod
    def _get_messages_component_formatted(messages: List) -> str:
        if messages:
            return "Previous Messages, use these as context to the current message:\n\"\"\"\n" + "".join(
                messages) + "\n\"\"\""
        return ""

    @staticmethod
    def _get_tools_component_formatted(tools: List[ToolAdapter]) -> str:
        return "\n".join([f"name: {tool.name}\ndescription: {tool.description}\n\n" for tool in tools])

    def build_prompt(self, persona: Persona, memory: Memory, tools: List[ToolAdapter]) -> str:
        last_thought = memory.data.get("last_thought") if memory.data.exists("last_thought") else ""
        notes = memory.scratch_pad.get()
        messages = [f"{message.name}: {message.content}\n" for message in memory.data.get_all_messages()]
        current_message_content = memory.data.get_current_message().content

        thought_component_prompt = self._get_thought_component_prompt(last_thought)
        notes_component_prompt = self._get_notes_component_prompt(notes)
        messages_component_formatted = self._get_messages_component_formatted(messages)
        tools_component_formatted = self._get_tools_component_formatted(tools)

        prompt = f'''{persona.prompt()}

{thought_component_prompt}

Thought instructions:
- Provides exact details on the task to best answer the message, do not forget important details.
- You have access to tools that can help you answer the message. Always try to use a tool.
-- In your thought, recommend a tool that will help.
-- If the problem can be solved using the tool, always use the tool.
-- If the problem cannot be solved or if you are unsure, give your answer with why.
- If you keep running into issues with your tools, give your answer with the problems you're running into.
- If the message doesn't require any tool, just give your answer.

Message:
"""
{current_message_content}
"""
{notes_component_prompt}
{messages_component_formatted}

Here are the tools you have access to, you DO NOT have access to other tools, use the name of the tool you think can help:
"""
{tools_component_formatted}
"""

Provide a choice based on the following JSON schema:
"""
{ThoughtStateChoice.schema()}
"""

Your response should be in JSON matching the schema. It should include your reasoning for the choice. 
- If you don't know the answer, provide a short summary of the details for the action to take.
- If you know the answer, use your observations to help. The answer should include any important details on how you know.

Choice JSON:'''
        logger.debug(f"Using prompt for thought step: {prompt}")
        return prompt

    def handle_transition(self, response: str, memory: Memory, tools: List[ToolAdapter]) -> Transition:
        try:
            choice = ThoughtStateChoice.model_validate_json(response)
            if choice.type == ThoughtStateType.ACTION:
                memory.data.set("last_thought", choice.reason)
                return Transition(next_state=States.ACTION, updated_response=choice.reason)
            elif choice.type == ThoughtStateType.ANSWER:
                return Transition(next_state=States.FINAL_ANSWER, updated_response=choice.reason)
        except Exception as e:
            return Transition(updated_response=f"Failed to parse response as Choice: {e}",
                              next_state=States.THOUGHT)


class ReActAgentFactory:

    @staticmethod
    def start(llm: LLMAdapter,
              tools: List[ToolAdapter],
              persona: Persona = None,
              memory: Memory = None,
              clear_scratch_pad_after_answer: bool = False,
              clear_data_after_answer: bool = False,
              step_limit: int = 10) -> ActorRef[ReActAgent]:
        if persona is None:
            persona = Persona(description="You're a helpful assistant. You solve problems by breaking them down into "
                                          "multiple steps, thinking on those steps, acting on them, and observing. "
                                          "Given the problem, you will use your tools to solve it in as few steps as "
                                          "possible.")
        if memory is None:
            memory = Memory()

        thought_state = ThoughtState(
            generator=Generator(service=llm, use_json_model=True, temperature=0.1),
            tools=tools
        )
        action_state = ActionState(
            generator=Generator(service=llm, use_json_model=True, temperature=0.1),
            tools=tools
        )
        observe_state = ObserveState(
            generator=Generator(service=llm, use_json_model=False, temperature=0.1),
            tools=tools
        )
        final_answer_state = FinalAnswerState(Generator(service=llm, temperature=0.3))

        states = [thought_state, action_state, observe_state, final_answer_state]
        return ReActAgent.start(
            persona=persona,
            memory=memory,
            states=states,
            default_initial_state=States.THOUGHT.value,
            clear_scratch_pad_after_answer=clear_scratch_pad_after_answer,
            clear_data_after_answer=clear_data_after_answer,
            step_limit=step_limit,
            step_limit_state_name=States.FINAL_ANSWER.value
        )
