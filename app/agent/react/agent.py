import enum
import logging
from typing import Optional, List

from pykka import ActorRef

from app.agent.base import BaseAgent
from app.llm.adapter import LLMAdapter
from app.llm.generator import Generator
from app.memory.memory import Memory
from app.persona.persona import Persona
from app.state.defaults import default_tools_handler, default_text_handler
from app.state.final_answer.constants import FINAL_ANSWER_STATE
from app.state.final_answer.factory import FinalAnswerNodeFactory
from app.state.node import Node, GenerationHandlerResponse
from app.tools.adapter import ToolAdapter

logger = logging.getLogger(__name__)


class States(str, enum.Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVE = "observe"
    FINAL_ANSWER = FINAL_ANSWER_STATE


class ReActAgent(BaseAgent):
    name = "ReActAgent"
    description = "Reasons and performs actions with tools to solve problems."
    pass


def _observe_node(
        tools: List[ToolAdapter],
        llm: LLMAdapter,
) -> Node:
    def _prompt_handler(
            persona: Persona,
            memory: Memory,
            _: Optional[List[ToolAdapter]]) -> str:
        prompt = f'''{persona.prompt()}
        
Given the thought and the action from that thought, using what you know about the problem, reflect on what you observe.

The thought:
"""
{memory.data.get("last_thought")}
"""

The action from the thought:
"""
{memory.data.get("last_action")}
"""

Provide a concise observation of what you observe:'''
        logger.debug(f"Using prompt for action step: {prompt}")
        return prompt

    return Node(
        state_name=States.OBSERVE.value,
        prompt_handler=_prompt_handler,
        generation_handler=default_text_handler(
            next_state=States.THOUGHT.value,
            exclude_from_scratch_pad=False
        ),
        generator=Generator(
            service=llm,
            use_json_model=False,
            temperature=0.1,
        ),
        tools=tools,
    )


def _action_node(
        tools: List[ToolAdapter],
        llm: LLMAdapter,
) -> Node:
    def _prompt_handler(
            persona: Persona,
            memory: Memory,
            tools: Optional[List[ToolAdapter]]) -> str:
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

    return Node(
        state_name=States.ACTION.value,
        prompt_handler=_prompt_handler,
        generation_handler=default_tools_handler(
            next_state=States.OBSERVE.value,
            exclude_from_scratch_pad=False,
            save_data_key="last_action"
        ),
        generator=Generator(
            service=llm,
            use_json_model=True,
            temperature=0.1,
        ),
        tools=tools,
    )


def _thought_node(
        tools: List[ToolAdapter],
        llm: LLMAdapter,
) -> Node:
    def _prompt_handler(
            persona: Persona,
            memory: Memory,
            tools: List[ToolAdapter]) -> str:
        thought_component_prompt = "Create a thought for how to answer the message based on the previous steps you have taken." \
            if memory.data.exists("last_thought") else "Create a thought based on the message."
        notes_component_prompt = "\nHere are your notes so far in order from oldest to newest, use these to help create " \
                                 "your next thought: " + memory.scratch_pad.prompt() + "\n" \
            if len(memory.scratch_pad.get()) > 0 else ""
        messages_formatted = [f"{message.name}: {message.content}\n" for message in
                              memory.data.get_all_messages()]
        messages_component_formatted = f'''Previous Messages, use these as context to the current message:
"""
{messages_formatted}
"""
''' if len(messages_formatted) > 0 else ""
        tools_component_formatted = "\n".join([f"name: {tool.name}\ndescription: {tool.description}\n\n" for tool in tools])

        prompt = f'''{persona.prompt()}

{thought_component_prompt}

Your thought will be used to help you answer the message.

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
{memory.data.get_current_message().content}
"""
{notes_component_prompt}
{messages_component_formatted}

Here are the tools you have access to, you DO NOT have access to other tools, use the name of the tool you think can help:
"""
{tools_component_formatted}
"""

ALWAYS respond with only one of the following:
- "Answer: " if you know the answer to the message. Provide the full answer with all of the details on how you solved any problems.
- "Thought: " followed by your thought.

Provide your thought OR answer but only one.'''
        logger.debug(f"Using prompt for thought step: {prompt}")
        return prompt

    def _generation_handler(
            response: str,
            memory: Memory,
            _: List[ToolAdapter]) -> GenerationHandlerResponse:
        if "Answer:" in response:
            parse_answer = response.split("Answer:")[1].strip()
            return GenerationHandlerResponse(
                output=parse_answer,
                exclude_from_scratch_pad=False,
                next_state=States.FINAL_ANSWER.value
            )
        else:
            parse_thought = response.split("Thought:")[1].strip()
            memory.data.set("last_thought", parse_thought)
            return GenerationHandlerResponse(
                output=parse_thought,
                exclude_from_scratch_pad=False,
                next_state=States.ACTION.value
            )

    return Node(
        state_name=States.THOUGHT.value,
        prompt_handler=_prompt_handler,
        generation_handler=_generation_handler,
        generator=Generator(
            service=llm,
            use_json_model=False,
            temperature=0.1,
        ),
        tools=tools,
    )


class ReActAgentFactory:
    def __init__(self):
        pass

    @staticmethod
    def start(llm: LLMAdapter,
              tools: List[ToolAdapter],
              persona: Persona = None,
              memory: Memory = None,
              clear_scratch_after_answer: bool = False,
              clear_data_after_answer: bool = False,
              step_limit: int = 10) -> ActorRef[ReActAgent]:
        if persona is None:
            persona = Persona(description="You're a helpful assistant. You solve problems by breaking them down into "
                                          "multiple steps, thinking on those steps, acting on them, and observing. "
                                          "Given the problem, you will use your tools to solve it in as few steps as "
                                          "possible.")
        if memory is None:
            memory = Memory()

        thought_node = _thought_node(tools=tools, llm=llm)
        action_node = _action_node(tools=tools, llm=llm)
        observe_node = _observe_node(tools=tools, llm=llm)
        final_answer_node = FinalAnswerNodeFactory.get_node(Generator(
            service=llm,
            temperature=0.4,
        ))

        nodes = [thought_node, action_node, observe_node, final_answer_node]
        return ReActAgent.start(
            persona=persona,
            memory=memory,
            nodes=nodes,
            default_initial_state=States.THOUGHT.value,
            clear_scratch_after_answer=clear_scratch_after_answer,
            clear_data_after_answer=clear_data_after_answer,
            step_limit=step_limit,
            step_limit_state_name=States.FINAL_ANSWER.value
        )
