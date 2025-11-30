"""
LangGraph workflow for Slack messaging.

This module provides a LangGraph-based workflow for sending messages
to Slack and receiving responses.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
import operator

from agents.slack_agent import SlackAgent, SlackTimeoutError


class SlackState(TypedDict, total=False):
    """
    State schema for the Slack workflow.
    
    Attributes:
        user_message: The message to send to Slack
        messages: Conversation history
        awaiting_response: Whether we're waiting for a user response
        last_message_ts: Timestamp of the last sent message
        channel_id: The Slack channel ID
        timeout_occurred: Whether a timeout occurred waiting for response
    """
    user_message: str
    messages: List[Dict[str, str]]
    awaiting_response: bool
    last_message_ts: Optional[str]
    channel_id: Optional[str]
    timeout_occurred: bool


class SlackWorkflow:
    """
    LangGraph workflow for Slack conversations.
    
    This workflow orchestrates sending messages to Slack and optionally
    waiting for responses using a LangGraph state machine.
    
    Attributes:
        slack_agent: The SlackAgent instance for API calls
        wait_for_response: Whether to wait for a response after sending
        receive_timeout: Timeout for waiting for responses
        channel_override: Optional channel to use instead of agent default
        filter_bot_messages: Whether to filter out bot messages
        node_names: List of node names in the workflow
    """
    
    def __init__(
        self,
        slack_agent: SlackAgent,
        wait_for_response: bool = True,
        receive_timeout: int = 30,
        channel_override: Optional[str] = None,
        filter_bot_messages: bool = True
    ):
        """
        Initialize the Slack workflow.
        
        Args:
            slack_agent: Configured SlackAgent instance
            wait_for_response: If True, wait for response after sending
            receive_timeout: Timeout in seconds for receiving messages
            channel_override: Optional channel ID to override agent's default
            filter_bot_messages: If True, ignore messages from bots
        """
        self.slack_agent = slack_agent
        self.wait_for_response = wait_for_response
        self.receive_timeout = receive_timeout
        self.channel_override = channel_override
        self.filter_bot_messages = filter_bot_messages
        self.node_names = ["send_message", "receive_message"]
        self._graph = None
        self._compiled_graph = None
    
    async def send_message_node(self, state: SlackState) -> SlackState:
        """
        Node that sends a message to Slack.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with message sent
        """
        message = state.get("user_message", "")
        
        if not message:
            return state
        
        channel = self.channel_override or self.slack_agent.channel_id
        
        response = await self.slack_agent.send_message(
            message=message,
            channel_id=channel
        )
        
        # Update conversation history
        messages = list(state.get("messages", []))
        messages.append({
            "role": "assistant",
            "content": message
        })
        
        return {
            **state,
            "messages": messages,
            "last_message_ts": response.get("ts"),
            "awaiting_response": True,
            "channel_id": channel
        }
    
    async def receive_message_node(self, state: SlackState) -> SlackState:
        """
        Node that waits for a response message from Slack.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with received message
        """
        try:
            response = await self.slack_agent.wait_for_message(
                timeout=self.receive_timeout,
                after_ts=state.get("last_message_ts")
            )
            
            # Update conversation history
            messages = list(state.get("messages", []))
            messages.append({
                "role": "user",
                "content": response.get("text", "")
            })
            
            return {
                **state,
                "messages": messages,
                "awaiting_response": False,
                "last_message_ts": response.get("ts"),
                "timeout_occurred": False
            }
            
        except SlackTimeoutError:
            return {
                **state,
                "awaiting_response": False,
                "timeout_occurred": True
            }
    
    def _should_send(self, state: SlackState) -> str:
        """
        Routing function to determine if we should send a message.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name or END
        """
        if not state.get("user_message"):
            return END
        return "send_message"
    
    def _should_receive(self, state: SlackState) -> str:
        """
        Routing function to determine if we should wait for response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name or END
        """
        if self.wait_for_response:
            return "receive_message"
        return END
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow graph.
        
        Returns:
            Configured StateGraph instance
        """
        # Create the graph with our state schema
        graph = StateGraph(SlackState)
        
        # Add nodes
        graph.add_node("send_message", self.send_message_node)
        graph.add_node("receive_message", self.receive_message_node)
        
        # Set entry point with conditional routing
        graph.set_conditional_entry_point(
            self._should_send,
            {
                "send_message": "send_message",
                END: END
            }
        )
        
        # Add conditional edge from send to receive or end
        graph.add_conditional_edges(
            "send_message",
            self._should_receive,
            {
                "receive_message": "receive_message",
                END: END
            }
        )
        
        # Receive always ends
        graph.add_edge("receive_message", END)
        
        self._graph = graph
        return graph
    
    def compile(self):
        """
        Compile the workflow graph for execution.
        
        Returns:
            Compiled graph ready for invocation
        """
        if self._graph is None:
            self.build_graph()
        
        self._compiled_graph = self._graph.compile()
        return self._compiled_graph
    
    async def run(self, initial_state: SlackState) -> SlackState:
        """
        Execute the workflow with the given initial state.
        
        Args:
            initial_state: Starting state for the workflow
            
        Returns:
            Final state after workflow completion
        """
        if self._compiled_graph is None:
            self.compile()
        
        # LangGraph's ainvoke returns the final state
        result = await self._compiled_graph.ainvoke(initial_state)
        return result
