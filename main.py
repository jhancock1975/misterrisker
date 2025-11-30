"""
MisterRisker - LangGraph Slack Messaging Application

This is the main entry point for sending messages to Slack and
receiving responses using a LangGraph workflow.

Usage:
    python main.py "Your message here"
    
    Or in interactive mode:
    python main.py --interactive

Environment Variables Required:
    SLACK_BOT_TOKEN - Your Slack bot OAuth token (xoxb-...)
    SLACK_CHANNEL_ID - The channel ID to send messages to
    SLACK_TIMEOUT - (optional) Timeout in seconds (default: 30)
"""

import asyncio
import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.slack_agent import SlackAgent, SlackAuthError, SlackAPIError, SlackTimeoutError
from workflows.slack_workflow import SlackWorkflow, SlackState


async def send_and_receive(message: str, wait_for_response: bool = True) -> dict:
    """
    Send a message to Slack and optionally wait for a response.
    
    Args:
        message: The message to send
        wait_for_response: Whether to wait for a reply
        
    Returns:
        dict: Final workflow state with conversation history
    """
    try:
        # Create agent from environment variables
        agent = SlackAgent.from_env()
        
        # Create workflow
        workflow = SlackWorkflow(
            slack_agent=agent,
            wait_for_response=wait_for_response,
            receive_timeout=int(os.environ.get("SLACK_TIMEOUT", "30"))
        )
        
        # Run the workflow
        async with agent.conversation():
            initial_state: SlackState = {
                "user_message": message,
                "messages": [],
                "awaiting_response": False
            }
            
            final_state = await workflow.run(initial_state)
            return final_state
            
    except SlackAuthError as e:
        print(f"❌ Authentication Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SlackAPIError as e:
        print(f"❌ Slack API Error: {e}", file=sys.stderr)
        sys.exit(1)


async def interactive_mode():
    """
    Run an interactive conversation with Slack.
    
    Type messages to send to Slack, receive responses,
    and continue the conversation.
    """
    print("🚀 Starting interactive Slack conversation...")
    print("   Type 'quit' or 'exit' to end the session.\n")
    
    try:
        agent = SlackAgent.from_env()
        workflow = SlackWorkflow(
            slack_agent=agent,
            wait_for_response=True,
            receive_timeout=int(os.environ.get("SLACK_TIMEOUT", "30"))
        )
        
        async with agent.conversation():
            conversation_history = []
            
            while True:
                try:
                    # Get user input
                    message = input("📤 You: ").strip()
                    
                    if message.lower() in ('quit', 'exit', 'q'):
                        print("\n👋 Ending conversation. Goodbye!")
                        break
                    
                    if not message:
                        continue
                    
                    # Run workflow
                    state: SlackState = {
                        "user_message": message,
                        "messages": conversation_history,
                        "awaiting_response": False
                    }
                    
                    print("⏳ Waiting for response from Slack...")
                    final_state = await workflow.run(state)
                    
                    # Update history
                    conversation_history = final_state.get("messages", [])
                    
                    # Check for timeout
                    if final_state.get("timeout_occurred"):
                        print("⏱️  Timeout: No response received.")
                    else:
                        # Print the response
                        if conversation_history:
                            last_msg = conversation_history[-1]
                            if last_msg.get("role") == "user":
                                print(f"📥 Slack: {last_msg.get('content', '')}")
                    
                    print()  # Empty line for readability
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted. Goodbye!")
                    break
                    
    except SlackAuthError as e:
        print(f"❌ Authentication Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Send messages to Slack via LangGraph workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  SLACK_BOT_TOKEN    Slack bot OAuth token (required)
  SLACK_CHANNEL_ID   Target channel ID (required)
  SLACK_TIMEOUT      Response timeout in seconds (default: 30)

Examples:
  python main.py "Hello from LangGraph!"
  python main.py --no-wait "Fire and forget message"
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        "message",
        nargs="?",
        help="Message to send to Slack"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive conversation mode"
    )
    
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Send message without waiting for response"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if not os.environ.get("SLACK_BOT_TOKEN"):
        print("❌ Error: SLACK_BOT_TOKEN environment variable is required", file=sys.stderr)
        print("   Set it with: export SLACK_BOT_TOKEN='xoxb-your-token'", file=sys.stderr)
        sys.exit(1)
    
    if not os.environ.get("SLACK_CHANNEL_ID"):
        print("❌ Error: SLACK_CHANNEL_ID environment variable is required", file=sys.stderr)
        print("   Set it with: export SLACK_CHANNEL_ID='C0123456789'", file=sys.stderr)
        sys.exit(1)
    
    # Run appropriate mode
    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.message:
        result = asyncio.run(send_and_receive(args.message, not args.no_wait))
        
        print(f"\n✅ Message sent successfully!")
        
        if result.get("timeout_occurred"):
            print("⏱️  Timeout: No response received within the timeout period.")
        elif result.get("messages"):
            print("\n📝 Conversation:")
            for msg in result.get("messages", []):
                role = "You" if msg.get("role") == "assistant" else "Slack"
                print(f"   {role}: {msg.get('content', '')}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
