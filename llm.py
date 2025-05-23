from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from llm_tools import LLM_Tools
from llm_agents import LLM_Agents




class LLM_Model:
    def __init__(self, max_memory=5):
        self.model = None
        self.agent_handler = None
        self.memory = ChatMessageHistory()

    def load_model(self, temperature=0):
        self.model = ChatOpenAI(temperature=temperature)

    def load_agent(self):
        self.agent_handler = LLM_Agents(self.model)
        self.agent_handler.load_agent()

    def run_chatbot(self):
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                print('Shutting Down... Thank you!')
                break
            elif user_input.lower() == "toggle agent":
                self.toggle_agent_mode()
                break

            # Build conversation history as a string (optional, for context)
            history_str = ""
            for msg in self.memory.messages:
                if isinstance(msg, HumanMessage):
                    history_str += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_str += f"Assistant: {msg.content}\n"

            # Combine history and current input for context
            prompt = f"{history_str}User: {user_input}\nAssistant:"

            # Get response from the LLM
            response = self.model.invoke(prompt)
            print("\nAssistant:", response.content)

            # Save the turn to memory
            self.memory.add_user_message(user_input)
            self.memory.add_ai_message(response.content)

    def __getattr__(self, name):
        """Delegate attribute access to agent_handler if it exists"""
        if self.agent_handler and hasattr(self.agent_handler, name):
            return getattr(self.agent_handler, name)
        raise AttributeError(f"'LLM_Model' object has no attribute '{name}'")