from llm import LLM_Model
from dotenv import load_dotenv

load_dotenv()



def main(agent=False):
    llm = LLM_Model()
    llm.load_model()
    
    if agent == True:
        llm.load_agent()
        llm.agent_handler.run_agent()
    else:
        llm.run_chatbot()

if __name__ == "__main__":
    main(agent=True)