from llm import LLM_Model
from dotenv import load_dotenv

load_dotenv()



def main(agent=False):
    llm = LLM_Model()
    # llm.load_model(model_type='ollama', model="llama3.2")
    llm.load_model(model_type='openai')
    
    if agent == True:
        llm.load_agent()
        llm.agent_handler.run_agent()
    else:
        llm.run_chatbot()


def rag_test():
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from vector import retriever

    model = OllamaLLM(model="llama3.2")

    template = """
    You are an exeprt in answering questions about a pizza restaurant

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    while True:
        print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n\n")
        if question == "q":
            break
        
        reviews = retriever.invoke(question) # Allows us to get data from vector database 
        result = chain.invoke({"reviews": reviews, "question": question})
        print(result)



if __name__ == "__main__":
    # main(agent=True)
    rag_test()
    