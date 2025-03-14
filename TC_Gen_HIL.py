import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.prompt import Prompt
import time
from prompts_file import *
import warnings

from doc_loader import get_req_instance
from knowledge_base import create_vectorstore

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""


def llm_call(vectorstore, req_instance):
    llm = ChatGroq( model= "llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
        )

    memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10
            )

    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        condense_question_prompt=condense_question_prompt,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
        },
    )

    # Create prompt template
    chain_template = """
    {chat_history}
    Human: {human_input}
    AI:"""
    chain_prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=chain_template)

    llm_chain = LLMChain(llm=llm, prompt=chain_prompt, memory=memory)

    console.print("[bold green] Running the Model")

    memory.clear()

    response = conversation.invoke(
        {
            "question": req_instance,
            "chat_history": [],
        }
    )

    # console.print("[bold green] Model Output")
    # console.print(response)

    print(f"Output:\n{response['answer']}")
    analysis = extract_xml(response['answer'], "analysis")

    inputs = Prompt.ask("Are you satisfied with test scenarios ??", choices=["y", "n"], default="y").strip().lower()

    if inputs != 'y':
        feedback = Prompt.ask("Please provide the feedback")

        while True:
            console.print("[blue]Waiting for 60 sec")
            time.sleep(60)

            console.print("[bold green]Running the Model in loop")

            final_response = llm_chain.invoke(
                f"Here are the feedback from user: {feedback}. Now use these points, and re-generate the test scenarios again")
            # console.print(final_response)
            # print(f" History:\n{response['chat_history']}\n")
            print(f"Output:\n{final_response['text']}")

            rerun_model = Prompt.ask("What to re-run the Model for better results ? ?", choices=["y", "n"],
                                       default="n").strip().lower()

            if rerun_model == 'y':
                feedback = Prompt.ask("Please provide the feedback")
            else:
                console.print("[bold green] User Satisfied, continuing the execution")
                model_output.append(final_response['text'])
                return model_output
    else:
        test_scenarios = extract_xml(response['answer'], "tests")
        model_output.append(response['answer'])
        console.print("[bold green] User Satisfied, continuing the execution")

    return test_scenarios

# if __name__ == '__ main __':
model_output = []
req_variable = get_req_instance()

print("[bold green]STARTED EXECUTION.....")

vectorstore_variable = create_vectorstore()
console = Console()
final_output = llm_call(vectorstore_variable, req_variable)
print(final_output)