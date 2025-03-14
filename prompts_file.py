from langchain_core.prompts import ChatPromptTemplate

condense_question_template = """
Given the following conversation and a follow up feedback, rephrase the follow up feedback to be a standalone feedback.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

qa_template = """ You are an assistant who will generate test scenarios of given requirement text. Use the following pieces of retrieved context to understand the requirement. 
If you don't know can't understand the requirement fully, ask for clarifications / questions

Context:
{context}

Chat History:
{chat_history}

{context}

Requirement: {question}

You should return the response only in below format:

<analysis>
Explain your understanding of the given requirement in detail. Break down it in detail.
</analysis>

<tests>

Based on the analysis, provide the test scenarios such way that the respective requirement can be fully validated / tested. Provide the response in below format:
    Test Objective: Objective of the test.
    Test Condition: How to perform the Test.
    Expected outcome: Expected outcome after performing the test.
    Validation: How to valid that the ECU behaves as per requirement.
    
</tests>
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)