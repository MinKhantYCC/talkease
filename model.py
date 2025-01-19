from langchain.memory import ConversationSummaryMemory
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.api.base import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from utils import download_large_file
import warnings

warnings.filterwarnings("ignore")


if not (os.path.exists("./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")):
    download_large_file("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", "./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")

def load_chain()->LLMChain:
    llm = CTransformers(model="./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", model_type="Llama",
        config={
            "temperature": 0,
            "stop": ['User', '\nUser','\n\n', '\nHuman', "</s>", " Human:"],
            "stream":True,
        },
        streaming=True
    )

    summary_prompt = PromptTemplate.from_template(
        (
            "<|system|>Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n"
            "Current summary:\n"
            "{summary}</s>\n\n"
            "<|user|>\nNew lines of conversation:\n"
            "{new_lines}</s>\n\n"
            "New summary:\n"
        )
    )

    chat_history = ChatMessageHistory()
    memory = ConversationSummaryMemory(chat_memory=chat_history, llm=llm, prompt=summary_prompt)
    prompt = PromptTemplate.from_template(
        (
            "This is a conversation between human and AI. Reply to the user in NO MORE THAN 30 words.\n"
            "History: {history}</s>"
            "{input}</s>\n"
            "AI:"
        )
    )

    chain = LLMChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
    )

    return chain

def get_ai_response(input_text:str, chain:LLMChain):
    ai_response = chain.invoke(input={
        "input": input_text
    })['text']
    return ai_response.strip()
