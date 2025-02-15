from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
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
    llm = ChatLlamaCpp(
        model_path="./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        stop=['User', '\nUser','\n\n', '\nHuman', "</s>", " Human:"],
        temperature=0,
        streaming=True,
        max_tokens=256,
        top_p=0.9,
        verbose=False,
        n_ctx=2048,
        n_batch=32,
    )

    summary_prompt = PromptTemplate.from_template(
        (
            "<|system|>Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n"
            "Previous summary:\n"
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

if __name__ == "__main__":
    print("Enter 'q' to quit.")
    chain = load_chain()
    while True:
        input_text = input("You: ")
        if input_text == "q":
            break
        else:
            ai_response = get_ai_response(input_text, chain)
            print("AI: ", ai_response)