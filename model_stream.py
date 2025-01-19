from langchain.memory import ConversationSummaryMemory
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.api.base import LLMChain
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from utils import download_large_file
import warnings

warnings.filterwarnings("ignore")


if not (os.path.exists("./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")):
    download_large_file("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", "./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")

class StreamChain(LLMChain):
    def stream(self, inputs:dict[str, str]):
        tokens = ""
        messages = ""
        for i, msg in enumerate(self.memory.chat_memory.messages):
            if i%2!=0:
                messages = messages + "\nAI: "+msg.content
            else:
                messages = messages + "\nHuman: " + msg.content
        inputs['history'] = messages
        prompt_formatted = self.prompt.format(**inputs)
        for token in self.llm.stream(prompt_formatted):
            tokens = tokens + token.content
            yield token.content
        self.memory.save_context(inputs={"input": inputs["input"]}, outputs={"output": tokens})

def load_chain():
    llm = ChatLlamaCpp(
        model_path="./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        stop=['User', '\nUser','\n\n', '\nHuman', "</s>", " Human:"],
        temperature=0,
        streaming=True,
        max_tokens=256,
        top_p=0.9,
        verbose=False,
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
    prompt_template = PromptTemplate.from_template(
        (
            "<|system|>This is a conversation between human and AI. Reply to the user in NO MORE THAN 30 words.\n"
            "Previous Conversation Summary: {history}</s>"
            "<|user|>Human: {input}</s>\n"
            "<|assistant|>"
        )
    )

    chain = StreamChain(llm=llm, memory=memory, prompt=prompt_template)

    return chain
 