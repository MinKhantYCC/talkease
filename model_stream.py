from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
import os
from utils import download_large_file
import warnings

warnings.filterwarnings("ignore")

if not (os.path.exists("./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")):
    download_large_file("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", "./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf")

class StreamChain(ConversationChain):
    def stream(self, inputs:dict[str, str]):
        """Stream the conversation with the AI model."""
        tokens = ""

        # Load the conversation history from memory
        inputs['history'] = self.memory.load_memory_variables({})['history'] 

        # Format the prompt with the input
        prompt_formatted = self.prompt.format(**inputs)

        # Stream the conversation with the AI model
        for token in self.llm.stream(prompt_formatted):
            tokens = tokens + token.content
            yield token.content

        # Update memory with the new conversation
        self.memory.save_context({"input": inputs["input"]}, {"output": tokens})

def load_chain():
    # Load the model from the file
    llm = ChatLlamaCpp(
        model_path="./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        stop=['User', '\nUser','\n\n', '\nHuman', "</s>", "\nHuman:", " Human:"],
        temperature=0.7,
        streaming=True,
        max_tokens=256,
        top_p=0.95,
        verbose=False,
        n_ctx=2048,
        n_batch=32,
    )

    # Create a prompt template for the conversation summary
    summary_prompt = PromptTemplate.from_template(
        (
            "Summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n"
            "Previous summary:\n"
            "{summary}\n\n"
            "New lines of conversation:\n"
            "{new_lines}\n\n"
            "New summary:\n"
        )
    )

    # Create a memory object to store the conversation summary
    memory = ConversationSummaryBufferMemory(llm=llm, prompt=summary_prompt)

    # Create a prompt template for the conversation
    prompt_template = PromptTemplate.from_template(
        (
            "This is a conversation between human and AI. Reply to the user in NO MORE THAN 30 words.\n"
            "History: {history}\n\n"
            "Human: {input}\n\n"
        )
    )

    # Create a conversation chain object
    chain = StreamChain(llm=llm, memory=memory, prompt=prompt_template)

    return chain

if __name__ == "__main__":
    print("Enter 'q' to quit.")
    chain = load_chain()
    outputs = ""
    while True:
        input_text = input("You: ")
        if input_text == "q":
            break
        else:
            for token in chain.stream(inputs={"input": input_text}):
                print(token, end="", flush=True)
                outputs = outputs + token
            print()
            