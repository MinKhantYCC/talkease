import streamlit as st
from streamlit_mic_recorder import speech_to_text
from model import load_chain, get_ai_response
from speech2text import speak

chain = load_chain()

def main():
    st.title("Talk with me")
    st.text("Powered by Min Khant")
    text = speech_to_text(
        language='en',
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=False,
        callback=None,
        args=(),
        kwargs={},
        key=None
    )
    if text:
        st.text(text)
        ai_response = get_ai_response(text, chain)
        speak(ai_response)

if __name__ == "__main__":
    main()