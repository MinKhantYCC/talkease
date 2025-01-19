import streamlit as st
from streamlit_mic_recorder import speech_to_text
from model_stream import load_chain
from speech2text import speak
from time import sleep
from gtts import gTTS
import re
import io
import base64
import warnings

warnings.filterwarnings("ignore")

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
        for ai_text in chain.stream({"input": text}):
            flag = re.findall("[+*.!/|:;#$%^&(){}`?<>,=]", ai_text)
            print(ai_text)
            if not (ai_text in ['AI', 'Response']) and not flag and ai_text:
                # speak(ai_text)
                sound_file = io.BytesIO()
                tts = gTTS(ai_text, lang="en")
                tts.write_to_fp(sound_file)
                # st.audio(sound_file)
                sound_file.seek(0)
                b64 = base64.b64encode(sound_file.read()).decode()
                md = f"""
                    <audio id="audioTag" controls autoplay>
                    <source src="data:audio/mp3;base64,{b64}"  type="audio/mpeg" format="audio/mpeg">
                    </audio>
                """
                st.markdown(
                    md,
                    unsafe_allow_html=True,
                )
                sleep(0.2)



if __name__ == "__main__":
    main()