from nltk.tokenize import sent_tokenize
from gtts import gTTS
import streamlit as st
import tempfile
import nltk

@st.cache_data(show_spinner="Downloading dependencies...")
def install_dependencies():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

install_dependencies()

st.markdown("### Annotation")
st.info("This section helps you present your text clearly using state-of-the-art Text-to-Speech models. It focuses on improving pronunciation, but does not reflect tone, engagement, or emotional expression.")

if "reviewed_text" in st.session_state:
    text = st.session_state["reviewed_text"]
    sentences = sent_tokenize(text)

    for i, sentence in enumerate(sentences):
        st.markdown(f"**Sentence {i+1}**: {sentence}")    
         
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            gTTS(sentence).save(tmp.name)
            tmp.seek(0)
            audio_bytes = tmp.read()
            st.audio(audio_bytes, format="audio/mp3")
            
        st.divider()
        
else:
    st.warning("No text to synthesize. Please review and submit your text first using **Text Analysis** to activate this tab.")
