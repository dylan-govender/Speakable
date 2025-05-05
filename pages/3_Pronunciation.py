import streamlit as st
from nltk.tokenize import sent_tokenize
from streamlit_mic_recorder import mic_recorder

import torch
import torchaudio
import ffmpeg
import phonemizer
import Levenshtein
import pydub
import nltk
from io import BytesIO
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

st.markdown("### Pronunciation")

if "show_success" not in st.session_state:
    st.session_state.show_success = True

@st.cache_data(show_spinner="Downloading dependencies...")
def install_dependencies():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

install_dependencies()

@st.cache_data  
def dismiss():
    st.session_state.show_success = False

if "reviewed_text" in st.session_state:
    human = st.text_area(
        label="Paste Text",
        label_visibility="collapsed",
        value=st.session_state["reviewed_text"],
        height=300
    )

    if st.session_state.show_success:
        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            st.success("The above text was automatically filled in from **Text Analysis** using your previous reviewed text. Please ensure that it is correct.")
        with col2:
            st.button("❌", on_click=dismiss, help="Dismiss message")
    
else:
    human = st.text_area(
        label="Paste Text",
        label_visibility="collapsed",
        placeholder="Enter your text here...",
        height=300
    )

@st.cache_resource(show_spinner="Loading speech model...")
def load_model_ipa_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    return processor, model

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if "selected_model" not in st.session_state:
    selected_model = "gemini-1.5-pro"
    st.session_state["selected_model"] = selected_model

llm = ChatGoogleGenerativeAI(model=st.session_state["selected_model"])

def generate_feedback(human, ipa, sentence, ratio, contents_str):
    prompt = f"""
    Given the following ipa transcription that has been generated from the audio file: {human}.

    The user is trying to say the word: \"{sentence}\" and in ipa transcription, it is: {ipa}. The provided IPA transcription is modified to include spaces between each character.

    The similarity between the ipa transcription and the phonemes is: {ratio}.
    The user's native language is English, the user's target language is English, and the user's efficiency level is Beginner. See below the differences between the ipa transcription and the phonemes: {contents_str}

    Your return should be in the following format:
    \\n- Word that was mispronounced: description of how to improve pronunciation of word and what the word sounds like in simple transcription, like for the word \"How\" it can be pronounced as \"ow\".

    \\n- Lastly you should return a score to the user and why they achieved this score.

    Strictly ensure you follow the above format.
    Strictly ensure that you do not include any ipa transcription or complex symbols in your return, instead replace it with the actual part of the word or sentence for better understanding.
    """
    
    messages = [
        ("system", prompt),
        ("human", human),
    ]
    
    try:
        st.info("**Feedback**\n" + llm.invoke(messages).content)
    except ResourceExhausted as e:
        st.warning(
            "The model that is analysing your text is currently exhausted. Please go to the **Settings** tab and select a different model."
        )
        
def phonemize_audio(audio):
    waveform, sample_rate = torchaudio.load(audio)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    processor, model = load_model_ipa_model()
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_human_ipa = torch.argmax(logits, dim=-1)
    human = processor.batch_decode(predicted_human_ipa)[0]
    human = human.replace(":", "").replace("'", "").replace("ˌ", "").removeprefix(" ").removesuffix(" ")
    
    return human

def generate_content_str(human, ipa, non_matching):
    contents = []
    for tag, i1, i2, j1, j2 in non_matching:
        # print(tag, human[i1:i2], ipa[j1:j2])
        if tag == 'equal':
            contents.append(('equal',human[i1:i2]))
        elif tag == 'replace':
            contents.append(('replace',human[i1:i2], ipa[j1:j2]))
        elif tag == 'delete':
            contents.append(('delete',human[i1:i2]))
        elif tag == 'insert':
            contents.append(('insert',ipa[j1:j2]))
    
    contents_str = "The key differences in the string are highlighted below: "
    for tag, *args in contents:
        if tag == 'equal':
            contents_str += f" Same({args[0]}), "
        elif tag == 'replace':
            contents_str += f" Replace({args[0]} with {args[1]}), "
        elif tag == 'delete':
            contents_str += f" Delete({args[0]}), "
        elif tag == 'insert':
            contents_str += f" Insert({args[0]}), "
    contents_str = contents_str.removesuffix(", ")
    
    return contents_str

if human:
    sentences = sent_tokenize(human)

    with st.container():
        st.markdown("### Pronunciation Practice")
        st.write("")
        sentence_options = [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences)]
        selected_index = st.selectbox("**Select a sentence:**", range(len(sentences)), format_func=lambda i: sentence_options[i])
        selected_sentence = sentences[selected_index]

        st.write("")
        audio = mic_recorder(
            start_prompt="🎙️ Record Pronunciation",
            stop_prompt="⏹️ Stop",
            key=f"rec_{selected_index}"
        )

    st.write("")
    if audio:
        st.audio(audio["bytes"])

        audio_segment = pydub.AudioSegment.from_file(BytesIO(audio["bytes"]))
        wav_io = BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        human_ipa = phonemize_audio(wav_io)
        ipa = phonemizer.phonemize(selected_sentence, language="en-us", backend="espeak")
        ipa = ipa.replace(":", "").replace("'", "").replace("ˌ", "").strip()
        ratio = Levenshtein.ratio(human_ipa, ipa)
        ops = Levenshtein.editops(human_ipa, ipa)
        matching = Levenshtein.matching_blocks(ops, human_ipa, ipa)
        non_matching = Levenshtein.opcodes(ops, human_ipa, ipa)
        contents_str = generate_content_str(human_ipa, ipa, non_matching)
        
        st.write("")
        generate_feedback(human_ipa, ipa, selected_sentence, ratio, contents_str)
        

else:
    st.warning("Please paste in your text in the text field above. If you are using unreviewed text, please analyse your text first using **Text Analysis**.")
