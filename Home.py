import streamlit as st
import streamlit.components.v1 as components
import nltk

st.image("images/speakable_logo.png", caption="")

if st.session_state.get("show_success", False):
    st.session_state.show_success = True  
    
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "gemini-1.5-flash"

@st.cache_data(show_spinner="Downloading dependencies...")
def install_dependencies():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

install_dependencies()

# Title Section
st.title("About Speakable")
st.subheader("Clear, Confident Communication for Everyone—Anywhere, Anytime.")

# Mission & Vision
st.markdown("### Our Mission")
st.markdown("""
To empower language learners with interactive tools and **real-time AI feedback** to enhance pronunciation, confidence, and communication skills.
""")

st.markdown("### Our Vision")
st.markdown("""
To ensure everyone can communicate their thoughts effectively by **enhancing the way we speak and connect**.
""")

# Challenges & Solutions
st.markdown("### Challenges")
st.markdown("""
- Over 70% of language learners struggle with speaking confidently.
- There is a **lack of immediate, personalized feedback** on pronunciation and tone.
- Many tools aren't accessible, scalable, or structured enough to support real learning.
""")

st.markdown("### Solutions")
st.markdown("""
- **AI-powered text enhancement**: spelling, grammar, tone, and engagement.
- **Real-time speech analysis** using automatic phoneme recognition.
- **AI coach** for immediate and contextual feedback.
- **Personalized lessons** based on your uploaded material.
""")

# Value Proposition
st.markdown("### Why Speakable?")
st.markdown("""
- Tailored lessons & feedback.
- Scalable across languages.
- Personalized, adaptive learning experience.
""")

# How It Works
st.markdown("### How Does It Work?")
st.markdown("""
- **Speech & Text Analysis** using advanced NLP and speech recognition.
- **Text Enhancement** with real-time grammar and tone revision.
- **Feedback Generation** using Generative AI.
- **Pronunciation Analysis** through automatic phoneme recognition.
""")

# Pricing
st.markdown("### Plans & Pricing")
st.markdown("""
- **Freemium**: Limited lessons, ad-supported
- **Premium**: $5/month, unlimited access, ad-free
- **Student Plan**: $3/month with educational email
- **Business Plans**: Volume licensing and discounts
""")

# Contact
st.markdown("### Contact Us")
st.markdown("""
- Email: [govdyla@gmail.com](mailto:govdyla@gmail.com)  
- Share: https://youspeak.streamlit.app/
""")

