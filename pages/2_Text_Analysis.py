import os
import ast
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if "selected_model" not in st.session_state:
    selected_model = "gemini-1.5-flash"
    st.session_state["selected_model"] = selected_model

llm = ChatGoogleGenerativeAI(model=st.session_state["selected_model"])
       

sentence_structure = '''

Provided Information starts here.

**Sentence Structure**
- Subject: Every sentence must clearly state who or what it is about.
- Predicate: Every sentence must describe what the subject is doing or being.
- Clause Structure: Ensure each sentence has at least one independent clause.

**Grammar Correction**
- Subject-Verb Agreement: The verb must agree in number and person with its subject.
- Verb Tense: Use consistent and appropriate verb tenses.
- Pronoun Usage and Agreement: Pronouns must agree with their antecedents in number and gender.
- Article Use: Use 'a', 'an', and 'the' correctly before nouns.
- Modifier Placement: Place adjectives and adverbs correctly to avoid confusion.
- Punctuation: Ensure correct use of commas, periods, and other punctuation marks.
- Capitalization: Capitalize the first word of every sentence and all proper nouns.
- Spelling: Check all words for correct spelling.

**Semantic and Logical Coherence**
- Sentence Completeness: Each sentence should express a full, standalone thought.
- Clarity: Avoid vague or confusing phrases.
- Conciseness: Eliminate unnecessary words.
- Logical Flow: Ensure ideas follow a natural, logical progression.

**Style and Tone**
- Tone and Register: Keep the tone polite and appropriate to context.
- Sentence Variety: Use a mix of simple, compound, and complex sentences.
- Voice (Active/Passive): Prefer active voice unless passive is contextually better.
- Idiomatic Expressions: Use natural English phrases and collocations.
- Parallel Structure: Use consistent grammar in lists and paired elements.

Provided Information ends here.
'''

prompt = f''' You are an excellent English teacher, and have the following provided information about how to improve English sentences: {sentence_structure}

With the information provided, please review the following text and return it with corrections. Do not change the text structure.

Feel free to add or remove words or suggest different expressions if you think they will improve the clarity of the text, but do not change the text structure. Consider that the text needs to be polite and friendly.

Return a **Python list of dictionaries**, where each dictionary represents a sentence. If a sentence is being corrected or does not align with the provided information, the dictionary must include the following four keys:
- "reviewed_text": A string containing the corrected version of the sentence.
- "explanation": A Python list of corrections made, with each entry being a short, clear reason based on the provided information.
- "score": A number between 0 and 100 representing how well the original sentence was written.
- "sentences": A Python list of dictionaries, each representing a corrected sentence. Each dictionary in this list must include the following five keys:
    - "sentence number": The position of the sentence in the original text.
    - "sentence": The original sentence.
    - "review": The corrected or revised version of the sentence.
    - "section": A list of the relevant section(s) from the provided information that support the correction.
    - "subsection": A list of the relevant subsection(s) from the provided information, each with a clear description that justifies the correction based on the provided information.

**Example of the explanation property:**
"explanation": [
    "Moved the sentence to create a better flow of information.",
    "Changed \"too\" to \"also\" for more formal language.",
    "Shortened \"which is something I enjoy playing\" to \"which I enjoy playing\" for conciseness.",
    "Replaced \"it\" with \"this visit\" and \"the same as last time\" with \"as good as the last one\" for clarity and better style."
]

**Example of a dictionary in the sentences array:**
{{
    "sentence number": "4",
    "sentence": "This is an incorrect sentence.",
    "review": "This is a correct sentence.",
    "section": ["Semantic and Logical Coherence", "Style and Tone"],
    "subsection": [
        "Conciseness: Removed redundant words for a more concise sentence.",
        "Sentence Variety: Rephrased for better flow and variety.",
        "Clarity: Clarified the meaning by removing vague phrasing.",
        "Voice (Active/Passive): Changed to active voice for clarity and impact."
    ]
}}

**Rules:**
- Use the provided information: {sentence_structure}
- Only use the provided information to justify corrections.
- Do not change the text structure and do not summarise the text unless it does not conform with the provided information.
- Ensure that "section" values match exactly with those in the provided information.
- Ensure that "section" values match exactly with those in the provided information. Ensure that the description of the correction is based on the provided information.
- Ensure that the number of sentences is correct and that each sentence is numbered correctly.
- If a sentence requires no correction, **do not include it** in the output.
- Strictly follow the format of the examples.
- Return only valid Python syntax with properly formatted dictionaries and lists, including all quotation marks and commas.
'''

if st.session_state.get("show_success", False):
    st.session_state.show_success = True  
    
st.markdown("### Text Analysis")

if "show_results" not in st.session_state:
    st.session_state["show_results"] = False
    
if "human_text" not in st.session_state:
    st.session_state["human_text"] = ""

example_text = '''My name is John Doe. Today I am going to the toy store with my mother, who wants to buy a video game. Tomorrow is my birthday. I want to buy a video game too because my favorite game is about football, which is something I enjoy playing, although not everyone likes it. I hope it will be the same as last time.'''

human = st.text_area(
    label="Paste Text",
    label_visibility="collapsed",
    value=st.session_state.get("human_text", ""),
    placeholder="Enter your text here...",
    height=300
)

col1, col2 = st.columns([1, 7])

with col2:     
    if st.button("Try Example"):
        st.session_state["human_text"] = example_text
        st.rerun()

try:
    
    with col1:
        if st.button("Analyse"):
            if human.strip():
                messages = [
                    ("system", prompt),
                    ("human", human),
                ]

                result = llm.invoke(messages).content

                try:
                    result = result.strip().removeprefix("```python").removesuffix("```").strip()
                    
                    data = ast.literal_eval(result)[0]
                    
                    st.session_state["human_text"] = human
                    st.session_state["reviewed_text"] = data['reviewed_text']
                    st.session_state["explanation"] = data['explanation']
                    st.session_state["score"] = data['score']
                    st.session_state["sentences"] = data['sentences']
                    st.session_state["show_results"] = True
                    
                except ValueError as e:
                    st.warning(
                        "There has been an error in parsing your text. Please analyse your text again."
                    )
                    
                except SyntaxError as e:
                    st.warning(
                        "There has been an error in parsing your text. Please analyse your text again."
                    )

            else:
                st.warning("Please paste in your text in the text field above.")
            
except ResourceExhausted as e:
    st.warning(
        "The model that is analysing your text is currently exhausted. Please go to the **Settings** tab and select a different model."
    )
    
if st.session_state.get("show_results", False):
    st.markdown("### Reviewed Text")
    st.success(st.session_state['reviewed_text'])

    st.markdown("### Explanation")
    full_explanation = ""
    for explanation in st.session_state['explanation']:
        full_explanation += "\n- " + explanation
    st.info(full_explanation)

    st.markdown("### Text Score")
    score = int(st.session_state['score'])
    if score > 49:
        st.success(f"Score: **{score}**")
    else:
        st.warning(f"Score: **{score}**")

    st.markdown("### Corrections")
    for sentence in st.session_state['sentences']:
        with st.expander(f"**Sentence {sentence['sentence number']}**"):
            st.markdown(f"**Original Sentence:** {sentence['sentence']}")
            st.markdown(f"**Reviewed Sentence:** {sentence['review']}")
            
            sections = ""
            for section in sentence["section"]:
                sections += "\n - " + "**" + section + "**"
            st.markdown(f"**Section:** {sections}")
            
            subsections = "\n".join([
                f"- **{s.split(':')[0].strip()}**: {':'.join(s.split(':')[1:]).strip()}"
                for s in sentence["subsection"]
            ])
            st.markdown(f"**Subsection:**\n{subsections}")

