import streamlit as st

st.markdown("### Settings")

models = {
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Flash-8B": "gemini-1.5-flash-8b",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite"
}

def get_model_name_by_value(value):
    for name, val in models.items():
        if val == value:
            return name
    return list(models.keys())[1]  

if "selected_model" in st.session_state:
    default_model_name = get_model_name_by_value(st.session_state["selected_model"])
else:
    default_model_name = "Gemini 1.5 Flash"

selected_model_name = st.selectbox(
    "**Select a Model:**",
    options=list(models.keys()),
    index=list(models.keys()).index(default_model_name)
)

st.session_state["selected_model"] = models[selected_model_name]

st.success(f"Selected: {selected_model_name} (`{models[selected_model_name]}`)")

st.markdown("### FAQ")
st.info("Gemini 1.5 Flash is the recommended model for this application.")
