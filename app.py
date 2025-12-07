import streamlit as st
import pandas as pd
from io import BytesIO
import json
import re
from openai import OpenAI
import google.generativeai as genai

genai.configure(api_key="AIzaSyCAi_HIMsp2CQ54WBjSrRmyiKSWsbxynQ0")

for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)
# ---------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Kanji Reader", layout="wide")
st.title("🈶 Japanese Kanji Reader & Translator")

st.write(
    "Paste Japanese text below. The app will extract words containing kanji "
    "and return **reading (kana)**, **romaji**, and **English translation**."
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("🔑 API Settings")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

model_choice = st.sidebar.selectbox(
    "Model",
    ["OpenAI GPT", "Google Gemini"]
)


# ---------------------------------------------------------
# JSON Extraction Helper
# ---------------------------------------------------------
def extract_json(text):
    """Attempts to extract a JSON array from the model output."""
    match = re.search(r'(\[\s*\{.*\}\s*\])', text, re.DOTALL)
    if match:
        json_text = match.group(1)
    else:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            json_text = text[start:end+1]
        else:
            return None

    try:
        return json.loads(json_text)
    except:
        try:
            cleaned = json_text.replace("'", '"')
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            return json.loads(cleaned)
        except:
            return None


# ---------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------
def build_prompt(text):
    return f"""
You are a Japanese linguist.

Extract useful Japanese words from the text below. Focus on words that contain kanji or important vocabulary.

For each word, return:
- "word": original word
- "reading_kana": hiragana/katakana pronunciation
- "romaji": romanization
- "translation_en": short English meaning

FORMAT STRICTLY AS JSON ARRAY:
[
  {{
    "word": "...",
    "reading_kana": "...",
    "romaji": "...",
    "translation_en": "..."
  }}
]

Text:
{text}
"""


# ---------------------------------------------------------
# Model Calls
# ---------------------------------------------------------
def call_openai(prompt):
    client = OpenAI(api_key=openai_key)
    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content


def call_gemini(prompt, gemini_key):
    import google.generativeai as genai

    genai.configure(api_key=gemini_key)

    # widely supported model
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    response = model.generate_content(prompt)
    return response.text






# ---------------------------------------------------------
# Main Input
# ---------------------------------------------------------
text_input = st.text_area("Enter Japanese text here", height=200)

if st.button("Generate"):
    if not text_input.strip():
        st.error("Please enter Japanese text.")
    else:
        prompt = build_prompt(text_input)

        try:
            if model_choice == "OpenAI GPT":
                if not openai_key:
                    st.error("Please enter your OpenAI API key in the sidebar.")
                    st.stop()
                raw_output = call_openai(prompt)
            else:
                if not gemini_key:
                    st.error("Please enter your Gemini API key in the sidebar.")
                    st.stop()
                raw_output = call_gemini(prompt, gemini_key)

            parsed = extract_json(raw_output)

            if parsed is None:
                st.error("Model output could not be parsed as JSON.")
                st.code(raw_output)
            else:
                df = pd.DataFrame(parsed)
                st.success("Done!")
                st.dataframe(df, use_container_width=True)

                # Download CSV
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    "Download CSV",
                    data=buffer,
                    file_name="kanji_readings.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")

