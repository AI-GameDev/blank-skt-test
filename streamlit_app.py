import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Streamlit App Configuration
st.set_page_config(page_title="Text Summarizer with GPT-4o-mini", page_icon=":memo:", layout="wide")

# Sidebar: API Key Input
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Streamlit Main Section
st.title("Text Summarizer with GPT-4o-mini")
st.write("Use this tool to summarize any text using the power of GPT-4o-mini!")

# Text Area Input
input_text = st.text_area("Enter the text you want to summarize", height=200)

# Check if the API key is provided
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
else:
    # Create the LangChain LLM with OpenAI API Key
    llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key, model_name="gpt-4o-mini")

    # Prompt Template for Summarization
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="다음 텍스트에 대한 간결한 요약을 입력하세요:\n{text}\n요약:"
    )

    # Create an LLM Chain
    summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Button to trigger the summarization
    if st.button("Summarize Text"):
        if input_text.strip():
            # Run the summarization chain
            summary = summarization_chain.run(text=input_text)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")