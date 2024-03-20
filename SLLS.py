import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Set OpenAI API key and Eleven API key
os.environ["OPENAI_API_KEY"] = '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'

# App framework
st.title("📕 Smart Language Learning System")
prompt = st.text_input("Put your English paragraph for getting correct grammar and pronunciation.")

# Prompt templates for the first sequence
title_template_first = PromptTemplate(
    input_variables=['topic'],
    template='Write the correct sentence only: {topic}'
)
script_template_first = PromptTemplate(
    input_variables=['title'],
    template='Analyze the paragraph titled "Para" below, focusing on the grammatical structure and tense of each line. Provide the analysis for each line as follows:Para Line 1:breakTense:breakSentence structure:break{title}'
)

# LLMS for the sequence
llm = OpenAI(temperature=0.9)

# Chains for the sequence
title_chain_first = LLMChain(llm=llm, prompt=title_template_first, verbose=True)
script_chain_first = LLMChain(llm=llm, prompt=script_template_first, verbose=True)

# Define the sequence
sequence = SimpleSequentialChain(chains=[title_chain_first, script_chain_first], verbose=True)

# Show stuff on the screen if there is a prompt
if prompt:
    # Run the sequence
    response_title_first = title_chain_first.run(topic=prompt)
    response_script_first = script_chain_first.run(title=response_title_first)
    
    # Display the responses
    st.title("Corrected Paragraph:")
    st.write(response_title_first)
    st.title("Sentence Structure analysis:")
    st.write(response_script_first)
