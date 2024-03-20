import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Import pyttsx3 library
import pyttsx3

# Set OpenAI API key and Eleven API key
os.environ["OPENAI_API_KEY"] = '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'
os.environ["ELEVEN_API_KEY"] = "3ffa54ac53786416ead207dca5ebc721"

# Initialize pyttsx3 engine
engine = pyttsx3.init()

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

# Prompt templates for the second sequence
title_template_second = PromptTemplate(
    input_variables=['topic'],
    template='Write the correct sentence only: {topic}'
)
audio_template_second = PromptTemplate(
    input_variables=['title'],
    template='{title}'
)

# LLMS for both sequences
llm = OpenAI(temperature=0.9)

# Chains for the first sequence
title_chain_first = LLMChain(llm=llm, prompt=title_template_first, verbose=True)
script_chain_first = LLMChain(llm=llm, prompt=script_template_first, verbose=True)

# Chains for the second sequence
title_chain_second = LLMChain(llm=llm, prompt=title_template_second, verbose=True)

# Define the two sequences
first_sequence = SimpleSequentialChain(chains=[title_chain_first, script_chain_first], verbose=True)
second_sequence = SimpleSequentialChain(chains=[title_chain_second], verbose=True)

# Show stuff on the screen if there is a prompt
if prompt:
    # Run the first sequence
    response_title_first = title_chain_first.run(topic=prompt)
    response_script_first = script_chain_first.run(title=response_title_first)
    
    # Display the responses from the first sequence
    st.title("Corrected Paragraph:")
    st.write(response_title_first)
    st.title("Sentence Structure analysis:")
    st.write( response_script_first)
    
    # Run the second sequence
    response_title_second = title_chain_second.run(topic=prompt)
    response_title_second_text = response_title_second  # Store the title response for the second sequence
    
    # Display the response from the second sequence
    #st.write("Response from the second sequence:")
    #st.write("Title:", response_title_second)
    
    # Convert text to speech using pyttsx3
    engine.say(response_title_second_text)
    engine.runAndWait()
