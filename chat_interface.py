import json
import time
import pandas as pd
from PIL import Image
import streamlit as st
from agents.MainBot import MainBot

path="http://localhost:8050/"
bot_img_path = "https://azure.blob.core.windows.net/ai_demo/logo.webp"
user_img_path = "https://azure.blob.core.windows.net/ai_demo/logo.png"

page_logo = Image.open("static/favicon.ico")

st.set_page_config(
        page_title='Main Demo',
        page_icon=page_logo
    )

#Hiding Made With Streamlit icon
hide_streamlit_style = """
            <style>
                footer {
                    visibility: hidden;
                    height: 0%;
                }
                .css-1li7dat {
                    visibility: hidden !important;
                }
            </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "file_df" not in st.session_state:
    st.session_state["file_df"] = None
if "log_text" not in st.session_state:
    st.session_state["log_text"] = None
if "chart_data" not in st.session_state:
    st.session_state["chart_data"] = []
if "chart_data_indices" not in st.session_state:
    st.session_state["chart_data_indices"] = []
if "system_message" not in st.session_state:
    st.session_state["system_message"] = None
if "past" not in st.session_state:
    st.session_state["past"] = []


main_bot = MainBot(st.session_state.chat_history, st.session_state.system_message, st.session_state.file_df, st.session_state.log_text)


def clear_text():
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""

def clear_conversation():
    del st.session_state.past[:]
    del st.session_state.chat_history[:]
    del st.session_state.chart_data[:]
    del st.session_state.chart_data_indices[:]

    del st.session_state.system_message
    del st.session_state.file_df
    del st.session_state.log_text


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)


def write_response(message, response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        message.bar_chart(df)

        expander = message.expander("Click to see data used in chart")
        expander.dataframe(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        message.line_chart(df)

        expander = message.expander("Click to see data used in chart")
        expander.dataframe(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


def handle_prompt(message, prompt: str) -> str:
    st.session_state.past.append(prompt)

    try:
        start_time = time.time()
        intent, response, history, system, df, log_text = main_bot.query_bot(prompt)
        end_time = time.time()
        print(f"Total time taken to execute request: {end_time - start_time}")
        st.session_state.chat_history=history
        st.session_state.system_message=system
        st.session_state.file_df=df
        st.session_state.log_text=log_text

        if intent == 'DrawChart' or intent == 'LineChart' or intent == 'BarChart':
            print("Type of response: ", type(response))
            print("Response: ", response)
            key = len(st.session_state.chat_history) - 1
            st.session_state.chart_data_indices.append(key)
            decoded_response = decode_response(response.replace("'", '"'))
            st.session_state.chart_data.append({ "key": key, "data": decoded_response})
            write_response(message, response_dict=decoded_response)
        
        else:
            message.markdown(response)

        
    except Exception as e:
        if len(st.session_state.chart_data_indices) > 0:
            st.session_state.chart_data_indices.pop()

        if len(st.session_state.chart_data_indices) > 0:
            st.session_state.chart_data.pop()
        
        msg = st.session_state.chat_history.pop()
        error_msg = 'Sorry an unexpected error has occured. Please try again or ask a different question.'
        msg["content"] = error_msg
        message.write(error_msg)
 

titleHtml="""
        <h1 style='text-align: center; margin:0 10px'>
            <img src="https://azure.blob.core.windows.net/ai_demo/favicon.ico" alt="Company" style='margin:10px'>
            Provision Demo
        </h1>
    """
st.markdown(titleHtml, unsafe_allow_html=True)

doc_link = 'https://google.com'
default_message = f"Based on your previous search history, <a href='{doc_link}' target='_blank'>here</a> are a few entity snapshots you might be interested in."


chat_container = st.container()

with chat_container:
    with st.chat_message("assistant", avatar=bot_img_path):
        st.markdown(default_message, unsafe_allow_html=True)
        st.write("Hi, how may I help you?")

    #non_system_messages = [message for message in st.session_state.chat_history if message["role"] != "system"]
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] != "system":
            chat_message = st.chat_message(message["role"], avatar=user_img_path if message["role"] == "user" else bot_img_path)
            if idx in st.session_state.chart_data_indices:
                response = next((response for response in st.session_state.chart_data if response["key"] == idx), None)
                write_response(chat_message, response['data'])
            else:
                chat_message.markdown(message["content"])


    if prompt := st.chat_input("Ask here"):
        # Add user message to chat history
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_img_path):
            st.markdown(prompt)
        # Display assistant response in chat message container
        message = st.chat_message("assistant", avatar=bot_img_path)
        full_response = handle_prompt(message, prompt)

# with st.container():
#     st.button("Clear Conversation", on_click=clear_conversation, type='secondary')

