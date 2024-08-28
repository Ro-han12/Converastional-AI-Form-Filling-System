import os
import json
import sqlite3
import streamlit as st
from langchain.chains import ConversationChain, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

# Setup OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY")

llm = None

# Define the PersonalDetails model
class PersonalDetails(BaseModel):
    name: str = Field(None, description="The name of the user.")
    city: str = Field(None, description="The name of the city where someone lives.")
    email: str = Field(None, description="An email address that the person associates as theirs.")
    age: int = Field(None, description="Age of that person associates as theirs.")

# Define the default template
_DEFAULT_TEMPLATE = """You are an interactive conversational chatbot. Your goal is to collect user information in a conversational and non-intrusive manner, one piece at a time. When asking for details, explain why you need them, and be persuasive yet empathetic. Build rapport by transitioning into small talk when appropriate, but aim to gather data smoothly as the conversation progresses. If a user hesitates or is unsure, provide reassurance, offer alternatives, and if the user wishes to correct or update their details, be flexible and handle it trustworthily. If no information is needed, thank the user and ask how you can further assist them. Do not repeat same questions to the same users.

Previous conversation:
{history}
Recent user input:
{input}
Information to ask for (do not ask as a list):
### ask_for list: ask_for_list
Available information of user: avl_info_list
"""

def create_db():
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            session_id TEXT PRIMARY KEY,
            name TEXT,
            city TEXT,
            email TEXT,
            age INT
        )
    ''')
    conn.commit()
    conn.close()

def save_data_to_db(session_id, details):
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO user_data (session_id, name, city, email, age) VALUES (?, ?, ?, ?,?)
    ''', (session_id, details.get('name'), details.get('city'), details.get('email'),details.get('age')))
    conn.commit()
    conn.close()

def load_data_from_db():
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    c.execute('SELECT * FROM user_data')
    data = c.fetchall()
    conn.close()
    return data

def view_user_data(session_id):
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    c.execute('SELECT * FROM user_data WHERE session_id = ?', (session_id,))
    data = c.fetchone()
    conn.close()
    if data:
        return {
            'session_id': data[0],
            'name': data[1],
            'city': data[2],
            'email': data[3],
            'age': data[4]
        }
    return None

create_db()  # Create the database and tables

def check_what_is_empty(user_personal_details: PersonalDetails) -> list:
    ask_for = []
    for field, value in user_personal_details.dict().items():
        if value in [None, "", 0]:
            ask_for.append(field)
    return ask_for

def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails) -> PersonalDetails:
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details

def conversation_chat(input: str, session_id: str, llm=None) -> str:
    if session_id:
        existing_info_of_user = view_user_data(session_id)
        if existing_info_of_user:
            existing_info_of_user = PersonalDetails(**existing_info_of_user)
        else:
            existing_info_of_user = PersonalDetails()

    # Check if user wants to end the conversation
    if input.lower() in ["stop", "exit"]:
        save_data_to_db(session_id, existing_info_of_user.dict())
        return f"Your data has been saved. Here is your stored information:\n\n{json.dumps(existing_info_of_user.dict(), indent=4)}\n\nThank you for chatting with us. Goodbye!"

    ner_chain = create_tagging_chain_pydantic(PersonalDetails, llm)
    extractions = ner_chain.run(input)  # Extract information using your NER chain
    existing_info_of_user = add_non_empty_details(existing_info_of_user, extractions)
    
    # Check if we have enough information, if not ask for it
    ask_for = check_what_is_empty(existing_info_of_user)

    # Update in-memory user data
    save_data_to_db(session_id, existing_info_of_user.dict())

    memories = ConversationBufferMemory(k=3)
    PROMPT = PromptTemplate(
        input_variables=["history", "input"],
        template=_DEFAULT_TEMPLATE.replace("ask_for_list", f"{ask_for}").replace("avl_info_list", f"{existing_info_of_user}"),
    )
    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=PROMPT,
        memory=memories
    )

    conv = conversation.predict(input=input)
    return conv

st.title("Conversational Form-Filling AI System‍")

def initialize_session_state():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Am here  🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! Am here to improve your conversational experience & gather details regarding you. 👋"]

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = None

def display_chat_history(session_id: str, llm=None):
    if not session_id:
        st.warning("Please enter a session ID in the sidebar")
        return

    reply_container = st.container()
    container = st.container()

    with container:
        # Form for user input
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about what you want to learn", key='input')
            submit_button = st.form_submit_button(label='Send')

        # Handle form submission
        if submit_button and user_input:
            output = conversation_chat(user_input, session_id=session_id, llm=llm)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        # Stop button outside the form
        stop_button = st.button("Stop")

        # Handle stop button click
        if stop_button:
            output = conversation_chat("stop", session_id=session_id, llm=llm)
            st.session_state['past'].append("stop")
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="User")
                message(st.session_state["generated"][i], key=str(i), avatar_style="initials", seed="AI")

def main():
    global llm  # Declare llm as global here to modify it within this function
    initialize_session_state()
    with st.sidebar:
        with st.form(key='sidebar_form'):
            session_id = st.text_input("Session ID", value=st.session_state.get('session_id', ''))
            gpt_token = st.text_input("GPT Token", value='')
            submit_button = st.form_submit_button(label='Update')

        if submit_button:
            st.session_state['session_id'] = session_id
            if gpt_token:
                os.environ["OPENAI_API_KEY"] = gpt_token

    if gpt_token:
        llm = ChatOpenAI(model_name="gpt-4", api_key=gpt_token)

    if st.session_state.get('session_id'):
        display_chat_history(session_id=st.session_state['session_id'], llm=llm)
        
        with st.sidebar:
            if st.button("View Data"):
                user_data_str = view_user_data(st.session_state['session_id'])
                if user_data_str:
                    st.sidebar.text_area("User Data", value=json.dumps(user_data_str, indent=4), height=300)
                else:
                    st.sidebar.text_area("User Data", value="No data found for this session ID", height=300)

if __name__ == "__main__":
    main()
