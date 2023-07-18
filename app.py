import streamlit as st
from transformers import pipeline

# Load the pre-trained model

classfier = pipeline("sentiment-analysis")

# Define the Streamlit app

# create the app interface
st.title('Sentiment Analysis App')
usr_inp = st.text_input('User:')
#chat_area = st.text_area("Chatbot:", disabled=True)

predict_btn = st.button('Predict')

# perform prediction on user input
if predict_btn:
    sentiment = classfier(usr_inp)
    for res in sentiment:        
      #st.write('The Sentiment is:', res['label'])
      response = str(res['label'])

if usr_inp:
   # Display the response in the chat area
   st.text_area("Chatbot:",value=response, height=100, key="chat_area" , )

st.markdown(
   """
    <style>
    .stTextInput>div>div>input {background-color: #f0f0f0; color :#000;}
    .stTextArea>div>div>textarea {background-color: #f0f0f0; color :#000;}
    </style>
   """,
   unsafe_allow_html=True
)



