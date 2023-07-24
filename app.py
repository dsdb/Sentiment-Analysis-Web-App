import streamlit as st
#from preprocess import preprocess_text
import torch
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification #2407


# Load the pre-trained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# save the model 
# Saved model size excceds the allowed limit in the git repo
#save_directory = "Models"
#model.save_pretrained(save_directory)
#tokenizer.save_pretrained(save_directory)

#@st.cache_resource
#def get_model():
    #model = AutoModelForSequenceClassification.from_pretrained(save_directory)

#@st.cache_resource
#def get_tokenizer():
    #tokenizer = AutoModelForSequenceClassification.from_pretrained(save_directory)

# Inference using pipeline()
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
X_train = ("Congratulations again on your promotion")
res_postive = classifier(X_train)

X_train =("We regret to inform you that we no longer have the item from your order in the stock")
res_negative = classifier(X_train)


# Define the Streamlit app


# create the app interface
st.title('Sentiment Analysis App')
usr_inp = st.text_area('Input Text:')
#chat_area = st.text_area("Chatbot:", disabled=True)

predict_btn = st.button('Predict')

# perform prediction on user input
if predict_btn:
#    input_text = preprocess_text(usr_inp)
    sentiment = classifier(usr_inp)
    response = sentiment[0]['label']      
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



