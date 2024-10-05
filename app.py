import os
import streamlit as st
from src.utils import decodeImage
from src.pipelines.prediction_pipeline import PredictionPipeline
import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
load_dotenv()

from langchain_groq import ChatGroq
api_key1 = os.getenv("GROQ_API_KEY")

# Set up the environment variables  s

# Title for the app
st.title("Hi, I am Dr. Maria, Orthopedic Specialist")
st.subheader("Please upload the X-ray for analysis")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Save the image
    image_path = "inputImage.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    #st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Create the classifier and predict
    classifier = PredictionPipeline(image_path)

    # Button to make prediction
    if st.button("Predict"):
        result = classifier.predict()
        st.write("Prediction Result:", result)
        if 'responses' not in st.session_state:
         if result == "Fractured":
            st.session_state['responses'] = ["Hi there, I am Dr. Maria! A fracture has been detected. If you'd like to ask anything or need further guidance, feel free to ask!"]
         elif result == "Non_Fractured":
            st.session_state['responses'] = ["Hi there, I am Dr. Maria! No fracture was detected. However, you can share your symptoms or ask any questions, and I’ll be happy to assist you."]
        if 'requests' not in st.session_state:
         st.session_state['requests'] = []

          # Initialize the language model
        llm = ChatGroq(groq_api_key=api_key1, model_name="llama3-8b-8192", temperature=0.6)

         # Initialize conversation memory
        if 'buffer_memory' not in st.session_state:
         st.session_state.buffer_memory = ConversationBufferWindowMemory(k=5, return_messages=True)

         # Define prompt templates
        system_msg_template = SystemMessagePromptTemplate.from_template(
         template=f"""You are Dr. Maria, an Orthopedic Specialist. After examining an X-ray report, it has been found that there is {result}. Please treat the patient accordingly and offer any further guidance as needed."""
         )
        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
 
        link = 'startup_logo1.jpg'

        # Create conversation chain
        conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

        # Container for chat history
        response_container = st.container()
        # Container for text box
        text_container = st.container()

        with text_container:
          user_query = st.chat_input("Enter your query")

          if user_query:
            with st.spinner("typing..."):
                # Pass the 'result' variable explicitly into the conversation
                response = conversation.predict(input=f"Query:\n{user_query}")
                
            # Append the new query and response to the session state  
            st.session_state.requests.append(user_query)
            st.session_state.responses.append(response)

        st.markdown(
         """
         <style>
         [data-testid="stChatMessageContent"] p{
             font-size: 1rem;
         }
         </style>
         """, unsafe_allow_html=True
         )

        # Display chat history
        with response_container:
         if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                with st.chat_message('Momos', avatar=link):
                    st.write(st.session_state['responses'][i])
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

