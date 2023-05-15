import nltk
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# nltk.download('omw-1.4')
# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Read in dataset
with open('C:/Users/Windows/Desktop/GOMYCODE/Personal Projects/Streamlit Dev/Mychatbox/full_dialogs (1).txt', 'r', encoding='utf8', errors='ignore') as file:
    dataset = file.read()



sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

# Vectorize corpus
vectorize = TfidfVectorizer()
X = vectorize.fit_transform(corpus)



# Define chatbot function
def chatbot_response(user_input):
    # Preprocess user input
    user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

    # Vectorize user input
    user_vector = vectorize.transform([user_input])

    # Calculate cosine similarity between user input and corpus
    similarities = cosine_similarity(user_vector, X)

    # Get index of most similar sentence
    idx = similarities.argmax()

    # Return corresponding sentence from corpus
    return sent_tokens[idx]

# Run chatbot
print("Welcome to the chatbot! How can I help you today?")

while True:
    user_input = input("> ")
    if user_input.lower() == 'quit':
        break
    response = chatbot_response(user_input)
    print(response)


#-----------Streamlit Implementation ---------------
import streamlit as st


st.title("CHATBOT MACHINE.")
st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

quit_sentences = ['quit', 'bye', 'Goodbye', 'exit']

history = []

st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# Get the user's question    
user_input = st.text_input(f'Input your response')
if user_input not in quit_sentences:
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot_response(user_input)
        st.write("Chatbot: " + response)

        # Create a history for the chat
        history.append(('User: ', user_input))
        history.append(('Bot: ', chatbot_response(user_input)))
else:
    st.write('Bye')

st.markdown('<hr><hr>', unsafe_allow_html= True)
st.subheader('Chat History')

chat_history_str = '\n'.join([f'{sender}: {message}' for sender, message in history])

st.text_area('Conversation', value=chat_history_str, height=300)




# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# import numpy as np
# import time
# import os
# import torch

# class ChatBot():
#     # initialize
#     def __init__(self):
#         # once chat starts, the history will be stored for chat continuity
#         self.chat_history_ids = None
#         # make input ids global to use them anywhere within the object
#         self.bot_input_ids = None
#         # a flag to check whether to end the conversation
#         self.end_chat = False
#         # greet while starting
#         self.welcome()
        
#     def welcome(self):
#         print("Initializing ChatBot ...")
#         print('Type "bye" or "quit" or "exit" to end chat \n')

#         # Greet and introduce
#         greeting = np.random.choice([
#             "Welcome, I am ChatBot, here for your kind service",
#             "Hey, Great day! I am your virtual assistant",
#             "Hello, it's my pleasure meeting you",
#             "Hi, I am a ChatBot. Let's chat!"
#         ])
#         print("ChatBot >>  " + greeting)
        
#     def user_input(self):
#         # receive input from user
#         text = input("User    >> ")
#         # end conversation if user wishes so
#         if text.lower().strip() in ['bye', 'quit', 'exit']:
#             # turn flag on 
#             self.end_chat=True
#             # a closing comment
#             print('ChatBot >>  See you soon! Bye!')
#             # time.sleep(1)
#             print('\nQuitting ChatBot ...')
#         else:
#             # continue chat, preprocess input text
#             # encode the new user input, add the eos_token and return a tensor in Pytorch
#             self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, \
#                                                        return_tensors='pt')

#     def bot_response(self):
#         # append the new user input tokens to the chat history
#         # if chat has already begun
#         if self.chat_history_ids is not None:
#             self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1) 
#         else:
#             # if first entry, initialize bot_input_ids
#             self.bot_input_ids = self.new_user_input_ids
        
#         # define the new chat_history_ids based on the preceding chats
#         # generated a response while limiting the total chat history to 1000 tokens, 
#         self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, \
#                                                pad_token_id=tokenizer.eos_token_id)
            
#         # last ouput tokens from bot
#         response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], \
#                                skip_special_tokens=True)
#         # in case, bot fails to answer
#         if response == "":
#             response = self.random_response()
#         # print bot response
#         print('ChatBot >>  '+ response)
        
#     # in case there is no response from model
#     def random_response(self):
#         i = -1
#         response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
#                                skip_special_tokens=True)
#         # iterate over history backwards to find the last token
#         while response == '':
#             i = i-1
#             response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
#                                skip_special_tokens=True)
#         # if it is a question, answer suitably
#         if response.strip() == '?':
#             reply = np.random.choice(["I don't know", 
#                                      "I am not sure"])
#         # not a question? answer suitably
#         else:
#             reply = np.random.choice(["Great", 
#                                       "Fine. What's up?", 
#                                       "Okay"
#                                      ])
#         return reply

# # build a ChatBot object
# bot = ChatBot()
# # start chatting
# while True:
#     # receive user input
#     bot.user_input()
#     # check whether to end chat
#     if bot.end_chat:
#         break
#     # output bot response
#     bot.bot_response()