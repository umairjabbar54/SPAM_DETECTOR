import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()


# Text preprocessing function
def transform_text(text):
  text = text.lower()
  text = text.split()

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return ' '.join(y)


# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit User Interface Design
st.title('📧 SMS Spam Detector')
st.markdown(
    'Type any text message below to check whether it is **Spam** or **Not Spam'
    ' (Ham)**.'
)

# Text box input for the user
input_sms = st.text_area('Enter the SMS message:')

if st.button('Predict'):
  if input_sms.strip() == '':
    st.warning('Please enter a message to analyze!')
  else:
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display result
    if result == 1:
      st.error('🚨 This is a SPAM message!')
    else:
      st.success('✅ This is NOT SPAM (Ham)!')