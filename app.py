import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    from wordcloud import WordCloud
except ImportError:
    os.system("pip install wordcloud")
    from wordcloud import WordCloud

# Ensure required NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean and preprocess text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Load Dataset
def load_dataset():
    try:
        df = pd.read_csv("email.csv")
        df['Message'] = df['Message'].apply(clean_text)
        return df
    except FileNotFoundError:
        st.error("🚨 Error: Dataset file 'email.csv' not found!")
        return pd.DataFrame(columns=["Message", "Category"])

# Train & Save Model
def train_spam_detector(df):
    if df.empty:
        st.warning("⚠️ No data available for training!")
        return None, 0
    X = df['Message']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, "spam_model_v2.pkl")
    return model, acc

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", layout="wide")
st.sidebar.title("🔍 Menu")
page = st.sidebar.radio("Navigate to", ["Overview", "Interactive Features", "Data Analysis", "Spam Classifier"])

if page == "Overview":
    st.title("📨 Email Spam Classification App")
    df = load_dataset()
    st.write("### Preview of the Dataset:")
    st.dataframe(df.head())
    
    if os.path.exists("spam_model_v2.pkl"):
        model = joblib.load("spam_model_v2.pkl")
        st.write("✅ Model Successfully Loaded!")
    else:
        st.write("🚀 Training Model... Please wait.")
        model, accuracy = train_spam_detector(df)
        if model:
            st.write(f"🎯 Model Trained! Accuracy: **{accuracy:.2f}**")

elif page == "Interactive Features":
    st.title("🛠️ User Interactive Features")
    name = st.text_input("Your Name:")
    if st.button("Say Hello"):
        st.write(f"Hello {name}, welcome to the Spam Detection App!")
    
    # Slider widget
    number = st.slider("Select a number", 1, 100)
    st.write(f"You selected: {number}")
    
    # Checkbox widget
    checkbox = st.checkbox("Show a message")
    if checkbox:
        st.write("✅ Checkbox is checked!")
    
    # Radio button for expertise level
    level = st.radio("Choose your skill level:", ["Beginner", "Intermediate", "Advanced"])
    st.write(f"Your skill level: {level}")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file of emails", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview:")
        st.dataframe(df_uploaded.head())

elif page == "Data Analysis":
    st.title("📊 Data Insights")
    df = load_dataset()
    
    if not df.empty:
        st.subheader("📊 Spam vs. Ham Frequency")
        fig, ax = plt.subplots()
        sns.countplot(x=df['Category'], palette=['blue', 'orange'], ax=ax)
        plt.xlabel("Email Type")
        plt.ylabel("Count")
        st.pyplot(fig)
    
        st.subheader("🔠 Most Frequent Words in Spam Emails")
        spam_words = " ".join(df[df['Category'] == 'spam']['Message'])
        if spam_words.strip():
            wordcloud = WordCloud(width=900, height=500, background_color="black").generate(spam_words)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough spam messages for a word cloud.")

elif page == "Spam Classifier":
    st.title("🔎 Detect Spam Emails")
    user_input = st.text_area("📩 Enter Email Content:")
    
    if st.button("🚀 Predict"):
        if user_input.strip():
            cleaned_input = clean_text(user_input)
            if os.path.exists("spam_model_v2.pkl"):
                model = joblib.load("spam_model_v2.pkl")
                prediction = model.predict([cleaned_input])[0]
                if prediction == "spam":
                    st.error("🚨 This is a SPAM message!")
                else:
                    st.success("✅ This is a HAM (safe) message.")
            else:
                st.warning("⚠️ Model file not found. Please train the model first.")
        else:
            st.warning("⚠️ Please enter a valid email message.")

# Styling
st.markdown(
    """
    <style>
    .stTitle { color: #009688; text-align: center; }
    .stButton { background-color: #ff5722; color: white; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)
