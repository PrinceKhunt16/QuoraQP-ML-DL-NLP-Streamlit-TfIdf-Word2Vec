import streamlit as st
import xgboost as xgb
import tensorflow as tf
from modules.text2vec import create_feature_row_ml_model, create_feature_row_dl_model
from utils.ensemble_decision import prdiction

st.set_page_config(page_title='QuoraQP')

st.markdown(
    "<h3 style='font-size:34px;'>Quora Question Pair Detector - "
    "<a href='https://www.linkedin.com/in/prince-khunt-linked-in/' target='_blank'>Prince Khunt</a></h3>", 
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='font-size:24px;'>Enter ✌🏻 questions to see if both express the same idea.</h3>
    """, 
    unsafe_allow_html=True
)

question1 = st.text_input("Question 1:")
question2 = st.text_input("Question 2:")
result = None

xgb_model = xgb.Booster()
xgb_model.load_model('models/xgboost_model.json')
dl_model = tf.keras.models.load_model('models/rnn_model.h5')
dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

if st.button('Check'):
    questions = [question1, question2]

    if question1 and question2:  
        feature_row_ml = create_feature_row_ml_model(question1, question2)
        feature_row_dl = create_feature_row_dl_model(question1, question2)
        feature_dmatrix = xgb.DMatrix(feature_row_ml)

        xgb_predict = xgb_model.predict(feature_dmatrix)
        dl_predict = dl_model.predict(feature_row_dl)

        if prdiction({'x': xgb_predict, 'l': dl_predict[0]}):
            result = "👍🏻 Both questions express the same idea!" 
        else:
            result = "👎🏻 The questions express different ideas."

    else:
        result = "😠 Please enter both questions."

if result:
    st.markdown(f"<h4 style='font-size:20px;'>{result}</h4>", unsafe_allow_html=True)