import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle

model = pickle.load(open('StartUp_Model.pkl', 'rb'))
st.markdown("<h1 style = 'text-align: center; color: 3D0C11'>START UP PROJECT</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: center; color: #FFB4B4'>Built by GoMyCode Sanaith Wizard</h6>", unsafe_allow_html = True)
st.image('pngwing.com (2).png', width = 400)


st.subheader('Project Brief')

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #FFB4B4'> In the dynamic and ever-evolving landscape of entrepreneurship, startups represent the vanguard of innovation and economic growth. The inception of a new venture is often accompanied by great enthusiasm and ambition, as entrepreneurs strive to transform their groundbreaking ideas into successful businesses. However, one of the central challenges faced by startups is the uncertainty surrounding their financial sustainability and profitability. This uncertainty is exacerbated by a myriad of factors,<br> ranging from market volatility and competition to operational costs and customer acquisition.</p>", unsafe_allow_html = True)

st.markdown("<br><br>", unsafe_allow_html = True)

username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

data = pd.read_csv('https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv')
heat = plt.figure(figsize = (14, 7))
sns.heatmap(data.drop('State', axis = 1).corr(), annot = True, cmap = 'BuPu')

st.write(heat)

st.write(data.sample(10))

st.sidebar.image('pngwing.com (4).png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your preffered Input type', ['Slider Input', 'Number Input'])

if input_type == "Slider Input":
    research = st.sidebar.slider("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.slider("Administration", data['Administration'].min(), data['Administration'].max())
    market = st.sidebar.slider("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())
else:
    research = st.sidebar.number_input("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.number_input("Administration", data['Administration'].min(), data['Administration'].max())
    market = st.sidebar.number_input("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())
    
input_variable = pd.DataFrame([{"R&D Spend":research, "Administration": admin, "Marketing Spend": market}])
st.write(input_variable)

pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
with pred_result:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with interpret:
    st.subheader('Model Interpretation')
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")
    # s = pd.read_csv('https://docs.google.com/spreadsheets/d/1KPmqPZpfGLtAPXSnbpdM3i7kaV6cxE8IxRAuvsHkfxY/edit?usp=sharing')