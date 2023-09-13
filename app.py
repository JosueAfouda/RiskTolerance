import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.express as px
import cvxopt as opt
from cvxopt import solvers

investors = pd.read_csv('data/InputData.csv', index_col = 0)

assets = pd.read_csv('data/CAC40Data.csv',index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
assets=assets.fillna(method='ffill')

def predict_riskTolerance(X_input):

    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    # estimate accuracy on validation set
    predictions = loaded_model.predict(X_input)
    return predictions

#Asset allocation given the Return, variance
def get_asset_allocation(riskTolerance,stock_ticker):
    #ipdb.set_trace()
    assets_selected = assets.loc[:,stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1-riskTolerance

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    w=portfolios['x'].T
    print (w)
    Alloc =  pd.DataFrame(data = np.array(portfolios['x']),index = assets_selected.columns)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    returns_final=(np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final,axis =1)
    returns_sum_pd = pd.DataFrame(returns_sum, index = assets.index )
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + 100
    return Alloc,returns_sum_pd

# Define the Streamlit app
st.title('Investment Advisor in the CAC40 stock market')

st.header('Step 2: Asset Allocation and Portfolio Performance')
st.sidebar.title('Step 1: Enter Investor Characteristics')

# Investor Characteristics
with st.sidebar:

    age = st.slider('Age:', min_value=investors['AGE07'].min(), max_value=70, value=25)
    net_worth = st.slider('NetWorth:', min_value=-1000000, max_value=3000000, value=10000)
    income = st.slider('Income:', min_value=-1000000, max_value=3000000, value=100000)
    education = st.slider('Education Level (scale of 4):', min_value=1, max_value=4, value=2)
    married = st.slider('Married:', min_value=1, max_value=2, value=1)
    kids = st.slider('Kids:', min_value=investors['KIDS07'].min(), max_value=investors['KIDS07'].max(), value=3)
    occupation = st.slider('Occupation:', min_value=1, max_value=4, value=3)
    willingness = st.slider('Willingness to take Risk:', min_value=1, max_value=4, value=3)

    if st.sidebar.button('Calculate Risk Tolerance'):
        X_input = [[age, education, married, kids, occupation, income, willingness, net_worth]]
        risk_tolerance_prediction = predict_riskTolerance(X_input)
        st.sidebar.write(f'Predicted Risk Tolerance: {round(float(risk_tolerance_prediction[0]*100), 2)}')

# Risk Tolerance Charts

risk_tolerance_text = st.text_input('Risk Tolerance (scale of 100):')
selected_assets = st.multiselect('Select the assets for the portfolio:', 
                                 options=list(assets.columns), 
                                 default=['Air Liquide', 'Airbus', 'Alstom', 'AXA', 'BNP Paribas'])

# Asset Allocation and Portfolio Performance

if st.button('Submit'):
    Alloc, returns_sum_pd = get_asset_allocation(float(risk_tolerance_text), selected_assets)

    # Display Asset Allocation chart
    st.subheader('Asset Allocation: Mean-Variance Allocation')
    fig_alloc = px.bar(Alloc, x=Alloc.index, y=Alloc.iloc[:, 0], 
                       labels={'index': 'Assets', '0': 'Allocation'})
    st.plotly_chart(fig_alloc)

    # Display Portfolio Performance chart
    st.subheader('Portfolio value of €100 investment')
    fig_performance = px.line(returns_sum_pd, x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0], labels={'index': 'Date', '0': 'Portfolio Value'})
    st.plotly_chart(fig_performance)