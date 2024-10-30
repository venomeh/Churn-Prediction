import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from groq import Groq
import utils as ut


client = Groq(
  api_key="Groq_API_KEY",
)


def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)



xgboost_model = load_model('models/xgb_model.pkl')
xgboost_SMOTE_model = load_model('models/xgb_model_SMOTE.pkl')

naive_bayes_model = load_model('models/nb_model.pkl')
naive_bayes_SMOTE_model = load_model('models/nb_model_SMOTE.pkl')

random_forest_model = load_model('models/rf_model.pkl')
random_forest_SMOTE_model = load_model('models/rf_model_SMOTE.pkl')

decision_tree_model = load_model('models/dt_model.pkl')
decision_tree_SMOTE_model = load_model('models/dt_model_SMOTE.pkl')

svm_model = load_model('models/svm_model.pkl')
svm_SMOTE_model = load_model('models/svm_model_SMOTE.pkl')

knn_model = load_model('models/knn_model.pkl')
knn_SMOTE_model = load_model('models/knn_model_SMOTE.pkl')

voting_classifier_SOFT_model = load_model('models/voting_clf_soft.pkl')
voting_classifier_HARD_model = load_model('models/voting_clf_hard.pkl')

lgbm_model = load_model('models/lgbm_model.pkl')
lgbm_SMOTE_model= load_model('models/lgbm_model_SMOTE.pkl')

cb_model = load_model('models/catboost_model.pkl')
cb_SMOTE_model= load_model('models/catboost_model_SMOTE.pkl')





def prepare_input(credit_score, location, gender ,age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
  
  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': int(has_credit_card),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location == 'France' else 0,
    'Geography_Germany': 1 if location == 'Germany' else 0,
    'Geography_Spain': 1 if location == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Gender_Female': 1 if gender == 'Female' else 0
  }

  input_df = pd.DataFrame([input_dict])

  return input_df, input_dict
 





def make_predictions(input_df,input_dict):

  probabilities = {
    'XGBoost' : xgboost_SMOTE_model.predict_proba(input_df)[0][1],
    'Random Forest' : random_forest_SMOTE_model.predict_proba(input_df)[0][1],
    'K-Nearest': knn_SMOTE_model.predict_proba(input_df)[0][1],
    'LightGBM' : lgbm_SMOTE_model.predict_proba(input_df)[0][1],
    'CatBoost' : cb_SMOTE_model.predict_proba(input_df)[0][1]
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1,col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig,use_container_width=True)
    st.write(f"the customer has a {avg_probability:.2%} probability of churning")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs,use_container_width=True)

  return avg_probability



def explain_pred(probability, input_dict, surname):
  prompt=f""" You are an expert data scientist at a bank where you specialize in interpreting and explaining the predictions made by a machine learning model. 
    your machine learning model has predicted that a customer named {surname} has a probability of {round(probability*100,1)}% chance of churning, based on the information provided below.

    here is the customer's information:
    {input_dict}

    here are the machine learning models top 10 most important features for predicting churn:
    feature	importance:
    
    features - importance
    NumOfProducts -	0.323888
    IsActiveMember - 0.164146
    Age -	0.109550
    Geography_Germany	- 0.091373
    Balance	- 0.052786
    Geography_France	- 0.046463
    Gender_Female	- 0.045283
    Geography_Spain	- 0.036855
    CreditScore	- 0.035005
    EstimatedSalary	- 0.032655

    
    {pd.set_option('display.max_columns', None)}

    here are the summary statistics for churned customers:
    {df[df['Exited']==1].describe()}

    - if a customer has over a 40% chance of churning, generate a 3 sentence explanation of why they are at risk of churning.
    - if the customer has a probability of churning less than 40% , generate a 3 sentence explanation of why they are not at risk of churning.
    - your explanation should be based on customer's information, the summary statistics of the of the customers who have churned and not churned, and the features - importance provided.

    Don't mention the probability of churning ,or the machine learning model, or say anything like 'based on the model's prediction and the top 10 most important features', just explain the predictions.

    
  """

  print("Explanation Prompt: ", prompt)

  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[{
        "role": "user",
        "content": prompt
      }],
  )

  return raw_response.choices[0].message.content



def generate_email(probability, input_dict, explanation,surname):
  prompt=f""" You are a manager at HS bank. You are responsible for ensuring customers stay with the bank and are incentivized with various efforts.

    You noticed a customer named {surname} has a {round(probability*100,1)} % chance of churning.

    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

    Make sure to list out several incentives to stay based on their information, in bullet points format. Don't ever mention the probability of churning, or the machine learning model to the customer.   
  """
  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages= [{
        "role": "user",
        "content": prompt
      }]
  )

  print("\n\n EMAIL PROMPT: ", raw_response.choices[0].message.content)

  return raw_response.choices[0].message.content
  





st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_opt =st.selectbox("Select a customer", customers)


if selected_customer_opt:

  selected_customer_id = int(selected_customer_opt.split(" - ")[0])
  
  print("selected customer id",selected_customer_id)

  selected_customer_surname =selected_customer_opt.split(" - ")[1]
  
  print("selected customer surname - " ,selected_customer_surname)

  selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

  print("selected customer : ",selected_customer)

  col1, col2 = st.columns(2)

  with col1:
    
    credit_score = st.number_input(
      "Credit Score",
      min_value=300,
      max_value=850,
      value=int(selected_customer["CreditScore"])
    )
    

    location = st.selectbox(  
      "Locations",["Spain", "France", "Germany"],
      
      index=["Spain", "France", "Germany"].index(
        selected_customer['Geography']) 
    )
    
    gender = st.radio("Gender", ["Male", "Female"],
                     index=0 if selected_customer["Gender"] == "Male" else 1)

    age = st.number_input(
      "Age",
      min_value=18,
      max_value=100,
      value=int(selected_customer["Age"])
    )


    tenure = st.number_input(
      "tenure (years)",
      min_value=0,
      max_value=50,
      value=int(selected_customer['Tenure'])
    )


  
  #                              col2
  with col2:
    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value=float(selected_customer["Balance"])
    )

    num_products = st.number_input(
      "Number of products",
      min_value=1,
      max_value=10,
      value=int(selected_customer["NumOfProducts"])
    )

    has_credit_card = st.checkbox(
      "Has Credit Card",
      value=bool(selected_customer["HasCrCard"])
    )

    is_active_member = st.checkbox(
      'Is Active Member',
      value=bool(selected_customer['IsActiveMember'])
    )
    
    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value=float(selected_customer["EstimatedSalary"])
    )

  input_df, input_dict = prepare_input(credit_score, location, gender ,age, tenure, balance, num_products, has_credit_card,is_active_member,estimated_salary)

  avg_probability=make_predictions(input_df,input_dict)


  explanation = explain_pred(avg_probability, input_dict, selected_customer['Surname'])

  st.markdown("------------")

  st.subheader("Explanation of prediction")

  st.markdown(explanation)

  email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

  st.markdown("------------")
  st.subheader("Personalized Email")

  st.markdown(email)
 
 