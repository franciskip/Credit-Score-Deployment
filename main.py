#You need this to use FastApi, work with statues and be able to end HTTPExceptions
from fastapi import FastAPI, status, HTTPException

#Both used for BaseModel
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.responses import FileResponse

# You need to be able to turn classes into JSONs and return 
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import os

main_folder = os.path.dirname(os.path.abspath(__file__))
myfile = os.path.join(main_folder, 'data/loan_data.csv')

app = FastAPI()

###################################################################################################################
#Ponto Endpoint
import pandas as pd
import numpy as np
import warnings
import math
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from math import sqrt
import category_encoders as ce
#from utils import encode_cat_variables


data = pd.read_csv(myfile)

r = 0.13
interestRate=r/data['Number_of_monthly_installments']
data['loan_limit']            = abs(np.pv(interestRate/data['Number_of_monthly_installments'],

                                data['Number_of_monthly_installments']*1,

                                data['monthly_installments'],

                                when='end').round(0)
)
#data = pd.read_excel(myfile)
data.head()

#data=df.drop(['id', 'first_name', 'last_name', 'email','ip_address','Grade'],1)

data.columns=data.columns.str.replace (" ", "_")
data.columns

objList=data.select_dtypes(include='object')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    data[feat] = le.fit_transform(data[feat].astype(str))

print (data.info())    

X=data.drop(['Score','loan_limit'],1)
Y=data[['Score','loan_limit']]

X.shape, Y.shape

#Modeling and predictions

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=300)


# rforest=RandomForestRegressor(random_state=2,n_estimators=200,min_samples_split=3)
# rforest.fit(x_train,y_train)
# y_pred=rforest.predict(x_test)

# MAE=mean_absolute_error(y_test,y_pred),
# RMSE=np.sqrt(mean_squared_error(y_test,y_pred)),
# R2_SCORE=r2_score(y_test, y_pred)
# print(MAE,RMSE,R2_SCORE)

etree=ExtraTreesRegressor(random_state=2)
etree.fit(x_train,y_train)
y_pred=etree.predict(x_test)

MAE=mean_absolute_error(y_test,y_pred),
RMSE=np.sqrt(mean_squared_error(y_test,y_pred)),
R2_SCORE=r2_score(y_test, y_pred)
print(MAE,RMSE,R2_SCORE)

pred=np.int_(etree.predict(data.drop(['Score','loan_limit'],axis=1)))
#pred=np.int_(rforest.predict(data.drop(['Score','Current_balance_Amt'],axis=1)))

pred=pd.DataFrame(pred, columns=['Score_pred','loan_limit_pred'])
pred['Score']=data['Score']
pred['loan_limit']=data['loan_limit']
pred.head(20)

pred.shape

# Import pickle Package
import pickle

# Save the Model to file in the current working directory
Pkl_Filename = os.path.join(main_folder, 'saved_model/Pickle_RL_Model.pkl')  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(etree, file)

# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_ET_Model = pickle.load(file)

Pickled_ET_Model


x_train.info()

# data model of predictors
class ScoreFeatures(BaseModel):
    gender: int 
    Age: int
    Years_in_job: int
    Income: int
    Province: int
    Savings: int
    Home_ownership: int
    Credit_history: int    
    Number_of_accs: int
    Credit_cards: int
    Home_loan: int
    Overdraft: int
    Student_Loan: int
    Non_perfoming_Accs: int
    Open_Accounts: int
    Current_In_Arrears: int
    Current_balance_Amt: int
    Past_due_Amt: int
    No_of_enquiries: int
    Def: int
    monthly_installments: int
    Number_of_monthly_installments: int


# POST method for model prediction
@app.post("/predict")
async def predict(payload: ScoreFeatures):
    # convert the payload to pandas DataFrame
    #input_df = pd.DataFrame([payload.dict()])

    
    input_df = payload.dict()
    # encoded all the categorical variables

    #TypeError: float() argument must be a string or a number, not 'dict'
    data_in = [[input_df['gender'], input_df['Age'], input_df['Years_in_job'], input_df['Income'], input_df['Province'],input_df['Savings'], input_df['Home_ownership'], 
    input_df['Credit_history'], input_df['Number_of_accs'], input_df['Credit_cards'], input_df['Home_loan'], input_df['Overdraft'],
    input_df['Student_Loan'], input_df['Non_perfoming_Accs'],  input_df['Open_Accounts'],
    input_df['Current_In_Arrears'], input_df['Current_balance_Amt'], input_df['Past_due_Amt'], input_df['No_of_enquiries'], input_df['Def'], input_df['monthly_installments'], input_df['Number_of_monthly_installments']]]

    prediction = (Pickled_ET_Model.predict(data_in)).round()

    #probability = Pickled_RF_Model.predict_proba(data_in).max()

    #input_df_encoded, _ = encode_cat_variables(input_df, list(le.keys()), le)
    # output the prediction score
    #score = np.int_(Pickled_RF_Model.predict(input_df_encoded)[0])
    
    return {
                'prediction': prediction
            }

##############################################################################################################

#create a basemodel how the customer class will look like

# class Customer(BaseModel):
#     customer_id: str
#     country: str
#     # city : Optional[str] = None


# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Optional[bool] = None

# class URLlink(BaseModel):
#     #url: str
#     url: Optional[str] = None

# class Invoice(BaseModel):
#     invoice_no : int
#     invoice_date : str
#     customer: Optional[URLlink] = None

# fakeInvoiceTable = dict()


# @app.post("/customer")
# async def create_customer(item: Customer): #body awaits json with customer:
# #This is how to work with and return a item
# #   country = item.country
# #   return {item.country}

#         # You will add here the code for created a customer in the database


#         # Encode the created customer item if succesful into a json and return
#         json_compatible_item_data = jsonable_encoder(item)
#         return JSONResponse(content=json_compatible_item_data, status_code=201)

# #Get a customer by customer_id
# @app.get("/customer/{customer_id}") # Customer ID will be a path parameter
# async def read_customer(customer_id: str):

#     # Only succeed if the item is  12345
#     if customer_id == "12345" :
#         # Create a fake customer (usually you would get this from a database)
#         item = Customer(customer_id = "12345", country= "Germany")

#         # Encode the customer into json and send it back 
#         json_compatible_item_data = jsonable_encoder(item)
#         return JSONResponse(content= json_compatible_item_data)

#     else: 
#         # Raise a 404 exception
#         raise HTTPException(status_code=404, detail= "Item not found")

# #Create a new invoice for a customer
# @app.post("/customer/{customer_id}/invoice")
# async def create_invoice(customer_id: str, invoice: Invoice):

#     #Add the customer link to the invoice
#     invoice.customer.url = "/customer/" + customer_id

#     #Turn the invoice instance into a JSON string and store it 
#     jsonInvoice = jsonable_encoder(invoice)
#     fakeInvoiceTable[invoice.invoice_no] = jsonInvoice

#     #Read it from the store and return a stored item
#     ex_invoice = fakeInvoiceTable[invoice.invoice_no] 

#     return JSONResponse(content= ex_invoice)


# @app.get("/invoice/{invoice_no}")
# async def read_invoice(invoice_no:int):
#     #Option to manually create an invoice
#         #ex_inv = invoice(invoice_no = invoice_no, invoice_data ="2021-01-05")
#         #json_compatible_item_data = json_encoder(ex_inv)

#     #Read invoice from the dictionary
#     ex_invoice = fakeInvoiceTable[invoice_no]

#     #Return the json that we stored
#     return JSONResponse(content=ex_invoice)

# #Return all invoices 
# @app.get("/customer/ {customer_id}/invoice")
# async def get_invoices(customer_id: str):
#     #Create links to the actual invoice(get from DB)
#     ex_json = {
#         "id_123456" : "/invoice/123456",
#         "id_789101" : "/invoice/789101",
#     }
#     return JSONResponse(content=ex_json)




    

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}