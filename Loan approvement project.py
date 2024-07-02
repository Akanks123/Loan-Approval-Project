#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd


# In[164]:


data= pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Aku Assignment\Data Mining\Loan Approval Project\Loan_approval_dataset.csv")


# In[165]:


data


# In[166]:


data.drop(columns=['Loan_ID'], inplace=True)


# In[167]:


data.columns


# In[168]:


data.isnull().sum()


# In[169]:


data.dropna(inplace=True)


# In[170]:


data.isnull().sum()


# In[171]:


data.Education.unique()


# In[172]:


data.Education.unique()


# In[173]:


data.Education= data.Education.replace(['Graduate', 'Not Graduate'],[1,0])


# In[174]:


data


# In[175]:


data.Self_Employed.unique()


# In[176]:


data.Self_Employed= data.Self_Employed.replace(['No', 'Yes'],[0,1])


# In[177]:


data


# In[178]:


data.Gender.unique()


# In[179]:


data.Gender= data.Gender.replace(['Male', 'Female'],[0,1])


# In[180]:


data.Married.unique()


# In[181]:


data.Married= data.Married.replace(['Yes', 'No'],[1,0])


# In[182]:


data.info()


# In[183]:


data.Loan_Status.unique()


# In[184]:


data.Loan_Status= data.Loan_Status.replace(['N', 'Y'],[0,1])


# In[185]:


from sklearn.preprocessing import LabelEncoder
columns= ['Gender','Married','Dependents', 'Property_Area']
le= LabelEncoder()
for x in columns:
    data[x]= le.fit_transform(data[x])


# In[186]:


data


# In[187]:


from sklearn.model_selection import train_test_split


# In[188]:


input_data= data.drop(columns=['Loan_Status'])


# In[208]:


input_data


# In[189]:


output_data= data["Loan_Status"]


# In[190]:


output_data


# In[191]:


x_train, x_test,y_train,y_test= train_test_split(input_data, output_data,test_size= 0.2)


# In[192]:


x_train.shape, x_test.shape


# In[193]:


from sklearn.preprocessing import StandardScaler


# In[194]:


scaler= StandardScaler()


# In[195]:


x_train_scaled= scaler.fit_transform(x_train)


# In[196]:


x_test_scaled= scaler.transform(x_test)


# In[197]:


#create Model
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train_scaled,y_train)


# In[198]:


model.score(x_test_scaled,y_test)


# In[204]:


from sklearn.metrics import accuracy_score
y_train_pred = model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# In[209]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Example with Random Forest classifier and GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train_scaled, y_train)


# In[212]:


best_rf_model = grid_search.best_estimator_
y_pred_train = best_rf_model.predict(x_train_scaled)
y_pred_test = best_rf_model.predict(x_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# In[225]:


import pandas as pd
pred_data= pd.DataFrame([['0','1' , '1' ,'1' ,'0','3000','1508.0','133.0','360.0','1.0','0']],columns=['Gender', 'Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History', 'Property_Area'])


# In[226]:


pred_data= scaler.transform(pred_data)


# In[227]:


model.predict(pred_data)


# In[ ]:




