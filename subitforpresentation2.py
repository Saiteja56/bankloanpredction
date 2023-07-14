#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline

import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r"C:\Users\Saite\OneDrive\Documents\aloandatasets1111111.csv")
df.info()


# In[3]:


df = df.set_index('Loan_ID')
df.head()


# In[4]:


df.describe()


# In[5]:


ncols = 4
nrows = np.ceil(len(df.columns)/ncols).astype(int)
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, figsize=(10,5))

for idx, c in enumerate(df.columns):
    row = idx // ncols
    col = idx % ncols
    sns.histplot(df, x=c, ax=axs[row, col])
plt.tight_layout()


# In[ ]:





# In[6]:



df['Credit_History'] = df['Credit_History'].astype(object)

df.isnull().sum()
target = 'Loan_Status'
y = df[target]
df.drop(columns=[target], inplace=True)


# In[7]:


y.value_counts(normalize=True)


# In[8]:


num_attributes = [c for c in df.columns if df[c].dtype != object]
num_attributes


# In[9]:


print(df['Gender'])


# In[10]:


df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()


# In[11]:


df.Married=df.Married.map({'Yes':1,'No':0})
df['Married'].value_counts()


# In[12]:


df.Dependants=df.Dependants.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependants'].value_counts()


# In[13]:


df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()


# In[14]:


df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()


# In[15]:


df.Propery_Area=df.Propery_Area.map({'Urban':2,'Semiurban':1,'Rural':0})
df['Propery_Area'].value_counts()


# In[16]:


df['Credit_History'].value_counts()


# In[17]:


print(df)


# In[18]:



df=df.fillna(df.median())
print(df['Credit_History'])
print(df)


# In[19]:


plt.matshow(df.corr())
print(df.corr())
plt.show()


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
X_num = df.values
x_train, x_test,y_train, y_test = train_test_split(X_num,y ,random_state=104, test_size=0.25, shuffle=True)
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix
from sklearn.preprocessing import StandardScaler



model = LogisticRegression(max_iter=1000, random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
cross_val_score(model, X_num, y, cv=skf).mean()
model.fit(x_train,y_train)
ypred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print ("accuracy: ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)
        
print ("recall : ", recall)
print ("precision : ", ps)

fi=2*ps*recall/(ps+recall)

print("f1score:", fi)






conf_matrix=confusion_matrix(y_test,ypred)     
fig=sns.heatmap(conf_matrix,annot=True,fmt='d')





# In[21]:


from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(random_state=0)
X_num = df.values
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
cross_val_score(model3, X_num, y, cv=skf).mean()
model3.fit(x_train,y_train)
ypred=model3.predict(x_test)
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)

print ("recall : ", recall)
print ("precision : ", ps)

fi=2*ps*recall/(ps+recall)

print("f1score:", fi)





conf_matrix=confusion_matrix(y_test,ypred)     
fig=sns.heatmap(conf_matrix,annot=True,fmt='d')


# In[22]:


from sklearn.naive_bayes import GaussianNB
model2 = GaussianNB()
X_num = df.values
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
cross_val_score(model2, X_num, y, cv=skf).mean()
model2.fit(x_train,y_train)
ypred=model2.predict(x_test)
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)

print ("recall : ", recall)
print ("precision : ", ps)
fi=2*ps*recall/(ps+recall)

print("f1score:", fi)



conf_matrix=confusion_matrix(y_test,ypred)     
fig=sns.heatmap(conf_matrix,annot=True,fmt='d')


# In[23]:


from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

svc_model1=SVC(C=.1,kernel='linear',gamma=1)
svc_model1.fit(x_train,y_train)
ypred=svc_model1.predict(x_test)
print ("Accuracy : ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)

print ("recall : ", recall)
print ("precision : ", ps)

fi=2*ps*recall/(ps+recall)

print("f1score:", fi)




conf_matrix=confusion_matrix(y_test,ypred)
fig=sns.heatmap(conf_matrix,annot=True,fmt='d')










    







# In[24]:


svc_model2=SVC(C=.1,kernel='poly',gamma=1)
svc_model2.fit(x_train,y_train)
ypred=svc_model2.predict(x_test)
print ("Accuracy : ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)

print ("recall : ", recall)
print ("precision : ", ps)
fi=2*ps*recall/(ps+recall)

print("f1score:", fi)



conf_matrix=confusion_matrix(y_test,ypred)
fig2=sns.heatmap(conf_matrix,annot=True,fmt='d')


# In[25]:


svc_model3=SVC(kernel='sigmoid')
svc_model3.fit(x_train,y_train)
ypred=svc_model3.predict(x_test)
print ("Accuracy : ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)

print ("recall : ", recall)
print ("precision : ", ps)
fi=2*ps*recall/(ps+recall)

print("f1score:", fi)



conf_matrix=confusion_matrix(y_test,ypred)
fig3=sns.heatmap(conf_matrix,annot=True,fmt='d')


# In[26]:


svc_model4=SVC(kernel='rbf')
svc_model4.fit(x_train,y_train)
ypred=svc_model4.predict(x_test)
print ("Accuracy : ", accuracy_score(y_test, ypred))
trueneg=0
truepas=0
falsepas=0
falseneg=0

for i in range(len(y_test)):
    if y_test[i]=='Y' and ypred[i]=='Y':
        truepas=truepas+1
    if y_test[i]=='Y' and ypred[i]=='N':
        falsepas=falsepas+1
    if y_test[i]=='N' and ypred[i]=='N':
        
        trueneg=trueneg+1
    if y_test[i]=='N' and ypred[i]=='Y':
        falseneg=falseneg+1
        
ps=truepas/(truepas+falsepas)
recall=truepas/(truepas+falseneg)


print ("recall : ", recall)
print ("precision : ", ps)
fi=2*ps*recall/(ps+recall)

print("f1score:", fi)


conf_matrix=confusion_matrix(y_test,ypred)
fig4=sns.heatmap(conf_matrix,annot=True,fmt='d')


# In[27]:


attributes=["Gender","Are_you_Married", "no_of_Dependents" ,"Did_you_graduate",  "Self_Employemd?","what_is_ApplicantIncome","what_is_CoapplicantIncome" , "LoanAmount" , "Loan_Amount_Term", "Credit_History","Property_Area"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(X_num,y ,random_state=104, test_size=0.25, shuffle=True)
model1=LogisticRegression(max_iter=1000, random_state=0)
model1.fit(x_train,y_train)

ypred=model1.predict(x_test)
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, ypred))


# In[28]:


user_details=[]

i=0
while i!=11:
     x=input(f"{attributes[i]}:")
     if i==0:
        if x=='male':
            user_details.append(1)
        elif x=='female':
            user_details.append(0)
        else:
            print("invalid input")
            
            continue
     elif i==1 or i==3 or i==4 or i==9:
        if x=='yes':
            user_details.append(1)
        elif x=='no':
            user_details.append(0)
        else:
            print("invalid input")
            
            continue
     elif i==10:
        if x=="urban":
            user_details.append(2)
        elif x=="semiurban":
            user_details.append(1)
        elif x=="rural":
            user_details.append(0)
        else:
            print("invalid input")
            
            continue
            
        
     else:
        if x.isdigit()==True:
            user_details.append(int(x))
            
        else:
            print("invalid input")
            
            continue
     i=i+1
     
        
            
loan_amount=user_details[7]           
user_details1=[user_details]
output=model1.predict(user_details1)
output[0]=output[0].strip()   

if output[0]=='Y':
            print("yes,you are eligible for taking loan from bank")
else:
            print("no,you are not eligible for taking loan from bank")
            

            
 


# In[29]:


if output[0]=='N':
    while output[0]!='Y':
        print(loan_amount)
        loan_amount=loan_amount-10
        user_details[7]=loan_amount
        user_details1=[user_details]
        output=model1.predict(user_details1)
    print(f"the minimum loan you can borrow is {loan_amount}")
    








# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




