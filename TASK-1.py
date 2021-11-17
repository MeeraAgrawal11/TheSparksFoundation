#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[15]:


data = pd.read_csv ('C:\\Users\\meera agarwal\\Desktop\\scores.csv')
data.head(10)


# In[16]:


data.isnull == True


# In[17]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Scores Vs Study-Hours',size=20)
plt.ylabel('Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[18]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# In[19]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[20]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# In[21]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[22]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[23]:


plt.scatter(x=val_X, y=val_y, color='red')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[24]:


print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# In[25]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# In[ ]:




