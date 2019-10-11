#!/usr/bin/env python
# coding: utf-8

# In[13]:


import mysql.connector
import pandas as pd
import numpy as np
from mysql.connector import Error


# In[14]:


mydb = mysql.connector.connect(
  host = 'localhost',
  user = 'root',
  passwd = '',
  database = 'startup')
print(mydb)


# In[15]:


df = pd.read_sql_query("select * from start", mydb)
df


# In[16]:


district = df["District"][0].upper()
year = df["Expected_Year"][0]
crop = df["Crop_type"][0]


# In[17]:


import pandas as pd
data = pd.read_csv("products.csv")
len(data)


# In[18]:


tamil_nadu = data[data["State_Name"] == "Tamil Nadu"]
tamil_nadu


# In[19]:


dist_wise = tamil_nadu[tamil_nadu["District_Name"]== district]
dist_wise


# In[20]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


dist_wise.dropna(axis = 0, inplace = True)
len(dist_wise)


# In[22]:


dist_wise = dist_wise[(dist_wise["Crop"] == 'Banana') | (dist_wise["Crop"] == 'Sugarcane') | (dist_wise["Crop"] == 'Turmeric') |
                     (dist_wise["Crop"] == 'Groundnut') | (dist_wise["Crop"] == 'Onion')]
dist_wise


# In[23]:


mean_1 = np.mean(dist_wise[dist_wise["Crop"]=='Banana'])[2]
mean_2 = np.mean(dist_wise[dist_wise["Crop"]=='Sugarcane'])[2]
mean_3 = np.mean(dist_wise[dist_wise["Crop"]== 'Groundnut'])[2]
mean_4 = np.mean(dist_wise[dist_wise["Crop"]=='Turmeric'])[2]
mean_5 = np.mean(dist_wise[dist_wise["Crop"]=='Onion'])[2]


# In[24]:


import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
import os

labels = ['Tobacco', 'Banana','Sugarcane','Groundnut','Onion']
values = [mean_5, mean_1, mean_2, mean_3, mean_4]


# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, name=district,textfont=dict(
        family="sans serif",
        size=18,
        color="black"
    ))])

fig.update_layout(
    title_text=" TOP CULTIVATION OF " +district +" DISTRICT")


fig.show()
plotly.offline.plot(fig, auto_open=False, filename=r'C:\wamp64\www\employee - Copy\images\pie', image='png')


# In[25]:


crop_wise = dist_wise[dist_wise["Crop"]==crop]
crop_wise


# In[26]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
X = crop_wise["Crop_Year"]
y = crop_wise["Production"]
print(X)


# In[27]:


X = np.array(X).reshape(-1,1)


# In[28]:


reg.fit(X,y)
reg.score(X,y)


# In[29]:


year = int(year)
i = reg.predict(np.array([[year]]))
print(i)
y= str(i)


# In[30]:


y


# In[31]:


mySql_insert_query = """INSERT INTO start (predicted) 
                                VALUES (prediction.y) """


# In[32]:


cursor = mydb.cursor()
cursor.execute("update start set predicted ='"+y+"' where Search_id='1'")


# In[ ]:





# In[ ]:





# In[ ]:




