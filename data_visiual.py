#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


data = pd.read_csv("veri16.txt",sep=';')

data = data.drop(["no"] , axis = 1)

data["PT_orani"] = data["Prim"] / data["Teminat"]  

data.describe()
data.isnull().any()  # null değer yok

data.head()


# In[4]:


# siniflama = data.groupby('Musteri_Sinifi').get_group('Sınıf1')
sinif_1 = data.loc[data.Musteri_Sinifi == "Sınıf1" , ]  
sinif_2 = data.loc[data.Musteri_Sinifi == "Sınıf2" , ]
sinif_3 = data.loc[data.Musteri_Sinifi == "Sınıf3" , ]
sinif_4 = data.loc[data.Musteri_Sinifi == "Sınıf4" , ]
sinif_5 = data.loc[data.Musteri_Sinifi == "Sınıf5" , ]


# In[13]:


# dağılım grafiği
f, ax = plt.subplots(figsize=(8, 5))
sns.despine(f)
sns.histplot(data["PT_orani"], bins=100, color="tab:orange", kde=True)
plt.show()


# In[63]:


#boxplot
bplot = [sinif_1["PT_orani"], sinif_2["PT_orani"], sinif_3["PT_orani"], sinif_4["PT_orani"],sinif_5["PT_orani"]]
fig = plt.figure(figsize =(10, 6))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(bplot)
plt.xlabel("Musteri_Sinifi")
plt.ylabel("PT_orani")
plt.show()


# In[5]:


data['ln_PT_orani'] = data['PT_orani'].apply(np.log)
# siniflama = data.groupby('Musteri_Sinifi').get_group('Sınıf1')
sinif_1 = data.loc[data.Musteri_Sinifi == "Sınıf1" , ]  
sinif_2 = data.loc[data.Musteri_Sinifi == "Sınıf2" , ]
sinif_3 = data.loc[data.Musteri_Sinifi == "Sınıf3" , ]
sinif_4 = data.loc[data.Musteri_Sinifi == "Sınıf4" , ]
sinif_5 = data.loc[data.Musteri_Sinifi == "Sınıf5" , ]


# In[62]:


#violinplot
siniflar = pd.concat([sinif_1,sinif_2,sinif_3,sinif_4,sinif_5])
fig = plt.figure(figsize =(12, 5))
sns.violinplot(data=siniflar, x = "Musteri_Sinifi", y="PT_orani" ,inner="points" , linewidth = 1.5 , palette="pastel")
plt.show()


# In[21]:


data.head()


# In[22]:

# dönüşümlü boxplot
bplot_ln = [sinif_1["ln_PT_orani"], sinif_2["ln_PT_orani"], sinif_3["ln_PT_orani"], sinif_4["ln_PT_orani"],sinif_5["ln_PT_orani"]]
fig = plt.figure(figsize =(10, 6))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(bplot_ln)
plt.xlabel("Musteri_Sinifi")
plt.ylabel("PT_orani")
plt.show()



# In[23]:


siniflar = pd.concat([sinif_1,sinif_2,sinif_3,sinif_4,sinif_5])
fig = plt.figure(figsize =(12, 5))
sns.violinplot(data=siniflar, x = "Musteri_Sinifi", y="ln_PT_orani" ,inner="points" , linewidth = 1.5 , palette="pastel")
plt.show()


# In[7]:


istenilen = data[["Yas","Gelir","Prim","Teminat","ln_PT_orani"]]
istenilen.head()




# In[25]:


sc_istenilen = pd.DataFrame(sc_istenilen)

sc_istenilen.columns = ["Yas","Gelir","Prim","Teminat","ln_PT_orani"]

sc_istenilen.head()


# In[75]:


sns.pairplot(istenilen)
plt.show()


# In[76]:


istenilen_altkume = istenilen.sample(n=100, random_state=42)
sns.pairplot(istenilen_altkume)
plt.show()


# In[8]:


corr_matrix = istenilen.corr()
sns.heatmap(corr_matrix , annot=True , cmap= "bwr", annot_kws={'fontsize':10, 'fontweight': 'bold', 'color': 'black'} , square=True)
plt.title("Heatmap")


# In[ ]:


#deneme grafikleri


# In[ ]:


'''fig = go.Figure()

for col in istenilen.columns:
    fig.add_trace(go.Scatter(x=istenilen[col], y=istenilen.index, mode='markers', name=col))

fig.update_layout(title='Değişkenlerin İnteraktif Scatter Plotu',
                  xaxis_title='Değerler',
                  yaxis_title='Index')

fig.show()
'''


# In[ ]:


'''fig = go.Figure()

for col in istenilen.columns:
    fig.add_trace(go.Scatter(x=istenilen[col], y=istenilen.index, mode='markers', name=col))

fig.update_layout(title='Değişkenlerin İnteraktif Scatter Plotu',
                  xaxis_title='Değerler',
                  yaxis_title='Index')
fig.show()
'''


# In[ ]:


"""# Paralel koordinatlar grafiği
fig = go.Figure(data=go.Parcoords(
                  line=dict(color=data['Yas'], colorscale='Viridis', showscale=True),
                  dimensions=list([
                      dict(range=[min(data['Yas']), max(data['Yas'])],
                           label='Yas', values=data['Yas']),
                      dict(range=[min(data['Gelir']), max(data['Gelir'])],
                           label='Gelir', values=data['Gelir']),
                      dict(range=[min(data['Prim']), max(data['Prim'])],
                           label='Prim', values=data['Prim']),
                      dict(range=[min(data['Teminat']), max(data['Teminat'])],
                           label='Teminat', values=data['Teminat']),
                      dict(range=[min(data['ln_PT_orani']), max(data['ln_PT_orani'])],
                           label='PT_Oranı', values=data['ln_PT_orani'])
                  ])
              ))

fig.update_layout(title='Değişkenlerin Paralel Koordinatlar Grafiği')
fig.show()
"""


# In[ ]:





# In[ ]:




