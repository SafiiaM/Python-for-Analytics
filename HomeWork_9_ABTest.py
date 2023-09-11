#!/usr/bin/env python
# coding: utf-8

# ##Проанализируйте результаты эксперимента и напишите свои рекомендации менеджеру. Mobile Games AB Testing with Cookie Cats
# 

# #Загрузка библиотек 

# In[12]:


from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd
import math
import statsmodels.stats.power as smp
from tqdm.auto import tqdm
import seaborn as sns
import plotly.express as px

plt.style.use('ggplot')


from scipy.stats import norm, t, kstest, shapiro
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats


import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


# #Проверка данных на нормальность
# 

# In[2]:


df=pd.read_csv( r"D:\Business Analyst\AB тестирование\Семинары\cookie_cats.csv", encoding="latin-1"
)
df


# In[3]:


df.describe() # основные статистики


# 'version'
# вариант А - game_30
# вариант B - game_40
# 
# 'sum_gamerounds' - проверка на нормальность 
# 
# 'retention_7' - ключевая метрика

# In[4]:


df.info() # количество ячеек, колонок, типы данных 


# In[6]:


# разделим группы на контрольную и тестовую 
A = df[df['version'] == 'gate_30']
B = df[df['version'] == 'gate_40']


# In[7]:


df[df['version']=='gate_30'].describe() # основные статистики варианта А


# In[8]:


df[df['version']=='gate_40'].describe() # основные статистики варианта B


# In[9]:


df.isna().sum()# количество ненулевых значений


# ## Визуализация

# In[35]:


plt.figure(figsize=(6, 2))
control_sum_gamerounds = df['sum_gamerounds'][df['version'] == 'gate_30']
test_sum_gamerounds =df['sum_gamerounds'][df['version'] == 'gate_40']


plt.hist(control_sum_gamerounds, color='r', bins=20)
plt.hist(test_sum_gamerounds,  bins=20)


plt.show()


# In[22]:


stats.ttest_ind(control_sum_gamerounds, test_sum_gamerounds, equal_var = False)


# pvalue > alpha (0.05), распределение не отличается от нормального 

# In[57]:


plt.boxplot([control_sum_gamerounds,test_sum_gamerounds],
            labels=['Контроль','Тест'],
            widths=0.5
           )
plt.title('Boxplot по пользователям',  loc='center')
plt.grid(axis  ='both')


# In[65]:


display(df.groupby('version').count())


# In[72]:


plt.figure(figsize = (10, 5))
ax = sns.countplot(x = 'version', data = df)
for p in ax.patches:
   ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.02))


# Нет значительной разницы между количеством игроком в тестовой и контрольной группах 

# In[53]:


ax = plot_df.head(100).plot()
plt.title("Распределение игроков", fontweight="bold", size=14)
plt.xlabel("Общее количество геймраундов", size=12)
plt.ylabel("Количество игроков", size=12)
plt.show()


# In[37]:


df[df['sum_gamerounds']==0]['userid'].count() # количество пользователей, которые вообще не играют 


# In[ ]:


## Анализ retention_7
Гипотеза Н0 - нет существенной разницы в количестве игроков между двумя группами, которые возвращаются  после 7 дней инсталляции игры 
Гипотеза Н1 - есть существенная разница в количестве игроков между двумя группами, которые возвращаются  после 7 дней инсталляции игры


# In[10]:


df.retention_7.value_counts() # количество значений в retention_7 


# In[73]:


plt.figure(figsize = (10, 5))
ax = sns.countplot(x = 'version', data = df, hue = 'retention_7')
for p in ax.patches:
   ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.02))


# In[40]:


df.groupby('version')['retention_7'].sum()# количество вернувшихся в контрольной и в тестовой группе через 7 дней 


# In[81]:


df_retention_ab = df.groupby("version").agg({"userid":"count", "retention_1":"mean","retention_7":"mean", "sum_gamerounds":"sum"})
df_retention_ab


# In[84]:


from statsmodels.stats.proportion import proportions_ztest
retention_7_userid = np.array([8502, 8279])
version_userid = np.array([44700, 45489])

proportions_ztest(count=retention_7_userid, nobs=version_userid)


# p-value = 0.001 < 0.05 - гипотеза H0 отвергается

# Непараметрический Хи-квадрат

# In[85]:


chisq, pvalue, table = proportion.proportions_chisquare(np.array([8502, 8279]), 
                                                   np.array([44700, 45489]))

print('Results are ','chisq =%.3f, pvalue = %.3f'%(chisq, pvalue))


# In[86]:


if abs(pvalue) < 0.05:
    print("We may reject the null hypothesis!")
else:
    print("We have failed to reject the null hypothesis")


# Выводы и рекомендации: 
# Эти данные содержат 90 189 строк игроков, которые установили игру во время выполнения AB-теста. Там нет дубликатов и пропущенных значений, поэтому очистка не требуется. Тем не менее, есть выбросы в количестве игровых раундов, сыгранных игроком в течение первых 14 дней после установки
# 
# есть существенная разница в количестве игроков между двумя группами, которые возвращаются  после 7 дней установки.
# Нужно выкатывать изменения версии игры. 

# In[ ]:





# 

# In[77]:





# In[78]:





# In[ ]:





# In[ ]:




