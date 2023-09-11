#!/usr/bin/env python
# coding: utf-8

# In[69]:


from scipy.stats import ttest_1samp
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# # На сайте запущен А/В тест с целью увеличить доход. В приложенном excel файле вы найдете сырые данные по результатам эксперимента – user_id, тип выборки variant_name и доход принесенный пользователем revenue. Проанализируйте результаты эксперимента и напишите свои рекомендации менеджеру.
# 

# In[ ]:





# In[15]:


data = pd.read_excel("D:\Business Analyst\AB тестирование\Семинары\AB_Test_Results (2).xlsx")
data.head(10)         


# In[16]:


data.shape


# In[19]:


data.VARIANT_NAME.value_counts()# объем выборки в тестовой и контрольной группах


# In[21]:


data.info()


# In[33]:


data.describe() # основные статистики


# In[34]:


data.isna().sum() # количество ненулевых значений


# In[57]:


data[data['REVENUE']>20] # посмотрим у кого доход > 20


# In[25]:


# Избавимся от некорретных строк. Посчитаем, сколько типов встречается у каждого пользователя.

v = df.    groupby('USER_ID', as_index=False).    agg({'VARIANT_NAME': pd.Series.nunique})
v.head(10)


# In[88]:


more_than_one_types = v.query('VARIANT_NAME > 1') # почистим данные, уберем поля, у которых у пользователя больше одного типа имени


# In[90]:


df_new = df[~df.USER_ID.isin(more_than_one_types.USER_ID)].sort_values('USER_ID')
df_new.shape # новый массив данных 


# In[91]:


df.shape # старый массив данных 


# In[ ]:





# In[ ]:


Проверка на нормальность распределения и применение статистических критериев


# Проверим на нормальность при помощи теста Шапиро-Уилко

# In[93]:


control = df_new.query('VARIANT_NAME == "control"')
variant = df_new.query('VARIANT_NAME == "variant"')


# In[95]:


len(control),len(variant)


# In[97]:


alpha = 0.05

st = shapiro(df_new.REVENUE)
print('Distribution is {}normal\n'.format( {True:'not ',
False:''}[st[1] < alpha]));


# In[98]:


control.REVENUE.hist(bins = 25, alpha =0.7, label='Control')
variant.REVENUE.hist(bins = 25, alpha =0.7, label='Variant')
plt.title('Доход на юзера по группам')
plt.xlabel('Доход')
plt.ylabel('Число пользователей')
plt.legend();


# In[99]:


plt.boxplot([control.REVENUE,variant.REVENUE],
            labels=['Контроль','Вариант'],
            widths=0.5
           )
plt.title('Boxplot по пользователям',  loc='center')
plt.grid(axis  ='both')


# In[100]:


stats.ttest_ind(control.REVENUE.values, variant.REVENUE.values, equal_var = False)


# pvalue > alpha = 0.05, стат. значимых различий нет

# Посмотрим на группы отдельно

# In[37]:


data[data['VARIANT_NAME']=='control'].describe() # контрольная группа


# In[38]:


data[data['VARIANT_NAME']=='variant'].describe() # тестовая группа


# 

# Метрика доход, принесенный пользователем 

# In[56]:


plt.figure(figsize=(6, 6))
control_revenue = data['REVENUE'][data['VARIANT_NAME']=='control']
test_revenue =data['REVENUE'][data['VARIANT_NAME']=='variant']


plt.hist(control_revenue, color ='r', bins=20)
plt.hist(test_revenue,  bins=20)


plt.show()


# In[47]:


stats.ttest_ind(control_revenue, test_revenue, equal_var = False) # - pvalue > alpha 0,05, нет стат. значимых различий


# In[59]:


data['REVENUE'].value_counts()


# Непараметрический критерий Манна-Уитни

# In[63]:


data.groupby('VARIANT_NAME')['REVENUE'].describe()


# In[76]:


mw_stats = mannwhitneyu(x=data[data['VARIANT_NAME'] == 'control']['REVENUE'].values, 
y=data[data['VARIANT_NAME'] == 'variant']['REVENUE'].values)
mw_stats
# стат. значимых различий нет, pvalue > alpha 0.05
# Критерий Манна-Уитни не позволяет сделать вывод в пользу альтернативной гипротезы о разнице доходов 


# ##Расчет мощности

# In[ ]:


d = (M1 – M2) / S_pooled
S_pooled = math.sqrt[(S1^2+S2^2)/2]


# In[101]:


C_mean = control.REVENUE.values.mean()

T_mean = variant.REVENUE.values.mean()


C_std = control.REVENUE.values.std()

T_std = variant.REVENUE.values.std()


# In[102]:


print(len(control.REVENUE.values), len(variant.REVENUE.values))


# In[103]:


n =  len(control.REVENUE.values)


# In[104]:


##S = np.sqrt((sd_t**2 / n_t) + (sd_c**2 / n_c))

S = np.sqrt((T_std**2 + C_std **2)/ 2)


# In[109]:


effect =float((T_mean-C_mean)/ S) # Эффект Дека-Хенна


# # parameters for power analysis
# 
# alpha = 0.05
# 
# # perform power analysis
# from statsmodels.stats.power import TTestIndPower
# analysis = TTestIndPower()
# result = analysis.solve_power(effect, power=None,
# nobs1=n, ratio=1.0, alpha=alpha)
# 
# result

# Низкая мощность теста (около 30%)- недостаточная для выкатывания изменений на сайте для увеличения доходов
# 
# Рекомендации менеджеру: 
# 
# Пользователи попадали и в контрольную и в тестовую группу 
# Возникают вопросы к дизайну и к запуску теста, тест мог быть поставлен не корректно
# Большая дисперсия в данных 
# Нет возможности сделать достоверные заключения при данной мощности теста. 
# Перезапустить тест
# Рассчитать объем выборки до начала теста для тестовой и контрольной групп, период длительности теста 
