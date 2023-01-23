#!/usr/bin/env python
# coding: utf-8

# [인프런 - 공공데이터로 파이썬 데이터 분석 시작하기](https://www.inflearn.com/course/%EA%B3%B5%EA%B3%B5%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0) 의 질문에 대한 답변 입니다.
# 
# * [Reshaping and pivot tables — pandas documentation](https://pandas.pydata.org/docs/user_guide/reshaping.html)

# In[1]:


# 전체 Data에서 필요한 부분만 뽑아 낸 다음에 , Tiny Data로 만들고 시각화 하는 과정에서 
# 언제는 unstack과 transpose를 사용하고, 언제는 reset_index를 사용하시는데
# 각각 언제 어떤 방법을 사용하는지 궁금합니다.


# In[2]:


import pandas as pd
import seaborn as sns


# sns.set(font="Malgun Gothic")
sns.set(font="AppleGothic")


# In[3]:


df_first = pd.read_csv("data/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", 
                       encoding="cp949")
df_first.shape


# In[4]:


df_first


# In[5]:


df = pd.melt(df_first, id_vars=["지역"])
df.head()


# In[6]:


group = df.groupby(["지역", "variable"])["value"].sum()
group


# In[7]:


group.index


# In[8]:


group[:10].plot.bar()


# In[9]:


group_unstack = group.unstack()
group_unstack


# In[10]:


group_unstack.T


# In[11]:


group_unstack.transpose().plot(figsize=(15, 4))


# In[12]:


type(group)


# In[13]:


group.to_frame()


# In[14]:


pd.DataFrame(group)


# In[15]:


df_group = group.reset_index()
df_group


# In[16]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_group, x="variable", y="value", hue="지역")


# In[ ]:




