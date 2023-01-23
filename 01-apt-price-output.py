#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/open-data-01-apt-output)
# 
# 
# # 전국 신규 민간 아파트 분양가격 동향
# 
# 2013년부터 최근까지 부동산 가격 변동 추세가 아파트 분양가에도 반영될까요? 공공데이터 포털에 있는 데이터를 Pandas 의 melt, concat, pivot, transpose 와 같은 reshape 기능을 활용해 분석해 봅니다. 그리고 groupby, pivot_table, info, describe, value_counts 등을 통한 데이터 요약과 분석을 해봅니다. 이를 통해 전혀 다른 형태의 두 데이터를 가져와 정제하고 병합하는 과정을 다루는 방법을 알게 됩니다. 전처리 한 결과에 대해 수치형, 범주형 데이터의 차이를 이해하고 다양한 그래프로 시각화를 할 수 있게 됩니다.
# 
# 
# ## 다루는 내용
# * 공공데이터를 활용해 전혀 다른 두 개의 데이터를 가져와서 전처리 하고 병합하기
# * 수치형 데이터와 범주형 데이터를 바라보는 시각을 기르기
# * 데이터의 형식에 따른 다양한 시각화 방법 이해하기
# 
# ## 실습
# * 공공데이터 다운로드 후 주피터 노트북으로 로드하기
# * 판다스를 통해 데이터를 요약하고 분석하기
# * 데이터 전처리와 병합하기
# * 수치형 데이터와 범주형 데이터 다루기
# * 막대그래프(bar plot), 선그래프(line plot), 산포도(scatter plot), 상관관계(lm plot), 히트맵, 상자수염그림, swarm plot, 도수분포표, 히스토그램(distplot) 실습하기
# 
# ## 데이터셋
# * 다운로드 위치 : 
#     * 공공데이터 포털 : https://www.data.go.kr/data/15061057/fileData.do
#     * 수업과 같은 데이터로 실습하기 위해 아래 링크의 데이터로 실습하는 것을 권장합니다.
#     * 데이터셋 다운로드(구글드라이브) : http://bit.ly/open-data-set-folder
#     * 데이터셋 다운로드(드랍박스) : https://bit.ly/dropbox-open-data-set
# 
# 
# ### 전국 평균 분양가격(2013년 9월부터 2015년 8월까지)
# * 전국 공동주택의 3.3제곱미터당 평균분양가격 데이터를 제공
# 
# ###  주택도시보증공사_전국 평균 분양가격(2019년 12월)
# * 전국 공동주택의 연도별, 월별, 전용면적별 제곱미터당 평균분양가격 데이터를 제공
# * 지역별 평균값은 단순 산술평균값이 아닌 가중평균값임

# In[1]:


# %가 들어가는 명령은 command line 명령어 입니다. 
# 터미널에서 사용할 수 있는 명령을 주피터 노트북 안에서 사용할 수 있습니다.
# 매직커맨드라고 불리는 이 명령이 운영체제에 따라 다르게 동작하거나 동작하지 않을 수도 있으니 그렇다면 다음 셀을 참고해서 파일 목록을 실행해 보세요.
get_ipython().run_line_magic('ls', 'data')


# In[3]:


import os
# os.walk 를 통해 특정 폴더 안의 파일을 읽어옵니다.
for root, dirs, files in os.walk("data"):
    print(files)


# In[2]:


# 파이썬에서 쓸 수 있는 엑셀과도 유사한 판다스 라이브러리를 불러옵니다.
import pandas as pd
import numpy as np

# 경고 메시지는 출력되지 않게 합니다.
import warnings
warnings.filterwarnings("ignore")


# ## 데이터 로드
# ### 최근 파일 로드
# 공공데이터 포털에서 "주택도시보증공사_전국 평균 분양가격"파일을 다운로드 받아 불러옵니다.
# 이 때, 인코딩을 설정을 해주어야 한글이 깨지지 않습니다.
# 보통 엑셀로 저장된 한글의 인코딩은 cp949 혹은 euc-kr로 되어 있습니다.
# df_last 라는 변수에 최근 분양가 파일을 다운로드 받아 로드합니다.
# 
# * 한글인코딩 : [‘설믜를 설믜라 못 부르는’ 김설믜씨 “제 이름을 지켜주세요” : 사회일반 : 사회 : 뉴스 : 한겨레](http://www.hani.co.kr/arti/society/society_general/864914.html)
# 
# 데이터를 로드한 뒤 shape를 통해 행과 열의 갯수를 출력합니다.

# In[3]:


# 최근 분양가 파일을 로드해서 df_last 라는 변수에 담습니다.
# 파일로드시 OSError가 발생한다면, engine="python"을 추가해 보세요.
df_last = pd.read_csv("data/주택도시보증공사_전국 평균 분양가격(2019년 12월).csv", encoding="cp949", engine="python")
df_last.shape


# In[4]:


# head 로 파일을 미리보기 합니다.

df_last.head()


# In[5]:


# tail 로도 미리보기를 합니다.

df_last.tail()


# ### 2015년 부터 최근까지의 데이터 로드
# 전국 평균 분양가격(2013년 9월부터 2015년 8월까지) 파일을 불러옵니다.
# df_first 라는 변수에 담고 shape로 행과 열의 갯수를 출력합니다.

# In[6]:


# 해당되는 폴더 혹은 경로의 파일 목록을 출력해 줍니다.

get_ipython().run_line_magic('ls', 'data')


# In[7]:


df_first = pd.read_csv("data/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", 
                       encoding="cp949")
df_first.shape


# In[8]:


# df_first 변수에 담긴 데이터프레임을 head로 미리보기 합니다.

df_first.head()


# In[9]:


# df_first 변수에 담긴 데이터프레임을 tail로 미리보기 합니다.

df_first.tail()


# ### 데이터 요약하기

# In[10]:


df_last.info()


# ### 결측치 보기

# isnull 혹은 isna 를 통해 데이터가 비어있는지를 확인할 수 있습니다.
# 결측치는 True로 표시되는데, True == 1 이기 때문에 이 값을 다 더해주면 결측치의 수가 됩니다.

# In[11]:


True == 1


# In[12]:


False == 0


# In[13]:


True + True + False


# In[14]:


df_last.isnull()


# In[15]:


# isnull 을 통해 결측치를 구합니다.

df_last.isnull().sum()


# In[16]:


# isna 를 통해 결측치를 구합니다.

df_last.isna().sum()


# ### 데이터 타입 변경
# 분양가격이 object(문자) 타입으로 되어 있습니다. 문자열 타입을 계산할 수 없기 때문에 수치 데이터로 변경해 줍니다. 결측치가 섞여 있을 때 변환이 제대로 되지 않습니다. 그래서 pd.to_numeric 을 통해 데이터의 타입을 변경합니다.

# In[17]:


type(np.nan)


# In[18]:


df_last["분양가격"] = pd.to_numeric(df_last["분양가격(㎡)"], errors='coerce')
df_last["분양가격"].head(1)


# ### 평당분양가격 구하기
# 공공데이터포털에 올라와 있는 2013년부터의 데이터는 평당분양가격 기준으로 되어 있습니다.
# 분양가격을 평당기준으로 보기위해 3.3을 곱해서 "평당분양가격" 컬럼을 만들어 추가해 줍니다.

# In[19]:


df_last["평당분양가격"] = df_last["분양가격"] * 3.3
df_last.head(1)


# ### 분양가격 요약하기

# In[20]:


# info를 통해 분양가격을 봅니다.

df_last.info()


# In[21]:


# 변경 전 컬럼인 분양가격(㎡) 컬럼을 요약합니다.

df_last["분양가격(㎡)"].describe()


# In[22]:


df_last["분양가격(㎡)"].nunique()


# In[23]:


# 수치데이터로 변경된 분양가격 컬럼을 요약합니다.

df_last["분양가격"].describe()


# ### 규모구분을 전용면적 컬럼으로 변경
# 규모구분 컬럼은 전용면적에 대한 내용이 있습니다. 전용면적이라는 문구가 공통적으로 들어가고 규모구분보다는 전용면적이 좀 더 직관적이기 때문에 전용면적이라는 컬럼을 새로 만들어주고 기존 규모구분의 값에서 전용면적, 초과, 이하 등의 문구를 빼고 간결하게 만들어 봅니다.
# 
# 이 때 str 의 replace 기능을 사용해서 예를들면 "전용면적 60㎡초과 85㎡이하"라면 "60㎡~85㎡" 로 변경해 줍니다.
# 
# * pandas 의 string-handling 기능을 좀 더 보고 싶다면 :
# https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

# In[24]:


df_last["규모구분"].unique()


# In[25]:


df_last["전용면적"] = df_last["규모구분"].str.replace("전용면적", "")
df_last["전용면적"] = df_last["전용면적"].str.replace("초과", "~")
df_last["전용면적"] = df_last["전용면적"].str.replace("이하", "")
df_last["전용면적"] = df_last["전용면적"].str.replace(" ", "").str.strip()
df_last["전용면적"]


# ### 필요없는 컬럼 제거하기
# drop을 통해 전처리 해준 컬럼을 제거합니다. pandas의 데이터프레임과 관련된 메소드에는 axis 옵션이 필요할 때가 있는데 행과 열중 어떤 기준으로 처리를 할 것인지를 의미합니다. 보통 기본적으로 0으로 되어 있고 행을 기준으로 처리함을 의미합니다. 메모리 사용량이 줄어들었는지 확인합니다.

# In[26]:


df_last.info()


# In[27]:


# drop 사용시 axis에 유의 합니다.
# axis 0:행, 1:열
df_last = df_last.drop(["규모구분", "분양가격(㎡)"], axis=1)


# In[28]:


# 제거가 잘 되었는지 확인 합니다.
df_last.head(1)


# In[29]:


# 컬럼 제거를 통해 메모리 사용량이 줄어들었는지 확인합니다.
df_last.info()


# ## groupby 로 데이터 집계하기
# groupby 를 통해 데이터를 그룹화해서 연산을 해봅니다.

# In[30]:


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
# df.groupby(["인덱스로 사용할 컬럼명"])["계산할 컬럼 값"].연산()
df_last.groupby(["지역명"])["평당분양가격"].mean()


# In[31]:


# 전용면적으로 분양가격의 평균을 구합니다.
df_last.groupby(["전용면적"])["평당분양가격"].mean()


# In[32]:


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
df_last.groupby(["전용면적", "지역명"])["평당분양가격"].mean().unstack().round()


# In[33]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.

g = df_last.groupby(["연도", "지역명"])["평당분양가격"].mean()
g
# g.unstack().transpose()


# In[34]:


g.loc[2018]


# In[35]:


g.unstack()


# In[36]:


g.unstack().transpose()


# ## pivot table 로 데이터 집계하기
# * groupby 로 했던 작업을 pivot_table로 똑같이 해봅니다.

# In[37]:


pd.pivot_table(df_last, index=["지역명"], values=["평당분양가격"], aggfunc="mean")


# In[38]:


# df_last.groupby(["전용면적"])["평당분양가격"].mean()


# In[39]:


pd.pivot_table(df_last, index="전용면적", values="평당분양가격")


# In[40]:


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
# df_last.groupby(["전용면적", "지역명"])["평당분양가격"].mean().unstack().round()


# In[41]:


df_last.pivot_table(index="전용면적", columns="지역명", values="평당분양가격").round()


# In[42]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.
# g = df_last.groupby(["연도", "지역명"])["평당분양가격"].mean()
p = pd.pivot_table(df_last, index=["연도", "지역명"], values="평당분양가격")
p.loc[2017]


# ## 최근 데이터 시각화 하기
# ### 데이터시각화를 위한 폰트설정
# 한글폰트 사용을 위해 matplotlib의 pyplot을 plt라는 별칭으로 불러옵니다.

# In[43]:


import matplotlib.pyplot as plt

# plt.rc("font", family="Malgun Gothic")
plt.rc("font", family="AppleGothic")


# In[ ]:


# 폰트 설정이 잘 안 된다면 해당 셀의 마지막 줄 주석을 풀고 실행해 보세요.
# colab 사용시에도 아래 함수를 활용해 보세요.
def get_font_family():
    """
    시스템 환경에 따른 기본 폰트명을 반환하는 함수
    """
    import platform
    system_name = platform.system()

    if system_name == "Darwin" :
        font_family = "AppleGothic"
    elif system_name == "Windows":
        font_family = "Malgun Gothic"
    else:
        # Linux(colab)
        get_ipython().system('apt-get install fonts-nanum -qq  > /dev/null')
        get_ipython().system('fc-cache -fv')

        import matplotlib as mpl
        mpl.font_manager._rebuild()
        findfont = mpl.font_manager.fontManager.findfont
        mpl.font_manager.findfont = findfont
        mpl.backends.backend_agg.findfont = findfont
        
        font_family = "NanumBarunGothic"
    return font_family

# plt.rc("font", family=get_font_family())


# ### Pandas로 시각화 하기 - 선그래프와 막대그래프
# pandas의 plot을 활용하면 다양한 그래프를 그릴 수 있습니다.
# seaborn을 사용했을 때보다 pandas를 사용해서 시각화를 할 때의 장점은 미리 계산을 하고 그리기 때문에 속도가 좀 더 빠릅니다.

# In[44]:


# 지역명으로 분양가격의 평균을 구하고 선그래프로 시각화 합니다.
g = df_last.groupby(["지역명"])["평당분양가격"].mean().sort_values(ascending=False)
g.plot()


# In[45]:


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
g.plot.bar(rot=0, figsize=(10, 3))


# 전용면적별 분양가격의 평균값을 구하고 그래프로 그려봅니다.

# In[46]:


# 전용면적으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
df_last.groupby(["전용면적"])["평당분양가격"].mean().plot.bar()


# In[47]:


# 연도별 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
# 연도에 소숫점이 생기지 않게 표시하고자 한다면 ax.xaxis.set_major_locator를 사용해서 integer로 설정합니다.
from matplotlib.ticker import MaxNLocator

ax = plt.figure().gca()
df_last.groupby(["연도"])["평당분양가격"].mean().plot()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# ### box-and-whisker plot | diagram
# 
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html
# 
# * [상자 수염 그림 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EC%83%81%EC%9E%90_%EC%88%98%EC%97%BC_%EA%B7%B8%EB%A6%BC)
# * 가공하지 않은 자료 그대로를 이용하여 그린 것이 아니라, 자료로부터 얻어낸 통계량인 5가지 요약 수치로 그린다.
# * 5가지 요약 수치란 기술통계학에서 자료의 정보를 알려주는 아래의 다섯 가지 수치를 의미한다.
# 
# 
# 1. 최솟값
# 1. 제 1사분위수
# 1. 제 2사분위수( ), 즉 중앙값
# 1. 제 3 사분위 수( )
# 1. 최댓값
# 
# * Box plot 이해하기 : 
#     * [박스 플롯에 대하여 :: -[|]- Box and Whisker](https://boxnwhis.kr/2019/02/19/boxplot.html)
#     * [Understanding Boxplots – Towards Data Science](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)

# In[48]:


df_last.pivot_table(index="월", columns="연도", values="평당분양가격")


# In[49]:


df_last.pivot_table(index="월", columns="연도", values="평당분양가격").plot.box()


# In[50]:


p = df_last.pivot_table(index="월", columns=["연도", "전용면적"], values="평당분양가격")
p.plot.box(figsize=(15, 3), rot=30)


# In[51]:


p = df_last.pivot_table(index="연도", columns="지역명", values="평당분양가격")
p.plot(figsize=(15, 3), rot=30)
# 그래프의 밖에 legend 표시하도록 설정
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ### Seaborn 으로 시각화 해보기

# In[52]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


# barplot으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(10, 3))
sns.barplot(data=df_last, x="지역명", y="평당분양가격", ci=None)


# In[54]:


# barplot으로 연도별 평당분양가격을 그려봅니다.

sns.barplot(data=df_last, x="연도", y="평당분양가격")


# In[55]:


sns.catplot(data=df_last, x="연도", y="평당분양가격", kind="bar", col="지역명", col_wrap=4)


# https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot

# In[56]:


# lineplot으로 연도별 평당분양가격을 그려봅니다.
# hue 옵션을 통해 지역별로 다르게 표시해 봅니다.
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_last, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# In[57]:


sns.relplot(data=df_last, x="연도", y="평당분양가격", 
            hue="지역명", kind="line", col="지역명", col_wrap=4, ci=None)


# ### boxplot과 violinplot

# In[58]:


# 연도별 평당분양가격을 boxplot으로 그려봅니다.
# 최솟값
# 제 1사분위수
# 제 2사분위수( ), 즉 중앙값
# 제 3 사분위 수( )
# 최댓값

sns.boxplot(data=df_last, x="연도", y="평당분양가격")


# In[59]:


# hue옵션을 주어 전용면적별로 다르게 표시해 봅니다.
plt.figure(figsize=(12, 3))
sns.boxplot(data=df_last, x="연도", y="평당분양가격", hue="전용면적")


# In[60]:


# 연도별 평당분양가격을 violinplot으로 그려봅니다.

sns.violinplot(data=df_last, x="연도", y="평당분양가격")


# ### lmplot과 swarmplot 

# In[61]:


df_last["연도"].unique()


# In[62]:


sns.regplot(data=df_last, x="연도", y="평당분양가격")


# In[63]:


sns.regplot(data=df_last, x="연도", y="평당분양가격", x_jitter=.1)


# In[64]:


# 연도별 평당분양가격을 lmplot으로 그려봅니다. 
# hue 옵션으로 전용면적을 표현해 봅니다.
sns.lmplot(data=df_last, x="연도", y="평당분양가격", hue="전용면적", col="전용면적", col_wrap=3,  x_jitter=.1)


# In[65]:


# 연도별 평당분양가격을 swarmplot 으로 그려봅니다. 
# swarmplot은 범주형(카테고리) 데이터의 산점도를 표현하기에 적합합니다.
plt.figure(figsize=(15, 3))
sns.swarmplot(data=df_last, x="연도", y="평당분양가격", hue="전용면적", size=1)


# ### 이상치 보기

# In[66]:


# 평당분양가격의 최대값을 구해서 max_price 라는 변수에 담습니다.

df_last["평당분양가격"].describe()


# In[67]:


max_price = df_last["평당분양가격"].max()
max_price


# In[68]:


# 서울의 평당분양가격이 특히 높은 데이터가 있습니다. 해당 데이터를 가져옵니다.
df_last[df_last["평당분양가격"] == max_price]


# ### 수치데이터 히스토그램 그리기

# distplot은 결측치가 있으면 그래프를 그릴 때 오류가 납니다. 
# 따라서 결측치가 아닌 데이터만 따로 모아서 평당분양가격을 시각화하기 위한 데이터를 만듭니다.
# 데이터프레임의 .loc를 활용하여 결측치가 없는 데이터에서 평당분양가격만 가져옵니다.

# In[69]:


h = df_last["평당분양가격"].hist(bins=10)


# In[70]:


# 결측치가 없는 데이터에서 평당분양가격만 가져옵니다. 그리고 price라는 변수에 담습니다.
# .loc[행]
# .loc[행, 열]
price = df_last.loc[df_last["평당분양가격"].notnull(), "평당분양가격"]


# In[71]:


# distplot으로 평당분양가격을 표현해 봅니다.

sns.distplot(price)


# In[72]:


# sns.distplot(price, hist=False, rug=True)
sns.kdeplot(price, shade=True)
sns.rugplot(price)


# In[73]:


sns.displot(price, kde=True, rug=True)


# In[74]:


sns.displot(data=df_last, x="평당분양가격", kde=True, rug=True, hue="전용면적")


# In[75]:


sns.displot(data=df_last, x="평당분양가격", kde=True, rug=True, hue="전용면적", col="전용면적", col_wrap=3)


# In[76]:


sns.displot(data=df_last, x="평당분양가격", kde=True, rug=True,
            hue="지역명", col="지역명", col_wrap=1, aspect=5, height=1)


# * distplot을 산마루 형태의 ridge plot으로 그리기
# * https://seaborn.pydata.org/tutorial/axis_grids.html#conditional-small-multiples
# * https://seaborn.pydata.org/examples/kde_ridgeplot.html

# In[77]:


# subplot 으로 표현해 봅니다.
g = sns.FacetGrid(df_last, row="지역명",
                  height=1.7, aspect=4,)
g.map(sns.kdeplot, "평당분양가격");


# In[78]:


# pairplot
df_last_notnull = df_last.loc[df_last["평당분양가격"].notnull(), 
                          ["연도", "월", "평당분양가격", "지역명", "전용면적"]]
sns.pairplot(df_last_notnull, hue="지역명")


# In[79]:


# 규모구분(전용면적)별로 value_counts를 사용해서 데이터를 집계해 봅니다.
df_last["전용면적"].value_counts()


# ## 2015년 8월 이전 데이터 보기

# In[80]:


pd.options.display.max_columns = 25


# In[81]:


df_last.head()


# In[82]:


df_first.head()


# In[83]:


# df_first 변수에 담겨있는 데이터프레임의 정보를 info를 통해 봅니다.
df_first.info()


# In[84]:


# 결측치가 있는지 봅니다.
df_first.isnull().sum()


# ### melt로 Tidy data 만들기
# pandas의 melt를 사용하면 데이터의 형태를 변경할 수 있습니다. 
# df_first 변수에 담긴 데이터프레임은 df_last에 담겨있는 데이터프레임의 모습과 다릅니다. 
# 같은 형태로 만들어주어야 데이터를 합칠 수 있습니다. 
# 데이터를 병합하기 위해 melt를 사용해 열에 있는 데이터를 행으로 녹여봅니다.
# 
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-by-melt
# * [Tidy Data 란?](https://vita.had.co.nz/papers/tidy-data.pdf)

# In[85]:


df_first.head(1)


# In[86]:


df_first.melt(id_vars="지역", var_name="기간", value_name="평당분양가격")


# In[87]:


# pd.melt 를 사용하며, 녹인 데이터는 df_first_melt 변수에 담습니다. 

df_first_melt = df_first.melt(id_vars="지역", var_name="기간", value_name="평당분양가격")
df_first_melt.head()


# In[88]:


# df_first_melt 변수에 담겨진 컬럼의 이름을 
# ["지역명", "기간", "평당분양가격"] 으로 변경합니다.

df_first_melt.columns = ["지역명", "기간", "평당분양가격"]
df_first_melt.head(1)


# ### 연도와 월을 분리하기
# * pandas 의 string-handling 사용하기 : https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

# In[89]:


date = "2013년12월"
date


# In[90]:


# split 을 통해 "년"을 기준으로 텍스트를 분리해 봅니다.
date.split("년")


# In[91]:


# 리스트의 인덱싱을 사용해서 연도만 가져옵니다.
date.split("년")[0]


# In[92]:


# 리스트의 인덱싱과 replace를 사용해서 월을 제거합니다.

date.split("년")[-1].replace("월", "")


# In[93]:


# parse_year라는 함수를 만듭니다.
# 연도만 반환하도록 하며, 반환하는 데이터는 int 타입이 되도록 합니다.

def parse_year(date):
    year = date.split("년")[0]
    year = int(year)
    return year

y = parse_year(date)
print(type(y))
y


# In[94]:


# 제대로 분리가 되었는지 parse_year 함수를 확인합니다.
parse_year(date)


# In[95]:


# parse_month 라는 함수를 만듭니다.
# 월만 반환하도록 하며, 반환하는 데이터는 int 타입이 되도록 합니다.
def parse_month(date):
    month = date.split("년")[-1].replace("월", "")
    month = int(month)
    return month


# In[96]:


# 제대로 분리가 되었는지 parse_month 함수를 확인합니다.
parse_month(date)


# In[97]:


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 연도만 추출해서 새로운 컬럼에 담습니다.

df_first_melt["연도"] = df_first_melt["기간"].apply(parse_year)
df_first_melt.head(1)


# In[98]:


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 월만 추출해서 새로운 컬럼에 담습니다.

df_first_melt["월"] = df_first_melt["기간"].apply(parse_month)
df_first_melt.head(1)


# In[99]:


# df_last와 병합을 하기 위해서는 컬럼의 이름이 같아야 합니다.
# sample을 활용해서 데이터를 미리보기 합니다.
df_last.sample()


# In[100]:


# 버전에 따라 tolist() 로 동작하기도 합니다.
# to_list() 가 동작하지 않는다면 tolist() 로 해보세요.
df_last.columns.to_list()


# In[101]:


cols = ['지역명', '연도', '월', '평당분양가격']
cols


# In[102]:


# 최근 데이터가 담긴 df_last 에는 전용면적이 있습니다. 
# 이전 데이터에는 전용면적이 없기 때문에 "전체"만 사용하도록 합니다.
# loc를 사용해서 전체에 해당하는 면적만 copy로 복사해서 df_last_prepare 변수에 담습니다.
df_last_prepare = df_last.loc[
    df_last["전용면적"] == "전체", cols].copy()
df_last_prepare.head(1)


# In[103]:


# df_first_melt에서 공통된 컬럼만 가져온 뒤
# copy로 복사해서 df_first_prepare 변수에 담습니다.
df_first_prepare = df_first_melt[cols].copy()
df_first_prepare.head(1)


# ### concat 으로 데이터 합치기
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

# In[104]:


# df_first_prepare 와 df_last_prepare 를 합쳐줍니다.

df = pd.concat([df_first_prepare, df_last_prepare])
df.shape


# In[105]:


# 제대로 합쳐졌는지 미리보기를 합니다.

df


# In[106]:


# 연도별로 데이터가 몇개씩 있는지 value_counts를 통해 세어봅니다.

df["연도"].value_counts(sort=False)


# ### pivot_table 사용하기
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-and-pivot-tables

# In[107]:


# 연도를 인덱스로, 지역명을 컬럼으로 평당분양가격을 피봇테이블로 그려봅니다.
t = pd.pivot_table(df, index="연도", columns="지역명", 
                   values="평당분양가격").round()
t


# In[108]:


# 위에서 그린 피봇테이블을 히트맵으로 표현해 봅니다.
plt.figure(figsize=(15, 7))
sns.heatmap(t, cmap="Blues", annot=True, fmt=".0f")


# In[109]:


# transpose 를 사용하면 행과 열을 바꿔줄 수 있습니다.

t.transpose()


# In[110]:


# 바뀐 행과 열을 히트맵으로 표현해 봅니다.

plt.figure(figsize=(15, 7))
sns.heatmap(t.T, cmap="Blues", annot=True, fmt=".0f")


# In[111]:


# Groupby로 그려봅니다. 인덱스에 ["연도", "지역명"] 을 넣고 그려봅니다.
g = df.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack().round()
g


# In[112]:


plt.figure(figsize=(15, 7))
sns.heatmap(g.T, annot=True, fmt=".0f", cmap="Greens")


# ## 2013년부터 최근 데이터까지 시각화하기
# ### 연도별 평당분양가격 보기

# In[113]:


# barplot 으로 연도별 평당분양가격 그리기
sns.barplot(data=df, x="연도", y="평당분양가격")


# In[114]:


# pointplot 으로 연도별 평당분양가격 그리기
plt.figure(figsize=(12, 4))
sns.pointplot(data=df, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# In[115]:


# 서울만 barplot 으로 그리기
df_seoul = df[df["지역명"] == "서울"].copy()
print(df_seoul.shape)

sns.barplot(data=df_seoul, x="연도", y="평당분양가격", color="b")
sns.pointplot(data=df_seoul, x="연도", y="평당분양가격")


# In[116]:


# 연도별 평당분양가격 boxplot 그리기
sns.boxplot(data=df, x="연도", y="평당분양가격")


# In[117]:


sns.boxenplot(data=df, x="연도", y="평당분양가격")


# In[118]:


# 연도별 평당분양가격 violinplot 그리기
plt.figure(figsize=(10, 4))
sns.violinplot(data=df, x="연도", y="평당분양가격").set_title("연도별 전국 평균 평당 분양가격(단위:천원)")


# In[119]:


# 연도별 평당분양가격 swarmplot 그리기
plt.figure(figsize=(12, 5))
sns.swarmplot(data=df, x="연도", y="평당분양가격", hue="지역명", size=1)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# ### 지역별 평당분양가격 보기

# In[120]:


# barplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(12, 4))
sns.barplot(data=df, x="지역명", y="평당분양가격")


# In[121]:


# 위와 같은 그래프를 미리 연산을 하고 결과값으로 그리는 방법
# groupby 로 구하기
mean_price = df.groupby(["지역명"])["평당분양가격"].mean().to_frame().sort_values(by="평당분양가격", ascending=False)


# In[122]:


# groupby 로 했던 방법을 똑같이
# pivot_table 로 구하기
mean_price = df.pivot_table(index="지역명", values="평당분양가격").sort_values(by="평당분양가격", ascending=False)


# In[123]:


plt.figure(figsize=(12, 4))
sns.barplot(data=mean_price, x=mean_price.index, y="평당분양가격", palette="Blues_r")


# In[124]:


# boxplot 으로 지역별 평당분양가격을 그려봅니다.

plt.figure(figsize=(12, 4))
sns.boxplot(data=df, x="지역명", y="평당분양가격")


# In[125]:


plt.figure(figsize=(12, 4))
sns.boxenplot(data=df, x="지역명", y="평당분양가격")


# In[126]:


# violinplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(24, 4))
sns.violinplot(data=df, x="지역명", y="평당분양가격")


# In[127]:


# swarmplot 으로 지역별 평당분양가격을 그려봅니다.

plt.figure(figsize=(24, 4))
sns.swarmplot(data=df, x="지역명", y="평당분양가격", hue="연도", size=2)


# In[ ]:




