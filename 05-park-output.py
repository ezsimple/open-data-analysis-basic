#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/open-data-05-park-output)
# 
# 
# # 전국 도시 공원 표준 데이터
# https://www.data.go.kr/dataset/15012890/standard.do
# * 전국 도시 공원 표준데이터에는 데이터를 전처리 해보기에 좋은 데이터가 많습니다.
# * 시간, 결측치, 이상치, 수치형, 범주형 데이터를 고르게 볼 수 있으며 다양한 텍스트 데이터 처리를 해볼 수 있습니다.
# * 또 정규표현식을 활용해서 텍스트 데이터 전처리와 데이터 마스킹 기법에 대해 다룹니다.
# * 그리고 이렇게 전처리한 내용을 바탕으로 전국 도시공원에 대한 분포를 시각화해 봅니다.
# * 어떤 공원이 어느 지역에 어떻게 분포되어 있는지를 위경도로 표현해 봅니다.
# 
# ## 이번 챕터에서 설치가 필요한 도구
# 
# * 별도의 설치가 필요합니다.(folium 을 설치했던 것 처럼 따로 설치해야지만 사용할 수 있습니다.)
# * 윈도우
#     * <font color="red">주피터 노트북 상에서 설치가 되지 않으니</font> anaconda prompt 를 열어서 설치해 주세요.
#     * <font color="red">관리자 권한</font>으로 아나콘다를 설치하셨다면 다음의 방법으로 anaconda prompt 를 열어 주세요.
#     <img src="https://i.imgur.com/GhoLwsd.png">
# * 맥
#     * terminal 프로그램을 열어 설치해 주세요. 
# 
# 
# ### Pandas Profiling
# * [pandas-profiling/pandas-profiling: Create HTML profiling reports from pandas DataFrame objects](https://github.com/pandas-profiling/pandas-profiling)
# 
# * 아나콘다로 주피터를 설치했다면 : `conda install -c conda-forge pandas-profiling`
# * pip로 주피터를 설치했다면 : `pip install pandas-profiling`
# 
# ### 워드클라우드
# [amueller/word_cloud: A little word cloud generator in Python](https://github.com/amueller/word_cloud)
# 
# * 다음 명령어로 설치가 가능합니다. conda prompt 혹은 터미널을 열어 설치해 주세요.
# 
# * conda : `conda install -c conda-forge wordcloud`
# * pip : `pip install wordcloud`
# 

# ## 분석에 사용할 도구를 불러옵니다.

# In[1]:


# 필요한 라이브러리를 로드합니다.
# pandas, numpy, seaborn, matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Seaborn 설정으로 시각화의 스타일, 폰트 설정하기
# * [matplotlib.pyplot.rc — Matplotlib 3.1.3 documentation](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.rc.html)

# In[2]:


# seaborn 의 set 기능을 통해 폰트, 마이너스 폰트 설정, 스타일 설정을 합니다.
# "Malgun Gothic"
sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')


# In[3]:


# 한글폰트 설정 확인을 합니다.

pd.Series([1, -1, 0, 5, -5]).plot(title="한글폰트")


# In[4]:


from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")


# ## 데이터 로드

# In[5]:


# 데이터를 로드해서 df 라는 변수에 담습니다.

df = pd.read_csv("data/전국도시공원표준데이터.csv", encoding="cp949")
df.shape


# In[6]:


# 미리보기를 합니다.

df.head()


# ## Pandas Profiling
# * [pandas-profiling/pandas-profiling: Create HTML profiling reports from pandas DataFrame objects](https://github.com/pandas-profiling/pandas-profiling)
# 
# * 별도의 설치가 필요합니다.(folium 을 설치했던 것 처럼 따로 설치해야지만 사용할 수 있습니다.)
# * conda : `conda install -c conda-forge pandas-profiling`
# * pip : `pip install pandas-profiling --upgrade`

# In[7]:


# 버전에 따라 기능의 차이가 있습니다. 
# 이 강좌에서 사용하는 버전은 다음과 같습니다.

import pandas_profiling

pandas_profiling.__version__


# In[ ]:


get_ipython().system('pip show pandas_profiling')


# * 미리 생성해 놓은 리포트 보기 : https://corazzon.github.io/open-data-analysis-basic/05-park_pandas_profile.html
# 
# * <font color="red">한글 폰트 설정 문제 해결 영상 : </font> https://youtu.be/BhZvZpNF9jU

# In[10]:


# pandas_profiling 의 ProfileReport 를 불러와 표현합니다.
# 이 때 title은 "도시공원 표준 데이터" 로 하고 주피터 노트북에서 바로 보면 iframe을 통해 화면이 작게 보이기 때문에
# 별도의 html 파일로 생성해서 그려보세요.
from pandas_profiling import ProfileReport

# 운영체제에 맞는 폰트를 설정해 주세요.
# Win
# plt.rc("font", family="Malgun Gothic")
# Mac
# plt.rc("font", family="AppleGothic")
# 위 폰트 설정 방법으로 오류가 난다면 아래의 seaborn set을 통해 설정해 보세요.
# Win
# sns.set(font="Malgun Gothic")
# Mac
# sns.set(font="AppleGothic")

get_ipython().run_line_magic('time', 'profile = ProfileReport(df, title="도시공원 표준 데이터")')
profile.to_file(output_file="05-park_pandas_profile.html")


# ## 기본 정보 보기

# In[9]:


# info로 기본 정보를 봅니다.

df.info()


# In[10]:


# 결측치의 수를 구합니다.
df.isnull().sum()


# In[11]:


# 결측치 비율 구하기
# 결측의 평균을 통해 비율을 구하고 100을 곱해줍니다.

round(df.isnull().mean() * 100, 2)


# ## 결측치 시각화
# * [ResidentMario/missingno: Missing data visualization module for Python.](https://github.com/ResidentMario/missingno)

# In[12]:


sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')


# In[13]:


# 이전 챕터에서 설치하지 않았다면 아나콘다에 missingno를 설치합니다. 
# !conda install -c conda-forge missingno
# 라이브러리를 로드합니다.

import missingno

missingno.matrix(df)


# * 그래프의 색상 선택 : [Choosing Colormaps in Matplotlib — Matplotlib 3.1.0 documentation](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)

# In[14]:


# seaborn을 통해 위와 같은 결측치 그래프를 그려봅니다.

null = df.isnull()

plt.figure(figsize=(15, 4))
sns.heatmap(null, cmap="Blues_r")


# # 데이터 전처리
# * 불필요한 컬럼 제거
# * 시도 결측치 처리
#     * 다른 데이터로 대체
#     * 도로명 혹은 지번 둘 중 하나만 있다면 있는 데이터로 대체
# * 아웃라이어 데이터 제거 혹은 대체
#     * 위경도가 국내 범위를 벗어나는 데이터의 경우 제외하고 그리도록 처리

# ## 불필요한 컬럼 제거하기

# In[15]:


# 전체 컬럼명을 출력해 봅니다.

df.columns


# In[16]:


# drop 으로 'Unnamed: 19' 를 제거하기

print(df.shape)
df = df.drop(["Unnamed: 19"], axis=1)
print(df.shape)


# ## 결측치 대체
# ### 도로명 주소와 지번 주소 
# * 둘 중 하나만 있을 때 나머지 데이터로 결측치 대체하기

# In[17]:


# 도로명 주소의 널값 수

df['소재지도로명주소'].isnull().sum()


# In[18]:


# 지번 주소의 널값 수

df['소재지지번주소'].isnull().sum()


# In[19]:


# "소재지도로명주소"와 "소재지지번주소"가 모두 결측치가 아닌 데이터를 찾습니다.

df[df['소재지도로명주소'].notnull() & df['소재지지번주소'].notnull()].shape


# In[20]:


# "소재지도로명주소"의 결측치를 fillna 를 통해 "소재지지번주소"로 채웁니다.
df['소재지도로명주소'] = df['소재지도로명주소'].fillna(df['소재지지번주소'])


# In[21]:


# "소재지도로명주소"의 결측치수를 세어봅니다.
df['소재지도로명주소'].isnull().sum()


# In[22]:


# "소재지도로명주소"와 "소재지지번주소"가 모두 결측치인 데이터를 찾습니다.

df[df['소재지도로명주소'].isnull() & df['소재지지번주소'].isnull()].shape


# ## 파생변수 만들기
# ### 주소를 통한 시도, 구군 변수 생성하기

# In[23]:


# 소재지도로명주소로 시도, 구군 변수 생성하기
# .str.split(' ', expand=True)[0] 을 통해 공백문자로 분리하고 리스트의 첫번째 값을 가져오도록 하기
df["시도"] = df["소재지도로명주소"].str.split(expand=True)[0]
df[["소재지도로명주소", "시도"]].head(3)


# In[24]:


# 구군 가져오기
df["구군"] = df["소재지도로명주소"].str.split(expand=True)[1]
df[["소재지도로명주소", "시도", "구군"]].sample(3)


# In[25]:


# 시도 데이터의 빈도수 세어보기
df["시도"].value_counts()


# In[26]:


# 강원은 "강원도"로 변경해줄 필요가 보입니다.
df["시도"] = df["시도"].replace("강원", "강원도")
df["시도"].value_counts()


# ## 이상치 제거
# * 경도, 위도의 이상치 처리하기

# In[27]:


# 위경도 시각화
sns.scatterplot(data=df, x="경도", y="위도")


# In[28]:


# 위 지도로 위도와 경도의 아웃라이어 데이터를 제외하고 출력해 봅니다.
# 좀 더 정확하게 출력하려면 대한민국 위경도 데이터 범위를 다시 넣어줍니다. 
# 이상치를 제거한 데이터를 df_park 라는 새로운 변수에 담습니다.
df_park = df[(df["경도"] < 132) & 
             (df["위도"] > 32)].copy()


# In[29]:


# 위도 경도의 아웃라이어 데이터가 제거되었는지 확인함

sns.scatterplot(data=df_park, x="경도", y="위도")


# In[30]:


df[["위도", "경도"]].describe()


# In[31]:


# 위경도가 잘못입력된 데이터를 봅니다.
# 주소가 잘못되지는 않았습니다.
# 주소를 통해 위경도를 다시 받아올 필요가 있습니다.
df[(df["경도"] > 132) | (df["위도"] < 32)]


# ## 날짜 데이터 다루기

# In[32]:


# 지정고시일의 데이터 타입을 봅니다.
df["지정고시일"].head(1)


# In[33]:


# 지정고시일의 데이터 타입을 datetime 형태로 변경합니다.
df["지정고시일"] = pd.to_datetime(df["지정고시일"])
df["지정고시일"].head(1)


# In[34]:


# 고시연도와 고시월을 지정고시일에서 추출해서 파생변수를 만듭니다.
df["고시연도"] = df["지정고시일"].dt.year
df["고시월"] = df["지정고시일"].dt.month

df[["지정고시일", "고시연도", "고시월"]].head()


# In[35]:


# 화랑공원의 고시연도를 찾아봅니다. 데이터 전처리가 맞게 되었는지 확인해 봅니다.
df.loc[df["공원명"].str.contains("화랑"), ["공원명", "고시연도"]]


# In[36]:


# 고시연도의 분포를 distplot 으로 그립니다.
# 이 때, 결측치가 있으면 오류가 나기 때문에 결측치가 없는 데이터만 따로 모아 그립니다.
sns.distplot(df.loc[df["고시연도"].notnull(), "고시연도"], rug=True)


# * 판다스 스타일링 : [Styling — pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html)
# * 숫자의 많고 적음에 따라 heatmap 처럼 스타일을 표현 가능 합니다.

# In[37]:


# 연도와 월별 고시 현황을 pivot_table로 그려봅니다.
# year_month 라는 변수에 담아 재사용 합니다.
# 데이터가 많기 때문에 2000년보다 크고 2019년보다 작은 연도의 데이터만 봅니다.
year_month = pd.pivot_table(df, index="고시연도", columns="고시월", 
               values="공원명", aggfunc="count")
year_month = year_month[(year_month.index > 2000) & 
                        (year_month.index < 2019)]
year_month.astype(int).style.background_gradient()


# ## 텍스트 데이터 다루기
# ### 정규표현식 
# 
# * [정규 표현식 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C_%ED%91%9C%ED%98%84%EC%8B%9D)
# 
# 
# * 파이썬 공식문서 정규표현식 참고하기 : 
#     * https://docs.python.org/3.8/library/re.html#re.sub
#     
#     
# * 문자열 바꾸기 : re.sub("규칙", "패턴", "데이터")
#     * https://docs.python.org/3.8/library/re.html#text-munging
# 
# 
# * 정규표현식 문자열 패턴
#     * https://docs.python.org/3.8/howto/regex.html#matching-characters
# 
#     
#     
# * [    ] : 일치시킬 문자 세트의 패턴
# * [가나다] : 가 or 나 or 다 중에 하나를 포함하고 있는지
# * [가-힣] : 한글 가부터 힣까의 문자 중 하나를 포함하고 있는지
# * [0-9] : 0~9까지의 숫자 중 하나를 포함하고 있는지
# * [^0-9] : 숫자를 포함하고 있지 않음
# * [^가-힣] : 한글이 포함되어 있지 않음
# * [가-힣+] : 한글이 하나 이상 포함되는지
# 
# 
# | 클래스 | 표현식 | 설명 |
# |----|-----|----|
# |\d | [0-9]|십진수와 일치|
# |\D| [^0-9] | 숫자가 아닌 문자와 일치|
# |\s |[\t\n\r\f\v] | 공백 문자와 일치 |
# |\S |[^\t\n\r\f\v]| 공백이 아닌 문자와 일치|
# | \w| [a-zA-Z0-9_] | 모든 숫자 영문자와 일치|
# | \W| [^a-zA-Z0-9_]| 영문자, 숫자가 아닌 것과 일치|
# 
# 
# 표현식	설명
# 
# |표현식 | 설명 |	
# |----|-----|
# |^x	| 문자열의 시작을 표현하며 x 문자로 시작|
# |x$	| 문자열의 종료를 표현하며 x 문자로 종료|
# |.x	| 임의의 한 문자의 자리수를 표현하며 문자열이 x 로 끝남|
# |x+	| 반복을 표현하며 x 문자가 한번 이상 반복됨|
# |x*	| 반복여부를 표현하며 x 문자가 0번 또는 그 이상 반복됨|
# |x?	| 존재여부를 표현하며 x 문자가 존재할 수도, 존재하지 않을 수도 있음|
# |x\|y	| or 를 표현하며 x 또는 y 문자가 존재함|
# |(x)	| 그룹을 표현하며 x 를 그룹으로 처리함|
# |(x)(y)	| 그룹들의 집합을 표현하며 순서대로 번호를 부여하여 x, y 는 각 그룹의 데이터로 관리|
# |(x)(?:y)	| 그룹들의 집합에 대한 예외를 표현하며 그룹 집합으로 관리되지 않음|
# |x{n}	| 반복을 표현하며 x 문자가 n번 반복됨|
# |x{n,}	| 반복을 표현하며 x 문자가 n번 이상 반복됨
# |x{n,m}	| 반복을 표현하며 x 문자가 최소 n번 이상 최대 m 번 이하로 반복됨|
# 

# In[38]:


# 정규표현식 라이브러리를 로드합니다.
# 설치가 필요 없이 파이썬에 내장되어 있는 라이브러리 입니다.
# 파이썬에 배터리 포함됨 이라는 특징이 있는데 이런 여러 표준 라이브러리가 잘 갖춰져 있습니다.
import re


# ### 공원 보유시설

# In[39]:


# 컬럼이 너무 많으면 끝까지 보이지 않습니다. 
# options.display 를 사용해 max_columns 값을 채워줍니다.

pd.options.display.max_columns = 100


# In[40]:


# 데이터를 미리보기 합니다.
df.head(1)


# ### 운동시설
# * 텍스트 데이터의 빈도수를 구합니다.
# * 한 컬럼에 여러 텍스트가 , 혹은 - 등의 특수문자로 구분되어 있습니다.
# * 데이터 전처리를 통해 텍스트의 빈도수를 구합니다.

# In[41]:


df["공원보유시설(운동시설)"].value_counts(sort=False).head(5)


# In[43]:


# "공원보유시설(운동시설)"의 unique 값을 구하고 리스트 형태로 만듭니다.
# 그리고 다시 str 형태로 변경하고 gym 이라는 변수에 담습니다.
gym = df["공원보유시설(운동시설)"].unique().tolist()
gym = str(gym)
gym[:1000]


# In[104]:


# replace 기능을 사용해서 ,를 빈문자로 공백을 , 로 +를 ,로 변경합니다.
gym = gym.replace("+", ",").replace("'", ",").replace(" ", ",").replace("/", ",")
gym[:1000]


# In[106]:


# 제거할 특수문자를 빈문자로 대체합니다.
# re.sub('[?.,;:|\)*~`’!^\-_+<>@\#$%&-=#}※]', '', text)
# 정규표현식에서 +, *, . 등은 특별한 의미를 갖습니다. 
# 이런 문자를 제거할 때는 \를 통해 예외처리를 합니다. 
# 예를들어 .이라면 \. 으로 사용합니다.
# 여기에서는 숫자와 .을 제거하도록 합니다.
gym = re.sub("[\[\d\-nan\.\]]", "", gym)
gym = re.sub(",{2,}", ",", gym)
gym[:1000]


# In[46]:


# split을 사용해 문자열을 ,로 리스트형태로 만든 후에 
# 판다스의 시리즈 형태로 데이터를 만들면 빈도수를 구할 수 있습니다.
# 빈도수는 gym_count 라는 변수에 담습니다.

gym_count = pd.Series(gym.split(",")).value_counts()
gym_count.head()


# In[47]:


# 빈도수를 구한 gym_count 변수에서 head를 통해 상위 50개 단어에 대한 그래프를 그립니다.
gym_count.head(50).plot.bar(figsize=(15, 4))


# ### 워드클라우드로 빈도수 표현하기
# [amueller/word_cloud: A little word cloud generator in Python](https://github.com/amueller/word_cloud)
# 
# * 별도의 설치가 필요합니다. 
# * 다음 명령어로 설치가 가능합니다. conda prompt 혹은 터미널을 열어 설치해 주세요.
# 
# * conda 사용시 : `conda install -c conda-forge wordcloud`
# * pip 사용시 : `pip install wordcloud`

# In[48]:


# 공식문서의 튜토리얼을 보고 wordcloud를 그리는 함수를 만들어 봅니다.
# 이때 폰트 설정시 폰트명이 아닌 폰트의 설치 경로를 입력해 주셔야 합니다.
# 윈도우 : r"C:\Windows\Fonts\malgun.ttf" 해당 경로에 폰트가 있는지 확인을 해주세요.
# 맥 : r"/Library/Fonts/AppleGothic.ttf"
# 나눔고딕 등의 폰트를 설치했다면 : '/Library/Fonts/NanumBarunGothic.ttf'

from wordcloud import WordCloud

def wordcloud(data, width=1200, height=500):
    word_draw = WordCloud(
        font_path=r"/Library/Fonts/NanumBarunGothic.ttf",
        width=width, height=height,
        stopwords=["운동기구", "개소", "외종"], 
        background_color="white",
        random_state=42
    )
    word_draw.generate(data)

    plt.figure(figsize=(15, 7))
    plt.imshow(word_draw)
    plt.axis("off")
    plt.show()


# In[49]:


# 위에서 만든 함수에 전처리한 text인 gym을 넣어 그려봅니다.
wordcloud(gym, width=1200, height=700)


# ### 편익시설

# In[50]:


# 편익시설에서 필요 없는 문자를 제거하고
# 토큰화를 위해 각 시설을 "," 로 구분하도록 전처리를 하는 함수를 구현합니다. 
# 함수명은 get_string 으로 합니다.

def get_string(text):
    if pd.isna(text):
        return text
    
    text = re.sub("[\d\.'\-(nan)\[\]\/]", "", text)
    text = re.sub(" ", "", text)
    text = re.sub(",{2,}", ",", text)
    return text


# In[51]:


# 함수가 잘 구현되었는지 확인해 봅니다.
# 다음의 문자를 전처리 했을 때  
# "[1. '화장실' 5, 2. 주차장, -, nan]" 전처리 후 '화장실,주차장,' 가 출력되는지 확인해 주세요.

get_string("[1. '화장실' 5, 2. 주차장, -, nan, /]")


# In[52]:


# 결측치를 넣어봅니다. 오류가 나지않는지 확인해 주세요.

get_string(np.nan)


# In[53]:


# map을 통해 위에서 구현한 함수를 적용해 줍니다.
# 원본과 비교해 보기 위해 "편익시설"이라는 새로운 컬럼을 만듭니다.

df["편익시설"] = df["공원보유시설(편익시설)"].map(get_string)
df["편익시설"].head()


# In[54]:


# 전처리 한 "편익시설"의 빈도수를 구하고 
# tail 로 끝부분에 있는 데이터를 미리보기 합니다.

df["편익시설"].value_counts().tail()


# In[55]:


type(np.nan)


# In[56]:


# 문자열을 연결해 주는 join 으로 편익시설을 연결해 주는 방법이 있습니다.
# 이 때, 결측치가 있으면 결측치는 숫자형태이기 때문에 문자와 연결되지 않아 오류가 납니다.
# 따라서 결측치를 제외한 값만 가져와서 연결합니다.

fac = ",".join(df.loc[df["편익시설"].notnull(), "편익시설"])
fac = get_string(fac)
fac[:100]


# In[57]:


# 위에서 전처리한 "편익시설" 컬럼을 리스트 형태로 만들고 다시 str 으로 변환 합니다. 
# 위에서 만든 get_string 함수로 전처리를 합니다.
fac = str(df["편익시설"].tolist())
fac = get_string(fac)
fac[:100]


# In[58]:


# ,가 2개 이상 들어가면 하나로 변경합니다.
# '화장실,주차장,,,' 텍스트를 정규표현식으로 바꿔봅니다.

re.sub(",{2,}", ",", '화장실,주차장,,,')


# In[59]:


# 다시 ","로 텍스트를 분리하고 판다스의 Series 형태로 만들고 value_counts 로 빈도수를 계산합니다.

fac_count = pd.Series(fac.split(",")).value_counts()
fac_count.head()


# In[60]:


# 상위 50개 단어에 대해 막대그래프를 그려봅니다.

fac_count.head(50).plot.bar(figsize=(15, 4))


# In[61]:


wordcloud(fac)


# ### 키워드 별 빈도수 구하기
# * 위에서 빈도수를 구해보니 "파고라", "파고라등" 이라는 두 개의 단어가 다른 빈도로 세어졌습니다.
# * "화장실"도 "화장실동", "공중화장실" 등 다양한 텍스트가 있는 것을 볼 수 있습니다.
# * 일일이 전처리를 해주면 좋지만 데이터 전처리에는 많은 예외가 등장합니다.
# * 그래서 때로는 보고자 하는 키워드를 넣고 빈도수를 구하는게 전처리를 하는 작업보다 빠를 때도 있습니다.
# * 여기에서는 운동시설이나 편의시설이 있는데 특정 시설을 키워드에 넣고 구하는 방법을 사용해 봅니다.

# In[62]:


# "공원보유시설(편익시설)" 에서 문자열 처리를 하기 위해 결측치를 제외한 값을 가져옵니다.
# df_fac 이라는 변수에 담습니다.
facilities = ["화장실", "주차장", "음수대", "파고라", "정자", 
              "벤치", "의자", "놀이터", "축구장", "야영장", "CCTV"]

df_fac = df[df["공원보유시설(편익시설)"].notnull()]


# In[63]:


# facilities 리스트에 있는 값으로 for문을 활용해 컬럼을 만듭니다.
# 해당 단어가 들어가면 True, 아니면 False로 값이 표현되도록 합니다.

for f in facilities:
    df[f] = df_fac["공원보유시설(편익시설)"].str.contains(f)


# In[64]:


# facilities 리스트로 만든 각 컬럼의 빈도수를 구합니다. 

df[facilities].sum()


# ## 특정 정보 마스킹
# ### 전화번호 마스킹

# In[65]:


# 가운데가 4자리일 때 번호 마스킹 "010-1234-5678"
# re.sub("규칙", "패턴", "데이터")
# 규칙
# (\d{3}) 는 숫자 3자리를 의미합니다.
# (\d{1,2}) 는 숫자 1~2자리를 의미합니다.
# 패턴
# r"\1-\2**-**\5" 의 "\숫자"는 앞에서 () 안에 정의한 값의 순서를 의미합니다. 
# 변수의 순서라고 생각해도 됩니다.
# 여기에서 \3, \4는 쓰지 않고 * 로 대체했습니다.
# r은 raw를 의미합니다.

re.sub("(\d{3})-(\d{2})(\d{2})-(\d{2})(\d{2})", r"\1-\2**-**\5", "010-1234-5678")


# In[66]:


# 가운데가 3자리일 때 번호 마스킹이 잘 동작하는지 확인 "010-123-5678"

re.sub("(\d{3})-(\d{1,2})(\d{2})-(\d{2})(\d{2})", r"\1-\2**-**\5", "010-123-5678")


# In[67]:


# 가운데가 2자리일 때는 마스킹 없이 보이는지 확인하기 "010-12-5678"

re.sub("(\d{3})-(\d{1,2})(\d{2})-(\d{2})(\d{2})", r"\1-\2**-**\5", "010-12-5678")


# In[68]:


# 국가 번호가 들어갈 때 확인하기 "+82-10-1234-5678"
# - 가 들어가거나 여러 예외사항이 있을 때 마스킹 되지 않을 수 있음 그렇다면 함수를 만들어서 해결해 보기

re.sub("(\d{3})-(\d{1,2})(\d{2})-(\d{2})(\d{2})", r"\1-\2**-**\5", "+82-10-1234-5678")


# In[69]:


# 함수를 통해 가운데 들어오는 번호를 마스킹 하도록 처리
# 국제번호 등 다양한 형식의 번호가 들어왔을 때 
# - 를 기준으로 뒤에서 두 번째 항목을 마스킹 처리 하되, 
# 마지막 자리는 앞 두자리만 마스킹 합니다.
# 함수명은 mask_phone_number 로 합니다.

phone = "+82-10-1234-5678"

def mask_phone_number(phone):
    
    if pd.isnull(phone):
        return phone
    
    num = phone.split("-")
    num_len = len(num) // 2
    
    num[num_len] = re.sub("\w", "*", num[num_len])
    num[-1] = re.sub("(\d{2})(\d{2})", r"**\2", num[-1])
    
    return "-".join(num)

mask_phone_number(phone)


# In[70]:


# 결측치가 들어갔을 때 오류가 날 수 있기 때문에 결측치를 체크합니다.

mask_phone_number(np.nan)


# In[71]:


# 위에서 만든 함수를 map을 통해 적용합니다.

df["전화번호(전처리)"] = df["전화번호"].map(mask_phone_number)
df[["공원명", "전화번호", "전화번호(전처리)"]].sample(10)


# ### 이메일 주소 마스킹
# * 해당 데이터에는 이메일 주소가 없지만 정규표현식을 더 연습해 보기 위해 이메일주소도 마스킹처리를 해봅니다.

# In[72]:


# 이메일 주소의 형식만 남기고 모두 마스크처리 합니다.
email = "address@mail.com"

re.sub("[^@.]", "x", email)


# In[73]:


# 이메일 주소 아이디의 일부만 마스크처리 합니다. 
# 이 때 맨 앞과 맨 뒤 문자는 그대로 놔두고 나머지 문자에 대해 마스크 처리리를 합니다.
email = "address@mail.com"

address, domain = email.split("@")
mask = re.sub("\w", "*", address[1:-1])
print(f"{address[0]}{mask}{address[-1]}@{domain}")
print(email)


# In[74]:


# 위에서 작성한 예제를 함수로 만듭니다.
def mask_email(email):
    address, domain = email.split("@")
    mask = re.sub("\w", "*", address[1:-1])
    return f"{address[0]}{mask}{address[-1]}@{domain}"
    
mask_email(email)


# In[75]:


# "1234address_123@gmail.com"를 마스킹 처리 합니다.

mask_email("1234address_123@gmail.com")


# ### 자동차 등록번호 마스킹
# * 역시나 해당 데이터에는 자동차 등록번호가 없지만 정규표현식을 더 연습해 보기 위해 마스킹처리를 해봅니다.

# In[76]:


# 자동차 등록번호를 마스킹 합니다. 
# r'*\2\3**\5' 여기에서 \1 에 해당되는 한글 앞의 숫자는 
# 몇 자리가 들어가든 한글 바로 앞의 마지막 한자리만 봅니다.
# 따라서 앞에 있는 숫자는 마지막 번호만 마스킹 하게 됩니다.
car_num = "32가1234"
re.sub("([0-9])([가-힣])([\d])([\d]{2})([\d])", r"*\2\3**\5", car_num)


# In[77]:


# 자동차 등록번호를 마스킹 합니다. 
car_num = "안녕하세요. 자동차 등록 번호는 132가1234 입니다."

re.sub("([0-9])([가-힣])([\d])([\d]{2})([\d])", r"*\2\3**\5", car_num)


# # 데이터 요약하기
# ## 수치형, 범주형 변수 구분하기

# In[78]:


# data_types 변수에 데이터 타입만 따로 모으기

data_types = df.dtypes
data_types[:5]


# ## 수치형 변수 요약

# In[79]:


# 수치형 변수 구하기
# var_numbers 변수에 담습니다.

var_numbers = data_types[data_types != "object"]
var_numbers = var_numbers.index.tolist()
var_numbers


# In[80]:


# 히스토그램을 그리면 수치형 변수에 대한 빈도수 분포를 확인할 수 있습니다.
# 빈도수 분포 => 도수 분포표를 시각화 합니다.

h = df.hist(figsize=(12, 10))


# In[81]:


# describe 를 통해 요약합니다.

df.describe()


# ## 범주형 변수 요약

# In[82]:


# 범주형 변수 구하기
# var_objects 라는 변수에 저장합니다.

var_objects = data_types[data_types == "object"]
var_objects = var_objects.index.tolist()
var_objects[:5]


# In[83]:


# 문자열 타입의 데이터 describe 로 요약하기

df.describe(include="object")


# ## crosstab
# * 범주형 변수끼리의 빈도수 구하기

# In[84]:


# 관리기관과 공원구분에 따른 빈도수를 구합니다.
# 결과가 많기 때문에 "체육공원"을 5개 이상 관리하는 기관명만 따로 뽑습니다.
# 숫자의 많고 적음 표시를 위해 style.background_gradient() 를 사용합니다.
# 결과를 manage_category 변수에 담습니다.

manage_category = pd.crosstab(index=df["관리기관명"], columns=df["공원구분"])
manage_category[manage_category["체육공원"] > 5].style.background_gradient()


# In[85]:


# "시도" 별 "공원구분" 의 수를 구합니다.

city_category = pd.crosstab(index=df["시도"], columns=df["공원구분"])
city_category.style.background_gradient()


# # 시도별 공원 분포

# ## 시도별 공원 비율

# In[86]:


# 시도별로 합계 데이터를 출력
city_count = df["시도"].value_counts().to_frame()
city_mean = df["시도"].value_counts(normalize=True).to_frame()
city_count.head()


# In[87]:


# 합계와 비율을 함께 구합니다.
# city 라는 변수에 담습니다.

city = city_count.merge(city_mean, left_index=True, right_index=True)
city.columns = ["합계", "비율"]
city.style.background_gradient()


# ## 공원구분별 분포

# In[88]:


# "공원구분"별로 색상을 다르게 표현하고 "공원면적"에 따라 원의 크기를 다르게 그립니다.
# 제주도는 해안선과 유사한 모습으로 공원이 배치되어 있는 모습이 인상적입니다.
# df에는 전체 데이터 프레임이 df_park 에는 위경도의 이상치를 제거한 데이터가 들어있습니다.
plt.figure(figsize=(8, 9))
sns.scatterplot(data=df_park, x="경도", y="위도", 
                hue="공원구분", size="공원면적", sizes=(10, 100))


# ## 시도별 공원분포

# In[89]:


# 시도별로 scatterplot 의 색상을 다르게 표현하고 공원면적에 따라 원의 크기를 다르게 그립니다.

plt.figure(figsize=(8, 9))
sns.scatterplot(data=df_park, x="경도", y="위도", 
                hue="시도", size="공원면적", sizes=(10, 100))


# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# In[90]:


# countplot 으로 시도별 빈도수를 그립니다.
plt.figure(figsize=(9, 7))
sns.countplot(data=df, y="시도", order=city_count.index, palette="Greens_r")


# ## 특정 공원구분 가져오기

# In[91]:


# "공원구분"별로 빈도수를 구합니다.

df["공원구분"].value_counts()


# In[92]:


# str.match를 통해 특정 텍스트가 들어가는 공원만 가져오기
# ".*"는 앞뒤로 어떤 문자가 등장하든 상관없다는 의미입니다.
# 따라서 아래의 정규표현식 대신 r'(역사|체육|수변|문화)공원'를 사용해도 똑같이 동작합니다.
park_type = r'.*((역사|체육|수변|문화)공원).*'

park = df[df["공원구분"].str.match(park_type)]
park.shape


# In[93]:


# 위에서 정의한 var_numbers 라는 변수를 가져옵니다.
var_numbers


# In[94]:


# 리스트 다루기 - var_pair 라는 변수에 리스트를 만들어서 값을 추가하고 삭제해 봅니다.
# 기존 리스트에 "공원구분" 원소 추가하기 "공원구분"에 따라 색상을 다르게 표현해 보기 위함
# var_pair 라는 변수에 담습니다. 
var_pair = var_numbers
var_pair.append("공원구분")
var_pair


# In[95]:


# 기존 리스트의 원소를 제거하기 - '지정고시일', '고시연도', '고시월'
# 제거 한 후 다시 제거하려고 하면 다음과 같은 오류가 발생할 수 있습니다.
# ValueError: list.remove(x): x not in list
var_pair.remove("지정고시일")
var_pair.remove("고시연도")
var_pair.remove("고시월")
var_pair


# In[96]:


# 위에서 가져온 var_pair 라는 변수에 담긴 리스트에 있는 컬럼을 사용해 pairplot을 그립니다.

sns.pairplot(park[var_pair], hue="공원구분")


# ## 피봇테이블로 시도별 공원수와 평균면적 구하기

# In[97]:


# 시도별 공원수와 "공원면적" 컬럼의 평균값, 중앙값을 구합니다.
# style.background_gradient()를 사용해서 값의 많고 적음에 따라 다른 색상으로 표현되도록 합니다.
park_size = pd.pivot_table(df, index="시도", values="공원면적", 
               aggfunc=["count", "mean", "median"])
with pd.option_context('display.precision', 2):
    display(park_size.round(2).style.background_gradient())


# ## 내가 사는 지역의 공원을 지도에 표시해 보기

# In[98]:


# 경기도 성남시에 위치한 공원만 출력해 봅니다.
# df_sn 이라는 변수에 담아서 재사용 합니다.

df_sn = df[(df["시도"] == "서울특별시") & (df["구군"] == "강서구")]
df_sn.shape


# In[99]:


# "시도", "구군", "공원명" 컬럼만 미리보기 합니다.

df_sn[["시도", "구군", "공원명"]].head()


# ### 특정 공원 정보 찾아보기
# * 판교의 화랑공원을 찾아봅니다.

# In[100]:


# str.contains 로 "공원명"에 "화랑"이 들어가는 데이터를 찾습니다.

df_sn.loc[df_sn["공원명"].str.contains("염창"), ["공원명", "위도", "경도"]]


# In[101]:


# cols 컬럼에 특정 컬럼만 불러와서 봅니다.
# '공원보유시설(운동시설)', '공원보유시설(유희시설)', '공원보유시설(편익시설)', '공원보유시설(교양시설)' 
cols = "공원보유시설(운동시설)	공원보유시설(유희시설)	공원보유시설(편익시설)	공원보유시설(교양시설)"
cols = cols.split("\t")


df_sn.loc[df_sn["공원명"].str.contains("염창"), cols]


# ###  지도에 표현하기
# * 다음의 문서를 활용해서 지도를 표현합니다.
# * https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Quickstart.ipynb
# * Folium 사용예제 :
# http://nbviewer.jupyter.org/github/python-visualization/folium/tree/master/examples/

# In[102]:


# folium 으로 지도에 표현해 봅니다.
# 지도에서 한글이 깨지면 html 문서로 저장해서 보세요.
import folium

m = folium.Map([37.557808, 126.860243], zoom_start=13)


for n in df_sn.index:
    lat = df_sn.loc[n, "위도"]
    long = df_sn.loc[n, "경도"]
    tooltip = df_sn.loc[n, "공원명"] + " " + df_sn.loc[n, "소재지도로명주소"]
    
    folium.Marker([lat, long], tooltip=tooltip).add_to(m)

m.save('index.html')
m


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




