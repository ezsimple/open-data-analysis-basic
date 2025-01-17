#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/open-data-01-apt-price-input)
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
# ### 전국 평균 분양가격(2013년 9월부터 2015년 8월까지)
# * 전국 공동주택의 3.3제곱미터당 평균분양가격 데이터를 제공
# 
# ###  주택도시보증공사_전국 평균 분양가격(2019년 12월)
# * 전국 공동주택의 연도별, 월별, 전용면적별 제곱미터당 평균분양가격 데이터를 제공
# * 지역별 평균값은 단순 산술평균값이 아닌 가중평균값임

# In[ ]:





# In[ ]:


# 파이썬에서 쓸 수 있는 엑셀과도 유사한 판다스 라이브러리를 불러옵니다.


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

# In[ ]:


# 최근 분양가 파일을 로드해서 df_last 라는 변수에 담습니다.
# 파일로드시 OSError가 발생한다면, engine="python"을 추가해 보세요.
# 윈도우에서 파일탐색기의 경로를 복사해서 붙여넣기 했는데도 파일을 불러올 수 없다면
# 아마도 경로에 있는 ₩ 역슬래시 표시를 못 읽어왔을 가능성이 큽니다. 
# r"경로명" 으로 적어주세요.
# r"경로명"으로 적게 되면 경로를 문자 그대로(raw) 읽으라는 의미입니다.



# In[ ]:


# head 로 파일을 미리보기 합니다.
# 메소드 뒤에 ?를 하면 자기호출 이라는 기능을 통해 메소드의 docstring을 출력합니다.
# 메소드의 ()괄호 안에서 Shift + Tab키를 눌러도 같은 문서를 열어볼 수 있습니다.
# Shift + Tab + Tab 을 하게 되면 팝업창을 키울 수 있습니다.


# In[ ]:


# tail 로도 미리보기를 합니다.


# ### 2015년 부터 최근까지의 데이터 로드
# 전국 평균 분양가격(2013년 9월부터 2015년 8월까지) 파일을 불러옵니다.
# df_first 라는 변수에 담고 shape로 행과 열의 갯수를 출력합니다.

# In[ ]:


# 해당되는 폴더 혹은 경로의 파일 목록을 출력해 줍니다.


# In[ ]:


# df_first 에 담고 shape로 행과 열의 수를 출력해 봅니다.
df_first


# In[ ]:


# df_first 변수에 담긴 데이터프레임을 head로 미리보기 합니다.


# In[ ]:


# df_first 변수에 담긴 데이터프레임을 tail로 미리보기 합니다.


# ### 데이터 요약하기

# In[ ]:


# info 로 요약합니다.


# ### 결측치 보기

# isnull 혹은 isna 를 통해 데이터가 비어있는지를 확인할 수 있습니다.
# 결측치는 True로 표시되는데, True == 1 이기 때문에 이 값을 다 더해주면 결측치의 수가 됩니다.

# In[ ]:


# isnull 을 통해 결측치를 봅니다.


# In[ ]:


# isnull 을 통해 결측치를 구합니다.


# In[ ]:


# isna 를 통해 결측치를 구합니다.


# ### 데이터 타입 변경
# 분양가격이 object(문자) 타입으로 되어 있습니다. 문자열 타입을 계산할 수 없기 때문에 수치 데이터로 변경해 줍니다. 결측치가 섞여 있을 때 변환이 제대로 되지 않습니다. 그래서 pd.to_numeric 을 통해 데이터의 타입을 변경합니다.

# In[ ]:





# ### 평당분양가격 구하기
# 공공데이터포털에 올라와 있는 2013년부터의 데이터는 평당분양가격 기준으로 되어 있습니다.
# 분양가격을 평당기준으로 보기위해 3.3을 곱해서 "평당분양가격" 컬럼을 만들어 추가해 줍니다.

# In[ ]:





# ### 분양가격 요약하기

# In[ ]:


# info를 통해 분양가격을 봅니다.


# In[ ]:


# 변경 전 컬럼인 분양가격(㎡) 컬럼을 요약합니다.


# In[ ]:


# 수치데이터로 변경된 분양가격 컬럼을 요약합니다.



# ### 규모구분을 전용면적 컬럼으로 변경
# 규모구분 컬럼은 전용면적에 대한 내용이 있습니다. 전용면적이라는 문구가 공통적으로 들어가고 규모구분보다는 전용면적이 좀 더 직관적이기 때문에 전용면적이라는 컬럼을 새로 만들어주고 기존 규모구분의 값에서 전용면적, 초과, 이하 등의 문구를 빼고 간결하게 만들어 봅니다.
# 
# 이 때 str 의 replace 기능을 사용해서 예를들면 "전용면적 60㎡초과 85㎡이하"라면 "60㎡~85㎡" 로 변경해 줍니다.
# 
# * pandas 의 string-handling 기능을 좀 더 보고 싶다면 :
# https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

# In[ ]:


# 규모구분의 unique 값 보기



# In[ ]:


# 규모구분을 전용면적으로 변경하기



# ### 필요없는 컬럼 제거하기
# drop을 통해 전처리 해준 컬럼을 제거합니다. pandas의 데이터프레임과 관련된 메소드에는 axis 옵션이 필요할 때가 있는데 행과 열중 어떤 기준으로 처리를 할 것인지를 의미합니다. 보통 기본적으로 0으로 되어 있고 행을 기준으로 처리함을 의미합니다. 메모리 사용량이 줄어들었는지 확인합니다.

# In[ ]:


# info로 정보 보기



# In[ ]:


# drop 사용시 axis에 유의 합니다.
# axis 0:행, 1:열



# In[ ]:


# 제거가 잘 되었는지 확인 합니다.



# In[ ]:


# 컬럼 제거를 통해 메모리 사용량이 줄어들었는지 확인합니다.



# ## groupby 로 데이터 집계하기
# groupby 를 통해 데이터를 그룹화해서 연산을 해봅니다.

# In[ ]:


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
# df.groupby(["인덱스로 사용할 컬럼명"])["계산할 컬럼 값"].연산()



# In[ ]:


# 전용면적으로 분양가격의 평균을 구합니다.



# In[ ]:


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.



# In[ ]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.


# ## pivot table 로 데이터 집계하기
# * groupby 로 했던 작업을 pivot_table로 똑같이 해봅니다.

# In[ ]:


# 지역명을 index 로 평당분양가격 을 values 로 구합니다.



# In[ ]:


# df_last.groupby(["전용면적"])["평당분양가격"].mean()


# In[ ]:


# 전용면적을 index 로 평당분양가격 을 values 로 구합니다.



# In[ ]:


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
# df_last.groupby(["전용면적", "지역명"])["평당분양가격"].mean().unstack().round()


# In[ ]:





# In[ ]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.
# g = df_last.groupby(["연도", "지역명"])["평당분양가격"].mean()



# ## 최근 데이터 시각화 하기
# ### 데이터시각화를 위한 폰트설정
# 한글폰트 사용을 위해 matplotlib의 pyplot을 plt라는 별칭으로 불러옵니다.

# In[ ]:


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

# In[ ]:


# 지역명으로 분양가격의 평균을 구하고 선그래프로 시각화 합니다.



# In[ ]:


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.



# 전용면적별 분양가격의 평균값을 구하고 그래프로 그려봅니다.

# In[ ]:


# 전용면적으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.



# In[ ]:


# 연도별 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.



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

# In[ ]:


# index를 월, columns 를 연도로 구하고 평당분양가격 으로 pivot_table 을 구하고 상자수염그림을 그립니다.




# In[ ]:


# columns 에 "연도", "전용면적"을 추가해서 pivot_table 을 만들고 시각화 합니다.



# In[ ]:


# index를 월, columns 를 지역명으로 구하고 평당분양가격 으로 pivot_table 을 구하고 선그래프를 그립니다.




# ### Seaborn 으로 시각화 해보기

# In[ ]:


# 라이브러리 로드하기



# In[ ]:


# barplot으로 지역별 평당분양가격을 그려봅니다.



# In[ ]:


# barplot으로 연도별 평당분양가격을 그려봅니다.



# In[ ]:


# catplot 으로 서브플롯 그리기



# https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot

# In[ ]:


# lineplot으로 연도별 평당분양가격을 그려봅니다.
# hue 옵션을 통해 지역별로 다르게 표시해 봅니다.



# In[ ]:


# relplot 으로 서브플롯 그리기



# ### boxplot과 violinplot

# In[ ]:


# 연도별 평당분양가격을 boxplot으로 그려봅니다.
# 최솟값
# 제 1사분위수
# 제 2사분위수( ), 즉 중앙값
# 제 3 사분위 수( )
# 최댓값



# In[ ]:


# hue옵션을 주어 전용면적별로 다르게 표시해 봅니다.



# In[ ]:


# 연도별 평당분양가격을 violinplot으로 그려봅니다.



# ### lmplot과 swarmplot 

# In[ ]:


# 연도별 평당분양가격을 lmplot으로 그려봅니다. 
# hue 옵션으로 전용면적을 표현해 봅니다.



# In[ ]:


# 연도별 평당분양가격을 swarmplot 으로 그려봅니다. 
# swarmplot은 범주형(카테고리) 데이터의 산점도를 표현하기에 적합합니다.



# ### 이상치 보기

# In[ ]:


# 평당분양가격의 최대값을 구해서 max_price 라는 변수에 담습니다.


# In[ ]:





# In[ ]:


# 서울의 평당분양가격이 특히 높은 데이터가 있습니다. 해당 데이터를 가져옵니다.



# ### 수치데이터 히스토그램 그리기

# distplot은 결측치가 있으면 그래프를 그릴 때 오류가 납니다. 
# 따라서 결측치가 아닌 데이터만 따로 모아서 평당분양가격을 시각화하기 위한 데이터를 만듭니다.
# 데이터프레임의 .loc를 활용하여 결측치가 없는 데이터에서 평당분양가격만 가져옵니다.

# In[ ]:





# In[ ]:





# In[ ]:


# 결측치가 없는 데이터에서 평당분양가격만 가져옵니다. 그리고 price라는 변수에 담습니다.
# .loc[행]
# .loc[행, 열]



# In[ ]:


# distplot으로 평당분양가격을 표현해 봅니다.



# In[ ]:


# sns.distplot(price, hist=False, rug=True)



# * distplot을 산마루 형태의 ridge plot으로 그리기
# * https://seaborn.pydata.org/tutorial/axis_grids.html#conditional-small-multiples
# * https://seaborn.pydata.org/examples/kde_ridgeplot.html

# In[ ]:


# subplot 으로 표현해 봅니다.



# In[ ]:


# pairplot



# In[ ]:


# 규모구분(전용면적)별로 value_counts를 사용해서 데이터를 집계해 봅니다.



# ## 2015년 8월 이전 데이터 보기

# In[ ]:


# 모든 컬럼이 출력되게 설정합니다.



# In[ ]:


# head 로 미리보기를 합니다.



# In[ ]:


# df_first 변수에 담겨있는 데이터프레임의 정보를 info를 통해 봅니다.



# In[ ]:


# 결측치가 있는지 봅니다.



# ### melt로 Tidy data 만들기
# pandas의 melt를 사용하면 데이터의 형태를 변경할 수 있습니다. 
# df_first 변수에 담긴 데이터프레임은 df_last에 담겨있는 데이터프레임의 모습과 다릅니다. 
# 같은 형태로 만들어주어야 데이터를 합칠 수 있습니다. 
# 데이터를 병합하기 위해 melt를 사용해 열에 있는 데이터를 행으로 녹여봅니다.
# 
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-by-melt
# * [Tidy Data 란?](https://vita.had.co.nz/papers/tidy-data.pdf)

# In[ ]:


# head 로 미리보기 합니다.



# In[ ]:


# pd.melt 를 사용하며, 녹인 데이터는 df_first_melt 변수에 담습니다. 




# In[ ]:


# df_first_melt 변수에 담겨진 컬럼의 이름을 
# ["지역명", "기간", "평당분양가격"] 으로 변경합니다.



# ### 연도와 월을 분리하기
# * pandas 의 string-handling 사용하기 : https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

# In[ ]:


date = "2013년12월"
date


# In[ ]:


# split 을 통해 "년"을 기준으로 텍스트를 분리해 봅니다.



# In[ ]:


# 리스트의 인덱싱을 사용해서 연도만 가져옵니다.



# In[ ]:


# 리스트의 인덱싱과 replace를 사용해서 월을 제거합니다.



# In[ ]:


# parse_year라는 함수를 만듭니다.
# 연도만 반환하도록 하며, 반환하는 데이터는 int 타입이 되도록 합니다.


# In[ ]:


# 제대로 분리가 되었는지 parse_year 함수를 확인합니다.



# In[ ]:


# parse_month 라는 함수를 만듭니다.
# 월만 반환하도록 하며, 반환하는 데이터는 int 타입이 되도록 합니다.


# In[ ]:


# 제대로 분리가 되었는지 parse_month 함수를 확인합니다.



# In[ ]:


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 연도만 추출해서 새로운 컬럼에 담습니다.



# In[ ]:


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 월만 추출해서 새로운 컬럼에 담습니다.



# In[ ]:


# 컬럼명을 리스트로 만들때 버전에 따라 tolist() 로 동작하기도 합니다.
# to_list() 가 동작하지 않는다면 tolist() 로 해보세요.



# In[ ]:


# df_last와 병합을 하기 위해서는 컬럼의 이름이 같아야 합니다.
# sample을 활용해서 데이터를 미리보기 합니다.



# In[ ]:


cols = ['지역명', '연도', '월', '평당분양가격']
cols


# In[ ]:


# 최근 데이터가 담긴 df_last 에는 전용면적이 있습니다. 
# 이전 데이터에는 전용면적이 없기 때문에 "전체"만 사용하도록 합니다.
# loc를 사용해서 전체에 해당하는 면적만 copy로 복사해서 df_last_prepare 변수에 담습니다.



# In[ ]:


# df_first_melt에서 공통된 컬럼만 가져온 뒤
# copy로 복사해서 df_first_prepare 변수에 담습니다.



# ### concat 으로 데이터 합치기
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

# In[ ]:


# df_first_prepare 와 df_last_prepare 를 합쳐줍니다.



# In[ ]:


# 제대로 합쳐졌는지 미리보기를 합니다.



# In[ ]:


# 연도별로 데이터가 몇개씩 있는지 value_counts를 통해 세어봅니다.



# ### pivot_table 사용하기
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-and-pivot-tables

# In[ ]:


# 연도를 인덱스로, 지역명을 컬럼으로 평당분양가격을 피봇테이블로 그려봅니다.


# In[ ]:


# 위에서 그린 피봇테이블을 히트맵으로 표현해 봅니다.



# In[ ]:


# transpose 를 사용하면 행과 열을 바꿔줄 수 있습니다.



# In[ ]:


# 바뀐 행과 열을 히트맵으로 표현해 봅니다.



# In[ ]:


# Groupby로 그려봅니다. 인덱스에 ["연도", "지역명"] 을 넣고 그려봅니다.



# In[ ]:





# ## 2013년부터 최근 데이터까지 시각화하기
# ### 연도별 평당분양가격 보기

# In[ ]:


# barplot 으로 연도별 평당분양가격 그리기


# In[ ]:


# pointplot 으로 연도별 평당분양가격 그리기


# In[ ]:


# 서울만 barplot 으로 그리기


# In[ ]:


# 연도별 평당분양가격 boxplot 그리기



# In[ ]:





# In[ ]:


# 연도별 평당분양가격 violinplot 그리기



# In[ ]:


# 연도별 평당분양가격 swarmplot 그리기



# ### 지역별 평당분양가격 보기

# In[ ]:


# barplot 으로 지역별 평당분양가격을 그려봅니다.



# In[ ]:


# boxplot 으로 지역별 평당분양가격을 그려봅니다.


# In[ ]:





# In[ ]:


# violinplot 으로 지역별 평당분양가격을 그려봅니다.


# In[ ]:


# swarmplot 으로 지역별 평당분양가격을 그려봅니다.



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




