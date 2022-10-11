import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


st.set_page_config(
    page_title="VoC sentiment analysis",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded")



st.title("VoC: Sentiment Analysis POC")
st.markdown("------------------------------------------------------------------------------------")
# asa = st.sidebar.radio('Select company', ('Nokia','Samsung'))

# st.sidebar.markdown("##Upload reviews data :")

filename = st.sidebar.file_uploader("Upload reviews data:", type=("csv", "xlsx"))

if filename is not None:
    data = pd.read_csv(filename)
    data["body"] = data["body"].astype("str")
    data["score"] = data["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    data["sentiment"] = np.where(data['score'] >= .5, "Positive", "Negative")
    data = data[['brand','body','sentiment','score','date']]
    data['date']=pd.to_datetime(data['date'])
    data['quarter'] = pd.PeriodIndex(data.date, freq='Q')

    per_dt = data.groupby(['brand','sentiment']).size().reset_index()
    per_dt = per_dt.sort_values(['sentiment'],ascending=False)
    per_dt1 = data.groupby(['brand']).size().reset_index()
    per_dt2 = pd.merge(per_dt,per_dt1,how = 'left', on = 'brand')
    per_dt2['Sentiment_Percentage'] = per_dt2['0_x']/per_dt2['0_y']
    per_dt2 = per_dt2[['brand','sentiment','Sentiment_Percentage']]

    brand_c = data.groupby(['brand']).size().reset_index()
    st.sidebar.write("Reviews count by brand:")
    st.sidebar.write("Nokia   : " + str(brand_c[0][1]))
    st.sidebar.write("HUAWEI  : " + str(brand_c[0][0]))
    st.sidebar.write("Samsung : " + str(brand_c[0][2]))

    st.subheader("Phone Reviews Sentiment distribution")

    col3, col4 = st.columns(2)

    with col4:
        data1 = data[data['brand'] == 'Nokia']
        sentiment_count = data1.groupby(['sentiment'])['sentiment'].count()
        sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'sentiment':sentiment_count.values})
        fig = px.pie(sentiment_count,values='sentiment',names='Sentiments',width=550, 
            height=400).update_layout(title_text='Sentiment distribution for Nokia', title_x=0.5)
        st.plotly_chart(fig,use_container_width=True)

    with col3:
        trend_dt = data[data['brand'] == 'Nokia']
        trend_dt['Review_Month'] = trend_dt['date'].dt.strftime('%m-%Y')
        trend_dt1 = trend_dt.groupby(['Review_Month','sentiment']).size().reset_index()
        trend_dt1 = trend_dt1.sort_values(['sentiment'],ascending=False)
        trend_dt1.rename(columns = {0:'Sentiment_Count'}, inplace = True)

        fig2 = px.line(trend_dt1, x="Review_Month", y="Sentiment_Count", color='sentiment',width=600, 
            height=400).update_layout(title_text='Trend analysis of sentiments for Nokia', title_x=0.5)
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown("------------------------------------------------------------------------------------")

    col1, col2 = st.columns(2)
    with col1:
        
        fig = px.histogram(data, x="brand", y="sentiment",
            histfunc="count", color="sentiment",facet_col="sentiment", 
            labels={"sentiment": "sentiment"},width=550, height=400).update_layout(title_text='Distribution by count of sentiment', title_x=0.5)
        st.plotly_chart(fig,use_container_width=True)

    with col2:

        fig1 = px.histogram(per_dt2, x= "brand", y="Sentiment_Percentage",color="sentiment" ,facet_col="sentiment", labels={"sentiment": "sentiment"},
        width=550, height=400).update_layout(yaxis_title="Percentage",title_text='Distribution by percentage of sentiment', title_x=0.5)
        st.plotly_chart(fig1,use_container_width=True)

    
    st.markdown("------------------------------------------------------------------------------------")

    st.subheader("Word Cloud for reviews Sentiment")

    word_ls = ['phone.','phone,','will','window','really','andoid','tracfone','minute','best','time','amazon','need','still','work','phone','huawei','samsung','nokia','windows phone','great','good','use','love','one','amazing','still used','lumia','iphone']
    data['body1'] = data['body'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in (word_ls)]))
    data['body1'] = data['body1'].str.replace('phone',' ')
    
    col5, col6 = st.columns(2)

    with col5:
        # st.text("Positive reviews word cloud for Nokia")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df = data[(data["sentiment"]=="Positive") & (data["brand"]=="Nokia") & (data['score'] > .9)]
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.title("Positive reviews word cloud for Nokia")
        st.pyplot()

    with col6:        
        # st.text("Negative reviews word cloud for Nokia:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df = data[(data["sentiment"]=="Negative") & (data["brand"]=="Nokia") & (data['score'] <=.2)]
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640,colormap="RdYlGn").generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.title("Negative reviews word cloud for Nokia")
        st.pyplot()


    col7, col8 = st.columns(2)

    with col7:
        # st.text("Positive reviews word cloud for HUAWEI:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df = data[(data["sentiment"]=="Positive") & (data["brand"]=="HUAWEI") & (data['score'] > .9)]
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.title("Positive reviews word cloud for HUAWEI")
        st.pyplot()

    with col8:
        # st.text("Negative reviews word cloud for HUAWEI:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df = data[(data["sentiment"]=="Negative") & (data["brand"]=="HUAWEI") & (data['score'] <= .2)]
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640,colormap="RdYlGn").generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.title("Negative reviews word cloud for HUAWEI")
        st.pyplot()
    
    col9, col10 = st.columns(2)
    
    with col9:
        # st.text("Positive reviews word cloud for Samsung:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df = data[(data["sentiment"]=="Positive") & (data["brand"]=="Samsung") & (data['score'] > .9)]
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.title("Positive reviews word cloud for Samsung")
        st.pyplot()


    with col10:
        # st.text("Negative reviews word cloud for Samsung:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df = data[(data["sentiment"]=="Negative") & (data["brand"]=="Samsung") & (data['score'] <= .2)]
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640,colormap="RdYlGn").generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.title("Negative reviews word cloud for Samsung")
        st.pyplot()



    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Top 5 positive reviews for Nokia :")

    pos = data[(data['brand'] == 'Nokia') & (data['score'] > .9)].reset_index()
    pos = pos.sort_values(['score'],ascending=False)
    st.write("1. " +str(pos['brand'][6]) +"| Positive | Sentiment Score: " +str(pos['score'][6]) + " - " + str(pos['body'][6]))
    st.write("2. " +str(pos['brand'][7]) +"| Positive | Sentiment Score: " +str(pos['score'][7]) + " - " + str(pos['body'][7]))
    st.write("3. " +str(pos['brand'][8]) +"| Positive | Sentiment Score: " +str(pos['score'][8]) + " - " + str(pos['body'][8]))
    st.write("4. " +str(pos['brand'][9]) +"| Positive | Sentiment Score: " +str(pos['score'][9]) + " - " + str(pos['body'][9]))
    st.write("5. " +str(pos['brand'][10]) +"| Positive | Sentiment Score: " +str(pos['score'][10]) + " - " + str(pos['body'][10]))


    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Top 5 negative reviews for Nokia :")

    neg = data[(data['brand'] == 'Nokia') & (data['score'] < .1)].reset_index()
    st.markdown("1. " +str(neg['brand'][1]) +"| Negative | Sentiment Score: " +str(neg['score'][1]) + " - " + str(neg['body'][1]))
    st.markdown("2. " +str(neg['brand'][2]) +"| Negative | Sentiment Score: " +str(neg['score'][2]) + " - " + str(neg['body'][2]))
    st.markdown("3. " +str(neg['brand'][3]) +"| Negative | Sentiment Score: " +str(neg['score'][3]) + " - " + str(neg['body'][3]))
    st.markdown("4. " +str(neg['brand'][4]) +"| Negative | Sentiment Score: " +str(neg['score'][4]) + " - " + str(neg['body'][4]))
    st.markdown("5. " +str(neg['brand'][5]) +"| Negative | Sentiment Score: " +str(neg['score'][5]) + " - " + str(neg['body'][5]))
