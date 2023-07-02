import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pivottablejs import pivot_ui
import plotly.express as px
import plotly.graph_objects as go
import re
from wordcloud import WordCloud, STOPWORDS
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
import joblib


st.set_page_config(page_title='HHI Dashboard', layout='wide')
# Load the trained model
model = joblib.load('akmal_HHI_2.pkl')


# Define the process_data function
def process_data(df):
    # Choose variables
    df2 = df[['onPageTime', 'nextTime', 'user_active_days',
              'total_visit_counts', 'total_unique_session', 'total_time_spent', 'time_spent_per_session', 'timeOnPage']]

    # Standard scaler
    sc = StandardScaler()
    dataset = sc.fit_transform(df2)
    x = pd.DataFrame(dataset, columns=df2.columns)
    return x

# Convert seconds to minutes and seconds format
def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


# Function to extract website name using regex
def extract_website_name(url):
    match = re.search(r'https?://(www\.)?([\w.-]+)\.', url)
    if match:
        return match.group(2)
    else:
        return ''


# def plot_wordcloud(text):
#     stopwords = set(STOPWORDS)
#     wc = WordCloud(stopwords=stopwords, background_color="white", max_words=500, width=800, height=500)
#     wc.generate(text)
#     # plt.imshow(wc, interpolation='bilinear')
#     # plt.axis("off")
#     # plt.show()
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     st.pyplot()
#     return wc

# Create the Streamlit application
def main():
    # Set the page title
    st.title("HouseHold Income Prediction Dashboard")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Preprocess the data
        processed_data = process_data(df)

        # Make predictions
        predictions = model.predict(processed_data)

        # Add predictions to the DataFrame
        df['prediction_value'] = predictions

        # Assign HHI values based on predictions
        df['HHI'] = ''
        df.loc[df['prediction_value'] == 0, 'HHI'] = '1.500-5000'
        df.loc[df['prediction_value'] == 1, 'HHI'] = '1500-'
        df.loc[df['prediction_value'] == 2, 'HHI'] = '5000+'



        ## Setting Key Metrics ##

        ##Counting HHIS
        hhis_counts = df['prediction_value'].count()
        count_1500_5000 = df.loc[df['prediction_value'] == 0, 'prediction_value'].count()
        count_1500_minus =df.loc[df['prediction_value'] == 1, 'prediction_value'].count()
        count_5000_plus = df.loc[df['prediction_value'] == 2, 'prediction_value'].count()

        ## count unique users
        count_user_id = df['fullVisitorId'].nunique()

        ##website metrics
        average_on_page_time = df['onPageTime'].mean()
        formatted_time_1 = format_time(average_on_page_time)
        average_time_on_page = df['timeOnPage'].mean()
        formatted_time_2 = format_time(average_time_on_page)
        average_session = df['time_spent_per_session'].mean()
        formatted_time_3 = format_time(average_session)

        ## Extracting website name from url
        df['website_name'] = df['url'].apply(extract_website_name)

        # Display the scorecard
        st.subheader("Key Metrics")
        col_01, col_02, col_03, col_04 = st.columns(4)
        st.write("--------------------------")
        col1, col2, col3, col4 = st.columns(4)

        with col_01:
            st.metric(label="Total No of HHI's predicted", value= hhis_counts)

        with col_02:
            st.metric(label="Total No of HHI (1500-)", value= count_1500_minus)

        with col_03:
            st.metric(label="Total No of HHI (1500-5000)", value= count_1500_5000)

        with col_04:
            st.metric(label="Total No of HHI (5000+", value= count_5000_plus)


        with col1:
            st.metric(label="Total Unique Visitors", value=count_user_id)

        with col2:
            st.metric(label="Average On Page Time", value=formatted_time_1)

        with col3:
            st.metric(label="Average Time on Page", value=formatted_time_2)

        with col4:
            st.metric(label="Average Time Spent Per Session", value=formatted_time_3)

        st.write("---------------------------")

        col1_row2, col2_row2 = st.columns(2)

        with col1_row2:
            fig = px.histogram(df, x='country', color="HHI")
            fig.update_layout(height=500, width=700)
            fig.update_layout(xaxis_title='Country', yaxis_title='HHI Count')
            st.subheader("HHI Groups across countries")
            st.plotly_chart(fig)

        with col2_row2:

            fig2 = px.histogram(df, x='region', color="HHI")
            fig2.update_layout(height=500, width=700)
            fig2.update_layout(xaxis_title='Region', yaxis_title='HHI Counts')
            st.subheader("HHI Groups across the region")
            st.plotly_chart(fig2)

        st.write("--------------------------")

        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            fig3 = px.histogram(df, x='mobileDeviceBranding', color="HHI")
            fig3.update_layout(height=500, width=700)
            fig3.update_layout(xaxis_title='Websites', yaxis_title='Website Visits')
            st.subheader("Mobile Devices By HHI Groups")
            st.plotly_chart(fig3)

        with row3_col2:
            fig3 = px.histogram(df, x='website_name', color="HHI")
            fig3.update_layout(height=500, width=700)
            fig3.update_layout(xaxis_title='Websites', yaxis_title='Website Visits')
            st.subheader("Website Visits By HHI Groups")
            st.plotly_chart(fig3)

        st.write("--------------------------")
        ## most read articles

        row4_col1, row4_col2, row4_col3 = st.columns([0.8, 0.1, 0.8])

        with row4_col1:
            st.subheader("Top 10 Most Article Visits")
            unique_pages = df.drop_duplicates(subset='total_visit_counts')
            top_10_pages = unique_pages.nlargest(10, 'total_visit_counts')[['total_visit_counts','website_name', 'pageTitle', 'HHI']]
            st.write(top_10_pages)

        with row4_col2:
            ""

        with row4_col3:
            st.subheader("Top 10 Article Screentime by timeOnPage")
            unique_pages = df.drop_duplicates(subset='timeOnPage')
            top_10_pages = unique_pages.nlargest(10, 'timeOnPage')[['timeOnPage','website_name', 'pageTitle', 'HHI']]
            st.write(top_10_pages)

        # Display the DataFrame with predictions

        st.write("--------------------------")

        st.subheader(':cloud: Word Cloud :cloud: of articles for each HHI Groups')
        ## Creating wordcloud for each HHI

        my_stopwords = set(
            ['the', 'and', 'in', 'is', 'will', ',', 'u', '.', 'go', 'still', '-', 'car', 'tu', 'ni', 'dah',
             'yg', 'yang', 'tak',
             'kereta', 'beli', 'la', 'nak', 'ada', 'dan', 'buat', 'dia', 'macam', 'mcm', 'x', 'xde', 'takde',
             'mana',
             'pun', 'nk', 'nak', 'je', 'ke', 'kat', 'dah', 'dh', 'lagi', 'lg', 'dengan', 'dgn', 'tapi', 'tp',
             'kalau', 'klu', 'klau', 'kita', 'kte', 'memang', 'mmg', 'kena', 'kna', 'dari', 'kata', 'di',
             'boleh', 'bole',
             'org', 'orang', 'apa', 'lah', 'aku', 'kau', 'pun', 'pon', 'itu', 'tu', 'tue', 'tu.', 'dlm',
             'dalam', 'utk', 'untuk', '&',
             'sbb', 'sebab', 'tk', 'makin', 'mkn', 'abis', 'tau', 'jadi', 'jd', 'kena', 'kene', 'ini', 'ni',
             'nie',
             'sy', 'saye', 'saya', 'ke?', 'ke', 'dpt', 'dapat', 'dpn', 'depan', 'now', 'even', 'got', 'want',
             'eh', 'ya',
             'de', '?', 'n', 'news straits time', 'new strait', 'news straits', 'new', 'straits', 'times', 'says', 'ohbulan', 'bh',
             'berita', 'malaysia', 'nation', 'nst', 'nsttv'
             ])
        STOPWORDS.update(my_stopwords)
        stopwords = set(STOPWORDS)

        # Create word clouds for each income bracket group
        income_groups = df['HHI'].unique()

        # Create columns for each word cloud
        columns = st.columns(len(income_groups))

        for i, group in enumerate(income_groups):
            # Filter data for the current income group
            group_data = df[df['HHI'] == group]

            # Concatenate the page titles into a single string
            text = ' '.join(group_data['pageTitle'])

            # Create the WordCloud with stop words
            wordcloud = WordCloud(
                width=500,
                height=400,
                background_color='white',
                max_words=50,
                stopwords=stopwords
            ).generate(text)

            # Plot the WordCloud in the corresponding column
            with columns[i]:
                st.subheader(f':cloud: {group} HHI')
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

        st.write("--------------------------")

        df2 = df[['HHI', 'prediction_value', 'browser', 'operatingSystem', 'mobileDeviceBranding',
                  'country', 'region', 'website_name', 'pageTitle', 'total_visit_counts',
                  'total_unique_session', 'total_time_spent', 'timeOnPage']]
        st.dataframe(df2)

        # # Add a "Clear" button
        # if st.button("Clear"):
        #     uploaded_file = None




# Run the Streamlit application
if __name__ == '__main__':
    main()