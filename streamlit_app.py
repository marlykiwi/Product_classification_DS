import streamlit as st 
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd 
import numpy as np
import plotly.express as px 
import cv2
import plotly.graph_objects as go
import plotly.offline as py


import warnings
warnings.filterwarnings('ignore')

#Headline
st.markdown("# Automated Product Classification")

#Reading the dataframe
file_path = 'C:/Python_3.10/DS_data/X_train_update.csv'
df = pd.read_csv(file_path, index_col=0)
lowercase = lambda x:str(x).lower()
df.rename(lowercase, axis='columns',inplace=True)
file_name = 'C:/Python_3.10/DS_data/Y_train_CVw08PX.csv'
df_y = pd.read_csv(file_name, index_col=0)
lowercase_y = lambda x:str(x).lower()
df_y.rename(lowercase_y, axis='columns',inplace=True)

df_combo = pd.concat([df, df_y], axis = 1)


fig_countplot = plt.figure(figsize=(10, 4))
sns.countplot(x="prdtypecode",data=df_combo, order = df_combo['prdtypecode'].value_counts().index)
plt.xticks(rotation =90)

fig1 = plt.figure(figsize=(5,5))
img1 = cv2.imread('C:/Python_3.10/DS_data/language_plot.jpg')
plt.imshow(img1)
plt.axis('off')

fig_dist = plt.figure(figsize=(5,5))
img_dist = cv2.imread('C:/Python_3.10/DS_data/image_distribution.png')
plt.imshow(img_dist)
plt.axis('off')


st.sidebar.markdown("# Menu")
#sidebar_names = ['Introduction', 'Dataset', 'Model processing', 'Conclusion']
#sidebar_line = st.radio(sidebar_names)


import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('french'))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

df_wc = pd.read_csv('C:/Python_3.10/DS_data/WC_streamlit.csv')
bag = " "
for row in df_wc['text'][:50]:
#    for element in row:
    bag += row + ' '

wc = WordCloud(background_color="white", max_words=100, 
                stopwords=stop_words, max_font_size=50, random_state=42)



cloud = plt.figure(figsize= (10,6)) # Initialization of a figure
wc.generate(bag)           # "Calculation" of the wordcloud
plt.imshow(wc) # Display
plt.axis('off')
#plt.show()
#st.pyplot(cloud)

#st.write("image plotting")
df_images = pd.read_csv("C:/Python_3.10/DS_data/images_w_bounding_box_streamlit.csv")
import cv2
img_bb = cv2.imread('C:/Python_3.10/DS_data/image_1263597046_product_3804725264.jpg')
img_resized = cv2.resize(img_bb, (500, 500))
fig_bb = plt.figure(figsize=(10, 4))
plt.subplot(1,4,2)
plt.imshow(img_resized)
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(img_resized)
x1 = df_images.x1[0]
x2 = df_images.x2[0]
y1 = df_images.y1[0]
y2 = df_images.y2[0]
plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],"r")
plt.axis('off')






if st.sidebar.checkbox('Introduction'):
    st.markdown(
    """
    E-commerce: 

- Online buying or selling of products and services.
- Two main stakeholders: vendors (marketplace) and customers
- Addition of products by multiple vendors
Challenge:
- Assigning the correct category to the product
Consequence:
- Lost user (user not finding the product/looking somewhere else)
- Incorrect product recommendation
Solution: Artifical Intelligence
- Automatising  the process
- Increasing the success rate
"""
    )
if st.sidebar.checkbox('Dataset'):
    if st.button('Click to display the text dataset'):
        st.subheader('Dataset')
        st.dataframe(df_combo) #Main way to display df
        #fig_df.show()
        st.pyplot(fig_countplot)
        st.pyplot(fig1)
        st.pyplot(fig_dist)
        st.pyplot(cloud)
     #   st.bar_chart(data=df_y, x='prdtypecode')#, use_container_width=True)
        #st.plotly_chart(fig3)
    if st.button('Click to view the image dataset'):
      #  st.write('Ok') 
        image_data = pd.read_csv('C:/Python_3.10/DS_data/image_files.csv', index_col=0)  
        st.dataframe(image_data)
if st.sidebar.checkbox('Model processing'):
    if st.button('Models & Results') is True:
    #   st.header('Model processing')
        st.subheader ('Text processing')
        st.markdown("""Text mining:
- converting strings to lowercase
- removing web links/html tags/e-mails/numbers through regular expressions
- removing punctuations and white spaces
- checking the languages present in the text
- removing stop words
- applying word tokenization
- retaining only words above a given length of letters
""")
        st.markdown("""Word2Vec:
- Vectorizes the data and provides the semantic & arithmetic properties
- Gives orientation/position of the words in space
- Reduces the size of the data to be processed
- Parameters: vector_size=100, min_count=1, window=5
- Creation of sentence vectors for input to algorithm
""")
## Word Embedding
        PCA =pd.read_csv('C:/Python_3.10/DS_data/PCA.csv', index_col=0)
    #   PCA.head()
        #word_emb = plt.figure(figsize =(10,10))
        #plt.figure(figsize=(15,15))
        fig,ax = plt.subplots()
        PCA.plot('pc1', 'pc2', kind='scatter', ax=ax, c='red', marker='x')
        for i, j in PCA.iterrows():
            ax.annotate(i, j)
        plt.show()
        st.pyplot(fig, ax)
        #st.image(fig, caption='Vector representation of words in 2D space')
        #st.select_slider('slide here', options=PCA['pc1'])

        # Model(s)
        st.markdown('Tested models & their scores')
        models = pd.DataFrame({'Model name': ['K-Nearest Neighbors', 'Random Forest', 'Support Vector Machine', 'Dense Neural Networks'], 
           'Train Score': [0.80, 0.99, 0.85, 0.76], 'Test Score': [0.75, 0.76, 0.79, 0.75] })
        st.table(models)
        st.markdown("""K-Nearest Neighbors:
- test size = 20%, random state = 1234
- Parameters: n_neighbors = 7 (GridSearchCV in range 1-20), metric = minkowski (default)')
- Output: the majority class among its nearest neighbors')
- Training score: 0.80, Test score: 0.75
""")
        knn_clf = pd.read_csv('C:/Python_3.10/DS_data/knn_text_report.csv', index_col=0)
        knn_clf
        knn_cm = pd.read_csv('C:/Python_3.10/DS_data/knn_cm.csv', index_col=0)
        cm = plt.figure(figsize=(25,20))
        #sns.heatmap(knn_cm, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        sns.heatmap(knn_cm, cmap=plt.cm.Blues, linewidths = 0.30, annot = True)
        st.pyplot(cm)

        
        st.subheader ('Image processing')
        df_bb= pd.read_csv('C:/Python_3.10/DS_data/images_w_bounding_box_streamlit.csv', index_col=0) 
        st.dataframe(df_bb)
        st.pyplot(fig_bb)

        st.markdown('Convolutional Neural Networks')
        fig_cv = plt.figure(figsize=(5,5))
        img_cv = cv2.imread('C:/Python_3.10/DS_data/cnn_summary.png')
        plt.imshow(img_cv)
        plt.axis('off')
        st.pyplot(fig_cv)

        cnn_clf= pd.read_csv('C:/Python_3.10/DS_data/cr_index.csv', index_col=0)
        cnn_clf

        cnn_cm = pd.read_csv('C:/Python_3.10/DS_data/crosstab_new_codes.csv', index_col=0)
        cm_1 = plt.figure(figsize=(25,20))
        #sns.heatmap(knn_cm, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        sns.heatmap(cnn_cm, cmap=plt.cm.Blues, linewidths = 0.30, annot = True)
        st.pyplot(cm_1)
     #  st.write('Great !!!')
        



if st.sidebar.checkbox('Conclusion'):
    st.markdown(""" Conclusion:
- Score: 0.75 for text, and 0.34 for images.
Suggestions for improving the result:
- Using a pre-trained model
- Balancing the text sample by oversampling
- Balancing of images through augmentation 
- Requiring a cleaner text input to the models.
""")
#Reading and showing the dataframe

