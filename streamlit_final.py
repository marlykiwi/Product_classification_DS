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
file_path = 'X_train_update.csv'
df = pd.read_csv(file_path, index_col=0)
lowercase = lambda x:str(x).lower()
df.rename(lowercase, axis='columns',inplace=True)
file_name = 'Y_train_CVw08PX.csv'
df_y = pd.read_csv(file_name, index_col=0)
lowercase_y = lambda x:str(x).lower()
df_y.rename(lowercase_y, axis='columns',inplace=True)

df_combo = pd.concat([df, df_y], axis = 1)

#Reading the dataframe in a different way
fig_df = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns), fill_color='paleturquoise',align='left'),
    cells=dict(values=[df.designation, df.description, df.productid, df.imageid],
    fill_color='lavender',
               align='left'))])


#images_df = pd.read_csv('')

#Creating the pie chart for links
num_links = 298
num_no_links = 84916 - num_links
x = [num_no_links, num_links]
labels = ['No links', 'Links']
explode = (0, 0.5)
#Creating the pie chart interactively
fig3 = px.pie(values=x, names=labels,
color_discrete_map={'Sendo':'cyan', 'Tiki':'royalblue','Shopee':'darkblue'})
fig3.update_layout(title="<b>Percentage ratio of the number of links</b>")
#st.plotly_chart(fig3)
#Countplot of product code
fig_countplot = plt.figure(figsize=(10, 4))
sns.countplot(x="prdtypecode",data=df_combo, order = df_combo['prdtypecode'].value_counts().index)
plt.xticks(rotation =90)


st.sidebar.markdown("# Menu")
#sidebar_names = ['Introduction', 'Dataset', 'Model processing', 'Conclusion']
#sidebar_line = st.radio(sidebar_names)


import nltk
nltk.download(‘stopwords’)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('french'))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

df_wc = pd.read_csv('WC_streamlit.csv')
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
df_images = pd.read_csv("images_w_bounding_box_streamlit.csv")
import cv2
img = cv2.imread(df_images["img"][0])
img_shape = img.shape
img = cv2.resize(img,(100,100))
fig_image = plt.figure(figsize= (10, 4))
plt.subplot(1,4,2)
plt.imshow(img[...,::-1])
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(img[...,::-1])
x1 = df_images.x1[0]/img_shape[1]*100
x2 = df_images.x2[0]/img_shape[1]*100
y1 = df_images.y1[0]/img_shape[1]*100
y2 = df_images.y2[0]/img_shape[1]*100
plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],"r")
plt.axis('off')
#st.pyplot(fig_image);


if st.sidebar.checkbox('Introduction'):
    st.markdown(
    """
    E-commerce: Online buying or selling of products and services

    There are two main stakeholders in order for the e-commerce to exist, the vendor or vendors (marketplace) and the customers. In the case of small owner-based e-commerces, the same person is usually responsible of classifying every product, however in the case of large sites, multiple vendors are adding their products on a frequent basis. 
Despite the existence of guidelines, vendors are prone to error when looking for the right category to fit their product. A wrongful classification derives in the user not finding the product and looking for it elsewhere (lost user), a mistaken tax application, or a useless product recommendation throughout the shopping journey. This is where artificial intelligence can play a key role in automatising the process and increasing the chance of success.
One of the main challenges of any digital business is the conversion of visiting users into paying customers. The possibility of creating a sale is related to multiple factors (catalogue curation, price, usability, etc.), however one of the key factors towards a product being purchased is the possibility to find it.
In order to improve the potential of sales, users need to be able to find those products, which they expect to locate within a specific category. The miss-placement of  a product in the wrong category can lead to an orphan and not visited product in the wrong category, a lack of sale opportunity and a frustrated user who might look for the product elsewhere and not return.

    """
    )
if st.sidebar.checkbox('Dataset'):
    if st.button('Click to display the text dataset'):
        st.subheader('Dataset')
        st.dataframe(df_combo) #Main way to display df
        #fig_df.show()
        st.pyplot(fig_countplot)
        st.pyplot(cloud)
     #   st.bar_chart(data=df_y, x='prdtypecode')#, use_container_width=True)
        #st.plotly_chart(fig3)
    if st.button('Click to view the image dataset'):
      #  st.write('Ok') 
        image_data = pd.read_csv('image_files.csv', index_col=0)  
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
        PCA =pd.read_csv('PCA.csv', index_col=0)
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
        knn_clf = pd.read_csv('knn_text_report.csv', index_col=0)
        knn_clf
        knn_cm = pd.read_csv('knn_cm.csv', index_col=0)
        cm = plt.figure(figsize=(25,20))
        #sns.heatmap(knn_cm, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        sns.heatmap(knn_cm, cmap=plt.cm.Blues, linewidths = 0.30, annot = True)
        st.pyplot(cm)

        
        st.subheader ('Image processing')
        image_bb= pd.read_csv('images_w_bounding_box_streamlit.csv', index_col=0) 
        st.dataframe(image_bb)
        st.pyplot(fig_image)

        st.markdown('Convolutional Neural Networks')
        fig_cv = plt.figure(figsize=(5,5))
        img_cv = cv2.imread('cnn_summary.png')
        plt.imshow(img_cv)
        plt.axis('off')
        st.pyplot(fig_cv)

        cnn_clf= pd.read_csv('cr_index.csv', index_col=0)
        cnn_clf

        cnn_cm = pd.read_csv('crosstab_new_codes.csv', index_col=0)
        cm_1 = plt.figure(figsize=(25,20))
        #sns.heatmap(knn_cm, cmap ='RdYlGn', linewidths = 0.30, annot = True)
        sns.heatmap(cnn_cm, cmap=plt.cm.Blues, linewidths = 0.30, annot = True)
        st.pyplot(cm_1)
     #  st.write('Great !!!')
        



if st.sidebar.checkbox('Conclusion'):
    st.markdown(""" Conclusion:
- Pre-processing of raw datasets gave us a maximum score of 0.75 for text, and 0.34 for images
- Suggestion for improving the scores: oversampling of text, image augmentation
""")
#Reading and showing the dataframe

