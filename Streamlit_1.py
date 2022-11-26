#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st 
from PIL import Image 


# In[2]:


import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd 
import numpy as np


# In[3]:

df = pd.read_csv("products_codes.csv", index_col = 0)

import warnings
warnings.filterwarnings('ignore')


# In[4]:


#st.markdown("# Automated Product Classification")
st.title('Automated Product Classification')


# In[5]:

#st.bar_chart(data=df, *, x=codes, y=codes.count(), width=0, height=0, use_container_width=True)
#DATA_URL = ('X_train_update - Copy.csv')


# In[6]:


#st.markdown("# Self Exploratory Visualization")
#st.markdown('Explore the dataset')
#st.markdown("# Exploring the dataset")


# In[7]:


st.markdown('**Information**: CSV format, text, images')


# In[18]:


st.header('Data Exploration')

st.write(df)
st.write(df["prdtypecode"].value_counts())

fig = plt.figure(figsize=(10, 4))
plt.xticks(rotation = 60)
sns.countplot(x="prdtypecode", data=df, order = df['prdtypecode'].value_counts().index)

st.pyplot(fig)

st.write("Length of the dataset", len(df))

df1 = pd.read_csv("products_codes.csv", index_col = 0)


def descr_length(el):
    if (pd.isna(el) == True):
        return 0
    else:
        return len(el)

df["len_prod"] = df["designation"].apply(lambda x: len(x))

df["len_descr"] = df["description"].apply(descr_length)

mean_ = lambda x: x.mean()

func_to_apply = {
    "len_prod": [mean_],
    "len_descr": [mean_]
}


fig = plt.figure(figsize=(10, 4))
plt.title("Character length distribution for Product")
sns.distplot(df.len_prod)
st.pyplot(fig)

fig = plt.figure(figsize=(10, 4))
plt.title("Character length distribution for Description")
sns.distplot(df.len_descr)
st.pyplot(fig);

st.write("image plotting")

df_images = pd.read_csv("images_w_bounding_box_streamlit.csv")

import cv2

fig = plt.figure(figsize= (10, 4))
plt.title("Image with bounding box")
img = cv2.imread(df_images["img"][0])
img_shape = img.shape
img = cv2.resize(img,(100,100))
plt.imshow(img[...,::-1])
x1 = df_images.x1[0]/img_shape[1]*100
x2 = df_images.x2[0]/img_shape[1]*100
y1 = df_images.y1[0]/img_shape[1]*100
y2 = df_images.y2[0]/img_shape[1]*100
plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],"r")
st.pyplot(fig);



#st.bar_chart(*, x="codes".index, y=df["codes"].value_counts(), width=0, height=0, use_container_width=True)





# In[19]:


st.subheader('Text dataset')


# In[20]:


if st.button("# Exploring the dataset"):
    img=Image.open('Product classes.png')
    st.image(img,width=400, caption="Distribution of product classes")


# In[21]:


#df = pd.DataFrame(
#   np.random.randn(50, 20),
#   columns=('col %d' % i for i in range(20)))

#st.dataframe(df)


# In[22]:


st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")

#df = pd.read_csv(DATA_URL, nrows = nrows)
#    lowercase = lambda x:str(x).lower()
#    df.rename(lowercase, axis='columns',inplace=True)
#    return df
#st.header("Now, Explore Yourself the Dataset")
#st.dataframe(df)
# Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading text dataset...')
    # Load 10,000 rows of data into the dataframe.
#df = load_data(84916)
    # Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading text dataset...Completed!')
#images=Image.open('images/meet.png')
#st.image(images,width=600)


# In[23]:


# Showing the original raw data
if st.checkbox("Show Raw Data", False):
    st.subheader('Raw data')
    st.write(df)
st.title('Quick  Explore')
st.sidebar.subheader(' Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")
if st.sidebar.checkbox('Basic info'):
    if st.sidebar.checkbox('Dataset Quick Look'):
        st.subheader('Dataset Quick Look:')
        st.write(df.head())
    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Show Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)
   
    if st.sidebar.checkbox('Statistical Description'):
        st.subheader('Statistical Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('Missing Values?'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())


# In[24]:


df.head()


# In[ ]:





# In[ ]:




