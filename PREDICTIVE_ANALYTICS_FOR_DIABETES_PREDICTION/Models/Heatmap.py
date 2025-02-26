import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df= pd.read_csv('diabetes_data_upload.csv')

df["Gender"].replace({"Female":1,"Male":0},inplace=True)
df['Polyuria'].replace({"Yes":1,"No":0},inplace=True)
df['Polydipsia'].replace({"Yes":1,"No":0},inplace=True)
df['sudden weight loss'].replace({"Yes":1,"No":0},inplace=True)
df['weakness'] .replace({"Yes":1,"No":0},inplace=True)
df['Polyphagia'].replace({"Yes":1,"No":0},inplace=True)
df['Genital thrush'].replace({"Yes":1,"No":0},inplace=True)
df['visual blurring'].replace({"Yes":1,"No":0},inplace=True)
df['Itching'].replace({"Yes":1,"No":0},inplace=True)
df['Irritability'].replace({"Yes":1,"No":0},inplace=True)
df['delayed healing'].replace({"Yes":1,"No":0},inplace=True)
df['partial paresis'].replace({"Yes":1,"No":0},inplace=True)
df['muscle stiffness'].replace({"Yes":1,"No":0},inplace=True)
df['Alopecia'].replace({"Yes":1,"No":0},inplace=True)
df['Obesity'].replace({"Yes":1,"No":0},inplace=True)
df['class'].replace({"Positive":1,"Negative":0},inplace=True)



#print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2g', linewidths=0.5,)
plt.show()

