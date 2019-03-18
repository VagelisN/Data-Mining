from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stopwords = ["said","say","says"]

stopwords=ENGLISH_STOP_WORDS.union(stopwords)

train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")
test_data = pd.read_csv('./datasets/test_set.csv', sep="\t")

#locate the texts that belong to the Politics Category
dataPol = train_data.loc[train_data["Category"] == "Politics"]
dataTec = train_data.loc[train_data["Category"] == "Technology"]
dataBus = train_data.loc[train_data["Category"] == "Business"]
dataFoo = train_data.loc[train_data["Category"] == "Football"]
dataFil = train_data.loc[train_data["Category"] == "Film"]

#get a string of all the content of each category
textPol = ''.join(dataPol["Content"])
textTec = ''.join(dataTec["Content"])
textBus = ''.join(dataBus["Content"])
textFoo = ''.join(dataFoo["Content"])
textFil = ''.join(dataFil["Content"])

#get the wordcloud without the stopwords for each category
wordcloud = WordCloud(stopwords= stopwords).generate(textPol)
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
fig.savefig("./WC_Politics.png")
plt.close()

wordcloud = WordCloud(stopwords= stopwords).generate(textTec)
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
fig.savefig("./WC_Technology.png")
plt.close()

wordcloud = WordCloud(stopwords= stopwords).generate(textBus)
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
fig.savefig("./WC_Business.png")
plt.close()

wordcloud = WordCloud(stopwords= stopwords).generate(textFoo)
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
fig.savefig("./WC_Football.png")
plt.close()

wordcloud = WordCloud(stopwords= stopwords).generate(textFil)
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
fig.savefig("./WC_Film.png")
plt.close()