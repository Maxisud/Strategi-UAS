import mariadb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

mydb = mariadb.connect(
  host="localhost",
  user="root",
  passwd="",
  database="knn"
)

#mengambil data dari database
text = []
sentiment = []
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM trkalimat")
myresult = mycursor.fetchall() 

for kalimat in myresult:
    text.append(kalimat[2])
    sentiment.append(kalimat[3])

dict = {'text': text, 'sentiment':sentiment}
df = pd.DataFrame(dict)


#mengubah teks string menjadi bentuk binary
tf = TfidfVectorizer()
text_counts = tf.fit_transform(df['text'])

#split data training dan testing
x_train, x_test, y_train, y_test= train_test_split(text_counts, df['sentiment'], test_size=0.25, random_state=5)

#modeling atau training data
knn = KNeighborsClassifier(n_neighbors=5) # you can change the number of neighbours
model = knn.fit(x_train.toarray(), y_train)
predicted = knn.predict(x_test.toarray())

#hasil training
akurasi = metrics.accuracy_score(predicted, y_test)
print(str(akurasi))

Recall = metrics.recall_score(predicted, y_test)
print(str(Recall))

Presicion = metrics.precision_score(predicted, y_test)
print(str(Presicion))

f_measure = metrics.f1_score(y_test, predicted)
print(str(f_measure))

#menyimpan model
filename = 'model.ict'
joblib.dump(model, filename)
joblib.dump(tf, "cv.ict") 