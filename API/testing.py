import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
import numpy as np
import sys

#memuat model
model = joblib.load("model.ict")
cv = joblib.load('cv.ict')
text = " "
for arg in sys.argv:
    text = text + " " + arg

temp = cv.transform([text])
temp = temp.toarray()
res = model.predict(temp)
if(res=="positive"):
    hasil = "Kalimat anda adalah positif"
else:
    hasil = "Kalimat anda adalah negatif"

print(str(hasil))