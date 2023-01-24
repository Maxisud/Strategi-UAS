import getopt, sys
import mariadb
import pandas as pd

mydb = mariadb.connect(
  host="localhost",
  user="root",
  passwd="",
  database="knn"
)

file_path = sys.argv[1]
df = pd.read_csv(file_path)
df.head()
