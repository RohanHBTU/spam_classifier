import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv("spam.csv",encoding="ISO-8859-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
profile=ProfileReport(df)
profile.to_file(output_file='Report.html')