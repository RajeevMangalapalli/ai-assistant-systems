import pandas as pd


#Printing the dataframe head
df = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\music_info.csv")
print(df.head())

#Dataframe information
print(df.info(), end = "\n\n")

#Checking for missing values
print(df.isnull().sum(), end = "\n\n")

#Cleaning the dataframe by removing the columns that can't be used for analysis
df_cleaned = df.drop(["spotify_id","tags","genre"], axis = "columns") #Drop multiple columns by passing a list of column names
print(df_cleaned.info(), end = "\n\n")
df_cleaned.to_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\cleaned_music_info.csv", index=False)
print(df_cleaned.isnull().sum(), end = "\n\n")



