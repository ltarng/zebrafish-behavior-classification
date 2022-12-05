from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


folder_path = "D:/Google Cloud (60747050S)/Research/FMTResult(csv)/mirror_biting/"
file_name = "mirror_S4_right.csv"
file_path = folder_path + file_name
df = pd.read_csv(file_path)

occur_times = []  # Declare a null array for counting locations occurence times
df = df.reset_index()  # make sure indexes pair with number of rows

for index, row in df.iterrows():
	occur_times[row['Fish0_x']][row['Fish0_y']] += 1
	# print(row['Fish0_x'], row['Fish0_y'])

# Normalize coordinate x and y
# scaler = MinMaxScaler()
# df[['Fish0_x','Fish0_y']] = scaler.fit_transform(df[['Fish0_x','Fish0_y']])

