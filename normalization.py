from sklearn.preprocessing import MinMaxScaler

def normalize_preprocessed_data(df, start_col, end_col):
    scaler = MinMaxScaler()
    df.iloc[:, start_col:end_col] = scaler.fit_transform(df.iloc[:, start_col:end_col].to_numpy())
    return df
