from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class TransformerDataPreparer:
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        
    def prepare_data(self, df):
        features = [col for col in df.columns if col != 'date']
       
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])

        X, y, dates, prices = [], [], [], []
        for i in range(self.sequence_length, len(scaled_data) - 1):
            X.append(scaled_data[i-self.sequence_length:i])
            target = 1 if df['price'].iloc[i + 1] > df['price'].iloc[i] else 0
            y.append(target)
            dates.append(df['date'].iloc[i + 1])
            prices.append(df['price'].iloc[i + 1])

        X, y = np.array(X), np.array(y)
        dates = pd.Series(dates).reset_index(drop=True)
        prices = pd.Series(prices).reset_index(drop=True)
        return X, y, scaler, dates, prices