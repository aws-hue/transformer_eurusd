import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.df_data = pd.read_csv(file_path)
        self.df_data.columns = map(str.lower, self.df_data.columns)
        self.df_data = self.df_data[['date', 'price', 'open', 'high', 'low']]
        self.df_data['date'] = pd.to_datetime(self.df_data['date'])
        self.df_data = self.df_data.sort_values('date').reset_index(drop=True)

    def get_data(self):
        return self.df_data
    
    