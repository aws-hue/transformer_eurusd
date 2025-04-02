import data_loader as DataLoader
import model_trainer as ModelTrainer
import pandas as pd

if __name__ == "main":
    
    file_path = 'EURUSD.csv'
    sequence_length = 90
    epochs = 2

    data_loader = DataLoader(file_path)
    df_data = data_loader.get_data()

    trainer = ModelTrainer(df_data, sequence_length, epochs)
    n_trials = 2
    
    best_params = trainer.optimize(n_trials)
    print("Best Parameters:", best_params)