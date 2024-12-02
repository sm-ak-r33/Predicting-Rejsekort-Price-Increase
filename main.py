import os

os.system("python pipeline/preprocessing.py")
os.system("python pipeline/autoarima.py")
os.system("python pipeline/selected_arima.py")
os.system("python pipeline/prophet_model.py")
os.system("python pipeline/BiLSTM.py")