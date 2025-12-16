import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def process_data(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'File tidak ditemukan pada lokasi {input_path}')
    
    df = pd.read_csv(input_path)
    
    # Menangani nilai kosong dan Membersihkan data duplikat
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Feature in dataset
    numeric_features = df.select_dtypes(include='number').columns
    categorical_features = df.select_dtypes(include='object').columns
        
    # Mengatasi Outlier
    Q1 = df[numeric_features].quantile(0.25)
    Q3 = df[numeric_features].quantile(0.75)
    IQR = Q3 - Q1
    
    condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_filtered_numeric = df.loc[condition, numeric_features] 

    df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)
    
    # Normalization
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])   
    
    # Label encoding
    le = LabelEncoder()
    
    for col in categorical_features:
        df[col] = le.fit_transform(df[col])
    
    # Rename to target
    df = df.rename(columns={'Weather Type': 'target'})
    
    
    # Simpan Data
    df.to_csv(output_path, index=False)
    

def main():
    INPUT_FILE = os.path.join('../weather_dataset.csv')
    OUTPUT_FILE = os.path.join('weather_dataset_preprocessing.csv')
    
    try:
        process_data(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    main()