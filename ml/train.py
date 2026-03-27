import glob #for file accessing
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    datasetDir = "../Datasets"
    modelOutputPath = "model.pkl"
    maxRecords = 2000
    
    print(f"[info] looking for CSV files in {datasetDir}")
    csvFiles = sorted(glob.glob(os.path.join(datasetDir, "*.csv")))
    
    if not csvFiles:
        print("[Error] No csv files found. check datasets path ")
        return

    print(f"[info] Found {len(csvFiles)} CSV files")
    
    dataframes = []
    totalFiles = len(csvFiles)
    
    for i , filePath in enumerate(csvFiles , start=1):
        print(f"[info] loading file {i}/{totalFiles}: {os.path.basename(filePath)}")
        df= pd.read_csv(filePath)
        dataframes.append(df)
    
    print("[info] Merging all files ....")
    data = pd.concat(dataframes,ignore_index=True)
    print(f"[info] combined shape before cleaning : {data.shape}")
    
    print("[info] cleaning data....")
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan , inplace=True)
    data.dropna(inplace=True)
    data = data.head(maxRecords)
    print(f"[info] Shape after removing missing values : {data.shape}")
    
    if data.shape[1] < 2 :
        print("[error] dataset must have at least one feature column and one label column. ")
        return
    
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    print("[info] splitting train/test sets.....")
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2,random_state=42,stratify=y)
    
    print("[INFO] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train,Y_train)
    
    print("[INFO] Evaluating model...")
    yPred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, yPred)
    print(f"[RESULT] Model Accuracy: {accuracy:.4f}")
    
    print(f"[INFO] Saving model to: {modelOutputPath}")
    joblib.dump(model,modelOutputPath)
    print("[DONE] Model saved successfully as model.pkl")


if __name__ == "__main__":
    main()