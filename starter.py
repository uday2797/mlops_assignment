import sys
import pandas as pd
import pickle

# --------- Step 1: Params from CLI ---------
year = int(sys.argv[1])  # Example: 2023
month = int(sys.argv[2]) # Example: 3

# --------- Step 2: Load Model and Vectorizer ---------

'''
# Load model and DictVectorizer from model2.bin
with open('model2.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
'''

model_path = '/app/model2.bin'
with open(model_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# --------- Step 3: Load Input Data ---------
url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
df = pd.read_parquet(url)

# --------- Step 4: Feature Engineering ---------
df = df[(df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() > 0]
df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
df = df[(df.duration >= 1) & (df.duration <= 60)]

categorical = ['PULocationID', 'DOLocationID']
df[categorical] = df[categorical].astype(str)
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)


# --------- Step 5: Predict ---------
y_pred = model.predict(X_val)

# --------- Step 6: Save Output ---------
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

output_file = f'predicted_{year:04d}_{month:02d}.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

# --------- Final Output ---------
print(f"✅ Q5 Mean prediction: {y_pred.mean():.2f}")
print(f"✅ Q1 Std deviation: {y_pred.std():.2f}")
print(f"✅ Output saved to {output_file}")