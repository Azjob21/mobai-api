import pandas as pd
import requests, json

segs = pd.read_csv('data/product_segments.csv')
hf = segs[segs['segment']=='HF']['id_produit'].head(10).tolist()
print('HF products:', hf)

r = requests.post('http://localhost:8000/predict', json={'product_ids': hf, 'date': '2026-02-14'})
for item in r.json():
    print(f"  Product {item['product_id']}: demand={item['predicted_demand']}, prob={item['probability']}")
