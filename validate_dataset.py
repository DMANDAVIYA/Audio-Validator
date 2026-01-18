import os
import json
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from audio_validator import validate
from audio_validator.viz import save_stability, save_pca, save_snr_dist

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def process_folder(src, out_prefix, exts):
    os.makedirs(f'plots/{out_prefix}', exist_ok=True)
    res = []
    files = [f for f in os.listdir(src) if any(f.endswith(e) for e in exts)]
    
    print(f"Processing {src}...")
    for f in tqdm(files):
        path = os.path.join(src, f)
        r = validate(path)
        
        save_stability(r['scores'], f"plots/{out_prefix}/{f}_stability.png")
        if len(r['embs']) > 2: save_pca(r['embs'], f"plots/{out_prefix}/{f}_pca.png")
        
        y, _ = librosa.load(path, sr=16000, duration=10)
        save_snr_dist(y, f"plots/{out_prefix}/{f}_dist.png")
        
        del r['embs'], r['scores']
        r['filename'] = f
        res.append(r)
    
    pd.DataFrame(res).to_csv(f'{out_prefix}_report.csv', index=False)
    with open(f'{out_prefix}_report.json', 'w') as f: json.dump(res, f, indent=2, cls=NpEncoder)

def run():
    process_folder('data_final/data_final/failed', 'failed', ['.mp3'])
    process_folder('data_final/data_final/mbti_data', 'mbti', ['.wav'])

if __name__ == "__main__":
    run()
