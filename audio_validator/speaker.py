import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerCheck:
    def __init__(self):
        self.enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    
    def check(self, path):
        sig, fs = torchaudio.load(path)
        if fs != 16000: sig = torchaudio.transforms.Resample(fs, 16000)(sig)
        sig = sig.mean(0) if sig.shape[0] > 1 else sig.squeeze()
        
        step, win = 24000, 24000
        if sig.shape[0] < win: return 1.0, 1, np.zeros((1, 192)), [1.0]
        
        chunks = [sig[i:i+win] for i in range(0, sig.shape[0] - win + 1, step)]
        if not chunks: return 1.0, 1, np.zeros((1, 192)), [1.0]
        
        embs = []
        with torch.no_grad():
            for i in range(0, len(chunks), 32):
                embs.append(self.enc.encode_batch(torch.stack(chunks[i:i+32])).squeeze(1).cpu().numpy())
        
        e = np.vstack(embs)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        
        sim_matrix = np.dot(e, e.T)
        triu_indices = np.triu_indices(len(e), k=1)
        sim = np.mean(sim_matrix[triu_indices]) if len(triu_indices[0]) > 0 else 1.0
        
        scores = [np.mean(sim_matrix[i, max(0, i-5):min(len(e), i+5)]) for i in range(len(e))]
        return float(sim), len(e), e, scores
