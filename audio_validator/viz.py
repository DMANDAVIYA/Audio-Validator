import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def save_stability(scores, out):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(scores)*2, len(scores)), scores, color='#2ecc71', linewidth=2)
    plt.axhline(y=0.7, color='#e74c3c', linestyle='--')
    plt.ylim(0, 1)
    plt.ylabel('Consistency Score')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_pca(embs, out):
    if len(embs) < 3: return
    p = PCA(n_components=2).fit_transform(embs)
    plt.figure(figsize=(6, 6))
    plt.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), cmap='viridis', alpha=0.6)
    plt.title('Voice Clusters')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_snr_dist(y, out):
    plt.figure(figsize=(6, 4))
    plt.hist(y, bins=50, color='#3498db', alpha=0.7)
    plt.title('Waveform Distribution')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
