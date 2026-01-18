import numpy as np
import librosa

def estimate_snr(path):
    y, sr = librosa.load(path, sr=16000)
    if len(y) == 0: return 0.0
    
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    energy = np.array([
        np.sum(y[i:i+frame_length]**2) 
        for i in range(0, len(y)-frame_length, hop_length)
    ])
    
    threshold = np.percentile(energy, 30)
    
    signal_frames = energy > threshold
    noise_frames = energy <= threshold
    
    if not np.any(signal_frames) or not np.any(noise_frames):
        return 0.0
    
    signal_rms = np.sqrt(np.mean(energy[signal_frames]))
    noise_rms = np.sqrt(np.mean(energy[noise_frames]))
    
    if noise_rms == 0:
        return 100.0
    
    snr_db = 20 * np.log10(signal_rms / noise_rms)
    return float(np.clip(snr_db, 0, 100))

def check_quality(path):
    return estimate_snr(path), np.mean(librosa.feature.spectral_flatness(y=librosa.load(path, sr=16000)[0]))
