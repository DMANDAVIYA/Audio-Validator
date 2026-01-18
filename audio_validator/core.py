from .snr import check_quality
from .speaker import SpeakerCheck

checker = SpeakerCheck()

def validate(path):
    snr, flatness = check_quality(path)
    sim, n_wins, embs, scores = checker.check(path)
    
    reasons = []
    warnings = []
    
    if n_wins == 0: 
        reasons.append("Silence")
    elif sim < 0.25: 
        reasons.append(f"Consistency {sim:.2f}")
    elif flatness > 0.5: 
        reasons.append(f"Flatness {flatness:.2f}")
    
    if snr < 15:
        warnings.append(f"Low SNR {snr:.1f}dB")
    
    return {'valid': not reasons, 'reasons': reasons, 'warnings': warnings, 'metrics': {'snr': snr, 'consistency': sim, 'flatness': flatness}, 'embs': embs, 'scores': scores}
