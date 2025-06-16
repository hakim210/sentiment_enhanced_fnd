from . import config

def calculate_emotion_weight(polarity: float, subjectivity: float, beta_override: float = None):

    alpha = config.ALPHA
    beta = beta_override if beta_override is not None else config.BETA
    
    if polarity < 0:
        weighted_polarity = beta * abs(polarity)
    else:
        weighted_polarity = abs(polarity)
        
    emotion_weight = 1 + alpha * subjectivity * weighted_polarity
    
    return emotion_weight

