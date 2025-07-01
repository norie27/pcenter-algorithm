# src/algorithms/genetic_operators.py

import random
import numpy as np



def M1_global(Pj, n, p):
    """Version globale de M1 pour la parallélisation"""
    q = random.randint(p // 2, p)
    selected = random.sample(Pj, min(q, len(Pj)))
    available = [v for v in range(n) if v not in selected]
    if len(selected) < p and available:
        needed = p - len(selected)
        selected.extend(random.sample(available, min(needed, len(available))))
    return selected

def M2_global(Pi, distances):
    """Version globale de M2 pour la parallélisation"""
    if len(Pi) < 2:
        return Pi.copy()
    
    min_dist = float('inf')
    ri, rj = 0, 1
    
    for i in range(len(Pi)):
        for j in range(i + 1, len(Pi)):
            d = distances[Pi[i], Pi[j]]
            if d < min_dist:
                min_dist = d
                ri, rj = i, j
    
    return [c for k, c in enumerate(Pi) if k not in (ri, rj)]

def X1_global(Pi, Pj, n, p):
    """Version globale de X1 pour la parallélisation"""
    combined = list(set(Pi) | set(Pj))
    
    if len(combined) >= p:
        return random.sample(combined, p)
    else:
        available = [v for v in range(n) if v not in combined]
        needed = p - len(combined)
        if needed > 0 and available:
            additional = random.sample(available, min(needed, len(available)))
            combined.extend(additional)
        return combined

def X2_global(Pi, Pj, n, p, distances):
    """Version globale de X2 pour la parallélisation - VERSION CORRIGÉE"""
    u1, u2 = random.sample(range(n), 2)
    q = random.uniform(0.1, 0.9)
    
    S1, S2 = [], []
    
    # Traiter Pi séparément
    for f in Pi:
        d1 = distances[f, u1]
        d2 = distances[f, u2]
        
        if d2 <= 1e-12:
            S1.append(f)
            continue
            
        ratio = d1 / d2
        if ratio <= q:
            S1.append(f)
        else:
            S2.append(f)
    
    # Traiter Pj séparément
    for f in Pj:
        d1 = distances[f, u1]
        d2 = distances[f, u2]
        
        if d2 <= 1e-12:
            S2.append(f)
            continue
            
        ratio = d1 / d2
        if ratio > q:
            S1.append(f)
        else:
            S2.append(f)
    
    # Enlever les doublons potentiels
    S1 = list(set(S1))
    S2 = list(set(S2))
    
    if len(S1) > p:
        S1 = random.sample(S1, p)
    if len(S2) > p:
        S2 = random.sample(S2, p)
        
    return S1, S2