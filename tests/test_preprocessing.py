
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.preprocessing import load_and_preprocess, preprocess_data

# Test 1 : Vérifier que la fonction retourne X et y avec les bonnes dimensions
def test_load_and_preprocess_shape(tmp_path):
    # Créer des fichiers temporaires simulant PS2, FS1 et profile
    ps2_file = tmp_path / "PS2.txt"
    fs1_file = tmp_path / "FS1.txt"
    profile_file = tmp_path / "profile.txt"

    # Simuler des données
    ps2_data = "\n".join(["\t".join(map(str, np.random.rand(5))) for _ in range(3)])
    fs1_data = "\n".join(["\t".join(map(str, np.random.rand(3))) for _ in range(3)])
    profile_data = "\n".join(["100\t100\t0\t10\t1" for _ in range(3)])

    # Écrire dans les fichiers
    ps2_file.write_text(ps2_data)
    fs1_file.write_text(fs1_data)
    profile_file.write_text(profile_data)

    # Appeler la fonction
    X, y = load_and_preprocess(ps2_file, fs1_file, profile_file)

    # Vérifications
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 3  # 3 lignes
    assert y.shape[0] == 3
    assert all(col in X.columns for col in ["ps2_mean", "fs1_mean"])  # Colonnes calculées

# Test 2 : Vérifier la standardisation
def test_preprocess_data():
    X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    X_scaled = preprocess_data(X)
    assert X_scaled.shape == X.shape
    # Moyenne proche de 0 et écart-type proche de 1
    assert np.isclose(X_scaled.mean(), 0, atol=1e-7)
    assert np.isclose(X_scaled.std(), 1, atol=1e-7)