import os
import numpy as np
import pickle
from brainaccess.core.eeg_manager import EEGManager


def get_PSD(raw, fmin, fmax):
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, picks='data')
    psd_data = [ch.mean() for ch in psd.get_data()]
    return psd_data

def predict(raw):
    theta_freq = (4, 8)
    alpha_freq = (8, 12)
    beta_freq = (12, 30)

    raw_theta = raw.copy().filter(l_freq=theta_freq[0], h_freq=theta_freq[1], picks='data')
    raw_beta = raw.copy().filter(l_freq=beta_freq[0], h_freq=beta_freq[1], picks='data')
    raw_alpha = raw.copy().filter(l_freq=alpha_freq[0], h_freq=alpha_freq[1], picks='data')


    theta_psd = get_PSD(raw_theta, theta_freq[0], theta_freq[1])
    alpha_psd = get_PSD(raw_beta, alpha_freq[0], alpha_freq[1])
    beta_psd = get_PSD(raw_alpha, beta_freq[0], beta_freq[1])

    theta_alpha_ratio = list(map(lambda x: x[0] / x[1], zip(theta_psd, alpha_psd)))
    alpha_beta_ratio = list(map(lambda x: x[0] / x[1], zip(alpha_psd, beta_psd)))


    X = theta_psd + alpha_psd + beta_psd + theta_alpha_ratio + alpha_beta_ratio

    scalers = []
    scaler_dir = "scalers"
    for i in range(20):
        with open(os.path.join(scaler_dir, f"scaler_{i}.pkl"), "rb") as f:
            scalers.append(pickle.load(f))

    normalized_X = np.array([scaler.transform(np.array(x).reshape(1, -1)) for scaler, x in zip(scalers, X)]).flatten()


    model_filename = 'svm_model.pkl'
    with open(model_filename, 'rb') as file:
        svm_classifier = pickle.load(file)

    prediction = svm_classifier.predict([normalized_X])
    return prediction
