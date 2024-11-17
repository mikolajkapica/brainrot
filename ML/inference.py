import os
import numpy as np
import keyboard  # Using keyboard module to detect key press
from brainaccess import core
import time
from brainaccess.core.eeg_manager import EEGManager
from brainaccess.utils import acquisition
import pickle


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

    # split x channel
    theta_psd = get_PSD(raw_theta, theta_freq[0], theta_freq[1])
    alpha_psd = get_PSD(raw_beta, alpha_freq[0], alpha_freq[1])
    beta_psd = get_PSD(raw_alpha, beta_freq[0], beta_freq[1])

    # theta to alpha for each channel in each split
    theta_alpha_ratio = list(map(lambda x: x[0] / x[1], zip(theta_psd, alpha_psd)))
    alpha_beta_ratio = list(map(lambda x: x[0] / x[1], zip(alpha_psd, beta_psd)))

    X = theta_psd + alpha_psd + beta_psd + theta_alpha_ratio + alpha_beta_ratio

    # normalize
    scalers = []
    scaler_dir = "scalers"
    for i in range(20):
        with open(os.path.join(scaler_dir, f"scaler_{i}.pkl"), "rb") as f:
            scalers.append(pickle.load(f))

    normalized_X = np.array([scaler.transform(np.array(x).reshape(1, -1)) for scaler, x in zip(scalers, X)]).flatten()

    # Load the trained SVM model from a file
    model_filename = 'svm_model.pkl'
    with open(model_filename, 'rb') as file:
        svm_classifier = pickle.load(file)

    # Make predictions using the loaded SVM model
    prediction = svm_classifier.predict([normalized_X])
    return prediction


device_name = "BA HALO 032"
cap: dict = {
    0: "Fp1",
    1: "Fp2",
    2: "O1",
    3: "O2",
}

eeg = acquisition.EEG()

core.init()

# scan for devices find the defined one
core.scan(0)
count = core.get_device_count()
port = None

print("Found devices:", count)
for i in range(count):
    name = core.get_device_name(i)
    print(name)
    if device_name in name:
        port = i
        print('Found your device!')

if port is None:
    print('Device not found!')
    raise Exception('Device not found!')

# start EEG acquisition setup
with EEGManager() as mgr:
    print("Connecting to device:", core.get_device_name(port))
    
    eeg.setup(mgr, device_name=device_name, cap=cap)

    # Start acquiring data
    eeg.start_acquisition()
    time.sleep(3)

    annotation = 1

    while True:
        time.sleep(0.1)
        print(f"Sending annotation {annotation} to the device")

        raw = eeg.get_mne(tim=10)
        
        print(predict(raw))

        if keyboard.is_pressed('q'):
            print("Stopping acquisition...")
            break

    eeg.stop_acquisition()
    mgr.disconnect()
