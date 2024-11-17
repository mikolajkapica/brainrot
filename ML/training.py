import matplotlib.pyplot as plt

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

import mne
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


eeg = acquisition.EEG()
fig, axs = plt.subplots(4, 1, figsize=(12, 16))

mean_theta = []
mean_alpha = []
mean_beta = []
mean_theta_alpha = []


# Dashboard for real time data visualization
# Not used for training but leaving it here for reference
def print_waves(raw):
    # Definicje zakresów częstotliwości dla fal
    theta_freq = (4, 8)
    alpha_freq = (8, 12)
    beta_freq = (12, 30)


    sfreq = raw.info['sfreq']
    start_sample = int(raw.n_times - 10 * sfreq)
    end_sample = raw.n_times

    # Zastosowanie filtrów pasmowych
    raw_theta = raw.copy().filter(l_freq=theta_freq[0], h_freq=theta_freq[1], picks='data')
    raw_alpha = raw.copy().filter(l_freq=alpha_freq[0], h_freq=alpha_freq[1], picks='data')
    raw_beta = raw.copy().filter(l_freq=beta_freq[0], h_freq=beta_freq[1], picks='data')

    # Tworzenie subplota
    for ax in axs:
        ax.clear()

    # Wyświetlanie fal theta
    # for ch in raw.ch_names[:4]:
    #     axs[0].plot(raw_theta.times[-window_size:], raw_theta.get_data(picks=ch, start=start_sample, stop=end_sample).T, label=ch)
    psd_data_theta, frequencies_theta = get_PSD(raw_theta, start_sample/sfreq, end_sample/sfreq, theta_freq[0], theta_freq[1])
    # for i in range(psd_data_theta.shape[0]):  # Ensure we stay within the bounds
    axs[0].plot(frequencies_theta, psd_data_theta)
    axs[0].set_title('Fale Theta (4-8 Hz)')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    # Wyświetlanie fal alfa
    psd_data_alpha, frequencies_alpha = get_PSD(raw_alpha, start_sample/sfreq, end_sample/sfreq, alpha_freq[0], alpha_freq[1])
    # for i in range(psd_data_alpha.shape[0]):  # Ensure we stay within the bounds
    axs[1].plot(frequencies_alpha, psd_data_alpha)
    # for ch in raw.ch_names[:4]:
    #     axs[1].plot(raw_alpha.times[-window_size:], raw_alpha.get_data(picks=ch, start=start_sample, stop=end_sample).T, label=ch)
    axs[1].set_title('Fale Alfa (8-12 Hz)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    # Wyświetlanie fal beta
    psd_data_beta, frequencies_beta = get_PSD(raw_beta, start_sample/sfreq, end_sample/sfreq, beta_freq[0], beta_freq[1])
    # for i in range(psd_data_beta.shape[0]):  # Ensure we stay within the bounds
    axs[2].plot(frequencies_beta, psd_data_beta)
    # for ch in raw.ch_names[:4]:
    #     axs[2].plot(raw_beta.times[-window_size:], raw_beta.get_data(picks=ch, start=start_sample, stop=end_sample).T, label=ch)
    axs[2].set_title('Fale Beta (12-30 Hz)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)
    axs[2].legend()


    global mean_theta
    global mean_alpha
    global mean_beta

    # Calculate and print the mean of all signals
    mean_theta.append(psd_data_theta.mean())
    mean_alpha.append(psd_data_alpha.mean())
    mean_beta.append(psd_data_beta.mean())
    mean_theta_alpha.append(psd_data_theta.mean() / psd_data_alpha.mean())

    # Plot all waves on the same plot
    axs[3].plot(range(len(mean_theta[-20:])), mean_theta[-20:], label='Theta (4-8 Hz)')
    axs[3].plot(range(len(mean_alpha[-20:])), mean_alpha[-20:], label='Alpha (8-12 Hz)')
    axs[3].plot(range(len(mean_beta[-20:])), mean_beta[-20:], label='Beta (12-30 Hz)')
    axs[3].plot(range(len(mean_theta_alpha[-20:])), mean_theta_alpha[-20:], label='Theta Alpha ratio')
    axs[3].set_title('All Waves')
    axs[3].legend()
    axs[3].grid(True)

    plt.draw()
    plt.pause(0.01)



def get_PSD(raw, fmin, fmax):
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, picks='data')
    frequencies = psd.freqs
    psd_data = [ch.mean() for ch in psd.get_data()]
    return psd_data


raws = [(data, mne.io.read_raw_fif(f"pythonAPI/brainaccess/data/{data}"), "nudny" if data.startswith("nudny") else "ciekawy") for data in os.listdir("pythonAPI/brainaccess/data") if data.endswith(".fif")]

train_data = []

grouped_theta = []
grouped_alpha = []
grouped_beta = []
grouped_theta_alpha = []
grouped_alpha_beta = []

for raw_number, (raw_name, raw, label) in enumerate(raws):
    raw.load_data()
    theta_freq = (4, 8)
    alpha_freq = (8, 12)
    beta_freq = (12, 30)

    sfreq = raw.info['sfreq']
    start_sample = int(raw.n_times - 10 * sfreq)
    end_sample = raw.n_times

    window_size = int(10 * sfreq)

    raw_theta = raw.copy().filter(l_freq=theta_freq[0], h_freq=theta_freq[1], picks='data')
    raw_beta = raw.copy().filter(l_freq=beta_freq[0], h_freq=beta_freq[1], picks='data')
    raw_alpha = raw.copy().filter(l_freq=alpha_freq[0], h_freq=alpha_freq[1], picks='data')


    def random_split(raw, seconds):
        start = np.random.randint(0, raw.n_times / raw.info['sfreq'] - seconds)
        cropped = raw.copy().crop(tmin=start, tmax=start + seconds)
        return cropped 

    seconds = 10
    n_splits = int((int(raw.n_times // (raw.info['sfreq'])) // seconds) * 1.5)
    print("splits: ", n_splits)

    raw_theta_splits = [random_split(raw_theta, seconds) for _ in range(n_splits)]
    raw_alpha_splits = [random_split(raw_alpha, seconds) for _ in range(n_splits)]
    raw_beta_splits = [random_split(raw_beta, seconds) for _ in range(n_splits)]

    # split x channel
    theta_psd_splits = [(get_PSD(split, theta_freq[0], theta_freq[1]), label) for split in raw_theta_splits]
    alpha_psd_splits = [(get_PSD(split, alpha_freq[0], alpha_freq[1]), label) for split in raw_alpha_splits]
    beta_psd_splits = [(get_PSD(split, beta_freq[0], beta_freq[1]), label) for split in raw_beta_splits]

    # theta to alpha for each channel in each split
    theta_alpha_ratio_splits = [(list(map(lambda x: x[0] / x[1], zip(theta_split[0], alpha_split[0]))), label) for theta_split, alpha_split in zip(theta_psd_splits, alpha_psd_splits)]
    alpha_beta_ratio_splits = [(list(map(lambda x: x[0] / x[1], zip(alpha_split[0], beta_split[0]))), label) for alpha_split, beta_split in zip(alpha_psd_splits, beta_psd_splits)]

    for split in range(n_splits):
        grouped_theta.append(theta_psd_splits[split])
        grouped_alpha.append(alpha_psd_splits[split])
        grouped_beta.append(beta_psd_splits[split])
        grouped_theta_alpha.append(theta_alpha_ratio_splits[split])
        grouped_alpha_beta.append(alpha_beta_ratio_splits[split])


X = []
y = []

for group in zip(grouped_theta, grouped_alpha, grouped_beta, grouped_theta_alpha, grouped_alpha_beta):
    grouped_features = []
    for g in group:
        grouped_features = grouped_features + (g[0])

    X.append(grouped_features)
    y.append(group[0][1])

# normalize
scalers = [StandardScaler() for _ in range(20)]

normalized_X = np.array([scaler.fit_transform(feature.reshape(-1, 1)) for scaler, feature in zip(scalers, np.array(X).T)]).T[0]

print(X[0])
print(normalized_X[0])
print(np.array(normalized_X).shape)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1}")

# Save the trained SVM model to a file
model_filename = 'svm_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(svm_classifier, file)

print(f"SVM model saved to {model_filename}")

# Save the scalers to a directory
scaler_dir = "scalers"
os.makedirs(scaler_dir, exist_ok=True)

for i, scaler in enumerate(scalers):
    with open(os.path.join(scaler_dir, f"scaler_{i}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

print(f"Scalers saved to {scaler_dir}")
