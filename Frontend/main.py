import eel
import os
import mne
import signal
import sys
from brainaccess import core
import time
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from threading import Thread
import keyboard  # Using keyboard module to detect key press


from inference.inference import predict


eel.init(f'{os.path.dirname(os.path.realpath(__file__))}/web')


data = None 
run = True

last_prediction = None
last_predictions = []

@eel.expose
def say_hello_py(x):
    print('Hello from %s' % x)

@eel.expose
def get_data():
    # raw = mne.io.read_raw_fif('20241116_1400-raw.fif', preload=True)
    # sfreq = raw.info['sfreq']

    # start_sample = int(raw.n_times - 4 * sfreq)
    # end_sample = raw.n_times
    # average_activity = raw.get_data(["data"], start=start_sample, stop=end_sample).mean(axis=1)
    # print("Average activity per channel:", average_activity)
    # return average_activity.tolist()
    if not data:
        return []
    return data.get_data().tolist()

@eel.expose
def get_devices():
    core.init()
    core.scan(0)
    count = core.get_device_count()
    print("Found devices:", count)
    devices = []
    for i in range(count):
        name = core.get_device_name(i)
        print(name)
        devices.append(name)
        # if device_name in name:
        #     port = i
        #     print('Found your device!')
    return devices

@eel.expose
def choose_device(device_name):
    print("Choosing device:", device_name)
    handle_thread(device_name)

@eel.expose
def get_prediction():
    global last_prediction
    print(last_prediction)
    return last_prediction

def handle_thread(device_name):
    th = Thread(target = handle_inference, args=(device_name,))
    
    def handle_shutdown_signal(signum, frame):
        print("\nShutdown signal received, stopping Eel app gracefully...")
        global run
        run = False
        th.join()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    th.start()




def handle_inference(device_name):
    cap: dict = {
        0: "Fp1",
        1: "Fp2",
        2: "O1",
        3: "O2",
    }
    
    eeg = acquisition.EEG()
    with EEGManager() as mgr:
        eeg.setup(mgr, device_name=device_name, cap=cap)

        # Start acquiring data
        eeg.start_acquisition()
        time.sleep(3)

        annotation = 1

        while True:
            time.sleep(0.1)
            print(f"Sending annotation {annotation} to the device")

            raw = eeg.get_mne(tim=10)
            
            prediction = predict(raw)[0]
            global last_prediction
            global last_predictions
            # last_prediction = prediction
            print(prediction)
            last_predictions.append(prediction)
            if last_prediction is None:
                last_prediction = prediction
            elif last_predictions[-3:].count(prediction) == 3:
                last_prediction = prediction


            if keyboard.is_pressed('q'):  # If 'q' key is pressed
                print("Stopping acquisition...")
                break
        eeg.stop_acquisition()
        mgr.disconnect()

# def collect_data(device_name=None):
#     global data
#     global run
#     eeg = acquisition.EEG()

#     # define electrode locations
#     cap: dict = {
#         0: "Fp1",
#         1: "Fp2",
#         2: "O1",
#         3: "O2",
#     }

#     # define device name
#     # device_name = "BA HALO 032"

#     # start EEG acquisition setup
#     with EEGManager() as mgr:
#         eeg.setup(mgr, device_name=device_name, cap=cap)


#         # Start acquiring data
#         eeg.start_acquisition()
#         time.sleep(3)

#         start_time = time.time()
#         annotation = 1
#         while run:
#             time.sleep(1)
#             # send annotation to the device
#             print(f"Sending annotation {annotation} to the device")
#             eeg.annotate(str(annotation))
#             annotation += 1
#             if annotation % 10 == 0:
#                 data = eeg.get_mne(tim=10).filter(1, 40)

#         # get all eeg data and stop acquisition
#         eeg.stop_acquisition()
#         mgr.disconnect()

#     eeg.close()






eel.start('main.html', size=(1000, 640))