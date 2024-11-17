# Brainrot for hackhathon Heroes of The Brain 16-17.11.2024

## Application
Brainrot is a standalone cross-platform application written in Eel but required BrainAccess Halo for reading the brain activity. \
It can be shipped to any machine and does not requires manully running any servers on the client computer. \
<br>
Application has two modes that can operate with:
- *Boredom mode* - When you watch an entertainment video it will allow you to easlily speed up parts of the video where you loose interest in and you are getting bored.
- *Attention mode* - When you watch a tutorial video it will allow you to easily go back few seconds every time you get distracted and can loose some information from the video.

<br>

## Machine learning
We are using Power Spectral Density calculated for theta, beta, alpha waves and their ratio of theta alpha and alpha beta. \
Normalized data are being fetched into Support Verctor Machine, then the trained model is used to make predictions on near real time data stream.

## Running the project
Attention: running all python scripts and (not built) application for this project require `bacore.dll` shipped with the `BrainAccessSDK` package. You need to either run them from a directory where the ddl exists (the target file does not have to be there, just the relative path based on the directory from where you run it) or add the directory with the dll to the path environment variable.
<br>
<br>

### Running client application
```
python ./Frontend/main.py
```

or build the application witn

```
python -m eel ./Frontend/main.py web
```

### Running training / inference
```
python ./ML/inference.py
```
```
python ./ML/training.py
```


## Additional
In `training.py` script there is also `print_waves` function that you can use for ploting near real time data stream from the BrainAccess. It contains i.e. PSD for used waves.