# PROTOGEN

🏳️‍🌈 🏳️‍⚧️ Trans Rights are Human Rights 🏳️‍⚧️ 🏳️‍🌈

//
[Introduction](#what-is-this-repo) //
[Installing](#installing-and-running) //

# What is this repo?
This is a repo for me to test out a collection of markerless facial tracking stuff, as well 
as controlling LED panels like a Raspi Pico Unicorn HAT

The end goal is to create my own Protogen mask for 3d printing, and use a Raspi (or Raspi-like alternative) to run 
    facial tracking real-time inside the mask, which will then affect the LED panels on the outside to show facial 
    poses on the mask.

3D Model blockout: \
`cabling: (pink: USBC from power bank (not in the head, thats a fire risk I dont want to have right next to my face), Red for an IR cam over usb to the raspi, Blue is a usb cable from the raspi to the raspi pico, and Yellow is GPIO from the pico controlling the led panels)`
![](readme_assets/3D_model.png)
Terminal Output:
![](readme_assets/terminal_output.png "An image of running the face_track.py script, and the printed result of the facial tracking pose estimation data")

# Installing and Running
To get this whole project, run these terminal commands
```
pip install opencv-python mediapipe Pillow serial numpy 
git clone https://github.com/CatAndDogSoup/PROTOGEN.git
cd PROTOGEN
python face_track.py
```

# Installing RPI4
```
sudo apt-get install cmake python3-dev python3-setuptoolslibtiff5-dev libjpeg-dev libopenjp2-7-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk libharfbuzz-dev libfribidi-dev libxcb1-dev

curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile

exec "$SHELL"

pyenv install 3.8
pyenv global 3.8

python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools wheel
python3.8 -m pip install opencv-python
python3.8 -m pip install mediapipe-rpi4
```
