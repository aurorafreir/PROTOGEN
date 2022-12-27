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

![img.png](assets/img.png "An image of running the face_track.py script, and the printed result of the facial tracking pose estimation data")

# Installing and Running
To get this whole project, run these terminal commands
```
pip install opencv-python mediapipe Pillow serial numpy 
git clone https://github.com/CatAndDogSoup/PROTOGEN.git
cd PROTOGEN
python face_track.py
```
