sudo apt-get install swig build-essential python-dev python3-dev
pip3 install box2d-py
echo "Installing gym and pytorch"
pip3 install gym[all]
pip3 install torch torchvision torchaudio
echo "Installing other dependencies"
pip3 install numpy matplotlib
