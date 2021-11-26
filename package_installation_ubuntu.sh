# install CUDA 11.3 (Stable)
sudo -H pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# install bacis python3 packages
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip
sudo -H pip3 install numpy --upgrade
sudo -H pip3 install ipython
sudo -H pip3 install matplotlib
sudo -H pip3 install pandas
sudo -H pip3 install h5py
sudo -H pip3 install leveldb
sudo -H pip3 install seaborn
sudo -H pip3 install tensorflow-gpu
sudo -H pip3 install Theano
sudo -H pip3 install keras
sudo -H pip3 install -U scikit-learn
sudo -H pip3 install cython
sudo -H pip3 install opencv-python
sudo -H pip3 install lmdb
sudo -H pip3 install sympy
sudo -H pip3 install pydotplus
sudo -H pip3 install gpustat
sudo -H pip3 install xlrd
sudo -H pip3 install sacred
sudo -H pip3 install pymongo
sudo -H pip3 install openpyxl
sudo -H pip3 install tqdm
sudo -H pip3 install nltk
sudo -H pip3 install pyspellchecker
sudo -H pip3 install -U spacy
sudo -H pip3 install textacy
sudo -H pip3 install joblib
sudo -H pip3 install dataclasses
sudo -H pip3 install typing
sudo python3 -m spacy download en

# install Plotly
sudo pip3 install plotly

# install Huggingface packages
sudo pip3 install transformers
sudo pip3 install datasets

# install image manipulations
sudo apt install ffmpeg -y
sudo -H pip3 install pydub

# apt-get
sudo apt-get install -y p7zip-full
sudo apt install unzip
sudo apt-get install gedit -y

