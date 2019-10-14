Source : https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd

# Install requirements

	apt update
	apt install python3-pip cmake libopenblas-dev liblapack-dev libjpeg-dev

	git clone https://github.com/JetsonHacksNano/installSwapfile
	./installSwapfile/installSwapfile.sh
	reboot


	sudo pip3 install numpy

Now we are ready to install dlib, a deep learning library created by Davis King that does the heavy lifting for the face_recognition library.

However, there is currently a bug in Nvidia’s own CUDA libraries for the Jetson Nano that keeps it from working correctly. To work around the bug, we’ll have to download dlib, edit a line of code, and re-compile it. But don’t worry, it’s no big deal.

	wget http://dlib.net/files/dlib-19.17.tar.bz2
	tar jxvf dlib-19.17.tar.bz2
	cd dlib-19.17

	gedit dlib/cuda/cudnn_dlibapi.cpp

This will open up the file that we need to edit in a text editor. Search the file for the following line of code (which should be line 854):

	forward_algo = forward_best_algo;

And comment it out by adding two slashes in front of it, so it looks like this:

	//forward_algo = forward_best_algo;

	sudo python3 setup.py install


	sudo pip3 install face_recognition

