# how to run group 5 code in Rasberry Pi

# if runing with group 5's resperry pi board, go to line 7. If not, go to line 12.

# Because we download packages in virtual environment, you need to activate it first.

cd Lab1
source ./Lab1/bin/activate
export OMP_NUM_THREADS=1

# There is no need for reinstalling packages if having activate the virtual environment.
# the requrirements are list in requirements.txt, install the package before executing codes.

# Run YOLOv4+CNN
cd ./Lab1_code/YOLOv4_tiny/
# Run code with example video
python detectvideo.py --video ./data/street.mp4
# Run code with webcam
python detectvideo.py