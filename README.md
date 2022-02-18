# Real Time Face Mask Detection
This project involves making a mask detection model using python libraries using OpenCV, Tensorflow and Keras.
This project is made in 2 phases
1. The Train Model phase : This phsae involves making of the Training Model which is trained to detect faces in the Frame. This model trained with the dataset to get a good accuaracy.
2. The Face Mask Detect phase : The model obtained from training phsae is embedded in this script which take the detected Face, the presence of face mask and the percent of face coved with mask as input. This will classify the output as wearing mask or not. 

### To get started:
1. Clone this repository by clicking [here](https://github.com/vedantpople4/Mask_detection.git)
2. Navigate to the correct directory.
3. Install the required packages using <code>pip install -r requirements.txt</code>
4. To Run detection Model <code>python run detect_mask_video.py</code>
5. To End the stream, use <code> Ctrl+c or q</code>
This Face Mask Detection Model works on Real Time Video streams. 
