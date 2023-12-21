# ADWM (Algorithms of digital multimedia processing)
## Content:
 - Description:
   - Introduction cv
   - tracking red object
   - Image blurring (GaussBlur)
   - Canny algorithm
   - Motion detection on video
   - neural network training
   - Haar's work (workKhaara)
   - Text Recognition(opticalRecognationText)
   - Contours (several different operators)
   - Hand detectors (Detectros) **-(it's maybe interesting for HR)**
   - Tracking (mil,csrt,kcf) and histogram-based color tracking **-(it's maybe interesting for HR)**

### Introduction cv
This folder include introduction with openCV. 
Done work: show img and test 3 difeerent flags for image and screen, 
show video(movie,webcam), record movie, display the Red Cross in the center of the screen and inside cross accept blur
Fill the cross with one of the 3 colors - red, green, blue using the following rule: BASED ON RGB FORMAT
determine which central pixel is closer to which color red, green, blue and fill the cross with this color.

### Tracking red object
Done work:  Apply filtering to images using the inRange command and leaving only the red part
morphological transformations (opening and closing) of the filtered image
Find the moments on the resulting image of 1st order, find the area of ​​the object.
Based on an analysis of the area of ​​the object, find its center and build a black rectangle around the object.
Make sure that the resulting black rectangle is displayed on the video, with a new one on the new frame.

### Image blurring 
