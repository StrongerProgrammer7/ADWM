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
Done work: 
  - show img and test 3 difeerent flags for image and screen, 
  - show video(movie,webcam), record movie, display the Red Cross in the center of the screen and inside cross accept blur
  - Fill the cross with one of the 3 colors(RGB) - using the following rule: BASED ON RGB FORMAT
determine which central pixel is closer to which color red, green, blue and fill the cross with this color.
<br><img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/68c4617d-d14a-4dc3-9c6d-30ae532739fe" alt="" width="300"/>
### Tracking red object
Done work:  
- Apply filtering to images using the inRange command and leaving only the red part
- morphological transformations (opening and closing) of the filtered image
- Find the moments on the resulting image of 1st order, find the area of ​​the object.
- Based on an analysis of the area of ​​the object, find its center and build a black rectangle around the object.
- Make sure that the resulting black rectangle is displayed on the video, with a new one on the new frame.

<br><img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/20f87f7e-053f-4d01-b234-095fc06fbcd6" alt="" width="300"/>
<img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/41ffb369-2443-4b79-a67c-1db5d86a4fb5" alt="" width="300"/>

### Image blurring 
Done work: Realisation Gauss's filter and comparison with built-in Gauss's filter
<br> <img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/df7d9c23-2f2f-4db1-93f1-de0c374e0071" alt="" width="500"/>

### Canny's algorithm
Done work: Canny's algorithm: RGB -> GRAY -> Apply Gauss's filter -> calc gradient and gradient's angle -> around gradient's angle -> suppression of non-maximal -> double threshold -> get contours 

