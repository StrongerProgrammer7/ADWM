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
Done work: Canny's algorithm: 
1. RGB -> GRAY 
2.  Apply Gauss's filter
3.  calc gradient and gradient's angle 
4.  around gradient's angle
5.  suppression of non-maximal
6. double threshold
7. get contours 
<br><img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/c2c8ad50-533e-4044-a4e3-4d83c8387b33" alt="" width="300"/>

### Motion detection on video
Done work: 
1. read frame -> calc absdiff -> find contours
2. Walk along the contours of objects for the frame (frame_diff) and find a contour with an area larger than the previously specified parameter
3. If such a contour is found, it means there was movement, write the frame to a file

### Neural network training
1. Training standart NN recognation digit (build multilayer perceptron using the Keras library)
2. Training CNN recognation digit
3. Show digit and precision

### Text Recognition(opticalRecognationText)
1. Prepare dataset - capcha (include in folder) - standart image
2. Apply augmentation to dataset
3. Apply EasyOCR and Tesseract for recognation text on images
4. Record result EasyOcr and Tesseract .txt

### Contours 
***Theme: automobile (logo)***

This using Canny's algorithm (description up) and apply different operator and built-in canny
| built-in Canny | Sobel  | Prewitt | Kirch | Scharr |
| ------------- |:-------------: |:-------------:|:-------------:|:-------------:|
| <img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/cba11dcb-2dd1-4a23-b335-75a9a4dc03f0" alt="" width="300"/>| <img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/355b655d-1b6f-4390-9da4-a3da4ea4b37b" alt="" width="300"/> | <img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/4d35e841-d0d5-4d55-ac57-5c29a3bd35c5" alt="" width="300"/> | <img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/812c235c-7dfa-4281-acaa-7e40f17d8ce4" alt="" width="300"/> | <img src="https://github.com/StrongerProgrammer7/ADWM/assets/71569051/c0543d81-caee-4e0b-92bc-b615b9d3aa32" alt="" width="300"/> |

Also a comparison of the work of different operators.
Comparison of algorithms: on 3 images (small number of logos, large number of logos and cars on the street)
All 3 images were then processed with different borders (10, 100; 100, 200; 150, 230) and kernels (3x3.5x5, 7x7)
In total, 1 algorithm processed 27 images.

| Algorithm | Speed work(sec)  | Algorithm | MSE (the less, the more differences) |
| ------------- |:-------------:| :-------------:|:-------------:|
| Canny      | 2.3285114765167236  | Canny & Sobel |42416.498481999944 |
| Sobel      | 189,11079295476276 | Canny & Prewitt | 3615.576172236691 |
| Scharr      | 235,590115070343 | Canny & Scharr | 6270.624080584491 |
| Prewitt | 237,085066713 | Prewitt & Scharr | 7995.047592230903 |
| Kirsch | 1122,984922 | 
