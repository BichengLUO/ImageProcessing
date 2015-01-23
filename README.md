![ImageProcessing](/images/banner.png)
ImageProcessing
================
This is an MFC application based on OpenCV. It provides some basic functionality of image processing like Gaussian filter,  lifuifying and so on. Actually, it will be more like a simple OpenCV experiment platform for quick implementation of some ideas.

Functionality
-----------------
ImageProcessing project provides such functions:

#### Vignette Filter

![Vignette Filter Example](/images/ppppp.PNG "Before Processing")
![Vignette Filter Example](/images/p.PNG "After Processing")

#### Gaussian Filter

![Gaussian Filter Example](/images/g0.PNG "Before Processing")
![Gaussian Filter Example](/images/g1.PNG "After Processing")

#### Median Filter

![Median Filter Example](/images/m0.PNG "Before Processing")
![Median Filter Example](/images/m1.PNG "After Processing")

#### Saturation adjustment

![Saturation Example](/images/saturation1.PNG "Before Processing")
![Saturation Example](/images/saturation2.PNG "After Processing")
![Saturation Example](/images/saturation3.PNG "After Processing")

#### White Balance

![White Balance Example](/images/w0.PNG "Before Processing")
![White Balance Example](/images/w1.PNG "After Processing")

#### Image Sharpening

![Image Sharpening Example](/images/sharpen5.PNG "After Processing")
![Image Sharpening Example](/images/sharpen6.PNG "After Processing")

#### Skin Beautifying

The application will first detect all the faces in the image and consider the face regions as the mask for skin beautifying. Skin beautifying contains two features: Skin Blur and Skin Whiten.
> **Note:**
> - Skin Blur is implemented using Bilateral Filter
> - Skin Whiten is implemented simply using saturation and luminance

![Skin Beautifying Example](/images/skin1.PNG "Before Processing")
![Skin Beautifying Example](/images/skin2.PNG "After Processing")
![Skin Beautifying Example](/images/skin3.PNG "After Processing")

#### Liquifying

![Liquifying Example](/images/liquify1.PNG "After Processing")

#### Inpainting (based on PatchMatch)

The inpainting function is implemented based on PatchMatch algorithm.
> **Note:**
> PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing
> http://gfx.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php

![Inpaint Example](/images/pm1.png "After Processing")
![Inpaint Example](/images/pm2.png "After Processing")
![Inpaint Example](/images/pm3.png "After Processing")
![Inpaint Example](/images/pm4.png "After Processing")
![Inpaint Example](/images/pm5.png "After Processing")
![Inpaint Example](/images/pm6.png "After Processing")
![Inpaint Example](/images/pm7.png "After Processing")
