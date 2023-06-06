
denoise
openCV - fastNlMeansDenoisingColored

## 大纲
Given these requirements, here's a possible structure for your report:

1. **Introduction**: Briefly explain the task and its purpose.
    
2. **Methodology**: Describe the approach you took to create the function. Include any algorithms or techniques you used for image processing and color identification.
    
3. **Implementation**: Discuss the design decisions you made when writing your function. Include listings of your code with comments explaining how it works.
    
4. **Results and Performance Analysis**: Present the results of your function. Include examples of the color patterns it identified correctly, as well as any it struggled with. Assess the performance of your function in terms of the accuracy of the matches and the time taken to find them.
    
5. **Experiments**: Discuss any computational experiments you did and show any Matlab scripts you used that are not part of your final system.
    
6. **Discussion**: Analyze the results. Discuss any challenges you faced, any manual interventions you had to make, and why certain images may have been impossible for your function to solve. Suggest improvements that could be made to your function.
    
7. **Conclusion**: Summarize your findings and suggest any improvements or further research that could be done.
    
8. **Appendix**: Include the results tables and a full code with listings of your code.
    

Remember to include any relevant code snippets and screenshots in your report to illustrate your points.


## 代码分析
Great! You've shared your Python code that uses OpenCV for image processing to solve the task. Let's break down the code and its functionality to help you write your report:

1. **Function `get_ccorrected_image(img)`**: This function takes an image as input and returns a corrected version of the image. It applies bilateral filtering for noise reduction, converts the image to grayscale, applies Canny edge detection, and finds contours in the image. It then identifies circles and quadrilaterals in the image, sorts the circles based on their x-coordinates, and applies a perspective transform to correct the image.
    
2. **Function `determine_colorModule_range(corrected_img)`**: This function takes the corrected image as input and returns contours that represent the color modules in the image. It applies a median filter for further noise reduction, converts the image to RGB, applies Canny edge detection to each channel (R, G, B), combines the edges, and finds contours in the combined edge image.
    
3. **Function `image_enhanceSaturation(corrected_img)`**: This function enhances the saturation of the corrected image. It applies a bilateral filter, converts the image to the HSV color space, increases the saturation channel values by a factor, and converts the modified HSV image back to the BGR color space.
    
4. **Function `detect_colorModule(contours, enhanced_img)`**: This function takes the contours and the enhanced image as input and returns a 4x4 array representing the recognized colors in the image. It approximates each contour to a polygon, crops the region of interest (quadrilateral) from the image, converts it to RGB, and identifies the color of each cell within the quadrilateral.
    
5. **Main Program**: In the main program, you define a function `colourMatrix(filename)` that integrates all the above functions to process an image and return the recognized color pattern. You then call this function with a specific image file and print the result.
    

In your report, you can explain each function in detail, including the image processing techniques used (bilateral filtering, Canny edge detection, contour finding, perspective transform, saturation enhancement) and the logic behind the color detection rules. You can also discuss the performance of your program in terms of the accuracy of the color matches and the time taken to process an image. Include screenshots of the processed images and the output color pattern to illustrate your explanations. Finally, discuss any challenges you faced, any improvements you could make, and any techniques you learned about but didn't have time to implement.


## **Introduction**

The purpose of this project was to develop a function capable of interpreting color patterns from images. This task was inspired by the game "Life of George" by Lego, where players are required to construct a shape from Lego blocks based on an image that is displayed and then blanked out. The player's constructed shape is then photographed and compared to the original image. This project aims to automate the process of interpreting the color pattern from the original image.

The function developed in this project, named `colourMatrix(filename)`, takes an image file as input and returns a 4x4 array representing the color pattern in the image. The colors identified in the images are red, green, yellow, blue, and white. The function is designed to handle images of varying complexity and does not assume a specific orientation for the images.

The development of this function involved several image processing techniques, including noise reduction, edge detection, contour finding, and color identification. The performance of the function was evaluated based on the accuracy of the color matches and the time taken to process an image.

This report will detail the approach taken to develop the `colourMatrix(filename)` function, explain how it works, discuss the design decisions made during its development, and assess its performance. The report will also suggest potential improvements for the function and discuss any techniques that were learned about but not implemented due to time constraints.


**Methodology**

The development of the `colourMatrix(filename)` function involved a series of image processing steps, each of which contributed to the overall task of identifying the color pattern in an image.

1. **Image Correction**: The first step in the process was to correct the input image. This was achieved using the `get_ccorrected_image(img)` function, which applied bilateral filtering for noise reduction, converted the image to grayscale, applied Canny edge detection, and found contours in the image. The function then identified circles and quadrilaterals in the image, sorted the circles based on their x-coordinates, and applied a perspective transform to correct the image.
    
2. **Color Module Range Determination**: The next step was to determine the color module range in the corrected image. This was done using the `determine_colorModule_range(corrected_img)` function, which applied a median filter for further noise reduction, converted the image to RGB, applied Canny edge detection to each channel (R, G, B), combined the edges, and found contours in the combined edge image.
    
3. **Image Saturation Enhancement**: To improve the visibility of the colors in the image, the `image_enhanceSaturation(corrected_img)` function was used to enhance the saturation of the corrected image. This function applied a bilateral filter, converted the image to the HSV color space, increased the saturation channel values by a factor, and converted the modified HSV image back to the BGR color space.
    
4. **Color Module Detection**: The final step in the process was to detect the color modules in the enhanced image. This was done using the `detect_colorModule(contours, enhanced_img)` function, which approximated each contour to a polygon, cropped the region of interest (quadrilateral) from the image, converted it to RGB, and identified the color of each cell within the quadrilateral using a set of color detection rules.
    

The `colourMatrix(filename)` function integrated all these steps to process an image and return the recognized color pattern. The function was designed to handle images of varying complexity and did not assume a specific orientation for the images.

图像校正
程序的第一步是图像校正，这对于为后续步骤准备图像至关重要。此过程由get_ccorrected_image(img)函数执行。该函数依次执行下列过程。

降噪:该功能首先对输入图像应用双边滤波器进行降噪。双边滤波是一种非线性、边缘保持和降噪平滑滤波器。图像中每个像素的强度值由附近像素的强度值的加权平均值代替。至关重要的是，这个权重不仅取决于像素的欧几里得距离，还取决于辐射差异(例如，范围差异，如颜色强度，深度距离等)。这通过系统地循环每个像素并相应地调整相邻像素的权重来保持锐利的边缘。

灰度转换:将图像转换为灰度。这是因为许多图像处理技术都适用于灰度图像，因为它们更容易处理(只有2个维度:宽度和高度)，而彩色图像有三个颜色通道。而因为该阶段只需要对4个黑色圆形进行识别，所以不需要采用rgb颜色通道。

边缘检测:该函数然后对灰度图像应用Canny边缘检测。Canny方法通过寻找图像梯度的局部最大值来找到边缘。梯度是用高斯滤波器的导数来计算的。该方法使用两个阈值检测强边缘和弱边缘，只有当弱边缘与强边缘相连时，弱边缘才会被包含在输出中。因此，这种方法比其他方法更不容易被噪声欺骗，更容易检测到真正的弱边缘。

轮廓检测:通过边缘检测，在图像中找到轮廓(基于cv2.find。等高线可以简单地解释为连接所有连续点(沿着边界)的曲线，具有相同的颜色或强度。轮廓是形状分析和目标检测与识别的有用工具。

圆和四边形识别:该功能然后识别检测轮廓中的圆和四边形。这是通过将每个轮廓近似为多边形并检查多边形的顶点数量来完成的。设定如果一个多边形有超过6个顶点，它初步被认为是一个圆。如果一个多边形有4个顶点，它初步被认为是一个四边形。然后的所有圆进行质心计算，再根据最大凸边形算法(ConvexHull())求出4个黑色锚点的坐标值.

透视校正:最后，该函数应用透视变换(cv2.warpPerspective())来校正图像。通过定义一组变换的目标点(（490，10），（490，490），（10，490），（10，10）），并根据图像中锚点的位置或最大四边形的顶点计算透视变换矩阵（少数图片无法完全识别4个锚点，需要通过颜色模块边缘进行透视转换见图[Fig. 3.]）来完成的。然后，该函数使用计算矩阵对图像应用透视变换。

此函数的输出是输入图像的更正版本，为图像处理管道中的下一步做好准备。



结果与性能分析

从两个主要方面评估了colormatrix (filename)函数的性能:结果准确性和算法复杂性。

结果准确性分析

使用image2中的所有图像对colormatrix(文件名)函数进行了一系列测试，以评估其性能，特别是其在识别和解释颜色模式方面的准确性。结果非常令人印象深刻，该函数显示出100%的准确率。（输出结果可见附录）这是通过将函数的输出与每个测试图像中的已知颜色模式进行比较来确定的。在任何情况下，该功能都能够正确识别测试集中的所有颜色模块，从而获得完美的准确率。这种卓越的性能可归因于鲁棒图像处理技术在该功能中使用，包括降噪，边缘检测，轮廓查找和颜色识别。

在程序的各个阶段使用并优化最佳方法，如在校正图像时，进行两次不同方法的锚点识别（先识别黑色原型锚点，当部分透视变化的图片无法识别出4个圆形锚点转为识别颜色模块四边形）最大程度确保图像能够完成校正以进行下一步操作。在颜色识别阶段，因为部分图片的黄色和绿色在RGB值中数值接近（见[Fig.])，故对图像进行饱和度增强操作，拉大黄色和绿色的RGB值的差距，使颜色判断规则能100%判断成功具体的颜色。故无论图像的复杂程度如何，无论是颜色模块清晰、噪点最小的简单图像，还是颜色模块重叠、噪点高的复杂图像，该功能都能始终产生准确的结果。整个项目具有极大的适应性用于其他图集（如图片黑色锚点变换为了其他颜色锚点，依旧可以执行并输出正确结果）这证明了该函数的鲁棒性及其处理图像中真实世界可变性的能力。

算法复杂度分析

从时间复杂度和空间复杂度两方面分析了colormatrix (filename)函数的复杂度。算法的时间复杂度量化了算法运行所需的时间，作为程序输入大小的函数。算法的空间复杂度量化了算法运行所占用的空间或内存的数量，作为程序输入大小的函数。

该函数采用了几种图像处理技术，每种技术都有其自身的复杂性。然而，该函数的整体时间复杂度可以认为是线性的，因为图像中的每个像素都是单独处理的，处理时间随着图像的大小成比例地增加。

在空间复杂度方面，该函数在处理过程中存储输入图像和多个中间图像。然而，由于这些图像是单独处理和存储的，因此空间复杂度也与输入图像的大小成线性关系。

总之，colormatrix (filename)函数在准确性和复杂性方面都表现出了出色的性能。其完美的准确率和线性复杂性使其成为识别图像中颜色模式的高效可靠的工具。

[153.3623 214.911 153.3495]
[ 17.6339 212.8305 17.4485]