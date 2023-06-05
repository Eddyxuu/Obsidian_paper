
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

图像校正
程序的第一步是图像校正，这对于为后续步骤准备图像至关重要。此过程由get_ccorrected_image(img)函数执行。该函数依次执行下列过程。

降噪:该功能首先对输入图像应用双边滤波器进行降噪。双边滤波是一种非线性、边缘保持和降噪平滑滤波器。图像中每个像素的强度值由附近像素的强度值的加权平均值代替。至关重要的是，这个权重不仅取决于像素的欧几里得距离，还取决于辐射差异(例如，范围差异，如颜色强度，深度距离等)。这通过系统地循环每个像素并相应地调整相邻像素的权重来保持锐利的边缘。

灰度转换:将图像转换为灰度。这是因为许多图像处理技术都适用于灰度图像，因为它们更容易处理(只有2个维度:宽度和高度)，而彩色图像有三个颜色通道。而因为该阶段只需要对4个黑色圆形进行识别，所以不需要采用rgb颜色通道。

边缘检测:该函数然后对灰度图像应用Canny边缘检测。Canny方法通过寻找图像梯度的局部最大值来找到边缘。梯度是用高斯滤波器的导数来计算的。该方法使用两个阈值检测强边缘和弱边缘，只有当弱边缘与强边缘相连时，弱边缘才会被包含在输出中。因此，这种方法比其他方法更不容易被噪声欺骗，更容易检测到真正的弱边缘。

轮廓检测:通过边缘检测，在图像中找到轮廓。等高线可以简单地解释为连接所有连续点(沿着边界)的曲线，具有相同的颜色或强度。轮廓是形状分析和目标检测与识别的有用工具。

圆和四边形识别:该功能然后识别检测轮廓中的圆和四边形。这是通过将每个轮廓近似为多边形并检查多边形的顶点数量来完成的。设定如果一个多边形有超过6个顶点，它被认为是一个圆。如果一个多边形有4个顶点，它被认为是一个四边形。

透视校正:最后，该函数应用透视变换来校正图像。这是通过定义一组变换的目标点，并根据图像中圆形的位置或最大四边形的顶点计算透视变换矩阵来完成的。然后，该函数使用计算矩阵对图像应用透视变换。

此函数的输出是输入图像的更正版本，为图像处理管道中的下一步做好准备。