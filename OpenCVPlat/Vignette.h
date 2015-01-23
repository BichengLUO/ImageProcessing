#ifndef VIGNETTE_H_
#define VIGNETTE_H_
#include <opencv2\opencv.hpp>

double dist(CvPoint a, CvPoint b);
double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center);
void generateGradient(double *maskImg, int width, int height);

#endif