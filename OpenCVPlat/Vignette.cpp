#include "stdafx.h"
#include "Vignette.h"

// Helper function to calculate the distance between 2 points.
double dist(CvPoint a, CvPoint b)
{
	return sqrt((double)(a.x - b.x)*(double)(a.x - b.x) + (double)(a.y - b.y)*(double)(a.y - b.y));
}

// Helper function that computes the longest distance from the edge to the center point.
double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center)
{
	// given a rect and a line
	// get which corner of rect is farthest from the line

	std::vector<cv::Point> corners(4);
	corners[0] = cv::Point(0, 0);
	corners[1] = cv::Point(imgSize.width, 0);
	corners[2] = cv::Point(0, imgSize.height);
	corners[3] = cv::Point(imgSize.width, imgSize.height);

	double maxDis = 0;
	for (int i = 0; i < 4; ++i)
	{
		double dis = dist(corners[i], center);
		if (maxDis < dis)
			maxDis = dis;
	}

	return maxDis;
}

// Helper function that creates a gradient image.   
// firstPt, radius and power, are variables that control the artistic effect of the filter.
void generateGradient(double *maskImg, int width, int height)
{
	cv::Point firstPt = cv::Point(width / 2, height / 2);
	double radius = 1.0;
	double power = 1.0;
	//getMaxDisFromCorners that computes the longest distance from the edge to the center point.
	double maxImageRad = radius * getMaxDisFromCorners(cv::Size(width, height), firstPt);

	memset(maskImg, 1, sizeof(double) * width * height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double temp = dist(firstPt, cv::Point(j, i)) / maxImageRad;
			temp = temp * power;
			temp = cos(temp);
			temp *= temp;
			temp *= temp;
			double temp_s = temp;
			maskImg[i*width + j] = temp_s;
		}
	}
}