#pragma once
#include <opencv2\opencv.hpp>

void initialize_NNF(cv::Mat &nnf,
	int src_width, int src_height);
void initialize_NNF(cv::Mat &nnf,
	int src_width, int src_height,
	int dst_width, int dst_height);

void reconstruct_from_NNF(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst, int psize);
void reconstruct_from_NNF_no_avg(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst);

double ssd(int x1, int y1, int x2, int y2, int psize, const cv::Mat &m1, const cv::Mat &m2);
void propagation(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst, bool top_left);
void iterate_NNF(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst);

void initialize_hole_NNF(cv::Mat &nnf, const cv::Mat &src);
void initialize_hole_NNF(cv::Mat &nnf, const cv::Mat &src, const cv::Mat &dst, int psize);
void initialize_hole_NNF(cv::Mat &nnf, const cv::Mat &src, const cv::Mat &dst, int psize, const cv::Mat &last_nnf);
void fill_NNF(cv::Mat &nnf, const cv::Mat &src);

void reconstruct_from_hole_NNF(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst, int psize);
void reconstruct_from_hole_NNF_no_avg(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst, int psize);

bool in_hole(int x, int y, const cv::Mat &src);
bool around_hole(int x, int y, const cv::Mat &src, int psize);
double ssd_hole(int x1, int y1, int x2, int y2, int psize, const cv::Mat &m1, const cv::Mat &m2);
void propagation_hole(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst, bool top_left);
void random_search_hole(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst);
void iterate_hole_NNF(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst);
bool contains_hole(const cv::Mat &src);

void down_sample(const cv::Mat &input, cv::Mat &output);
void merge_original(const cv::Mat &bs, cv::Mat &src);

void visialize_NNF(const cv::Mat &nnf, const char *str);
bool in_NNF(int x, int y, const cv::Mat &nnf);
void sample_down(const cv::Mat &src, cv::Mat &dst);
void sample_up(const cv::Mat &src, cv::Mat &dst, const cv::Size &size);

void fill_image_with_image(cv::Mat &img, const cv::Mat &filling);