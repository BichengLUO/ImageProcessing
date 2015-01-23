#include "stdafx.h"
#include "Inpainting.h"
#include <limits>

void initialize_NNF(cv::Mat &nnf,
	int src_width, int src_height)
{
	initialize_NNF(nnf, src_width, src_height, src_width, src_height);
}

void initialize_NNF(cv::Mat &nnf,
	int src_width, int src_height,
	int dst_width, int dst_height)
{
	srand(time(NULL));
	nnf = cv::Mat::zeros(src_height, src_width, CV_32SC2);
	for (int i = 0; i < nnf.rows; i++)
	{
		int *data = nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			int b_x = rand() % dst_width;
			int b_y = rand() % dst_height;
			data[j * 2] = b_x - j;
			data[j * 2 + 1] = b_y - i;
		}
	}
}

void fill_NNF(cv::Mat &nnf, const cv::Mat &src)
{
	srand(time(NULL));
	for (int i = 0; i < nnf.rows; i++)
	{
		int *data = nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			if (in_hole(j, i, src) && in_hole(j + data[j * 2], i + data[j * 2 + 1], src))
			{
				int b_x = 0, b_y = 0;
				do
				{
					b_x = rand() % src.cols;
					b_y = rand() % src.rows;
				} while (in_hole(b_x, b_y, src) ||
					(b_x - j == 0 && b_y - i == 0));
				data[j * 2] = b_x - j;
				data[j * 2 + 1] = b_y - i;
			}
		}
	}
}

void reconstruct_from_NNF_no_avg(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst)
{
	for (int i = 0; i < src.rows; i++)
	{
		uchar *data = src.ptr<uchar>(i);
		const int *nnf_data = nnf.ptr<int>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (in_hole(j, i, dst))
			{
				int off_x = nnf_data[j * 2];
				int off_y = nnf_data[j * 2 + 1];
				int dst_i = i + off_y;
				int dst_j = j + off_x;

				const uchar *dst_data = src.ptr<uchar>(dst_i);
				uchar b = dst_data[dst_j * 3];
				uchar g = dst_data[dst_j * 3 + 1];
				uchar r = dst_data[dst_j * 3 + 2];

				data[j * 3] = b;
				data[j * 3 + 1] = g;
				data[j * 3 + 2] = r;
			}
		}
	}
}

void reconstruct_from_NNF(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst, int psize)
{
	for (int i = 0; i < src.rows; i++)
	{
		uchar *data = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			int bt = 0;
			int gt = 0;
			int rt = 0;
			int w = 0;
			for (int pi = i - psize / 2; pi <= i + psize / 2; pi++)
			for (int pj = j - psize / 2; pj <= j + psize / 2; pj++)
			{
				if (pi >= 0 && pi < nnf.rows &&
					pj >= 0 && pj < nnf.cols)
				{
					const int *nnf_data = nnf.ptr<int>(pi);
					int off_x = nnf_data[pj * 2];
					int off_y = nnf_data[pj * 2 + 1];
					int dst_i = i + off_y;
					int dst_j = j + off_x;
					if (dst_i >= 0 && dst_i < dst.rows &&
						dst_j >= 0 && dst_j < dst.cols)
					{
						const uchar *dst_data = dst.ptr<uchar>(dst_i);
						uchar b = dst_data[dst_j * 3];
						uchar g = dst_data[dst_j * 3 + 1];
						uchar r = dst_data[dst_j * 3 + 2];
						bt += b;
						gt += g;
						rt += r;
						w++;
					}
				}
			}
			data[j * 3] = bt / w;
			data[j * 3 + 1] = gt / w;
			data[j * 3 + 2] = rt / w;
		}
	}
}

double ssd(int x1, int y1, int x2, int y2, int psize, const cv::Mat &m1, const cv::Mat &m2)
{
	long br = 0;
	long gr = 0;
	long rr = 0;
	int w = 0;
	for (int i = y1 - psize / 2; i <= y1 + psize / 2; i++)
	{
		int i1 = i;
		int i2 = y2 + i - y1;
		if (i1 >= 0 && i1 < m1.rows &&
			i2 >= 0 && i2 < m2.rows)
		{
			const uchar *data1 = m1.ptr<uchar>(i1);
			const uchar *data2 = m2.ptr<uchar>(i2);
			for (int j = x1 - psize / 2; j <= x1 + psize / 2; j++)
			{
				int j1 = j;
				int j2 = x2 + j - x1;

				if (j1 >= 0 && j1 < m1.cols &&
					j2 >= 0 && j2 < m2.cols)
				{
					int b1 = data1[j1 * 3];
					int g1 = data1[j1 * 3 + 1];
					int r1 = data1[j1 * 3 + 2];

					int b2 = data2[j2 * 3];
					int g2 = data2[j2 * 3 + 1];
					int r2 = data2[j2 * 3 + 2];

					int bd = b1 - b2;
					int gd = g1 - g2;
					int rd = r1 - r2;

					br += bd * bd;
					gr += gd * gd;
					rr += rd * rd;
					w++;
				}
			}
		}
	}
	if (w == 0)
		return std::numeric_limits<double>::max();
	double result = (br + gr + rr) / (3.0 * w);
	return result;
}

void propagation(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst, bool top_left)
{
	if (top_left)
	{
		for (int i = 0; i < nnf.rows; i++)
		{
			int *nnf_data = nnf.ptr<int>(i);
			for (int j = 0; j < nnf.cols; j++)
			{
				int off_x1 = nnf_data[j * 2];
				int off_y1 = nnf_data[j * 2 + 1];
				double d1 = ssd(j, i, j + off_x1, i + off_y1, psize, src, dst);
				int off_x = off_x1;
				int off_y = off_y1;
				if (i > 0)
				{
					int off_x2 = nnf_data[j * 2 - nnf.cols * 2];
					int off_y2 = nnf_data[j * 2 + 1 - nnf.cols * 2];
					int dst_x = j + off_x2;
					int dst_y = i + off_y2;
					if (dst_y >= 0 && dst_y < dst.rows &&
						dst_x >= 0 && dst_x < dst.cols)
					{
						double d2 = ssd(j, i, dst_x, dst_y, psize, src, dst);
						if (d2 < d1)
						{
							d1 = d2;
							off_x = off_x2;
							off_y = off_y2;
						}
					}
				}
				if (j > 0)
				{
					int off_x3 = nnf_data[(j - 1) * 2];
					int off_y3 = nnf_data[(j - 1) * 2 + 1];
					int dst_x = j + off_x3;
					int dst_y = i + off_y3;
					if (dst_y >= 0 && dst_y < dst.rows &&
						dst_x >= 0 && dst_x < dst.cols)
					{
						double d3 = ssd(j, i, dst_x, dst_y, psize, src, dst);
						if (d3 < d1)
						{
							d1 = d3;
							off_x = off_x3;
							off_y = off_y3;
						}
					}
				}
				nnf_data[j * 2] = off_x;
				nnf_data[j * 2 + 1] = off_y;
			}
		}
	}
	else
	{
		for (int i = nnf.rows - 1; i >= 0; i--)
		{
			int *nnf_data = nnf.ptr<int>(i);
			for (int j = nnf.cols - 1; j >= 0; j--)
			{
				int off_x1 = nnf_data[j * 2];
				int off_y1 = nnf_data[j * 2 + 1];
				double d1 = ssd(j, i, j + off_x1, i + off_y1, psize, src, dst);
				int off_x = off_x1;
				int off_y = off_y1;
				if (i < nnf.rows - 1)
				{
					int off_x2 = nnf_data[j * 2 + nnf.cols * 2];
					int off_y2 = nnf_data[j * 2 + 1 + nnf.cols * 2];
					int dst_x = j + off_x2;
					int dst_y = i + off_y2;
					if (dst_y >= 0 && dst_y < dst.rows &&
						dst_x >= 0 && dst_x < dst.cols)
					{
						double d2 = ssd(j, i, dst_x, dst_y, psize, src, dst);
						if (d2 < d1)
						{
							d1 = d2;
							off_x = off_x2;
							off_y = off_y2;
						}
					}
				}
				if (j < nnf.cols - 1)
				{
					int off_x3 = nnf_data[(j + 1) * 2];
					int off_y3 = nnf_data[(j + 1) * 2 + 1];
					int dst_x = j + off_x3;
					int dst_y = i + off_y3;
					if (dst_y >= 0 && dst_y < dst.rows &&
						dst_x >= 0 && dst_x < dst.cols)
					{
						double d3 = ssd(j, i, dst_x, dst_y, psize, src, dst);
						if (d3 < d1)
						{
							d1 = d3;
							off_x = off_x3;
							off_y = off_y3;
						}
					}
				}
				nnf_data[j * 2] = off_x;
				nnf_data[j * 2 + 1] = off_y;
			}
		}
	}
}

void random_search(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst)
{
	for (int i = 0; i < nnf.rows; i++)
	{
		int *nnf_data = nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			int off_x = nnf_data[j * 2];
			int off_y = nnf_data[j * 2 + 1];
			int b_x = off_x;
			int b_y = off_y;
			double d = ssd(j, i, j + off_x, i + off_y, psize, src, dst);
			int radius = dst.rows > dst.cols ? dst.rows : dst.cols;
			while (radius >= 1)
			{
				int cx = rand() % (2 * radius + 1) - radius;
				int cy = rand() % (2 * radius + 1) - radius;
				int coff_x = off_x + cx;
				int coff_y = off_y + cy;
				int dst_x = j + coff_x;
				int dst_y = i + coff_y;
				if (dst_y >= 0 && dst_y < dst.rows &&
					dst_x >= 0 && dst_x < dst.cols)
				{
					double cd = ssd(j, i, dst_x, dst_y, psize, src, dst);
					if (cd < d)
					{
						d = cd;
						b_x = coff_x;
						b_y = coff_y;
					}
				}
				radius *= 0.5;
			}
			nnf_data[j * 2] = b_x;
			nnf_data[j * 2 + 1] = b_y;
		}
	}
}

void iterate_NNF(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst)
{
	for (int i = 0; i < 10; i++)
	{
		propagation(nnf, psize, src, dst, i % 2 == 0);
		random_search(nnf, psize, src, dst);
	}
}

void initialize_hole_NNF(cv::Mat &nnf, const cv::Mat &src)
{
	srand(time(NULL));
	nnf = cv::Mat::zeros(src.rows, src.cols, CV_32SC2);
	for (int i = 0; i < nnf.rows; i++)
	{
		int *data = nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			int b_x = 0, b_y = 0;
			do
			{
				b_x = rand() % src.cols;
				b_y = rand() % src.rows;
			} while (in_hole(b_x, b_y, src) ||
				(b_x - j == 0 && b_y - i == 0));
			data[j * 2] = b_x - j;
			data[j * 2 + 1] = b_y - i;
		}
	}
}

void initialize_hole_NNF(cv::Mat &nnf, const cv::Mat &src, const cv::Mat &dst, int psize)
{
	srand(time(NULL));
	nnf = cv::Mat::zeros(src.rows, src.cols, CV_32SC2);
	for (int i = 0; i < nnf.rows; i++)
	{
		int *data = nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			if (around_hole(j, i, src, psize))
			{
				int b_x = 0, b_y = 0;
				do
				{
					b_x = rand() % src.cols;
					b_y = rand() % src.rows;
				} while (in_hole(b_x, b_y, dst) ||
					(b_x - j == 0 && b_y - i == 0));
				data[j * 2] = b_x - j;
				data[j * 2 + 1] = b_y - i;
			}
		}
	}
}

void initialize_hole_NNF(cv::Mat &nnf, const cv::Mat &src, const cv::Mat &dst, int psize, const cv::Mat &last_nnf)
{
	if (last_nnf.rows == 0)
	{
		initialize_hole_NNF(nnf, src, dst, psize);
		return;
	}
	srand(time(NULL));
	nnf = cv::Mat::zeros(src.rows, src.cols, CV_32SC2);
	for (int i = 0; i < nnf.rows; i++)
	{
		int *data = nnf.ptr<int>(i);
		const int *last_data = last_nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			int last_x = last_data[j * 2];
			int last_y = last_data[j * 2 + 1];
			data[j * 2] = last_x;
			data[j * 2 + 1] = last_y;
			if (last_x == 0 &&
				last_y == 0 &&
				around_hole(j, i, src, psize))
			{
				int b_x = 0, b_y = 0;
				do
				{
					b_x = rand() % src.cols;
					b_y = rand() % src.rows;
				} while (in_hole(b_x, b_y, dst) ||
					(b_x - j == 0 && b_y - i == 0));
				data[j * 2] = b_x - j;
				data[j * 2 + 1] = b_y - i;
			}
		}
	}
}

bool in_hole(int x, int y, const cv::Mat &src)
{
	const uchar *src_data = src.ptr<uchar>(y);
	if (src_data[x * 3] == 0 &&
		src_data[x * 3 + 1] == 0 &&
		src_data[x * 3 + 2] == 255)
	{
		return true;
	}
	return false;
}

bool around_hole(int x, int y, const cv::Mat &src, int psize)
{
	int hole = 0;
	int no_hole = 0;
	for (int i = y - psize / 2; i <= y + psize / 2; i++)
	for (int j = x - psize / 2; j <= x + psize / 2; j++)
	{
		if (i >= 0 && i < src.rows &&
			j >= 0 && j < src.cols)
		{
			const uchar *src_data = src.ptr<uchar>(i);
			if (src_data[j * 3] == 0 &&
				src_data[j * 3 + 1] == 0 &&
				src_data[j * 3 + 2] == 255)
				hole++;
			else
				no_hole++;
		}
	}
	return hole > 0 && no_hole > 0 && 2 * no_hole >= hole;
}

void reconstruct_from_hole_NNF(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst, int psize)
{
	cv::Mat src_t = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		uchar *data = src_t.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (in_NNF(j, i, nnf) &&
				in_hole(j, i, dst))
			{
				int bt = 0;
				int gt = 0;
				int rt = 0;
				int w = 0;
				for (int pi = i - psize / 2; pi <= i + psize / 2; pi++)
				for (int pj = j - psize / 2; pj <= j + psize / 2; pj++)
				{
					if (pi >= 0 && pi < nnf.rows &&
						pj >= 0 && pj < nnf.cols)
					{
						const int *nnf_data = nnf.ptr<int>(pi);
						int off_x = nnf_data[pj * 2];
						int off_y = nnf_data[pj * 2 + 1];
						int dst_i = i + off_y;
						int dst_j = j + off_x;
						if (dst_i >= 0 && dst_i < dst.rows &&
							dst_j >= 0 && dst_j < dst.cols)
						{
							const uchar *dst_data = dst.ptr<uchar>(dst_i);
							uchar b = dst_data[dst_j * 3];
							uchar g = dst_data[dst_j * 3 + 1];
							uchar r = dst_data[dst_j * 3 + 2];
							if (b != 0 || g != 0 || r != 255)
							{
								bt += b;
								gt += g;
								rt += r;
								w++;
							}
						}
					}
				}
				data[j * 3] = bt / w;
				data[j * 3 + 1] = gt / w;
				data[j * 3 + 2] = rt / w;
			}
		}
	}
	src.release();
	src = src_t;
}

void reconstruct_from_hole_NNF_no_avg(const cv::Mat &nnf, cv::Mat &src, const cv::Mat &dst, int psize)
{
	cv::Mat src_t = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		uchar *data = src_t.ptr<uchar>(i);
		const int *nnf_data = nnf.ptr<int>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (in_NNF(j, i, nnf) &&
				in_hole(j, i, dst))
			{
				int off_x = nnf_data[j * 2];
				int off_y = nnf_data[j * 2 + 1];
				int dst_i = i + off_y;
				int dst_j = j + off_x;
				const uchar *dst_data = dst.ptr<uchar>(dst_i);
				uchar b = dst_data[dst_j * 3];
				uchar g = dst_data[dst_j * 3 + 1];
				uchar r = dst_data[dst_j * 3 + 2];
				data[j * 3] = b;
				data[j * 3 + 1] = g;
				data[j * 3 + 2] = r;
			}
		}
	}
	src.release();
	src = src_t;
}

double ssd_hole(int x1, int y1, int x2, int y2, int psize, const cv::Mat &m1, const cv::Mat &m2)
{
	unsigned long br = 0;
	unsigned long gr = 0;
	unsigned long rr = 0;
	int w = 0;
	for (int i = y1 - psize / 2; i <= y1 + psize / 2; i++)
	{
		int i1 = i;
		int i2 = y2 + i - y1;
		if (i1 >= 0 && i1 < m1.rows &&
			i2 >= 0 && i2 < m2.rows)
		{
			const uchar *data1 = m1.ptr<uchar>(i1);
			const uchar *data2 = m2.ptr<uchar>(i2);
			for (int j = x1 - psize / 2; j <= x1 + psize / 2; j++)
			{
				int j1 = j;
				int j2 = x2 + j - x1;

				if (j1 >= 0 && j1 < m1.cols &&
					j2 >= 0 && j2 < m2.cols)
				{
					int b1 = data1[j1 * 3];
					int g1 = data1[j1 * 3 + 1];
					int r1 = data1[j1 * 3 + 2];

					int b2 = data2[j2 * 3];
					int g2 = data2[j2 * 3 + 1];
					int r2 = data2[j2 * 3 + 2];

					if ((b1 != 0 || g1 != 0 || r1 != 255) &&
						(b2 != 0 || g2 != 0 || r2 != 255))
					{
						int bd = b1 - b2;
						int gd = g1 - g2;
						int rd = r1 - r2;

						br += bd * bd;
						gr += gd * gd;
						rr += rd * rd;
						w++;
					}
				}
			}
		}
	}
	if (w == 0)
		return std::numeric_limits<double>::max();
	double result = (br + gr + rr) / (3.0 * w * w);
	return result;
}

void propagation_hole(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst, bool top_left)
{
	if (top_left)
	{
		for (int i = 0; i < nnf.rows; i++)
		{
			int *nnf_data = nnf.ptr<int>(i);
			for (int j = 0; j < nnf.cols; j++)
			{
				if (in_NNF(j, i, nnf))
				{
					int off_x1 = nnf_data[j * 2];
					int off_y1 = nnf_data[j * 2 + 1];
					double d1 = ssd_hole(j, i, j + off_x1, i + off_y1, psize, src, dst);
					int off_x = off_x1;
					int off_y = off_y1;
					if (i > 0)
					{
						int off_x2 = nnf_data[j * 2 - nnf.cols * 2];
						int off_y2 = nnf_data[j * 2 + 1 - nnf.cols * 2];
						if (off_x2 != 0 || off_y2 != 0)
						{
							int dst_x = j + off_x2;
							int dst_y = i + off_y2;
							if (dst_y >= 0 && dst_y < dst.rows &&
								dst_x >= 0 && dst_x < dst.cols &&
								!in_hole(dst_x, dst_y, dst))
							{
								double d2 = ssd_hole(j, i, dst_x, dst_y, psize, src, dst);
								if (d2 < d1)
								{
									d1 = d2;
									off_x = off_x2;
									off_y = off_y2;
								}
							}
						}
					}
					if (j > 0)
					{
						int off_x3 = nnf_data[(j - 1) * 2];
						int off_y3 = nnf_data[(j - 1) * 2 + 1];
						if (off_x3 != 0 || off_y3 != 0)
						{
							int dst_x = j + off_x3;
							int dst_y = i + off_y3;
							if (dst_y >= 0 && dst_y < dst.rows &&
								dst_x >= 0 && dst_x < dst.cols &&
								!in_hole(dst_x, dst_y, dst))
							{
								double d3 = ssd_hole(j, i, dst_x, dst_y, psize, src, dst);
								if (d3 < d1)
								{
									d1 = d3;
									off_x = off_x3;
									off_y = off_y3;
								}
							}
						}
					}
					nnf_data[j * 2] = off_x;
					nnf_data[j * 2 + 1] = off_y;
				}
			}
		}
	}
	else
	{
		for (int i = nnf.rows - 1; i >= 0; i--)
		{
			int *nnf_data = nnf.ptr<int>(i);
			for (int j = nnf.cols - 1; j >= 0; j--)
			{
				if (in_NNF(j, i, nnf))
				{
					int off_x1 = nnf_data[j * 2];
					int off_y1 = nnf_data[j * 2 + 1];
					double d1 = ssd_hole(j, i, j + off_x1, i + off_y1, psize, src, dst);
					int off_x = off_x1;
					int off_y = off_y1;
					if (i < nnf.rows - 1)
					{
						int off_x2 = nnf_data[j * 2 + nnf.cols * 2];
						int off_y2 = nnf_data[j * 2 + 1 + nnf.cols * 2];
						if (off_x2 != 0 || off_y2 != 0)
						{
							int dst_x = j + off_x2;
							int dst_y = i + off_y2;
							if (dst_y >= 0 && dst_y < dst.rows &&
								dst_x >= 0 && dst_x < dst.cols &&
								!in_hole(dst_x, dst_y, dst))
							{
								double d2 = ssd_hole(j, i, dst_x, dst_y, psize, src, dst);
								if (d2 < d1)
								{
									d1 = d2;
									off_x = off_x2;
									off_y = off_y2;
								}
							}
						}
					}
					if (j < nnf.cols - 1)
					{
						int off_x3 = nnf_data[(j + 1) * 2];
						int off_y3 = nnf_data[(j + 1) * 2 + 1];
						if (off_x3 != 0 || off_y3 != 0)
						{
							int dst_x = j + off_x3;
							int dst_y = i + off_y3;
							if (dst_y >= 0 && dst_y < dst.rows &&
								dst_x >= 0 && dst_x < dst.cols &&
								!in_hole(dst_x, dst_y, dst))
							{
								double d3 = ssd_hole(j, i, dst_x, dst_y, psize, src, dst);
								if (d3 < d1)
								{
									d1 = d3;
									off_x = off_x3;
									off_y = off_y3;
								}
							}
						}
					}
					nnf_data[j * 2] = off_x;
					nnf_data[j * 2 + 1] = off_y;
				}
			}
		}
	}
}

void random_search_hole(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst)
{
	for (int i = 0; i < nnf.rows; i++)
	{
		int *nnf_data = nnf.ptr<int>(i);
		for (int j = 0; j < nnf.cols; j++)
		{
			if (in_NNF(j, i, nnf))
			{
				int off_x = nnf_data[j * 2];
				int off_y = nnf_data[j * 2 + 1];
				int b_x = off_x;
				int b_y = off_y;
				double d = ssd_hole(j, i, j + off_x, i + off_y, psize, src, dst);
				int radius = 50;
				while (radius >= 1)
				{
					int dst_x, dst_y;
					int coff_x, coff_y;
					int count = 0;
					bool cant_find = false;
					do
					{
						int cx = rand() % (2 * radius + 1) - radius;
						int cy = rand() % (2 * radius + 1) - radius;
						coff_x = off_x + cx;
						coff_y = off_y + cy;
						dst_x = j + coff_x;
						dst_y = i + coff_y;
						if (++count >= 20)
						{
							cant_find = true;
							break;
						}
					} while (dst_y < 0 || dst_y >= dst.rows ||
						dst_x < 0 || dst_x >= dst.cols ||
						in_hole(dst_x, dst_y, dst));
					if (!cant_find && coff_x != 0 && coff_y != 0)
					{
						double cd = ssd_hole(j, i, dst_x, dst_y, psize, src, dst);
						if (cd < d)
						{
							d = cd;
							b_x = coff_x;
							b_y = coff_y;
						}
					}
					radius *= 0.8;
				}
				nnf_data[j * 2] = b_x;
				nnf_data[j * 2 + 1] = b_y;
			}
		}
	}
}

void iterate_hole_NNF(cv::Mat &nnf, int psize, const cv::Mat &src, const cv::Mat &dst)
{
	for (int i = 0; i < 7; i++)
	{
		propagation_hole(nnf, psize, src, dst, i % 2 == 0);
		random_search_hole(nnf, psize, src, dst);
	}
}

bool contains_hole(const cv::Mat &src)
{
	for (int i = 0; i < src.rows; i++)
	{
		const uchar *src_data = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (src_data[j * 3] == 0 &&
				src_data[j * 3 + 1] == 0 &&
				src_data[j * 3 + 2] == 255)
				return true;
		}
	}
	return false;
}

void down_sample(const cv::Mat &input, cv::Mat &output)
{
	output = cv::Mat::zeros(input.rows / 2, input.cols / 2, input.type());
	for (int i = 0; i < output.rows; i++)
	{
		uchar *data = output.ptr<uchar>(i);
		const uchar *input_data = input.ptr<uchar>(2 * i);
		for (int j = 0; j < output.cols; j++)
		{
			data[j * 3] = input_data[j * 6];
			data[j * 3 + 1] = input_data[j * 6 + 1];
			data[j * 3 + 2] = input_data[j * 6 + 2];
		}
	}
}

void merge_original(const cv::Mat &bs, cv::Mat &src)
{
	for (int i = 0; i < src.rows; i++)
	{
		uchar *data = src.ptr<uchar>(i);
		const uchar *bs_data = bs.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (in_hole(j, i, src))
			{
				data[j * 3] = bs_data[j * 3];
				data[j * 3 + 1] = bs_data[j * 3 + 1];
				data[j * 3 + 2] = bs_data[j * 3 + 2];
			}
		}
	}
}

void visialize_NNF(const cv::Mat &nnf, const char *str)
{
	cv::Mat vis = cv::Mat::zeros(nnf.rows, nnf.cols, CV_8UC3);
	for (int i = 0; i < vis.rows; i++)
	{
		uchar *data = vis.ptr<uchar>(i);
		const int *nnf_data = nnf.ptr<int>(i);
		for (int j = 0; j < vis.cols; j++)
		{
			if (nnf_data[j * 2] != 0 ||
				nnf_data[j * 2 + 1] != 0)
			{
				data[3 * j] = 200;
				data[3 * j + 1] = (nnf_data[j * 2] + nnf.cols) * 255.0 / (2 * nnf.cols);
				data[3 * j + 2] = (nnf_data[j * 2 + 1] + nnf.rows) * 255.0 / (2 * nnf.rows);
			}
		}
	}
	cv::imwrite(str, vis);
	vis.release();
}

bool in_NNF(int x, int y, const cv::Mat &nnf)
{
	const int *nnf_data = nnf.ptr<int>(y);
	if (nnf_data[x * 2] != 0 ||
		nnf_data[x * 2 + 1] != 0)
		return true;
	return false;
}

void sample_down(const cv::Mat &src, cv::Mat &dst)
{
	dst = cv::Mat::zeros(src.rows / 2, src.cols / 2, src.type());
	for (int i = 0; i < dst.rows; i++)
	{
		uchar *data = dst.ptr<uchar>(i);
		const uchar *src_data = src.ptr<uchar>(2 * i);
		for (int j = 0; j < dst.cols; j++)
		{
			data[3 * j] = src_data[6 * j];
			data[3 * j + 1] = src_data[6 * j + 1];
			data[3 * j + 2] = src_data[6 * j + 2];
		}
	}
}

void sample_up(const cv::Mat &src, cv::Mat &dst, const cv::Size &size)
{
	dst = cv::Mat::zeros(size.height, size.width, src.type());
	for (int i = 0; i < dst.rows; i++)
	{
		int *data = dst.ptr<int>(i);
		if (i / 2 >= src.rows) continue;
		const int *src_data = src.ptr<int>(i / 2);
		for (int j = 0; j < dst.cols; j++)
		{
			int src_j = j / 2;
			if (src_j >= src.cols) continue;
			data[2 * j] = 2 * src_data[2 * src_j];
			data[2 * j + 1] = 2 * src_data[2 * src_j + 1];
		}
	}
}

void fill_image_with_image(cv::Mat &img, const cv::Mat &filling)
{
	for (int i = 0; i < img.rows; i++)
	{
		uchar *data = img.ptr<uchar>(i);
		const uchar *fill_data = filling.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (in_hole(j, i, img))
			{
				data[3 * j] = fill_data[3 * j];
				data[3 * j + 1] = fill_data[3 * j + 1];
				data[3 * j + 2] = fill_data[3 * j + 2];
			}
		}
	}
}