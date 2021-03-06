#pragma once
#include <opencv2\opencv.hpp>

// CDialogVignette 对话框

class CDialogVignette : public CDialogEx
{
	DECLARE_DYNAMIC(CDialogVignette)

public:
	CDialogVignette(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~CDialogVignette();

// 对话框数据
	enum { IDD = IDD_DIALOG_VIGNETTE };
	double *maskImg;
	cv::Mat labImg;
	cv::Mat old_labImg;
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	virtual BOOL OnInitDialog();
};
