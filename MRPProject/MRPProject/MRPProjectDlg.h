
// MRPProjectDlg.h : 헤더 파일
//

#include "GridCtrl\GridCtrl.h"
#pragma once


// CMRPProjectDlg 대화 상자
class CMRPProjectDlg : public CDialogEx
{
// 생성입니다.
public:
	CMRPProjectDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
	enum { IDD = IDD_MRPPROJECT_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()


public:
	void InitialMPS();
	void Cal_ProdA();
	void Cal_ProdB();
	void Cal_ProdC();
	void Cal_ProdD();

	void InitGridCtrl();
	CGridCtrl m_cGrid;

	void Print();
};

struct Product_A
{
	int nGross_Req[17];
	int nSchedule[17];
	int nProjectAvailable[17];
	int nNet_Req[17];
	int nPOReceipts[17];
	int nPOreleases[17];

	int nLeadTime = 2;
	int nOnHand = 50;
	int nSaftyStock = 0;
	int Quantity = 0;
};

struct Product_B
{
	int nGross_Req[17];
	int nSchedule[17];
	int nProjectAvailable[17];
	int nNet_Req[17];
	int nPOReceipts[17];
	int nPOreleases[17];
	
	int nLeadTime = 2;
	int nOnHand = 60;
	int nSaftyStock = 0;
	int Quantity = 0;
};

struct Product_C
{
	int nGross_Req[17];
	int nSchedule[17];
	int nProjectAvailable[17];
	int nNet_Req[17];
	int nPOReceipts[17];
	int nPOreleases[17];
	
	int nLeadTime = 1;
	int nOnHand = 40;
	int nSaftyStock = 5;
	int Quantity = 0;
};

struct Product_D
{
	int nGross_Req[17];
	int nSchedule[17];
	int nProjectAvailable[17];
	int nNet_Req[17];
	int nPOReceipts[17];
	int nPOreleases[17];
	
	int nLeadTime = 2;
	int nOnHand = 200;
	int nSaftyStock = 20;
	int Quantity = 0;
};