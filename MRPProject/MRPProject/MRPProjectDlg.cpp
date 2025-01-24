
// MRPProjectDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "MRPProject.h"
#include "MRPProjectDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMRPProjectDlg 대화 상자



CMRPProjectDlg::CMRPProjectDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CMRPProjectDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMRPProjectDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_CUSTOM1, m_cGrid);
}

BEGIN_MESSAGE_MAP(CMRPProjectDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
END_MESSAGE_MAP()

Product_A theProd_A;
Product_B theProd_B;
Product_C theProd_C;
Product_D theProd_D;
// CMRPProjectDlg 메시지 처리기

BOOL CMRPProjectDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	InitialMPS();
	Cal_ProdA();
	Cal_ProdB();
	Cal_ProdC();
	Cal_ProdD();

	InitGridCtrl();

	Print();
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CMRPProjectDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CMRPProjectDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CMRPProjectDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CMRPProjectDlg::InitialMPS()
{
	theProd_A.nGross_Req[9] = 1250;
	theProd_A.nGross_Req[13] = 850;
	theProd_A.nGross_Req[17] = 550;
	
	theProd_B.nGross_Req[9] = 470;
	theProd_B.nGross_Req[13] = 360;
	theProd_B.nGross_Req[17] = 560;
	
	theProd_D.nGross_Req[9] = 270;
	theProd_D.nGross_Req[13] = 250;
	theProd_D.nGross_Req[17] = 320;

	//가지고 있는 것이기 때문에 모든 주차에 넣기
	for (int i = 0; i <= 17; i++)
	{
		theProd_A.nProjectAvailable[i] = theProd_A.nOnHand;
		theProd_B.nProjectAvailable[i] = theProd_B.nOnHand;
		theProd_C.nProjectAvailable[i] = theProd_C.nOnHand;
		theProd_D.nProjectAvailable[i] = theProd_D.nOnHand;
	}
}

void CMRPProjectDlg::Cal_ProdA()
{
	for (int i = 0; i <= 17; i++)
	{
		//만약 총소요량이 있다면 LeadTime을 뺀 주차에 계획발주 넣기
		//넣을 때 전주차에 있는 예상 재고를 확인하고 계획 발주 넣기
		if(theProd_A.nGross_Req[i] > 0)
		{
			if(theProd_A.nGross_Req[i] > theProd_A.nProjectAvailable[i - 1])
			{
				theProd_A.nPOreleases[i - theProd_A.nLeadTime] =  theProd_A.nGross_Req[i] - theProd_A.nProjectAvailable[i - 1];
			}
			else
			{
				theProd_A.nProjectAvailable[i] = theProd_A.nProjectAvailable[i - 1] - theProd_A.nGross_Req[i];
			}
		}

		if (theProd_A.nPOreleases[i] > 0)
		{
			theProd_A.nPOReceipts[i + theProd_A.nLeadTime] = theProd_A.nPOreleases[i];
			theProd_A.nNet_Req[i + theProd_A.nLeadTime] = theProd_A.nGross_Req[i + theProd_A.nLeadTime] - theProd_A.nProjectAvailable[i + theProd_A.nLeadTime - 1];
			theProd_A.nProjectAvailable[i + theProd_A.nLeadTime] = theProd_A.nNet_Req[i + theProd_A.nLeadTime] - theProd_A.nPOReceipts[i + theProd_A.nLeadTime];
		}
	}
}

void CMRPProjectDlg::Cal_ProdB()
{
	for (int i = 0; i <= 17; i++)
	{
		//만약 총소요량이 있다면 LeadTime을 뺀 주차에 계획발주 넣기
		//넣을 때 전주차에 있는 예상 재고를 확인하고 계획 발주 넣기
		if(theProd_B.nGross_Req[i] > 0)
		{
			if(theProd_B.nGross_Req[i] > theProd_B.nProjectAvailable[i - 1])
			{
				theProd_B.nPOreleases[i - theProd_B.nLeadTime] =  theProd_B.nGross_Req[i] - theProd_B.nProjectAvailable[i - 1];
			}
			else
			{
				theProd_B.nProjectAvailable[i] = theProd_B.nProjectAvailable[i - 1] - theProd_B.nGross_Req[i];
			}
		}

		if (theProd_B.nPOreleases[i] > 0)
		{
			theProd_B.nPOReceipts[i + theProd_B.nLeadTime] = theProd_B.nPOreleases[i];
			theProd_B.nNet_Req[i + theProd_B.nLeadTime] = theProd_B.nGross_Req[i + theProd_B.nLeadTime] - theProd_B.nProjectAvailable[i + theProd_B.nLeadTime - 1];
			theProd_B.nProjectAvailable[i + theProd_B.nLeadTime] = theProd_B.nNet_Req[i + theProd_B.nLeadTime] - theProd_B.nPOReceipts[i + theProd_B.nLeadTime];
		}
	}
}

void CMRPProjectDlg::Cal_ProdC()
{
	for (int i = 0; i <= 17; i++)
	{
		//만약 총소요량이 있다면 LeadTime을 뺀 주차에 계획발주 넣기
		//넣을 때 전주차에 있는 예상 재고를 확인하고 계획 발주 넣기
		if(theProd_A.nPOreleases[i] > 0 || theProd_B.nPOreleases[i])
		{
			theProd_C.nGross_Req[i] = theProd_C.nGross_Req[i] + theProd_A.nPOreleases[i] + theProd_B.nPOreleases[i];
		}

		if(theProd_C.nGross_Req[i] > 0)
		{
			if(theProd_C.nGross_Req[i] > theProd_C.nProjectAvailable[i - 1])
			{
				theProd_C.nPOreleases[i - theProd_C.nLeadTime] =  2000;
			}
			else
			{
				theProd_C.nProjectAvailable[i] = theProd_C.nProjectAvailable[i - 1] - theProd_C.nGross_Req[i];
			}
		}

		if (theProd_C.nPOreleases[i] > 0)
		{
			theProd_C.nPOReceipts[i + theProd_C.nLeadTime] = theProd_C.nPOreleases[i];
			theProd_C.nNet_Req[i + theProd_C.nLeadTime] = theProd_C.nGross_Req[i + theProd_C.nLeadTime] - theProd_C.nProjectAvailable[i + theProd_C.nLeadTime - 1];
			theProd_C.nProjectAvailable[i + theProd_C.nLeadTime] = theProd_C.nNet_Req[i + theProd_C.nLeadTime] - theProd_C.nPOReceipts[i + theProd_C.nLeadTime];
		}
	}
}

void CMRPProjectDlg::Cal_ProdD()
{
	for (int i = 0; i <= 17; i++)
	{
		//만약 총소요량이 있다면 LeadTime을 뺀 주차에 계획발주 넣기
		//넣을 때 전주차에 있는 예상 재고를 확인하고 계획 발주 넣기
		
		if(theProd_A.nPOreleases[i] > 0 || theProd_C.nPOreleases[i])
		{
			theProd_D.nGross_Req[i] = theProd_D.nGross_Req[i] + theProd_A.nPOreleases[i] + (theProd_C.nPOreleases[i] * 2);
		}

		if(theProd_D.nGross_Req[i] > 0)
		{
			if(theProd_D.nGross_Req[i] > theProd_D.nProjectAvailable[i - 1])
			{
				theProd_D.nPOreleases[i - theProd_D.nLeadTime] =  5000;
			}
			else
			{
				theProd_D.nProjectAvailable[i] = theProd_D.nProjectAvailable[i - 1] - theProd_D.nGross_Req[i];
			}
		}

		if (theProd_D.nPOreleases[i] > 0)
		{
			theProd_D.nPOReceipts[i + theProd_D.nLeadTime] = theProd_D.nPOreleases[i];
			theProd_D.nNet_Req[i + theProd_D.nLeadTime] = theProd_D.nGross_Req[i + theProd_D.nLeadTime] - theProd_D.nProjectAvailable[i + theProd_D.nLeadTime - 1];
			theProd_D.nProjectAvailable[i + theProd_D.nLeadTime] = theProd_D.nNet_Req[i + theProd_D.nLeadTime] - theProd_D.nPOReceipts[i + theProd_D.nLeadTime];
		}
	}
}

void CMRPProjectDlg::InitGridCtrl()
{
	DWORD dwTextStyle = DT_CENTER | DT_VCENTER | DT_SINGLELINE;
	GV_ITEM Item;
	int nRow, i;
	CString str;


	//-- LOT NO ---------------------------------------------
	m_cGrid.SetEditable(FALSE);
	m_cGrid.SetListMode(TRUE);
	m_cGrid.EnableDragAndDrop(TRUE);
	m_cGrid.SetTextBkColor(RGB(255, 255, 255));
	m_cGrid.SetRowCount(18);
	m_cGrid.SetColumnCount(21);
	m_cGrid.SetHeaderSort(FALSE);
	m_cGrid.SetFixedColumnCount(1);
	m_cGrid.SetFixedRowCount(1);
	m_cGrid.SetFixedColumnSelection(FALSE);
	m_cGrid.SetFixedRowSelection(FALSE);
	m_cGrid.SetSingleRowSelection(FALSE);
	m_cGrid.EnableSelection(FALSE);
	m_cGrid.EnableScrollBar(FALSE);


	for(int i = 0; i < 21; i++)
	{
		m_cGrid.SetColumnWidth(i, 100);
		
	
	}
	m_cGrid.SetFont(GetFont());
	m_cGrid.SetColumnResize(FALSE);
	m_cGrid.SetRowResize(TRUE);

	Item.mask = GVIF_TEXT | GVIF_FORMAT;
	Item.col = 0;
	Item.nFormat = dwTextStyle;
	
	for (int i = 0; i < 19; i++)
	{
		Item.col = 0;
		Item.row = i + 1;
		Item.strText.Format(_T("%02d주차"), i+1);

		m_cGrid.SetItem(&Item);
	}

	for (int i = 0; i < 4; i++)
	{
		CString str;

		switch (i)
		{
			case 0:
			{
				str = _T("_A");
				break;
			}
			case 1:
			{
				str = _T("_B");
				break;
			}
			case 2:
			{
				str = _T("_C");
				break;
			}
			case 3:
			{
				str = _T("_D");
				break;
			}
			default:
				break;
		}

		Item.col = 5 * i + 1;
		Item.row = 0;
		Item.strText.Format(_T("총소요량%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 2;
		Item.row = 0;
		Item.strText.Format(_T("예정입고%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 3;
		Item.row = 0;
		Item.strText.Format(_T("예상재고%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 4;
		Item.row = 0;
		Item.strText.Format(_T("순소요량%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 5;
		Item.row = 0;
		Item.strText.Format(_T("계획수주%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 6;
		Item.row = 0;
		Item.strText.Format(_T("계획발주%s"), str);
		m_cGrid.SetItem(&Item); 
	}

	m_cGrid.Invalidate(FALSE);
}

void CMRPProjectDlg::Print()
{
	DWORD dwTextStyle = DT_CENTER | DT_VCENTER | DT_SINGLELINE;
	GV_ITEM Item;	
	
	m_cGrid.SetFont(GetFont());

	Item.mask = GVIF_TEXT | GVIF_FORMAT;
	Item.nFormat = dwTextStyle;

	for (int i = 0; i < 17; i++)
	{
		Item.col = 1;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_A.nGross_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 2;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_A.nSchedule[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 3;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_A.nProjectAvailable[i]);
		
		Item.col = 4;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_A.nNet_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 5;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_A.nPOReceipts[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 6;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_A.nPOreleases[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 7;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_B.nGross_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 8;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_B.nSchedule[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 9;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_B.nProjectAvailable[i]);
		
		Item.col = 10;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_B.nNet_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 11;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_B.nPOReceipts[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 12;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_B.nPOreleases[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 13;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_C.nGross_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 14;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_C.nSchedule[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 15;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_C.nProjectAvailable[i]);
		
		Item.col = 16;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_C.nNet_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 17;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_C.nPOReceipts[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 18;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_C.nPOreleases[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 19;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_D.nGross_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 20;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_D.nSchedule[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 21;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_D.nProjectAvailable[i]);
		
		Item.col = 22;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_D.nNet_Req[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 23;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_D.nPOReceipts[i]);
		m_cGrid.SetItem(&Item); 
		
		Item.col = 24;
		Item.row = i + 1;
		Item.strText.Format(_T("%d"),theProd_D.nPOreleases[i]);
		m_cGrid.SetItem(&Item); 

	}
	
	m_cGrid.Invalidate(FALSE);
}