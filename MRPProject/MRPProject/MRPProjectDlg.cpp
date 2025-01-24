
// MRPProjectDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "MRPProject.h"
#include "MRPProjectDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
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


// CMRPProjectDlg ��ȭ ����



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
// CMRPProjectDlg �޽��� ó����

BOOL CMRPProjectDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
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

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
	InitialMPS();
	Cal_ProdA();
	Cal_ProdB();
	Cal_ProdC();
	Cal_ProdD();

	InitGridCtrl();

	Print();
	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
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

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CMRPProjectDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
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

	//������ �ִ� ���̱� ������ ��� ������ �ֱ�
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
		//���� �Ѽҿ䷮�� �ִٸ� LeadTime�� �� ������ ��ȹ���� �ֱ�
		//���� �� �������� �ִ� ���� ��� Ȯ���ϰ� ��ȹ ���� �ֱ�
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
		//���� �Ѽҿ䷮�� �ִٸ� LeadTime�� �� ������ ��ȹ���� �ֱ�
		//���� �� �������� �ִ� ���� ��� Ȯ���ϰ� ��ȹ ���� �ֱ�
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
		//���� �Ѽҿ䷮�� �ִٸ� LeadTime�� �� ������ ��ȹ���� �ֱ�
		//���� �� �������� �ִ� ���� ��� Ȯ���ϰ� ��ȹ ���� �ֱ�
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
		//���� �Ѽҿ䷮�� �ִٸ� LeadTime�� �� ������ ��ȹ���� �ֱ�
		//���� �� �������� �ִ� ���� ��� Ȯ���ϰ� ��ȹ ���� �ֱ�
		
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
		Item.strText.Format(_T("%02d����"), i+1);

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
		Item.strText.Format(_T("�Ѽҿ䷮%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 2;
		Item.row = 0;
		Item.strText.Format(_T("�����԰�%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 3;
		Item.row = 0;
		Item.strText.Format(_T("�������%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 4;
		Item.row = 0;
		Item.strText.Format(_T("���ҿ䷮%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 5;
		Item.row = 0;
		Item.strText.Format(_T("��ȹ����%s"), str);
		m_cGrid.SetItem(&Item);
		
		Item.col = 5 * i + 6;
		Item.row = 0;
		Item.strText.Format(_T("��ȹ����%s"), str);
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