
// MRPProject.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// CMRPProjectApp:
// �� Ŭ������ ������ ���ؼ��� MRPProject.cpp�� �����Ͻʽÿ�.
//

class CMRPProjectApp : public CWinApp
{
public:
	CMRPProjectApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CMRPProjectApp theApp;