/*
 *  平成9年 電子技術総合研究所 植芝俊夫 著作権所有
 *
 *  著作者による許可なしにこのプログラムの第三者への開示、複製、改変、
 *  使用等その他の著作人格権を侵害する行為を禁止します。
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *
 *  Copyright 1996
 *  Toshio UESHIBA, Electrotechnical Laboratory
 *
 *  All rights reserved.
 *  Any changing, copying or giving information about source programs of
 *  any part of this software and/or documentation without permission of the
 *  authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damage in use of this program.
 */

/*
 *  $Id: plane_motion.cc,v 1.2 2002-07-25 02:38:01 ueshiba Exp $
 */
#include "TU/Calib++.h"

namespace TU
{
void
plane_motion(const Matrix<double>& ldata, const Matrix<double>& rdata,
	     Matrix<double>& Rt, Vector<double>& t, Vector<double>& n)
{
  /* compute A matrix (rotation, translation and plane normal are contained) */
    const int		ndata = ldata.nrow();
    Matrix<double>	P(9, 9);
    for (int i = 0; i < ndata; ++i)
    {
	Matrix<double>	Qt(9, 9);
	
	Qt[0](3, 3) -= rdata[i][2] * ldata[i];
	Qt[0](6, 3) += rdata[i][1] * ldata[i];
	Qt[1](0, 3) += rdata[i][2] * ldata[i];
	Qt[1](6, 3) -= rdata[i][0] * ldata[i];
	Qt[2](0, 3) -= rdata[i][1] * ldata[i];
	Qt[2](3, 3) += rdata[i][0] * ldata[i];

	P += Qt.trns() * Qt;
    }

    Matrix<double>	A(3, 3);
    Vector<double>	a((double*)A, 9);
    Vector<double>	evalue(9);
    a = P.eigen(evalue)[8];

#ifdef DEBUG
	std::cerr << "-----------------" << std::endl
		  << "  Eigen values = " << evalue;
#endif

  /* decompose A into Rt, t and n */
    Matrix<double>	AA = A * A.trns();
    Vector<double>	lambda(3);
    Matrix<double>	Vt = AA.eigen(lambda);

    
}
 
}
