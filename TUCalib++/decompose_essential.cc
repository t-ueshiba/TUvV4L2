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
 *  $Id: decompose_essential.cc,v 1.1 2002-08-09 01:28:05 ueshiba Exp $
 */
#include "TU/Calib++.h"

namespace TU
{
inline int
sign(double x)
{
    return (x > 0.0 ? 1 : x < 0.0 ? -1 : 0);
}

void
decompose_essential(const Matrix<double>& E,
		    const Matrix<double>& ldata,
		    const Matrix<double>& rdata,
		    Matrix<double>& Rt, Vector<double>& t)
{
    using namespace	std;
    
    SVDecomposition<double>	svd(E);
#ifdef DEBUG
    cerr << "=== BEGIN: decompose_essential ===\n"
	 << "  Singular values of E = " << svd.diagonal();
#endif

  /* Quasi identity matrix (det(I) = 1) */
    Matrix<double>	I(3, 3);
  /*    I[0][1] =  svd[1];
    I[1][0] = -svd[0];
    I[2][2] = (svd[0] + svd[1]) / 2;*/
    I[0][1] =  1.0;
    I[1][0] = -1.0;
    I[2][2] =  1.0;
    double	scale = pow(-I[0][1] * I[1][0] * I[2][2], 1.0 / 3);
    I /= scale;
#ifdef DEBUG
    cerr << " --- I ---\n" << I;
#endif

  /* Make rotation and translation */
    Rt = svd.Vt().trns() * I * svd.Ut();
    t = svd.Ut()[2];
#ifdef DEBUG
    cerr << " --- E ---\n" << E;
    cerr << " --- Rt * t.skew() ---\n" << Rt * t.skew();
#endif

  /* Reject spurious solutions */
    Matrix<double>	S = Rt * (Matrix<double>::I(3) - t % t);
    const u_int		ndata = ldata.nrow();
    double		sum = 0.0;
    for (int i = 0; i < ndata; ++i)
	sum += rdata[i] * S * ldata[i];
    if (sum < 0.0)
	Rt = Rt * (2.0 * (t % t) - Matrix<double>::I(3));
    int	sig = 0;
    for (int i = 0; i < ndata; ++i)
    {
	Matrix<double>	D(2, 2);
	D[0][0] = ldata[i] * ldata[i];
	D[0][1] = D[1][0] = -rdata[i] * Rt * ldata[i];
	D[1][1] = rdata[i] * Rt * Rt.trns() * rdata[i];
	Vector<double>	d(2);
	d[0] =  ldata[i] * t;
	d[1] = -rdata[i] * Rt * t;
	d.solve(D);
      /*#ifdef DEBUG
	cerr << " --- D ---\n" << D << "  d = " << d
	     << "    depth = " << d;
#endif*/
	sig += sign(d[0]) + sign(d[1]);
    }
    if (sig < 0)
    {
	t = -t;
      /*#ifdef DEBUG
	cerr << "*** tr reversed!! ***" << endl;
#endif*/
    }
#ifdef DEBUG
    cerr << "=== END:   decompose_essential ===" << endl;
#endif
}
 
}
