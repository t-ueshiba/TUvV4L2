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
 *  $Id$
 */
#include "TU/Calib++.h"

namespace TU
{
void
assess_projection(const Matrix<double>& data, const Matrix<double>& P)
{
    using namespace	std;
	
    const u_int	ndata = data.nrow();
    double	d_sum = 0.0;
    
    cerr << "=== Errror assessment ===" << endl;
    for (int i = 0; i < ndata; ++i)
    {
	Vector<double>	u = P * data[i](0, 4);
	u /= u[2];
	cerr << " -- Point: " << i << " --" << endl;
	double	d = u(0, 2).dist(data[i](4, 2));
	cerr << "  Distance = " << d << " pixel" << endl;
	d_sum += d;
    }
    cerr << " ** Average distance = " << d_sum / ndata << " pixel **" << endl;
}
 
}
