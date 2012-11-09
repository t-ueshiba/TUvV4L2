/*
 *  $Id$
 */
#include "TU/Calib++.h"

namespace TU
{
Matrix<double>
projection(const Matrix<double>& data)
{
    Normalization	norm3(data(0, 0, data.nrow(), 4)),
			norm2(data(0, 4, data.nrow(), 3));
    Matrix<double>	data3 = data(0, 0, data.nrow(), 4) * norm3.Tt(),
			data2 = data(0, 4, data.nrow(), 3) * norm2.Tt();
    Matrix<double>	A(12, 12);
    for (int i = 0; i < data.nrow(); ++i)
    {
	Matrix<double>	Ct(2, 12);
	
	Ct[0](0, 4) = Ct[1](4, 4) = data3[i];
	Ct[0](8, 4) = -data2[i][0] * data3[i];
	Ct[1](8, 4) = -data2[i][1] * data3[i];

	A += Ct.trns() * Ct;
    }

    Matrix<double>	P(3, 4);
    Vector<double>	p((double*)P, 12);
    Vector<double>	evalue;
    p = A.eigen(evalue)[11];
#ifdef DEBUG
    std::cerr << "Eigen values: " << evalue;
#endif
    return norm2.Tinv() * P * norm3.T();
}
 
}
