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
 *  $Id: Array++.inst.cc,v 1.3 2002-12-18 05:46:12 ueshiba Exp $
 */
#if defined __GNUG__ || defined __INTEL_COMPILER

#include "TU/Array++.cc"

namespace TU
{
template class Array<char>;
template class Array<u_char>;
template class Array<short>;
template class Array<u_short>;
template class Array<int>;
template class Array<u_int>;
template class Array<long>;
template class Array<u_long>;
template class Array<float>;
template class Array<double>;

template class Array2<Array<char> >;
template class Array2<Array<u_char> >;
template class Array2<Array<short> >;
template class Array2<Array<u_short> >;
template class Array2<Array<int> >;
template class Array2<Array<u_int> >;
template class Array2<Array<long> >;
template class Array2<Array<u_long> >;
template class Array2<Array<float> >;
template class Array2<Array<double> >;

template std::ostream&	operator <<(std::ostream&, const Array<char>&);
template std::ostream&	operator <<(std::ostream&, const Array<u_char>&);
template std::ostream&	operator <<(std::ostream&, const Array<short>&);
template std::ostream&	operator <<(std::ostream&, const Array<u_short>&);
template std::ostream&	operator <<(std::ostream&, const Array<int>&);
template std::ostream&	operator <<(std::ostream&, const Array<u_int>&);
template std::ostream&	operator <<(std::ostream&, const Array<long>&);
template std::ostream&	operator <<(std::ostream&, const Array<u_long>&);
template std::ostream&	operator <<(std::ostream&, const Array<float>&);
template std::ostream&	operator <<(std::ostream&, const Array<double>&);
}

#endif
