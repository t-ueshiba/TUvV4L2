/*
 *  平成19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  同所が著作権を所有する秘密情報です．著作者による許可なしにこのプロ
 *  グラムを第三者へ開示，複製，改変，使用する等の著作権を侵害する行為
 *  を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *  Copyright 2007
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Author: Toshio UESHIBA
 *
 *  Confidentail and all rights reserved.
 *  This program is confidential. Any changing, copying or giving
 *  information about the source code of any part of this software
 *  and/or documents without permission by the authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damages in the use of this program.
 *  
 *  $Id: Manip.h,v 1.3 2007-11-26 07:28:09 ueshiba Exp $
 */
#ifndef __TUManip_h
#define __TUManip_h

#include <iostream>

namespace TU
{
/************************************************************************
*  Manipulators								*
************************************************************************/
std::istream&	ign(std::istream& in)		;
std::istream&	skipl(std::istream&)		;

/************************************************************************
*  class IOManip							*
************************************************************************/
template <class S>
class IOManip
{
  public:
    IOManip(S& (S::*fi)(), S& (S::*fo)()) :_fi(fi), _fo(fo)	{}

    S&		iapply(S& s)		const	{return (s.*_fi)();}
    S&		oapply(S& s)		const	{return (s.*_fo)();}
    
  private:
    S&		(S::*_fi)();
    S&		(S::*_fo)();
};

template <class SS, class S> inline SS&
operator >>(SS& s, const IOManip<S>& m)
{
    m.iapply(s);
    return s;
}

template <class SS, class S> inline SS&
operator <<(SS& s, const IOManip<S>& m)
{
    m.oapply(s);
    return s;
}

/************************************************************************
*  class IManip1							*
************************************************************************/
template <class S, class T>
class IManip1
{
  public:
    IManip1(S& (S::*f)(T), T arg)	:_f(f), _arg(arg)	{}

    S&		apply(S& s)		const	{return (s.*_f)(_arg);}
    
  private:
    S&		(S::*_f)(T);
    const T	_arg;
};

template <class SS, class S, class T> inline SS&
operator >>(SS& s, const IManip1<S, T>& m)
{
    m.apply(s);
    return s;
}

/************************************************************************
*  class OManip1							*
************************************************************************/
template <class S, class T>
class OManip1
{
  public:
    OManip1(S& (S::*f)(T), T arg)	:_f(f), _arg(arg)	{}

    S&		apply(S& s)		const	{return (s.*_f)(_arg);}
    
  private:
    S&		(S::*_f)(T);
    const T	_arg;
};

template <class SS, class S, class T> inline SS&
operator <<(SS& s, const OManip1<S, T>& m)
{
    m.apply(s);
    return s;
}
    
/************************************************************************
*  class IOManip1							*
************************************************************************/
template <class S, class T>
class IOManip1
{
  public:
    IOManip1(S& (S::*fi)(T), S& (S::*fo)(T), T arg)
      :_fi(fi), _fo(fo), _arg(arg)				{}

    S&		iapply(S& s)		const	{return (s.*_fi)(_arg);}
    S&		oapply(S& s)		const	{return (s.*_fo)(_arg);}
    
  private:
    S&		(S::*_fi)(T);
    S&		(S::*_fo)(T);
    const T	_arg;
};

template <class SS, class S, class T> inline SS&
operator >>(SS& s, const IOManip1<S, T>& m)
{
    m.iapply(s);
    return s;
}

template <class SS, class S, class T> inline SS&
operator <<(SS& s, const IOManip1<S, T>& m)
{
    m.oapply(s);
    return s;
}

/************************************************************************
*  class IManip2							*
************************************************************************/
template <class S, class T, class U>
class IManip2
{
  public:
    IManip2(S& (S::*f)(T, U), T arg0, U arg1)
      :_f(f), _arg0(arg0), _arg1(arg1)				{}

    S&		apply(S& s)		const	{return (s.*_f)(_arg0, _arg1);}
    
  private:
    S&		(S::*_f)(T, U);
    const T	_arg0;
    const U	_arg1;
};

template <class SS, class S, class T, class U> inline SS&
operator >>(SS& s, const IManip2<S, T, U>& m)
{
    m.apply(s);
    return s;
}

/************************************************************************
*  class OManip2							*
************************************************************************/
template <class S, class T, class U>
class OManip2
{
  public:
    OManip2(S& (S::*f)(T, U), T arg0, U arg1)
      :_f(f), _arg0(arg0), _arg1(arg1)				{}

    S&		apply(S& s)		const	{return (s.*_f)(_arg0, _arg1);}
    
  private:
    S&		(S::*_f)(T, U);
    const T	_arg0;
    const U	_arg1;
};

template <class SS, class S, class T, class U> inline SS&
operator <<(SS& s, const OManip2<S, T, U>& m)
{
    m.apply(s);
    return s;
}
 
}

#endif	/* !__TUManip_h		*/
