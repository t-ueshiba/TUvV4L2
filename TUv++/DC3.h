/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: DC3.h,v 1.4 2007-11-26 08:11:50 ueshiba Exp $
 */
#ifndef __TUvDC3_h
#define __TUvDC3_h

#include "TU/Vector++.h"
#include "TU/Manip.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class DC3:		coorindate system for drawing			*
************************************************************************/
class DC3
{
  public:
    enum Axis	{X, Y, Z};
    
  public:
    DC3(Axis axis, double dist) :_axis(axis), _distance(dist)		{}
    virtual		~DC3()						;
    
    virtual DC3&	setInternal(int	   u0,	 int	v0,
				    double ku,	 double kv,
				    double near, double far=0.0)	= 0;
    virtual DC3&	setExternal(const Vector<double>& t,
				    const Matrix<double>& Rt)		= 0;
    virtual const DC3&	getInternal(int&    u0,	  int&	  v0,
				    double& ku,	  double& kv,
				    double& near, double& far)	const	= 0;
    virtual const DC3&	getExternal(Vector<double>& t,
				    Matrix<double>& Rt)		const	= 0;
    virtual DC3&	translate(double d)				;
    virtual DC3&	rotate(double angle)				= 0;

    
    friend OManip1<DC3, Axis>	axis(Axis)			;
    friend OManip1<DC3, double>	distance(double)		;
    friend OManip1<DC3, double>	translate(double)		;
    friend OManip1<DC3, double>	rotate(double)			;
    
    Axis		getAxis()	const	{return _axis;}
    double		getDistance()	const	{return _distance;}

  protected:
    DC3&		setAxis(Axis axis)	{_axis = axis;  return *this;}
    DC3&		setDistance(double d)	{_distance = d; return *this;}
	
  private:
    Axis		_axis;
    double		_distance;
};

}
}
#endif	// !__TUvDC3_h
