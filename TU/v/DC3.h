/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id$  
 */
#ifndef TU_V_DC3_H
#define TU_V_DC3_H

#include "TU/Geometry++.h"
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
    virtual DC3&	setExternal(const Point3d& t,
				    const Matrix33d& Rt)		= 0;
    virtual const DC3&	getInternal(int&    u0,	  int&	  v0,
				    double& ku,	  double& kv,
				    double& near, double& far)	const	= 0;
    virtual const DC3&	getExternal(Point3d& t, Matrix33d& Rt)	const	= 0;
    virtual DC3&	translate(double d)				;
    virtual DC3&	rotate(double angle)				= 0;

    
    friend OManip1<DC3, Axis>	axis(Axis axis)			;
    friend OManip1<DC3, double>	distance(double dist)		;
    friend OManip1<DC3, double>	translate(double d)		;
    friend OManip1<DC3, double>	rotate(double angle)		;
    
    Axis		getAxis()	const	{return _axis;}
    double		getDistance()	const	{return _distance;}

  protected:
    DC3&		setAxis(Axis axis)	{_axis = axis;  return *this;}
    DC3&		setDistance(double d)	{_distance = d; return *this;}
	
  private:
    Axis		_axis;
    double		_distance;
};

OManip1<DC3, DC3::Axis>	axis(DC3::Axis axis)		;
OManip1<DC3, double>	distance(double dist)		;
OManip1<DC3, double>	translate(double d)		;
OManip1<DC3, double>	rotate(double angle)		;

}
}
#endif	// !TU_V_DC3_H
