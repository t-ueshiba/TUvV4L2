/*
 *  $Id: DC3.h,v 1.2 2002-07-25 02:38:11 ueshiba Exp $
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
				    double near, double far)		= 0;
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
