/*
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
