/*
 *  $Id: S2200++.h,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#ifndef __TUS2200PP_h
#define __TUS2200PP_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class S2200								*
************************************************************************/
class S2200 : private Array2<Array<ABGR> >
{
  public:
    enum Channel	{OVERLAY = 0, BLUE = 1, GREEN = 2, RED = 3};
    enum		{DEFAULT_WIDTH = 640, DEFAULT_HEIGHT = 480};
    
    S2200(const char*)					;
    virtual ~S2200()					;

    Array2<Array<ABGR> >::operator [];
    
		operator const void*()		const	;
    int		operator !()			const	;
    u_int	u0()				const	{return _u0;}
    u_int	v0()				const	{return _v0;}
    u_int	width()				const	{return _w;}
    u_int	height()			const	{return _h;}
    Channel	channel()			const	{return _channel;}
    S2200&	set_roi(u_int, u_int, u_int, u_int)	;
    S2200&	set_channel(Channel)			;
    S2200&	flow()					;
    S2200&	freeze()				;
    S2200&	capture()				;
    S2200&	operator >>(Image<ABGR>&)		;
    S2200&	operator >>(Image<u_char>&)		;
    S2200&	operator <<(S2200& (*)(S2200&))	;

  private:
    S2200(const S2200&)				;
    S2200&		operator =(const S2200&)	;
 
    u_char*		mapon(u_int, u_int)		;
    virtual void	set_rows()			;
    
    int				_fd;
    volatile u_char* const	_vcon;
    volatile u_char* const	_vin;
    volatile u_char* const	_vmap;
    volatile u_char* const	_vout;
    volatile ABGR* const	_vram;
    u_int			_u0, _v0, _w, _h;
    Channel			_channel;
};

inline
S2200::operator const void*() const
{
    return (_fd != -1 ? this : 0);
}

inline int
S2200::operator !() const
{
    return !(operator const void*());
}

inline S2200&
S2200::operator <<(S2200& (*f)(S2200&))
{
    return (*f)(*this);
}

/************************************************************************
*  Manipulatros								*
************************************************************************/
inline S2200&
flow(S2200& s2200)
{
    return s2200.flow();
}

inline S2200&
freeze(S2200& s2200)
{
    return s2200.freeze();
}

inline S2200&
capture(S2200& s2200)
{
    return s2200.capture();
}

inline S2200&
red(S2200& s2200)
{
    return s2200.set_channel(S2200::RED);
}

inline S2200&
green(S2200& s2200)
{
    return s2200.set_channel(S2200::GREEN);
}

inline S2200&
blue(S2200& s2200)
{
    return s2200.set_channel(S2200::BLUE);
}

inline S2200&
overlay(S2200& s2200)
{
    return s2200.set_channel(S2200::OVERLAY);
}
 
}
#endif	/* !__TUS2200PP_h	*/
