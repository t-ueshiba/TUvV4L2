/*
 *  $Id: TUSnapper24++.h,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#ifndef __TUSnapper24PP_h
#define __TUSnapper24PP_h

#include <stdarg.h>
#include "TUsnp24.h"
#include "TUXil++.h"

/************************************************************************
*  class TUSnapper							*
************************************************************************/
class TUSnapper24
{
  public:
    enum Channel	{RED, GREEN, BLUE, BGR};
    
    TUSnapper24(u_int, ...)						;
    ~TUSnapper24()							;

    u_int		board_number()				const	;
    			operator const void*()			const	;
    int			operator !()				const	;
    u_int		u0()					const	;
    u_int		v0()					const	;
    u_int		width()					const	;
    u_int		height()				const	;
    Channel		channel()				const	;
    TUSnapper24&	set_roi(u_int, u_int, u_int, u_int)		;
    TUSnapper24&	set_channel(Channel)				;
    TUSnapper24&	capture()					;
    TUSnapper24&	operator >>(TUImage<u_char>&)			;
    TUSnapper24&	operator >>(TUImage<TUbgr>&)			;
    TUSnapper24&	operator >>(TUImage<TUabgr>&)			;
    TUSnapper24&	operator >>(TUImage<TUrgb>&)			;
    TUSnapper24&	operator >>(TUImage<TUrgba>&)			;
    TUSnapper24&	operator <<(TUSnapper24& (*)(TUSnapper24&))	;
    Terr		status()				const	;
    Thandle		handle()				const	;
    TUSnapper24&	set(...)					;
    
  private:
    TUSnapper24(const TUSnapper24&)					;
    TUSnapper24&	operator =(const TUSnapper24&)			;

    void		set_attributes(va_list)				;
    u_int		to_outer_coord(u_int)			const	;
    TUSnapper24&	wait_until_capture_complete(float=0.25)		;
    TUSnapper24&	read_raw(void*, u_int)				;
    
    const u_int		_board_number;
    Terr		_status;
    const Terr		_opened;
    const Thandle	_Hbase;
    const Thandle	_Hsnp24;
};

inline u_int
TUSnapper24::board_number() const
{
    return _board_number;
}

inline
TUSnapper24::operator const void*() const
{
    return (ASL_is_ok(_status) ? this : 0);
}

inline int
TUSnapper24::operator !() const
{
    return !(operator const void*());
}

inline u_int
TUSnapper24::u0() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[0]);
}

inline u_int
TUSnapper24::v0() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[1]);
}

inline u_int
TUSnapper24::width() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[2]);
}

inline u_int
TUSnapper24::height() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[3]);
}

inline TUSnapper24&
TUSnapper24::capture()
{
    _status = SNP24_capture(_Hsnp24, SNP24_START_AND_RETURN);
    return *this;
}

inline TUSnapper24&
TUSnapper24::operator <<(TUSnapper24& (*f)(TUSnapper24&))
{
    return (*f)(*this);
}

inline Terr
TUSnapper24::status() const
{
    return _status;
}

inline Thandle
TUSnapper24::handle() const
{
    return _Hsnp24;
}


/************************************************************************
*  Manipulators								*
************************************************************************/
inline TUSnapper24&
x1(TUSnapper24& snp24)
{
    return snp24.set("SNP24_CAPTURE", SNP24_SUB_X1);
}

inline TUSnapper24&
x2(TUSnapper24& snp24)
{
    return snp24.set("SNP24_CAPTURE", SNP24_SUB_X2);
}

inline TUSnapper24&
x4(TUSnapper24& snp24)
{
    return snp24.set("SNP24_CAPTURE", SNP24_SUB_X4);
}

inline TUSnapper24&
red(TUSnapper24& snp24)
{
    return snp24.set_channel(TUSnapper24::RED);
}

inline TUSnapper24&
green(TUSnapper24& snp24)
{
    return snp24.set_channel(TUSnapper24::GREEN);
}

inline TUSnapper24&
blue(TUSnapper24& snp24)
{
    return snp24.set_channel(TUSnapper24::BLUE);
}

inline TUSnapper24&
capture(TUSnapper24& snp24)
{
    return snp24.capture();
}

/************************************************************************
*  class TUSnapper24Image<T>						*
************************************************************************/
template <class T>
class TUSnapper24Image : public TUXilImage<T>
{
  public:
    TUSnapper24Image(u_int w=1, u_int h=1)
	:TUArray<TUNumericalArray<T,double> >(h),
	 TUXilImage<T>(memalign(sizeof(T) * w * h), w, h)		{}
    ~TUSnapper24Image()							;

    void		resize(u_int, u_int)				;

  protected:
    void		resize(T*, u_int, u_int)			;

  private:
    static T*		memalign(size_t)				;
};

#endif	/* !__TUSnapper24PP_h	*/
