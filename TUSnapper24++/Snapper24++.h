/*
 *  $Id: Snapper24++.h,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#ifndef __TUSnapper24PP_h
#define __TUSnapper24PP_h

#include <asl_inc.h>
extern "C" {
Terr EXPORT_FN	SNP24_read_video_data_raw(Thandle, void*, ui32);
Terr EXPORT_FN	SNP24_get_module_format(Thandle);
Terr EXPORT_FN	SNP24_get_TMG_format(Thandle);
}
#include <stdarg.h>
#include "TU/v/XilDC.h"

namespace TU
{
/************************************************************************
*  class Snapper							*
************************************************************************/
class Snapper24
{
  public:
    enum Channel	{ChRED, ChGREEN, ChBLUE, ChBGR};
    
    Snapper24(u_int, ...)					;
    ~Snapper24()						;

    u_int	board_number()				const	;
    		operator const void*()			const	;
    int		operator !()				const	;
    u_int	u0()					const	;
    u_int	v0()					const	;
    u_int	width()					const	;
    u_int	height()				const	;
    Channel	channel()				const	;
    Snapper24&	set_roi(u_int, u_int, u_int, u_int)		;
    Snapper24&	set_channel(Channel)				;
    Snapper24&	capture()					;
    Snapper24&	operator >>(Image<u_char>&)			;
    Snapper24&	operator >>(Image<BGR>&)			;
    Snapper24&	operator >>(Image<ABGR>&)			;
    Snapper24&	operator >>(Image<RGB>&)			;
    Snapper24&	operator >>(Image<RGBA>&)			;
    Snapper24&	operator <<(Snapper24& (*)(Snapper24&))		;
    Terr	status()				const	;
    Thandle	handle()				const	;
    Snapper24&	set(...)					;
    
  private:
    Snapper24(const Snapper24&)					;
    Snapper24&	operator =(const Snapper24&)			;

    void	set_attributes(va_list)				;
    u_int	to_outer_coord(u_int)			const	;
    Snapper24&	wait_until_capture_complete(float=0.25)		;
    Snapper24&	read_raw(void*, u_int)				;
    
    const u_int		_board_number;
    Terr		_status;
    const Terr		_opened;
    const Thandle	_Hbase;
    const Thandle	_Hsnp24;
};

inline u_int
Snapper24::board_number() const
{
    return _board_number;
}

inline
Snapper24::operator const void*() const
{
    return (ASL_is_ok(_status) ? this : 0);
}

inline int
Snapper24::operator !() const
{
    return !(operator const void*());
}

inline u_int
Snapper24::u0() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[0]);
}

inline u_int
Snapper24::v0() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[1]);
}

inline u_int
Snapper24::width() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[2]);
}

inline u_int
Snapper24::height() const
{
    i16	roi[ASL_SIZE_2D_ROI];
    SNP24_get_ROI(_Hsnp24, roi);
    return to_outer_coord(roi[3]);
}

inline Snapper24&
Snapper24::capture()
{
    _status = SNP24_capture(_Hsnp24, SNP24_START_AND_RETURN);
    return *this;
}

inline Snapper24&
Snapper24::operator <<(Snapper24& (*f)(Snapper24&))
{
    return (*f)(*this);
}

inline Terr
Snapper24::status() const
{
    return _status;
}

inline Thandle
Snapper24::handle() const
{
    return _Hsnp24;
}


/************************************************************************
*  Manipulators								*
************************************************************************/
inline Snapper24&
x1(Snapper24& snp24)
{
    return snp24.set("SNP24_CAPRE", SNP24_SUB_X1);
}

inline Snapper24&
x2(Snapper24& snp24)
{
    return snp24.set("SNP24_CAPRE", SNP24_SUB_X2);
}

inline Snapper24&
x4(Snapper24& snp24)
{
    return snp24.set("SNP24_CAPRE", SNP24_SUB_X4);
}

inline Snapper24&
red(Snapper24& snp24)
{
    return snp24.set_channel(Snapper24::ChRED);
}

inline Snapper24&
green(Snapper24& snp24)
{
    return snp24.set_channel(Snapper24::ChGREEN);
}

inline Snapper24&
blue(Snapper24& snp24)
{
    return snp24.set_channel(Snapper24::ChBLUE);
}

inline Snapper24&
capture(Snapper24& snp24)
{
    return snp24.capture();
}

/************************************************************************
*  class Snapper24Image<T>						*
************************************************************************/
template <class T>
class Snapper24Image : public XilImage<T>
{
  public:
    Snapper24Image(u_int w=1, u_int h=1)
	:XilImage<T>(memalign(sizeof(T) * w * h), w, h)			{}
    ~Snapper24Image()							;

    void		resize(u_int, u_int)				;

  protected:
    void		resize(T*, u_int, u_int)			;

  private:
    static T*		memalign(size_t)				;
};
 
}
#endif	/* !__TUSnapper24PP_h	*/
