/*
 *  $Id: Snapper24.cc,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#include <cstring>
#include "TU/Snapper24++.h"

namespace TU
{
/************************************************************************
*  Public member functions						*
************************************************************************/
Snapper24::Snapper24(u_int board_number, ...)
    :_board_number(board_number),
     _status(BASE_create(BASE_DEVICE | _board_number)),
     _opened(ASL_is_ok(_status)),
     _Hbase(ASL_get_ret(_status)),
     _Hsnp24(BASE_get_parameter(_Hbase, BASE_MODULE_HANDLE))
{
    if (!*this || _Hsnp24 == MODULE_NULL_HANDLE)
	return;
    
    SNP24_initialize(_Hsnp24, SNP24_EIA_DEFAULT);
    SNP24_set_sync(_Hsnp24, SNP24_SYNC_OFF_CSYNC_NEG);
    SNP24_set_format(_Hsnp24, SNP24_FORMAT_Y8_ON_RED, TMG_Y8);

    va_list	args;
    va_start(args, board_number);
    set_attributes(args);
    va_end(args);
    
    if (!*this)
	return;

    SNP24_capture(_Hsnp24, SNP24_START_AND_WAIT);
    SNP24_capture(_Hsnp24, SNP24_START_AND_WAIT);
    SNP24_capture(_Hsnp24, SNP24_START_AND_WAIT);
}

Snapper24::~Snapper24()
{
    if (_opened)
	BASE_destroy(_Hbase);
}

Snapper24::Channel
Snapper24::channel() const
{
    switch (ASL_get_ret(SNP24_get_module_format(_Hsnp24)))
    {
      case SNP24_FORMAT_Y8_ON_RED:
	return ChRED;

      case SNP24_FORMAT_Y8_ON_GRN:
	return ChGREEN;

      case SNP24_FORMAT_Y8_ON_BLU:
	return ChBLUE;
    }
    return ChBGR;
}

Snapper24&
Snapper24::set_roi(u_int u0, u_int v0, u_int w, u_int h)
{
    switch (ASL_get_ret(SNP24_get_subsample(_Hsnp24)))
    {
      case SNP24_SUB_X2:
	u0 *= 2;
	v0 *= 2;
	w  *= 2;
	h  *= 2;
	break;

      case SNP24_SUB_X4:
	u0 *= 4;
	v0 *= 4;
	w  *= 4;
	h  *= 4;
	break;
    }

    u_int	width_max  = SNP24_AREA_EIA_X_LENGTH,
		height_max = SNP24_AREA_EIA_Y_LENGTH;
    switch (ASL_get_ret(SNP24_get_video_standard(_Hsnp24)))
    {
      case SNP24_CCIR_DEFAULT:
	width_max  = SNP24_AREA_CCIR_X_LENGTH;
	height_max = SNP24_AREA_CCIR_Y_LENGTH;
	break;
    }
    if (w > width_max)
	w = width_max;
    if (h > height_max)
	h = height_max;
    if (u0 + w > width_max)
	u0 = width_max - w;
    if (v0 + h > height_max)
	v0 = height_max - h;

    i16	roi[ASL_SIZE_2D_ROI];
    roi[0] = u0;
    roi[1] = v0;
    roi[2] = w;
    roi[3] = h;
    _status = SNP24_set_ROI(_Hsnp24, SNP24_ROI_SET, roi);

    return *this;
}

Snapper24&
Snapper24::set_channel(Channel channel)
{
    switch (channel)
    {
      case ChRED:
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_Y8_ON_RED, TMG_Y8);
	break;

      case ChGREEN:
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_Y8_ON_GRN, TMG_Y8);
	break;

      case ChBLUE:
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_Y8_ON_BLU, TMG_Y8);
	break;
    }
    SNP24_reset_read_pointer(_Hsnp24, SNP24_READ_RESET_AUTO);
    
    return *this;
}

Snapper24&
Snapper24::operator >>(Image<u_char>& image)
{
    if (ASL_get_ret(SNP24_get_TMG_format(_Hsnp24)) != TMG_Y8)
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_Y8_ON_RED, TMG_Y8);
    image.resize(height(), width());
    return read_raw((u_char*)image, width()*height()*sizeof(u_char));
}

Snapper24&
Snapper24::operator >>(Image<BGR>& image)
{
    if (ASL_get_ret(SNP24_get_TMG_format(_Hsnp24)) != TMG_BGR24)
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_RGB, TMG_BGR24);
    image.resize(height(), width());
    return read_raw((BGR*)image, width()*height()*sizeof(BGR));
}

Snapper24&
Snapper24::operator >>(Image<ABGR>& image)
{
    if (ASL_get_ret(SNP24_get_TMG_format(_Hsnp24)) != TMG_XBGR32)
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_RGB, TMG_XBGR32);
    image.resize(height(), width());
    return read_raw((ABGR*)image, width()*height()*sizeof(ABGR));
}

Snapper24&
Snapper24::operator >>(Image<RGB>& image)
{
    if (ASL_get_ret(SNP24_get_TMG_format(_Hsnp24)) != TMG_RGB24)
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_RGB, TMG_RGB24);
    image.resize(height(), width());
    return read_raw((RGB*)image, width()*height()*sizeof(RGB));
}

Snapper24&
Snapper24::operator >>(Image<RGBA>& image)
{
    if (ASL_get_ret(SNP24_get_TMG_format(_Hsnp24)) != TMG_RGBX32)
	SNP24_set_format(_Hsnp24, SNP24_FORMAT_RGB, TMG_RGBX32);
    image.resize(height(), width());
    return read_raw((RGBA*)image, width()*height()*sizeof(RGBA));
}

Snapper24&
Snapper24::set(...)
{
    va_list	args;
    va_start(args, this);
    set_attributes(args);
    va_end(args);
    
    return *this;
}

/************************************************************************
*  Private member functions						*
************************************************************************/
void
Snapper24::set_attributes(va_list args)
{
    for (const char* attribute; (attribute = va_arg(args, const char*)) != 0; )
    {
	if (!strcmp(attribute, "SNP24_CAPTURE"))
	    _status = SNP24_set_capture(_Hsnp24, va_arg(args, Tparam));
	else if (!strcmp(attribute, "SNP24_ROI"))
	{
	    u_int	x = va_arg(args, u_int), y = va_arg(args, u_int),
			w = va_arg(args, u_int), h = va_arg(args, u_int);

	    set_roi(x, y, w, h);
	}
	else if (!strcmp(attribute, "SNP24_FORMAT"))
	{
	    Tparam	module_format = va_arg(args, Tparam);
	    ui16	TMG_format  = va_arg(args, ui16);

	    _status = SNP24_set_format(_Hsnp24, module_format, TMG_format);
	}
	else if (!strcmp(attribute, "SNP24_SYNC"))
	    _status = SNP24_set_sync(_Hsnp24, va_arg(args, Tparam));
	else if (!strcmp(attribute, "SNP24_VIDEO_SRC"))
	    _status = SNP24_set_video_src(_Hsnp24, va_arg(args, Tparam),
					  SNP24_252_ALL);
	else if (!strcmp(attribute, "SNP24_VIDEO_STANDARD"))
	    _status = SNP24_set_video_standard(_Hsnp24, va_arg(args, Tparam));
	else
	    _status = SNP24_make_ret(_Hsnp24, ASLERR_NOT_SUPPORTED, 0);

	if (!*this)
	    return;
    }
}

u_int
Snapper24::to_outer_coord(u_int value) const
{
    switch (ASL_get_ret(SNP24_get_subsample(_Hsnp24)))
    {
      case SNP24_SUB_X2:
	return value / 2;
	
      case SNP24_SUB_X4:
	return value / 4;
    }
    return value;
}

Snapper24&
Snapper24::wait_until_capture_complete(float timeout)
{
    clock_t	start = clock();
    while (SNP24_is_capture_complete(_Hsnp24) == FALSE)
	if (((float)(clock() - start) / (float)CLOCKS_PER_SEC) >= timeout)
	{
	    _status = SNP24_make_ret(_Hsnp24, ASLERR_TIMEOUT, 0);
	    break;
	}
    return *this;
}

Snapper24&
Snapper24::read_raw(void* data, u_int nbytes)
{
    if (!wait_until_capture_complete())
	return *this;

    _status = SNP24_read_video_data_raw(_Hsnp24, data, nbytes);
    return *this;
}
 
}
