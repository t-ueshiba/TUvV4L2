/*
 *  $Id: XilImage.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/XilDC.h"

namespace TU
{
/************************************************************************
*  class TUXilImage<T>							*
************************************************************************/
template <> void
XilImage<short>::set_storage()
{
    XilMemoryStorage	storage;
    const u_int		s = (height() > 1 ? &(*this)[1][0] - &(*this)[0][0] :
			     height() > 0 ? width() : 0);
    
    storage.shrt.data		 = (short*)*this;
    storage.shrt.scanline_stride = sizeof(short) * s;
    storage.shrt.pixel_stride	 = xil_nbands();
    xil_export(_xilimage);
    xil_set_memory_storage(_xilimage, &storage);
    xil_import(_xilimage, 1);
}
 
}
