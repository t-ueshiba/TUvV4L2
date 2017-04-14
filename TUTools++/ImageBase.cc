/*
 *  $Id: ImageBase.cc 1836 2015-07-06 06:42:09Z ueshiba $
 */
#include "TU/Image++.h"
#include "TU/Camera++.h"
#include "TU/Manip.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
inline static size_t	bit2byte(size_t i)	{return (i + 7)/8;}

inline static bool
isBigEndian()
{
    u_short	val = 0x0001;
    return ((*(u_char*)&val) != 0x01);
}

template <class T> static T	get(std::istream& in)			;

template <> inline short
get<short>(std::istream& in)
{
    u_int	c0 = in.get(), c1 = in.get();
    return c0 + (c1 << 8);
}
    
template <> inline int
get<int>(std::istream& in)
{
    u_int	c0 = in.get(), c1 = in.get(), c2 = in.get(), c3 = in.get();
    return c0 + (c1 << 8) + (c2 << 16) + (c3 << 24);
}

template <class T> static void	put(std::ostream& out, int val)		;

template <> inline void
put<short>(std::ostream& out, int val)
{
    char	c0 = u_int(val) & 0xff, c1 = (u_int(val) >> 8) & 0xff;
    out.put(c0);
    out.put(c1);
}
    
template <> inline void
put<int>(std::ostream& out, int val)
{
    char	c0 = u_int(val)		& 0xff, c1 = (u_int(val) >>  8) & 0xff,
		c2 = (u_int(val) >> 16) & 0xff, c3 = (u_int(val) >> 24) & 0xff;
    out.put(c0);
    out.put(c1);
    out.put(c2);
    out.put(c3);
}

/************************************************************************
*  class ImageBase							*
************************************************************************/
ImageBase::~ImageBase()
{
}

//! 入力ストリームから画像のヘッダを読み込む．
/*!
  \param in	入力ストリーム
  \return	読み込まれた画像の画素のタイプ
*/
ImageBase::TypeInfo
ImageBase::restoreHeader(std::istream& in)
{
  // Reset calibration parameters.
    P = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
    d1 = d2 = 0.0;
    
  // Read the first magic character.
    char	c;
    if (!in.get(c))
	return TypeInfo(DEFAULT);

  // Read image header.
    switch (c)
    {
      case 'P':
	return restorePBMHeader(in);
      case 'B':
	return restoreBMPHeader(in);
      default:
	throw std::runtime_error("TU::ImageBase::restoreHeader: neighter PBM nor BMP file!!");
	break;
    }

    return TypeInfo(DEFAULT);
}

//! 指定した画素タイプで出力ストリームに画像のヘッダを書き出す．
/*!
  \param out	出力ストリーム
  \param type	画素タイプ．ただし，#DEFAULTを指定した場合は，
		この画像オブジェクトの画素タイプで書き出される．
  \return	実際に書き出す場合の画素タイプ．
*/
ImageBase::Type
ImageBase::saveHeader(std::ostream& out, Type type) const
{
    if (type == DEFAULT)
	type = _defaultType();

    switch (type)
    {
      case BMP_8:
      case BMP_24:
      case BMP_32:
	return saveBMPHeader(out, type);
      default:
	return savePBMHeader(out, type);
    }

    return DEFAULT;
}

//! 指定されたタイプの1行あたりのデータバイト数またはpaddingバイト数を返す．
/*!
  \param type		画素のタイプ
  \param padding	falseならデータバイト数, trueならpaddingバイト数
  \return		1行あたりのデータバイト数またはpaddingバイト数
*/
size_t
ImageBase::type2nbytes(Type type, bool padding) const
{
    size_t	nbytes = _width(), align = 1;
    switch (type)
    {
      case SHORT:
	nbytes *= sizeof(short);
	break;
      case INT:
	nbytes *= sizeof(int);
	break;
      case FLOAT:
	nbytes *= sizeof(float);
	break;
      case DOUBLE:
	nbytes *= sizeof(double);
	break;
      case RGB_24:
	nbytes *= sizeof(RGB);
	break;
      case YUV_444:
	nbytes *= sizeof(YUV444);
	break;
      case YUV_422:
	nbytes *= sizeof(YUV422);
	break;
      case YUYV_422:
	nbytes *= sizeof(YUYV422);
	break;
      case YUV_411:
	(nbytes *= 3) /= 2;
	break;
      case BMP_8:
	align = 4;
	break;
      case BMP_24:
	nbytes *= sizeof(BGR);
	align = 4;
	break;
      case BMP_32:
	nbytes *= sizeof(BGRA);
	align = 4;
	break;
      default:
	break;
    }
    size_t	nbytesPerLine = align * ((nbytes + align - 1) / align);

    return (padding ? nbytesPerLine - nbytes : nbytesPerLine);
}

//! 指定されたタイプの画素のビット数を返す．
/*!
  \param type	画素のタイプ
  \return	画素のビット数
*/
size_t
ImageBase::type2depth(Type type)
{
    switch (type)
    {
      case SHORT:
	return 8*sizeof(short);
      case INT:
	return 8*sizeof(int);
      case FLOAT:
	return 8*sizeof(float);
      case DOUBLE:
	return 8*sizeof(double);
      case RGB_24:
      case YUV_444:
      case BMP_24:
	return 24;
      case YUV_422:
	return 16;
      case YUYV_422:
	return 16;
      case YUV_411:
	return 12;
      case BMP_32:
	return 32;
      default:
	break;
    }

    return 8;
}
 
/*
 *  private members
 */
ImageBase::TypeInfo
ImageBase::restorePBMHeader(std::istream& in)
{
  // Read pbm type.
    int		c;
    TypeInfo	typeInfo;
    in >> c >> std::ws;		// Read pbm type and skip trailing white spaces.
    switch (c)
    {
      case U_CHAR:
	typeInfo.type = U_CHAR;
	break;
      case RGB_24:
	typeInfo.type = RGB_24;
	break;
      default:
	throw std::runtime_error("TU::ImageBase::restorePBMHeader: unknown pbm type!!");
    }
    
  // Process comment lines.
    bool	legacy = false;	// legacy style of dist. param. representation
    for (; (c = in.get()) == '#'; in >> skipl)
    {
	char	key[256], val[256];
	in >> key;
	if (!strcmp(key, "DataType:"))		// pixel data type
	{
	    in >> val;
	    if (!strcmp(val, "Short"))
		typeInfo.type = SHORT;
	    else if (!strcmp(val, "Int"))
		typeInfo.type = INT;
	    else if (!strcmp(val, "Float"))
		typeInfo.type = FLOAT;
	    else if (!strcmp(val, "Double"))
		typeInfo.type = DOUBLE;
	    else if (!strcmp(val, "YUV444"))
		typeInfo.type = YUV_444;
	    else if (!strcmp(val, "YUV422"))
		typeInfo.type = YUV_422;
	    else if (!strcmp(val, "YUYV422"))
		typeInfo.type = YUYV_422;
	    else if (!strcmp(val, "YUV411"))
		typeInfo.type = YUV_411;
	}
	else if (!strcmp(key, "Endian:"))	// big- or little-endian
	{
	    in >> val;
	    switch (typeInfo.type)
	    {
	      case SHORT:
	      case INT:
	      case FLOAT:
	      case DOUBLE:
		if (isBigEndian())
		{
		    if (!strcmp(val, "Little"))
			throw std::runtime_error("TU::ImageBase::restore_epbm: big endian is not supported!!");
		}
		else
		{
		    if (!strcmp(val, "Big"))
			throw std::runtime_error("TU::ImageBase::restore_epbm: little endian is not supported!!");
		}
		break;
	      default:
		break;
	    }
	}
	else if (!strcmp(key, "PinHoleParameterH11:"))
	    in >> P[0][0];
	else if (!strcmp(key, "PinHoleParameterH12:"))
	    in >> P[0][1];
	else if (!strcmp(key, "PinHoleParameterH13:"))
	    in >> P[0][2];
	else if (!strcmp(key, "PinHoleParameterH14:"))
	    in >> P[0][3];
	else if (!strcmp(key, "PinHoleParameterH21:"))
	    in >> P[1][0];
	else if (!strcmp(key, "PinHoleParameterH22:"))
	    in >> P[1][1];
	else if (!strcmp(key, "PinHoleParameterH23:"))
	    in >> P[1][2];
	else if (!strcmp(key, "PinHoleParameterH24:"))
	    in >> P[1][3];
	else if (!strcmp(key, "PinHoleParameterH31:"))
	    in >> P[2][0];
	else if (!strcmp(key, "PinHoleParameterH32:"))
	    in >> P[2][1];
	else if (!strcmp(key, "PinHoleParameterH33:"))
	    in >> P[2][2];
	else if (!strcmp(key, "PinHoleParameterH34:"))
	    in >> P[2][3];
	else if (!strcmp(key, "DistortionParameterD1:"))
	{
	    in >> d1;
	    legacy = false;
	}
	else if (!strcmp(key, "DistortionParameterD2:"))
	{
	    in >> d2;
	    legacy = false;
	}
	else if (!strcmp(key, "DistortionParameterA:"))	// legacy dist. param.
	{
	    in >> d1;
	    legacy = true;
	}
	else if (!strcmp(key, "DistortionParameterB:"))	// legacy dist. param.
	{
	    in >> d2;
	    legacy = true;
	}
    }
    in.putback(c);

    if (legacy)
    {
	Camera<Intrinsic<double> >	camera(P);
	double	k = camera.k();
	d1 *= (k * k);
	d2 *= (k * k * k * k);
    }

    size_t	w, h;
    in >> w >> h;
    _resize(h, w, typeInfo);			// set width & height
    in >> w >> skipl;				// skip MaxValue

    return typeInfo;
}

ImageBase::TypeInfo
ImageBase::restoreBMPHeader(std::istream& in)
{
  // Read pbm type.
    int	c = in.get();				// Read second magic character.
    if (c != 'M')
	throw std::runtime_error("TU::ImageBase::restoreBMPHeader: not a BMP file!!");

  // Read file header.
    get<int>(in);				// Skip bfSize.
    get<short>(in);				// Skip bfReserved1.
    get<short>(in);				// Skip bfReserved2.
    get<int>(in);				// Skip bfOffBits.

  // Read information header.
    TypeInfo	typeInfo;
    int		w = 0, h = 0, d = 0;
    switch (c = get<int>(in))			// Read bcSize or biSize.
    {
      case 12:	// BMPCoreHeader:
	w = get<short>(in);			// Read bcWidth.
	h = get<short>(in);			// Read bcHeight.
	if (h > 0)
	    typeInfo.bottomToTop = true;
	else
	{
	    h = -h;
	    typeInfo.bottomToTop = false;
	}
	get<short>(in);				// Skip bcPlanes.
	switch (d = get<short>(in))		// Read bcBitCount.
	{
	  case 1:
	  case 4:
	  case 8:
	    typeInfo.ncolors = (1 << d);
	    break;
	}
	break;

      case 40:	// BMPInfoHeader:
	w = get<int>(in);			// Read biWidth.
	h = get<int>(in);			// Read biHeight.
	if (h > 0)
	    typeInfo.bottomToTop = true;
	else
	{
	    h = -h;
	    typeInfo.bottomToTop = false;
	}
	get<short>(in);				// Skip biPlanes.
	d = get<short>(in);			// Read biBitCount.
	if (get<int>(in) != 0)			// Read biCompression.
	    throw std::runtime_error("TUImageBase::restoreBMPHeader: compressed BMP file not supported!!");
	get<int>(in);				// Skip biSizeImage.
	get<int>(in);				// Skip biXPixPerMeter.
	get<int>(in);				// Skip biYPixPerMeter.
	if ((typeInfo.ncolors = get<int>(in)) == 0)	// Read biClrUsed.
	    switch (d)
	    {
	      case 1:
		typeInfo.ncolors = 2;
		break;
	      case 4:
		typeInfo.ncolors = 16;
		break;
	      case 8:
		typeInfo.ncolors = 256;
		break;
	    }
	
	get<int>(in);				// Read biClrImportant.
	break;

      default:	// Illegal information header size:
	throw std::runtime_error("TU::ImageBase::restoreBMPHeader: information header corrupted!!");
	break;
    }

  // Set type of the image.
    switch (d)
    {
      case 8:
	typeInfo.type = BMP_8;
	break;
      case 24:
	typeInfo.type = BMP_24;
	break;
      case 32:
	typeInfo.type = BMP_32;
	break;
      default:
	throw std::runtime_error("TU::ImageBase::restoreBMPHeader: unsupported depth!!");
	break;
    }

    _resize(h, w, typeInfo);		// Allocate image area of w*h size.
    
    return typeInfo;
}

ImageBase::Type
ImageBase::savePBMHeader(std::ostream& out, Type type) const
{
    out << 'P';
    switch (type)
    {
      case RGB_24:
	out << int(RGB_24) << std::endl;
	break;
      default:
	out << int(U_CHAR) << std::endl;
	break;
    }

    const size_t	depth = type2depth(type);
    out << "# PixelLength: " << bit2byte(depth) << std::endl;
    out << "# DataType: ";
    switch (type)
    {
      default:
	out << "Char" << std::endl;
	break;
      case RGB_24:
	out << "RGB24" << std::endl;
	break;
      case SHORT:
	out << "Short" << std::endl;
	break;
      case INT:
	out << "Int" << std::endl;
	break;
      case FLOAT:
	out << "Float" << std::endl;
	break;
      case DOUBLE:
	out << "Double" << std::endl;
	break;
      case YUV_444:
	out << "YUV444" << std::endl;
	break;
      case YUV_422:
	out << "YUV422" << std::endl;
	break;
      case YUYV_422:
	out << "YUYV422" << std::endl;
	break;
      case YUV_411:
	out << "YUV411" << std::endl;
	break;
    }
    out << "# Sign: ";
    switch (type)
    {
      case SHORT:
      case INT:
      case FLOAT:
      case DOUBLE:
	out << "Signed" << std::endl;
	break;
      default:
	out << "Unsigned" << std::endl;
	break;
    }
    if (isBigEndian())
	out << "# Endian: Big" << std::endl;
    else
	out << "# Endian: Little" << std::endl;
    out << "# PinHoleParameterH11: " << P[0][0] << std::endl
	<< "# PinHoleParameterH12: " << P[0][1] << std::endl
	<< "# PinHoleParameterH13: " << P[0][2] << std::endl
	<< "# PinHoleParameterH14: " << P[0][3] << std::endl
	<< "# PinHoleParameterH21: " << P[1][0] << std::endl
	<< "# PinHoleParameterH22: " << P[1][1] << std::endl
	<< "# PinHoleParameterH23: " << P[1][2] << std::endl
	<< "# PinHoleParameterH24: " << P[1][3] << std::endl
	<< "# PinHoleParameterH31: " << P[2][0] << std::endl
	<< "# PinHoleParameterH32: " << P[2][1] << std::endl
	<< "# PinHoleParameterH33: " << P[2][2] << std::endl
	<< "# PinHoleParameterH34: " << P[2][3] << std::endl;
    if (d1 != 0.0 || d2 != 0.0)
	out << "# DistortionParameterD1: " << d1 << std::endl
	    << "# DistortionParameterD2: " << d2 << std::endl;
    out << _width() << ' ' << _height() << '\n'
	<< 255 << std::endl;
    
    return type;
}

ImageBase::Type
ImageBase::saveBMPHeader(std::ostream& out, Type type) const
{
  // Write BMP magic characters.
    out << "BM";

  // Write file header.
    size_t	ncolors = (type == BMP_8 ? 256 : 0);
    put<int>(out, 14 + 40 + 4*ncolors
	     + type2nbytes(type, false)*_height());	// Write bfSize.
    put<short>(out, 0);					// Write bfReserved1.
    put<short>(out, 0);					// Write bfReserved2.
    put<int>(out, 14 + 40 + 4*ncolors);			// Write bfOffBits.

  // Write information header.
    put<int>(out, 40);					// Write biSize.
    put<int>(out, _width());				// Write biWidth.
    put<int>(out, _height());				// Write biHeight.
    put<short>(out, 1);					// Write biPlanes.
    put<short>(out, type2depth(type));			// Write biBitCount.
    put<int>(out, 0);					// Write biCompression.
    put<int>(out, type2nbytes(type, false)*_height());	// Write biSizeImage.
    put<int>(out, 0);					// Write biXPixPerMeter.
    put<int>(out, 0);					// Write biYPixPerMeter.
    put<int>(out, ncolors);				// Write biClrUsed.
    put<int>(out, 0);					// Write biClrImportant.
    
    return type;
}

/************************************************************************
*  class ImageBase::TypeInfo						*
************************************************************************/
//! 与えられた画素タイプに対して外部記憶へのデフォルト付加情報を作る．
/*!
  \param ty	画素タイプ
*/
__PORT
ImageBase::TypeInfo::TypeInfo(Type ty)
    :type(ty), bottomToTop(false), ncolors(0)
{
    switch (type)
    {
      case BMP_8:
	bottomToTop = true;
	ncolors = 256;
	break;
      case BMP_24:
      case BMP_32:
	bottomToTop = true;
	break;
      default:
	break;
    }
}

}
