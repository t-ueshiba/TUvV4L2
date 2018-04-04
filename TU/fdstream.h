/*!
  \file		fdstream.h
  \author	Toshio UESHIBA
  \brief	ファイル記述子付きストリームバッファに関するクラスの定義と実装
*/
#ifndef TU_FDSTREAM_H
#define TU_FDSTREAM_H

#include <iostream>
#include <streambuf>

namespace TU
{
/************************************************************************
*  class fdbuf								*
************************************************************************/
//! ファイル記述子を持つストリームバッファクラス
/*!
  #TU::fdistream, #TU::fdostream, #TU::fdstream の内部で使われる．
*/
class fdbuf : public std::streambuf
{
  public:
    typedef std::streambuf::traits_type	traits_type;	//!< 特性の型
    typedef traits_type::int_type	int_type;	//!< 整数の型

  public:
    fdbuf(int fd, bool closeFdOnClosing)				;
    virtual			~fdbuf()				;

    int				fd()				const	;
    
  protected:
    virtual int_type		underflow()				;
    virtual int_type		overflow(int_type c)			;
    virtual std::streamsize	xsputn(const char* s, std::streamsize n);

  protected:
    enum
    {
	pbSize	= 4,			//!< putback領域の最大文字数
	bufSize	= 1024			//!< 通常読み込み領域の最大文字数
    };
    
    const int	_fd;			//!< ファイル記述子
    const bool	_closeFdOnClosing;	//!< このバッファの破壊時に_fdをclose
    char	_buf[bufSize + pbSize];	//!< 読み込みデータ領域
};

//! ファイル記述子を返す．
/*!
  \return	ファイル記述子
*/
inline int
fdbuf::fd() const
{
    return _fd;
}
    
/************************************************************************
*  class fdistream							*
************************************************************************/
//! ファイル記述子を持つ入力ストリームクラス
class fdistream : public std::istream
{
  public:
    fdistream(const char* path)						;
    fdistream(int fd)							;
    
    int		fd()						const	;
    
  protected:
    fdbuf	_buf;		//!< ファイル記述子を持つストリームバッファ
};

//! 指定したファイル記述子から入力ストリームを作る．
/*!
  このストリームが破壊されてもファイル記述子はcloseされない．
  \param fd	入力可能なファイル記述子
*/
inline
fdistream::fdistream(int fd)
    :std::istream(0), _buf(fd, false)
{
    rdbuf(&_buf);
}

//! ファイル記述子を返す．
/*!
  \return	ファイル記述子
*/
inline int
fdistream::fd() const
{
    return _buf.fd();
}

/************************************************************************
*  class fdostream							*
************************************************************************/
//! ファイル記述子を持つ出力ストリームクラス
class fdostream : public std::ostream
{
  public:
    fdostream(const char* path)						;
    fdostream(int fd)							;
    
    int		fd()						const	;
    
  protected:
    fdbuf	_buf;		//!< ファイル記述子を持つストリームバッファ
};

//! 指定したファイル記述子から出力ストリームを作る．
/*!
  このストリームが破壊されてもファイル記述子はcloseされない．
  \param fd	出力可能なファイル記述子
*/
inline
fdostream::fdostream(int fd)
    :std::ostream(0), _buf(fd, false)
{
    rdbuf(&_buf);
}

//! ファイル記述子を返す．
/*!
  \return	ファイル記述子
*/
inline int
fdostream::fd() const
{
    return _buf.fd();
}

/************************************************************************
*  class fdstream							*
************************************************************************/
//! ファイル記述子を持つ入出力ストリームクラス
class fdstream : public std::iostream
{
  public:
    fdstream(const char* path)						;
    fdstream(int fd)							;
    
    int		fd()						const	;
    
  protected:
    fdbuf	_buf;		//!< ファイル記述子を持つストリームバッファ
};

//! 指定したファイル記述子から入出力ストリームを作る．
/*!
  このストリームが破壊されてもファイル記述子はcloseされない．
  \param fd	入出力可能なファイル記述子
*/
inline
fdstream::fdstream(int fd)
    :std::iostream(0), _buf(fd, false)
{
    rdbuf(&_buf);
}

//! ファイル記述子を返す．
/*!
  \return	ファイル記述子
*/
inline int
fdstream::fd() const
{
    return _buf.fd();
}

}
#endif	// !TU_FDSTREAM_H
