/*!
  \file		fdstream.cc
  \author	Toshio UESHIBA
  \brief	クラス TU::fdstream, TU::fdistream, TU::fdostream の実装
*/
#include "TU/fdstream.h"
#include <stdexcept>
#include <cstring>	// for memmove()
#include <fcntl.h>
#ifdef WIN32
#  include <io.h>	// for read() and write()
#else
#  include <unistd.h>	// for read() and write()
#endif

namespace TU
{
/************************************************************************
*  class fdbuf								*
************************************************************************/
//! 指定したファイル記述子からストリームバッファを作る．
/*!
  \param fd			ファイル記述子
  \param closeFdOnClosing	trueならばこのストリームバッファの破壊時に
				ファイル記述子をclose
*/
fdbuf::fdbuf(int fd, bool closeFdOnClosing)
    :_fd(fd), _closeFdOnClosing(closeFdOnClosing)
{
    using namespace	std;
    
    if (_fd < 0)
	throw runtime_error("TU::fdbuf::fdbuf: invalid file descriptor!");
    setg(_buf + pbSize, _buf + pbSize, _buf + pbSize);
}

//! ストリームバッファを破壊する．
fdbuf::~fdbuf()
{
    if (_closeFdOnClosing && _fd >= 0)
	::close(_fd);
}
    
//! ファイルから文字列をバッファに読み込む．
/*!
  \return	ユーザ側に返されていない文字があれば，その最初の文字．
		なければEOF．
*/
fdbuf::int_type
fdbuf::underflow()
{
#ifndef WIN32
    using std::memmove;
#endif
    if (gptr() < egptr())		// 現在位置はバッファ終端よりも前？
	return traits_type::to_int_type(*gptr());

    int	numPutback = gptr() - eback();	// 以前に読み込まれた文字数
    if (numPutback > pbSize)
	numPutback = pbSize;		// putback領域のサイズに切り詰め

  // 以前に読み込まれていた文字を高々pbSize個だけputback領域にコピー
    memmove(_buf + (pbSize - numPutback), gptr() - numPutback, numPutback);

  // 高々bufSize個の文字を新たに読み込む
    int	num = read(_fd, _buf + pbSize, bufSize);
    if (num <= 0)
	return traits_type::eof();

  // バッファのポインタをセットし直す
    setg(_buf + (pbSize - numPutback), _buf + pbSize, _buf + pbSize + num);

    return traits_type::to_int_type(*gptr());	// 次の文字を返す
}

//! ファイルに文字を書き出す．
/*!
  \param c	書き出す文字
  \return	書き出しに成功すればその文字．失敗すればEOF．
*/
fdbuf::int_type
fdbuf::overflow(int_type c)
{
    if (c != traits_type::eof())
    {
	char	z = c;
	if (write (_fd, &z, 1) != 1)
	    return traits_type::eof();
    }
    return c;
}

//! ファイルに文字列を書き出す．
/*!
  \param s	書き出す文字列
  \param n	書き出す文字数
  \return	実際に書き出した文字数
*/
std::streamsize
fdbuf::xsputn(const char* s, std::streamsize n)
{
    return write(_fd, s, n);
}

/************************************************************************
*  class fdistream							*
************************************************************************/
//! 指定したファイル名から入力ストリームを作る．
/*!
  このストリームが破壊されるとファイルもcloseされる．
  \param path	ファイル名
*/
fdistream::fdistream(const char* path)
    :std::istream(0), _buf(::open(path, O_RDONLY), true)
{
    rdbuf(&_buf);
}

/************************************************************************
*  class fdostream							*
************************************************************************/
//! 指定したファイル名から出力ストリームを作る．
/*!
  このストリームが破壊されるとファイルもcloseされる．
  \param path	ファイル名
*/
fdostream::fdostream(const char* path)
    :std::ostream(0), _buf(::open(path, O_WRONLY), true)
{
    rdbuf(&_buf);
}
    
/************************************************************************
*  class fdstream							*
************************************************************************/
//! 指定したファイル名から入出力ストリームを作る．
/*!
  このストリームが破壊されるとファイルもcloseされる．
  \param path	ファイル名
*/
fdstream::fdstream(const char* path)
    :std::iostream(0), _buf(::open(path, O_RDWR), true)
{
    rdbuf(&_buf);
}
    
}
