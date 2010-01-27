/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: fdstream.cc,v 1.3 2010-01-27 08:09:59 ueshiba Exp $
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
  \return	書き出した文字数
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
