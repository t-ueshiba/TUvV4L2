/*!
  \file		io.h
  \author	Toshio UESHIBA
  \brief	各種入出力関数の定義と実装
*/
#ifndef __TU_IO_H
#define __TU_IO_H

#include <fstream>
#include <string>

namespace TU
{
/************************************************************************
*  global functions							*
************************************************************************/
std::string	openFile(std::ifstream& in, const std::string& name,
			 const std::string& dirs, const char* ext)	;
}

#endif	// !__TU_IO_H
