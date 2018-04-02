/*!
  \file		io.cc
  \author	Toshio UESHIBA
  \brief	TU::openFile() の実装
*/
#include "TU/io.h"
#include <algorithm>
#include <stdexcept>

namespace TU
{
/************************************************************************
*  global functions							*
************************************************************************/
//! 指定した入力ファイルをオープンする．
/*!
  \param in	オープンされたファイルが結びつけられる入力ストリーム
  \param name	ファイル名(拡張子を含まず)
  \param dirs	':'で区切られたファイル探索ディレクトリの並び
  \param ext	ファイルの拡張子，0を指定すれば拡張子なし
  \return	オープンされたファイルのfull path名(拡張子を含まず)
*/
std::string
openFile(std::ifstream& in, const std::string& name,
	 const std::string& dirs, const char* ext)
{
    using namespace		std;

    string::const_iterator	p = dirs.begin();
    do
    {
	string::const_iterator	q = find(p, dirs.end(), ':');
	string			fullName = string(p, q) + '/' + name;
	in.open((ext ? fullName + ext : fullName).c_str());
	if (in)
	    return fullName;
	p = q;
    } while (p++ != dirs.end());

    throw runtime_error("Cannot open file \"" + name + ext +
			"\" in \"" + dirs + "\"!!");
    return string();
}

}
