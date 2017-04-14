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
 *  $Id$
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
