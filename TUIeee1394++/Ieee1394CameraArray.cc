/*
 *  $Id: Ieee1394CameraArray.cc,v 1.7 2011-01-16 23:43:47 ueshiba Exp $
 */
#include "TU/Ieee1394CameraArray.h"
#include <algorithm>

#ifdef HAVE_LIBTUTOOLS__
#  define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"
#  define DEFAULT_CAMERA_NAME	"IEEE1394Camera"

namespace TU
{
/************************************************************************
*  class Ieee1394CameraArray						*
************************************************************************/
//! 空のIEEE1394デジタルカメラの配列を生成する．
Ieee1394CameraArray::Ieee1394CameraArray()
    :Array<Ieee1394Camera*>()
{
}
    
//! IEEE1394デジタルカメラの配列を生成する．
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param i1394b		IEEE1394bモード (800Mbps)で動作
  \param ncameras	生成するカメラ台数．設定ファイルに記されている最初の
			ncameras台が生成される．-1を指定すると，設定ファイル
			中の全カメラが生成される．
*/
Ieee1394CameraArray::Ieee1394CameraArray(const char* name, const char* dirs,
					 bool i1394b, int ncameras)
    :Array<Ieee1394Camera*>(), _fullName()
{
    initialize(name, dirs, i1394b, ncameras);
}
    
//! IEEE1394デジタルカメラの配列を初期化する．
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param i1394b		IEEE1394bモード (800Mbps)で動作
  \param ncameras	生成するカメラ台数．設定ファイルに記されている最初の
			ncameras台が生成される．-1を指定すると，設定ファイル
			中の全カメラが生成される．
*/
void
Ieee1394CameraArray::initialize(const char* name, const char* dirs,
				bool i1394b, int ncameras)
{
    using namespace	std;

  // 現在設定されている全カメラを廃棄する．
    for (int i = 0; i < dim(); ++i)
	delete (*this)[i];

  // 設定ファイルのfull path名を生成し，ファイルをオープンする．
    ifstream	in;
    _fullName = openFile(in,
			 string(name != 0 ? name : DEFAULT_CAMERA_NAME),
			 string(dirs != 0 ? dirs : DEFAULT_CONFIG_DIRS),
			 ".conf");
    
  // 設定ファイルから遅延パラメータとカメラ数を読み込む．
    int	delay, n;
    in >> delay >> n;
    if ((ncameras < 0) || (ncameras > n))
	ncameras = n;
    resize(ncameras);
    
  // 設定ファイルに記された全カメラを生成する．
    for (int i = 0; i < dim(); ++i)
    {
	string		s;
	in >> s;			// global unique IDの読み込み
	u_int64_t	uniqId = strtoull(s.c_str(), 0, 0);
	(*this)[i] = new Ieee1394Camera(Ieee1394Camera::Monocular,
					i1394b, uniqId, delay);
	in >> *(*this)[i];		// カメラパラメータの読み込みと設定
    }
}

//! IEEE1394デジタルカメラの配列を破壊する．
Ieee1394CameraArray::~Ieee1394CameraArray()
{
    for (int i = 0; i < dim(); ++i)
	delete (*this)[i];
}

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
#endif	/* HAVE_LIBTUTOOLS__	*/    
