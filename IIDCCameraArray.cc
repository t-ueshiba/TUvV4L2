/*
 *  $Id: IIDCCameraArray.cc 1681 2014-10-17 02:16:17Z ueshiba $
 */
#ifdef HAVE_LIBTUTOOLS__
#include <cstdlib>
#include <iomanip>
#include "TU/IIDCCameraArray.h"
#include "TU/io.h"

namespace TU
{
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
//! 空のIIDCデジタルカメラの配列を生成する.
IIDCCameraArray::IIDCCameraArray()
    :Array<IIDCCamera*>()
{
}
    
//! IIDCデジタルカメラの配列を生成する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param i1394b		IEEE1394bモード (800Mbps)で動作
  \param ncameras	生成するカメラ台数. 設定ファイルに記されている最初の
			ncameras台が生成される. -1を指定すると, 設定ファイル
			中の全カメラが生成される. 
*/
IIDCCameraArray::IIDCCameraArray(const char* name, const char* dirs,
				 IIDCCamera::Speed speed, int ncameras)
    :Array<IIDCCamera*>(), _fullName()
{
    initialize(name, dirs, speed, ncameras);
}
    
//! IIDCデジタルカメラの配列を初期化する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param i1394b		IIDCbモード (800Mbps)で動作
  \param ncameras	生成するカメラ台数. 設定ファイルに記されている最初の
			ncameras台が生成される. -1を指定すると, 設定ファイル
			中の全カメラが生成される. 
*/
void
IIDCCameraArray::initialize(const char* name, const char* dirs,
			    IIDCCamera::Speed speed, int ncameras)
{
  // 設定ファイルのfull path名を生成し, ファイルをオープンする.
    std::ifstream	in;
    _fullName = openFile(in,
			 std::string(name != 0 ? name : DEFAULT_CAMERA_NAME),
			 std::string(dirs != 0 ? dirs : DEFAULT_CONFIG_DIRS),
			 ".conf");
    
  // 設定ファイルから遅延パラメータとカメラ数を読み込む.
    int	n;
    in >> n;
    if ((ncameras < 0) || (ncameras > n))
	ncameras = n;

    restore(in, ncameras, speed);
}

//! IIDCデジタルカメラの配列を破壊する.
IIDCCameraArray::~IIDCCameraArray()
{
    for (size_t i = 0; i < size(); ++i)
	delete (*this)[i];
}

std::istream&
IIDCCameraArray::restore(std::istream& in, int ncameras,
			 IIDCCamera::Speed speed)
{
  // 現在設定されている全カメラを廃棄する.
    for (size_t i = 0; i < size(); ++i)
	delete (*this)[i];

  // カメラ数を設定する．
    resize(ncameras);
    
  // 設定ファイルに記された全カメラを生成する.
    for (size_t i = 0; i < size(); ++i)
    {
	std::string	s;
	in >> s;			// global unique IDの読み込み
	u_int64_t	uniqId = strtoull(s.c_str(), 0, 0);
	(*this)[i] = new IIDCCamera(IIDCCamera::Monocular, uniqId, speed);
	in >> *(*this)[i];		// カメラパラメータの読み込みと設定
    }

    return in;
}
    
/************************************************************************
*  global functions							*
************************************************************************/
std::istream&
operator >>(std::istream& in, IIDCCameraArray& cameras)
{
    int	n;
    in >> n;		// カメラ数を読み込む.

    return cameras.restore(in, n, IIDCCamera::SPD_400M);
}

std::ostream&
operator <<(std::ostream& out, const IIDCCameraArray& cameras)
{
    using namespace	std;
    
    out << ' ' << cameras.size() << endl;
    for (size_t i = 0; i < cameras.size(); ++i)
	out << "0x" << setw(16) << setfill('0')
	    << hex << cameras[i]->globalUniqueId() << ' '
	    << dec << *cameras[i];

    return out;
}

}
#endif	/* HAVE_LIBTUTOOLS__	*/    
