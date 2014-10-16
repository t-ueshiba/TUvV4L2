/*
 *  $Id: V4L2CameraArray.cc 1219 2012-11-09 05:45:49Z ueshiba $
 */
#include <cstdlib>
#include "TU/V4L2CameraArray.h"

#ifdef HAVE_LIBTUTOOLS__

namespace TU
{
/************************************************************************
*  class V4L2CameraArray						*
************************************************************************/
//! 空のIEEE1394デジタルカメラの配列を生成する.
V4L2CameraArray::V4L2CameraArray()
    :Array<V4L2Camera*>()
{
}
    
//! IEEE1394デジタルカメラの配列を生成する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param ncameras	生成するカメラ台数. 設定ファイルに記されている最初の
			ncameras台が生成される. -1を指定すると, 設定ファイル
			中の全カメラが生成される. 
*/
V4L2CameraArray::V4L2CameraArray(const char* name, const char* dirs,
				 int ncameras)
    :Array<V4L2Camera*>(), _fullName()
{
    initialize(name, dirs, ncameras);
}
    
//! IEEE1394デジタルカメラの配列を初期化する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param ncameras	生成するカメラ台数. 設定ファイルに記されている最初の
			ncameras台が生成される. -1を指定すると, 設定ファイル
			中の全カメラが生成される. 
*/
void
V4L2CameraArray::initialize(const char* name, const char* dirs, int ncameras)
{
    using namespace	std;

  // 現在設定されている全カメラを廃棄する.
    for (size_t i = 0; i < size(); ++i)
	delete (*this)[i];

  // 設定ファイルのfull path名を生成し, ファイルをオープンする.
    ifstream	in;
    _fullName = openFile(in,
			 string(name != 0 ? name : TU_V4L2_DEFAULT_CAMERA_NAME),
			 string(dirs != 0 ? dirs : TU_V4L2_DEFAULT_CONFIG_DIRS),
			 ".conf");
    
  // 設定ファイルから遅延パラメータとカメラ数を読み込む.
    int	n;
    in >> n;
    if ((ncameras < 0) || (ncameras > n))
	ncameras = n;
    resize(ncameras);
    
  // 設定ファイルに記された全カメラを生成する.
    for (size_t i = 0; i < size(); ++i)
    {
	string	dev;
	in >> dev;			// device file名の読み込み
	(*this)[i] = new V4L2Camera(dev.c_str());
	in >> *(*this)[i];		// カメラパラメータの読み込みと設定
    }
}

//! IEEE1394デジタルカメラの配列を破壊する.
V4L2CameraArray::~V4L2CameraArray()
{
    for (size_t i = 0; i < size(); ++i)
	delete (*this)[i];
}

/************************************************************************
*  global functions							*
************************************************************************/
std::ostream&
operator <<(std::ostream& out, const V4L2CameraArray& cameras)
{
    out << cameras.size() << std::endl;
    for (size_t i = 0; i < cameras.size(); ++i)
	out << cameras[i]->dev() << ' ' << *cameras[i];

    return out;
}

}
#endif	/* HAVE_LIBTUTOOLS__	*/    
