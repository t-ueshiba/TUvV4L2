/*
 *  $Id: Ieee1394CameraArray.cc,v 1.11 2012-08-29 19:30:24 ueshiba Exp $
 */
#include <cstdlib>
#include "TU/Ieee1394CameraArray.h"

#ifdef HAVE_LIBTUTOOLS__

namespace TU
{
/************************************************************************
*  class Ieee1394CameraArray						*
************************************************************************/
//! 空のIEEE1394デジタルカメラの配列を生成する.
Ieee1394CameraArray::Ieee1394CameraArray()
    :Array<Ieee1394Camera*>()
{
}
    
//! IEEE1394デジタルカメラの配列を生成する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param i1394b		IEEE1394bモード (800Mbps)で動作
  \param ncameras	生成するカメラ台数. 設定ファイルに記されている最初の
			ncameras台が生成される. -1を指定すると, 設定ファイル
			中の全カメラが生成される. 
*/
Ieee1394CameraArray::Ieee1394CameraArray(const char* name, const char* dirs,
					 Ieee1394Node::Speed speed,
					 int ncameras)
    :Array<Ieee1394Camera*>(), _fullName()
{
    initialize(name, dirs, speed, ncameras);
}
    
//! IEEE1394デジタルカメラの配列を初期化する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param i1394b		IEEE1394bモード (800Mbps)で動作
  \param ncameras	生成するカメラ台数. 設定ファイルに記されている最初の
			ncameras台が生成される. -1を指定すると, 設定ファイル
			中の全カメラが生成される. 
*/
void
Ieee1394CameraArray::initialize(const char* name, const char* dirs,
				Ieee1394Node::Speed speed, int ncameras)
{
    using namespace	std;

  // 現在設定されている全カメラを廃棄する.
    for (int i = 0; i < dim(); ++i)
	delete (*this)[i];

  // 設定ファイルのfull path名を生成し, ファイルをオープンする.
    ifstream	in;
    _fullName = openFile(in,
			 string(name != 0 ? name : DEFAULT_CAMERA_NAME),
			 string(dirs != 0 ? dirs : DEFAULT_CONFIG_DIRS),
			 ".conf");
    
  // 設定ファイルから遅延パラメータとカメラ数を読み込む.
    int	delay, n;
    in >> delay >> n;
    if ((ncameras < 0) || (ncameras > n))
	ncameras = n;
    resize(ncameras);
    
  // 設定ファイルに記された全カメラを生成する.
    for (int i = 0; i < dim(); ++i)
    {
	string		s;
	in >> s;			// global unique IDの読み込み
	u_int64_t	uniqId = strtoull(s.c_str(), 0, 0);
	(*this)[i] = new Ieee1394Camera(Ieee1394Camera::Monocular,
					uniqId, speed, delay);
	in >> *(*this)[i];		// カメラパラメータの読み込みと設定
    }
}

//! IEEE1394デジタルカメラの配列を破壊する.
Ieee1394CameraArray::~Ieee1394CameraArray()
{
    for (int i = 0; i < dim(); ++i)
	delete (*this)[i];
}

}
#endif	/* HAVE_LIBTUTOOLS__	*/    
