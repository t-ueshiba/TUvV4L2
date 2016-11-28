/*
 *  $Id$
 */
#include "TU/V4L2CameraArray.h"
#include "TU/io.h"

namespace TU
{
/************************************************************************
*  class V4L2CameraArray						*
************************************************************************/
constexpr const char*	V4L2CameraArray::DEFAULT_CAMERA_NAME;
constexpr const char*	V4L2CameraArray::DEFAULT_CONFIG_DIRS;
    
//! 空のVideo for Linux v.2カメラの配列を生成する.
V4L2CameraArray::V4L2CameraArray(size_t ncameras)
    :Array<V4L2Camera>(ncameras), _fullName()
{
}
    
//! 設定ファイルを読み込んでVideo for Linux v.2カメラの配列を初期化する.
/*!
  \param name		カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
*/
void
V4L2CameraArray::restore(const char* name, const char* dirs)
{
  // 設定ファイルのfull path名を生成し, ファイルをオープンする.
    std::ifstream	in;
    _fullName = openFile(in,
			 std::string(name ? name : DEFAULT_CAMERA_NAME),
			 std::string(dirs ? dirs : DEFAULT_CONFIG_DIRS),
			 ".conf");
    
  // 設定ファイルに記された全カメラを生成する.
    in >> *this;
}

//! 設定ファイルにVideo for Linux v.2カメラ配列の設定を書き込む.
void
V4L2CameraArray::save() const
{
    std::ofstream	out(configFile().c_str());
    if (!out)
	throw std::runtime_error("V4L2CameraArray::save(): cannot open " +
				 configFile());

    out << *this;
}
    
}
