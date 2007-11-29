/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: FileSelection.h,v 1.4 2007-11-29 07:06:07 ueshiba Exp $
 */
#ifndef __TUvFileSelection_h
#define __TUvFileSelection_h

#include <fstream>
#include <string>
#include <dirent.h>
#include "TU/v/ModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class FileSelection							*
************************************************************************/
class FileSelection : public ModalDialog
{
  public:
    FileSelection(Window& parentWindow)					;
    virtual		~FileSelection()				;

    bool		open(std::ifstream& in)				;
    bool		open(std::ofstream& out)			;

    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    struct cmp
    {
	bool	operator ()(const char* a, const char* b)
						{return ::strcmp(a, b) < 0;}
    };
    
    void		changeDirectory(const std::string& dirname)	;
    void		getFileNames(DIR* dirp, int n)			;
    std::string		fullPathName(const char* filename)	const	;

    std::string		_fullname;  // fullpath file name currently selected.
    std::string		_dirname;   // directory name currently browsing.
    Array<char*>	_filenames; // file names under _dirname.
};

}
}
#endif // !__TUvFileSelection_h
