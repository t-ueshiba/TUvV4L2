/*
 *  $Id: FileSelection.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
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
