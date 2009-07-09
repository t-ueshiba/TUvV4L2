/*
 *  $Id: main.cc,v 1.2 2009-07-09 04:58:26 ueshiba Exp $
 */
/*!
  \mainpage	manustest
  \anchor	manustest
  キーボードインターフェースによる
  <a href="http://www.exactdynamics.nl/">Exact Dynamics社</a>製のManus
  マニピュレータのためのコントロールプログラムである．
  CAN(Control Area Network)デバイスのためのコントローラライブラリ:
  \ref libTUCan "libTUCan++"を利用している．
*/
#include <signal.h>
#include <unistd.h>
#include "TU/Can++.h"

void	init_kbhit();
void	term_kbhit();
int	kbhit();

//! メイン関数
/*!
  \param argc	コマンド名を含んだ引数の数
  \param argv	引数文字列の配列
*/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    char*		dev = "/dev/can0";
    extern char*	optarg;
    
    for (int c; (c = getopt(argc, argv, "d:")) != EOF; )
	switch (c)
	{
	  case 'd':
	    dev = optarg;
	    break;
	}

    try
    {
      // Manusマニピュレータオブジェクトの生成
	Manus		manus(dev);

	init_kbhit();

      // メインループ
	for (int c; (c = kbhit()) != 'x'; )
	{
	    Manus::Speed	speed;	// 速度指令値(すべて0に初期化)

	    switch (c)
	    {
	      case '0':
		manus.stillMode();	// STILLモードに移行
		continue;
	      case '4':
		manus.jointMode();	// JOINTモードに移行
		continue;
	      case '5':
		manus.foldOut();	// マニピュレータを拡げる
		continue;
	      case '6':
		manus.foldIn();		// マニピュレータを折り畳む
		continue;
	    
	      case 'q':
		manus.setBaseUp().tick();	// 台座を上げる
		continue;
	      case 'a':
		manus.setBaseDown().tick();	// 台座を下げる
		continue;

	      case 'w':
		speed[0] =  Manus::MAX_SPEED_JOINT_012;
		break;
	      case 's':
		speed[0] = -Manus::MAX_SPEED_JOINT_012;
		break;
	      case 'e':
		speed[1] =  Manus::MAX_SPEED_JOINT_012;
		break;
	      case 'd':
		speed[1] = -Manus::MAX_SPEED_JOINT_012;
		break;
	      case 'r':
		speed[2] =  Manus::MAX_SPEED_JOINT_012;
		break;
	      case 'f':
		speed[2] = -Manus::MAX_SPEED_JOINT_012;
		break;
	      case 't':
		speed[3] =  Manus::MAX_SPEED_JOINT_345;
		break;
	      case 'g':
		speed[3] = -Manus::MAX_SPEED_JOINT_345;
		break;
	      case 'y':
		speed[4] =  Manus::MAX_SPEED_JOINT_345;
		break;
	      case 'h':
		speed[4] = -Manus::MAX_SPEED_JOINT_345;
		break;
	      case 'u':
		speed[5] =  Manus::MAX_SPEED_JOINT_345;
		break;
	      case 'j':
		speed[5] = -Manus::MAX_SPEED_JOINT_345;
		break;
	      case 'i':
		speed[6] =  Manus::MAX_SPEED_JOINT_GRIP;
		break;
	      case 'k':
		speed[6] = -Manus::MAX_SPEED_JOINT_GRIP;
		break;
	    }
	    speed *= 0.2;	// 最高速度の1/5にする
	    
	  // 速度を設定してループを1回まわす．
	    manus.setSpeed(speed).tick();

	    if (manus.status() != Manus::OK)
		cerr << Manus::message(manus.status()) << endl;
	    cerr << "  Position: " << manus.position();
	}
	
      // プログラム終了時にはSTILLモードにする
	manus.stillMode();

	term_kbhit();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
