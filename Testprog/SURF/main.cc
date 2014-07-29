/*
 *  $Id: main.cc,v 1.4 2011-07-22 02:19:55 ueshiba Exp $
 */
#include "TU/SURFCreator.h"
#include "TU/FeatureMatch.h"
#include "TU/ICIA.h"
#include "MatchImage.h"
#include <fstream>

namespace TU
{
enum MapType		{PROJECTIVE, AFFINE};
static const MapType	DEFAULT_MAP_TYPE = AFFINE;
static const float	DEFAULT_INTENSITY_THRESH = 15.0;
    
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> Image<T>&
restoreFromFile(Image<T>& image, const std::string& fileName)
{
    using namespace	std;
    
    ifstream	in(fileName.c_str(), ios_base::in | ios_base::binary);
    if (!in)
	throw runtime_error("Cannot open the specified input image file: "
			    + fileName);
    image.restore(in);

    return image;
}

template <class T, class Map> static typename Map::point_type
warpImage(const Image<T>& image, Image<T>& warpedImage, const Map& map)
{
    using namespace	std;

    typedef typename Map::element_type	element_type;
    typedef typename Map::matrix_type	matrix_type;
    typedef typename Map::point_type	point_type;

    Map				inv = map.inv();
    BoundingBox<point_type>	bbox;
    bbox.expand(inv(point_type(0,	      0)))
	.expand(inv(point_type(image.width(), 0)))
	.expand(inv(point_type(image.width(), image.height())))
	.expand(inv(point_type(0,	      image.height())));
  //cerr << bbox.width() << 'x' << bbox.height() << endl;
    warpedImage.resize(size_t(bbox.height()), size_t(bbox.width()));
    warpedImage = 0;
    
  // 変形後の左上隅が(0，0)になるように原点を移動する．
    matrix_type	Ht = map.trns();
    (Ht[2] += bbox.min(0) * Ht[0]) += bbox.min(1) * Ht[1];

  // 変形画像の各画素値を計算する．
    const size_t	width1	= image.width()  - 1,
			height1	= image.height() - 1;
    for (size_t v = 0; v < warpedImage.height(); ++v)
    {
	Vector<element_type>	y = v * Ht[1] + Ht[2];
	for (size_t u = 0; u < warpedImage.width(); ++u)
	{
	    Point2<element_type>	p(y[0]/y[2], y[1]/y[2]);
	    if (0 <= p[0] && p[0] <= width1 && 0 <= p[1] && p[1] <= height1)
		warpedImage[v][u] = image.at(p);

	    y += Ht[0];
	}
    }

    return bbox.min();
}

//! 画像1を変形して画像0に貼り付け，一枚に統合する．
/*!
  \param image0	画像0
  \param image1	画像1
  \param map	image0をimage1に重ね合わせる変換
*/
template <class T, class Map> static Image<RGB>
integrateImages(const Image<T>& image0, const Image<T>& image1,
		const Map& map)
{
    using namespace	std;

    Image<T>	warpedImage;
    Point2i	origin = warpImage(image1, warpedImage, map);
    
    BoundingBox<Point2i>	bbox;
    bbox.expand(Point2i(0, 0))
	.expand(Point2i(image0.width(), image0.height()))
	.expand(origin)
	.expand(Point2i(origin[0] + warpedImage.width(),
			origin[1] + warpedImage.height()));

    Image<RGB>	result(bbox.width(), bbox.height());
    int		offset_u = -bbox.min(0), offset_v = -bbox.min(1);
    for (size_t v = 0; v < image0.height(); ++v)
    {
	const ImageLine<T>&	line0 = image0[v];
	      ImageLine<RGB>&	lineR = result[v + offset_v];
	for (size_t u = 0; u < image0.width(); ++u)
	    lineR[u + offset_u].r = line0[u];
    }

    offset_u += origin[0];
    offset_v += origin[1];
    for (size_t v = 0; v < warpedImage.height(); ++v)
    {
	const ImageLine<T>&	line1 = warpedImage[v];
	      ImageLine<RGB>&	lineR = result[v + offset_v];
	for (size_t u = 0; u < warpedImage.width(); ++u)
	    lineR[u + offset_u].g = line1[u];
    }

    return result;
}

template <class MAP, class T> static void
doJob(const Image<T> images[2],
      const SURFCreator::Parameters& surfParams,
      const FeatureMatch::Parameters& matchParams,
      bool refine, typename MAP::element_type intensityThresh)
{
    using namespace			std;

    typedef SURF::value_type		value_type;
    typedef typename MAP::element_type	element_type;
    
  // 画像からSURF特徴を取り出す．
    SURFCreator		surfCreator(surfParams);
    vector<SURF>	surfs0, surfs1;
    surfCreator.createSURFs<SURF>(images[0], back_inserter(surfs0));
    cerr << surfs0.size() << " SURFs extracted." << endl;
    surfCreator.createSURFs<SURF>(images[1], back_inserter(surfs1));
    cerr << surfs1.size() << " SURFs extracted." << endl;

  // 点対応を検出する．
    FeatureMatch		match(matchParams);
    MAP				map;
    vector<FeatureMatch::Match>	matchSet;
    match(map, surfs0.begin(), surfs0.end(), surfs1.begin(), surfs1.end(),
	  back_inserter(matchSet));

  // 点対応を示す画像を生成する．
    MatchImage	matchImage;
    Point2i	delta = matchImage.initializeH(images, images + 2);
    matchImage.drawMatches(matchSet.begin(), matchSet.end(),
			   delta, Point2i(0, 0), true);
    matchImage.save(cout);

    if (refine)
    {
	ICIA<MAP, T>	regist(images[0], intensityThresh, 0.5);
	element_type	err = regist(images[1], map, false, 1000);
	cerr << "RMS-err after refinement: " << sqrt(err) << endl;
    }
    
  // 2枚の画像を統合する．
    Image<RGB>	integratedImage = integrateImages(images[0], images[1], map);
    integratedImage.save(cout);
}
    
}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    typedef float	element_type;

    MapType			mapType		= DEFAULT_MAP_TYPE;
    SURFCreator::Parameters	surfParams;
    FeatureMatch::Parameters	matchParams;
    bool			refine		= false;
    element_type		intensityThresh	= DEFAULT_INTENSITY_THRESH;
    const element_type		RAD		= M_PI/180;
    extern char			*optarg;
    for (int c; (c = getopt(argc, argv, "PAt:a:s:i:c:rk:")) != -1; )
	switch (c)
	{
	  case 'P':
	    mapType = PROJECTIVE;
	    break;
	  case 'A':
	    mapType = AFFINE;
	    break;
	  case 't':
	    surfParams.scoreThresh = atof(optarg);
	    break;
	  case 'a':
	    matchParams.diffAngleMax = atof(optarg)*RAD;
	    break;
	  case 's':
	    matchParams.separation = atof(optarg);
	    break;
	  case 'i':
	    matchParams.inlierRate = atof(optarg);
	    break;
	  case 'c':
	    matchParams.conformThresh = atof(optarg);
	    break;
	  case 'r':
	    refine = true;
	    break;
	  case 'k':
	    intensityThresh = atof(optarg);
	    break;
	}

    try
    {
      // ファイルから画像を読み込む．
	Image<u_char>	images[2];
	extern int	optind;

	if (optind + 2 >= argc)
	{
	    restoreFromFile(images[0], argv[optind]);
	    restoreFromFile(images[1], argv[optind + 1]);
	}
	else
	{
	    images[0].restore(cin);
	    images[1].restore(cin);
	}

	switch (mapType)
	{
	  case PROJECTIVE:
	    doJob<Homography<element_type> >(images, surfParams, matchParams,
					     refine, intensityThresh);
	    break;
	  default:
	    doJob<Affinity2<element_type> >(images, surfParams, matchParams,
					    refine, intensityThresh);
	    break;
	}
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
