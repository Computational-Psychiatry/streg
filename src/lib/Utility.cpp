#include "RegFw.hpp"
#include "Utility.hpp"
#include <utility>
#include "Models.hpp"

#ifdef GUI
#include <opencv2/highgui/highgui.hpp>
#endif

/**
 * @brief imutil::cropCore - crop image from the center (after uspcaling it)
 * @param im
 * @return Mat
 */

/**
 * @brief imutil::cropCore - crop image from the center (after uspcaling it)
 * @param im
 * @return Mat
 */
cv::Mat imutil::cropCoreOld(const cv::Mat& im, int refImWidth)
{
    int64 t1 = cv::getTickCount();
    cv::Mat imc = im.clone();

    double R = refImWidth;

    if (imc.cols % 2 == 1)
    {
        cv::Mat trans = cv::Mat::zeros(2,3,CV_32F);

        trans.at<float>(0,0) = 1.;
        trans.at<float>(1,1) = 1.;
        trans.at<float>(0,2) = -0.5;
        trans.at<float>(1,2) = -0.5;

        cv::Mat timc;
        cv::warpAffine(imc, timc, trans, imc.size());
        imc = timc;
    }

    int imw = imc.cols;
    int buffer = (imw-R)/2;

    // pad if image is smaller than size to-be-cropped
    if (imc.cols < R)
    {
        int dw = 2*(R-imc.cols);
        //std::cout << "FaceWindow LARGER!" << std::endl;

        cv::copyMakeBorder(imc, imc, dw, dw, dw, dw, cv::BORDER_REFLECT);

        imw = imc.cols;
        buffer = (imw-R)/2;

        imc = imc(cv::Rect(buffer, buffer, R, R));
    }
    else
    {
        imc = imc(cv::Rect(buffer, buffer, R, R));
    }

#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs CROPCORE" << std::endl;
#endif
    return imc;
}


/**
 * @brief imutil::cropCore - crop image from the center (after uspcaling it)
 * @param im
 * @return Mat
 */
cv::Mat imutil::cropCore(const cv::Mat& im, int refImWidth, double *shiftBy)
{
    int64 t1 = cv::getTickCount();
    cv::Mat tmp1 = im.clone();
    double localRT = 2;

    if (refImWidth%2 == 1)
    {
        localRT = 4;
    }

    cv::resize(tmp1, tmp1, cv::Size(), localRT, localRT, cv::INTER_LANCZOS4); // cv::INTER_LINEAR // cv::INTER_LANCZOS4

    //    localRT = rfw::RT;
    double R = localRT*refImWidth;

    int imw = std::min<int>(tmp1.cols, tmp1.rows);
    int buffer = (imw-R)/2;

    cv::Mat tmp2;

    if (tmp1.cols >= R && tmp1.rows >= R)
    {
        tmp2 = tmp1(cv::Rect(buffer, buffer, R, R)).clone();
    }
    else
    {
//        int dw = 2*(R-std::min<int>(tmp1.cols,tmp1.rows));
        int dw = 2*(R-std::min<int>(tmp1.cols,tmp1.rows));
        //std::cout << "FaceWindow LARGER!" << std::endl;

        cv::copyMakeBorder(tmp1, tmp1, dw, dw, dw, dw, cv::BORDER_REFLECT);

        imw = tmp1.cols;
        buffer = (imw-R)/2;

        tmp2 = tmp1(cv::Rect(buffer, buffer, R, R)).clone();
    }

    cv::resize(tmp2, tmp2, cv::Size(), 1./localRT, 1./localRT, cv::INTER_LANCZOS4); // cv::INTER_LINEAR // cv::INTER_LANCZOS4

    imw = tmp2.cols;
//    tmp2 = tmp2(cv::Rect(imw/4,0,imw-imw/2,imw));

#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs CROPCORE" << std::endl;
#endif
    return tmp2;
}

/**
 * @brief imutil::resizeUpscaled
 * @param im - image to resize
 * @param r - resize ratio
 * @return
 */
cv::Mat imutil::resizeUpscaled(const cv::Mat &src, double r)
{
    cv::Mat im = src.clone();

    if (r == 0 || fabs(r) < 0.00001)
        return im;

    int64 t1 = cv::getTickCount();
    resizeUpscaledSelf(im, r);

#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs resizing" << std::endl;
#endif

    return im;
}

/**
 * @brief imutil::rotateSelf
 * @param im
 * @param theta - angle in degrees
 */
void imutil::rotateSelf(cv::Mat &im, double theta)
{
    if (theta == 0  || fabs(theta) < 0.00001)
        return;

    // resize rate
    // used to upscale image before translating. This will do interpolation for non-integer tx,ty.
    double rrt = rfw::RT;

    double diff = fabs(theta);

    double fine = 0.3;
    double superFine = 0.1;

    if (diff > fine) {
        rrt = 2;
    } else if (diff < fine && diff > superFine) {
        rrt = 4;
    } else if (diff < superFine ) {
        rrt = 8;
    }


    cv::resize(im, im, cv::Size(), rrt, rrt, cv::INTER_LANCZOS4);
    im = cvip::rotate(im, theta);
    cv::resize(im, im, cv::Size(), 1./rrt, 1./rrt, cv::INTER_LANCZOS4);
}

/**
 * @brief imutil::rotate
 * @param im
 * @param theta
 * @return rotated image - angle in degrees
 */
cv::Mat imutil::rotate(const cv::Mat &im, double theta)
{
    cv::Mat out = im.clone();
    imutil::rotateSelf(out,theta);
    return out;
}

/**
 * @brief imutil::resizeUpscaled
 * @param im - image to resize
 * @param r - resize ratio
 * @return
 */
void imutil::resizeUpscaledSelf(cv::Mat& im, double r)
{
    if (r == 1.  || fabs(r-1) < 0.000001)
        return;

    // resize rate
    // used to upscale image before translating. This will do interpolation for non-integer tx,ty.
    double rrt = rfw::RT;

    double diff = fabs(r-1.);

    double fine = 0.004;
    double superFine = 0.002;

    if (diff > fine) {
        rrt = 2;
    } else if (diff < fine && diff > superFine) {
        rrt = 4;
    } else if (diff < superFine ) {
        rrt = 8;
    }

    cv::resize(im, im, cv::Size(), r*rrt,  r*rrt, cv::INTER_LANCZOS4);
    cv::resize(im, im, cv::Size(), 1./rrt, 1./rrt, cv::INTER_LANCZOS4);
}

/**
 * @brief imutil::shift
 * @param im
 * @param tx
 * @param ty
 * @return
 */
cv::Mat imutil::shift(const cv::Mat& src, double tx, double ty)
{
    cv::Mat im = src.clone();
    if (tx == 0 && ty == 0)
        return im;

    imutil::shiftSelf(im, tx, ty);

    return im;
}

/**
 * @brief imutil::shift
 * @param im
 * @param tx
 * @param ty
 * @return
 */
void imutil::shiftSelf(cv::Mat& im, double tx, double ty)
{
    if (tx == 0 && ty == 0)
        return;

    // resize rate
    // used to upscale image before translating. This will do interpolation for non-integer tx,ty.
    double rrt = rfw::RT; // by default it's the largest

    double rem1 = std::max<double>(fabs(tx - (double)cvip::round(tx)), fabs(ty - (double)cvip::round(ty)));
    double rem2 = std::max<double>(fabs(tx - (double)cvip::round(tx*2)/2.), fabs(ty - (double)cvip::round(ty*2)/2.));
    double rem4 = std::max<double>(fabs(tx - (double)cvip::round(tx*4)/4.), fabs(ty - (double)cvip::round(ty*4)/4.));
    double rem8 = std::max<double>(fabs(tx - (double)cvip::round(tx*8)/8.), fabs(ty - (double)cvip::round(ty*8)/8.));

    if (rem1 == 0)
        rrt = 1.;
    else if (rem2 == 0)
        rrt = 2.;
    else if (rem4 == 0)
        rrt = 4.;
    else if (rem8 == 0)
        rrt = 8.;

    if (rrt != 1.)
        cv::resize(im, im, cv::Size(), rrt, rrt, cv::INTER_LANCZOS4);

    tx *= rrt;
    ty *= rrt;
    im = im.t();
    im = shiftRows(im, tx);
    im = im.t();
    im = shiftRows(im, ty);

    if (rrt != 1.)
        cv::resize(im, im, cv::Size(), 1./rrt, 1./rrt, cv::INTER_LANCZOS4);
}

/**
 * @brief imutil::shiftRows
 * @param im
 * @param t - translation size
 * @return
 */
cv::Mat imutil::shiftRows(cv::Mat& im, int t)
{
    cv::Mat nim = cv::Mat::zeros(im.rows, im.cols, im.type());

    for (int i=0; i<nim.rows; ++i)
    {
        int ni = (i+t)%nim.rows;
        if (ni<0)
            ni += nim.rows;

        cv::Mat dstRow = nim.row(ni);
        im.row(i).copyTo(dstRow);
    }

    return nim;
}

void imutil::applyLabelOldOld(cv::Mat &im, Label &l)
{
    int64 t1 = cv::getTickCount();

    cv::Mat trans = cv::Mat::zeros(2,3,CV_64F);
    trans.at<double>(0,0) = 1./l.sc;
    trans.at<double>(1,1) = 1./l.sc;

    trans.at<double>(0,2) = -l.tx;
    trans.at<double>(1,2) = -l.ty;


    double angle = -l.r;
    //    double angleRad = -l.r*cvip::PI/180.;
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(im.cols/2.,im.rows/2.), angle, 1.);
    //    trans = trans*rot;

    cv::Mat tim = im.clone();



    //    trans.push_back(cv::Mat<)



    cv::warpAffine(tim, im, trans, tim.size(), cv::INTER_LANCZOS4);
    tim = im.clone();
    cv::warpAffine(tim, im, rot, tim.size());

#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs ReSiZinG" << std::endl;
#endif
}



void imutil::applyLabelOld(cv::Mat &im, Label &l)
{
    im = imutil::shift(im, -l.tx, -l.ty);
    im = imutil::resizeUpscaled(im, 1./l.sc);
    im = imutil::rotate(im, -l.r);
}



void imutil::applyLabelReset(cv::Mat &im, const Label &l)
{


    int64 t1 = cv::getTickCount();

    cv::Mat trans = cv::Mat::zeros(2,3,CV_64F);
    trans.at<double>(0,0) = 1./l.sc;
    trans.at<double>(1,1) = 1./l.sc;

    trans.at<double>(0,2) = -l.tx;
    trans.at<double>(1,2) = -l.ty;

    cv::Mat tmp = (cv::Mat_<double>(1,3)<< 0., 0., 1);
    trans.push_back(tmp);


    double angle = -l.r;
    //    double angleRad = -l.r*cvip::PI/180.;
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(im.cols/2.,im.rows/2.), angle, 1.);
    cv::Mat H = rot*trans;

    cv::Mat tim = im.clone();







    // get rotation matrix for rotating the image around its center
    cv::Point2f center(tim.cols/2.0, tim.rows/2.0);



    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(center, tim.size(), angle).boundingRect();

    // adjust transformation matrix
//    H.at<double>(0,2) += bbox.width/2.0 - center.x;
//    H.at<double>(1,2) += bbox.height/2.0 - center.y;
    cv::warpAffine(tim, im, H, tim.size(), cv::INTER_LANCZOS4);












}




cv::Mat imutil::applyLabel(cv::Mat &im, const Label &l, double operationScale)
{
    int64 t1 = cv::getTickCount();

    cv::Mat trans = cv::Mat::zeros(2,3,CV_64F);
    trans.at<double>(0,0) = 1./(l.sc);
    trans.at<double>(1,1) = 1./(l.sc);

    trans.at<double>(0,2) = -l.tx;
    trans.at<double>(1,2) = -l.ty;

    cv::Mat tmp = (cv::Mat_<double>(1,3)<< 0., 0., 1);
    trans.push_back(tmp);


    double angle = -l.r;
    //    double angleRad = -l.r*cvip::PI/180.;
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(im.cols/2.,im.rows/2.), angle, 1.);
    cv::Mat H = rot*trans;

//    cvip::Image::printMat(H);




    cv::Mat tim = im.clone();

    cv::warpAffine(tim, im, H, tim.size(), cv::INTER_LANCZOS4);
//    tim = im.clone();
//    cv::warpAffine(tim, im, rot, tim.size());

#ifdef VERB_TIMING
    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs ReSiZinG" << std::endl;
#endif

    return H;
}


cv::Mat imutil::applyLabel(cv::Mat &im, const cv::Mat& H)
{
    cv::Mat tim = im.clone();

    cv::warpAffine(tim, im, H, tim.size(), cv::INTER_LANCZOS4);
//    tim = im.clone();
//    cv::warpAffine(tim, im, rot, tim.size());

//#ifdef VERB_TIMING
//    std::cout << (cv::getTickCount()-t1)/cv::getTickFrequency() << " secs ReSiZinG" << std::endl;
//#endif

    return H;
}



std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}



std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}



cv::Mat imutil::createCanonicalFace(const cv::Mat frame, const cv::Mat& pts)
{
    cv::Point leye = imutil::rectFromLandmarks(pts, "leye", cv::Size(frame.cols,frame.rows)).centerPoint();
    cv::Point reye = imutil::rectFromLandmarks(pts, "reye", cv::Size(frame.cols,frame.rows)).centerPoint();
    cv::Point mouth = imutil::rectFromLandmarks(pts, "mouth", cv::Size(frame.cols,frame.rows)).centerPoint();

    cv::Point faceCenter = (leye+reye+mouth)*0.33333;
    double iod = cvip::Image::dist(leye, reye);

    double alpha = 3.6;
    int xStart = faceCenter.x-alpha/2.*iod;
    int yStart = faceCenter.y-alpha/2.*iod;



    int xEnd = xStart+alpha*iod;
    int yEnd = yStart+alpha*iod;

    int fW = frame.cols;
    int fH = frame.rows;

    if (yStart <= 0) yStart = 0;
    if (yEnd >= fH-1) yEnd = fH-1;
    if (xStart <= 0) xStart = 0;
    if (xEnd >= fW-1) xEnd = fW-1;




    cv::Rect fr(xStart, yStart, xEnd-xStart, yEnd-yStart);
    leye.x -= xStart;
    reye.x -= xStart;
    mouth.x -= xStart;
    leye.y -= yStart;
    reye.y -= yStart;
    mouth.y -= yStart;


    double W = fr.width;
    double H = fr.height;



    double resizeRatio = (double)rfw::imWs["face"]/fr.width;


    cvip::Rect cfr(fr.x, fr.y, fr.width, fr.height);
    cfr = cfr.extended(frame.size());
    fr = cfr.toCvStyle();

//    cv::Point t_leye(0.25*W, 0.25*H);
//    cv::Point t_reye(0.75*W, 0.25*H);
//    cv::Point t_mouth(0.50*W, 0.75*H);

//    cv::Mat T = cv::getAffineTransform(std::vector<cv::Point2f>({leye,reye,mouth}),std::vector<cv::Point2f>({t_leye,t_reye,t_mouth}));

    //! no face OR face/landmark detection failed
    //!

    if (0 == fr.width)
    {

        cv::Mat dummy;
        return dummy;
#ifdef GUI
//        cv::imshow("face", frame);
//        cv::waitKey(0);
#endif
    }

    cv::Mat src = frame(fr).clone();
    cv::resize(src, src, cv::Size(0,0), resizeRatio, resizeRatio);

//    cv::Mat dst(H, W, src.type());
//    cv::warpAffine(src,dst,T,cv::Size(0,0));

//    cv::imshow("amanin  yandim", src);
//    cv::waitKey(0);

//    cv::resize(dst, dst, cv::Size(120,150));

    return src;
//    return dst;
}






cvip::Rect imutil::rectFromLandmarks(const cv::Mat &pts, const std::string& patchType, const cv::Size sz)
{
    // width/height to be used as a reference while cropping
    int w = sz.width;
    int h = sz.height;

    if (patchType == "face")
    {
        cv::Point2f leye = imutil::pointFromLandmarks(pts, "leye");
        cv::Point2f reye = imutil::pointFromLandmarks(pts, "reye");

        cv::Point2f midEyes(cvip::round((leye.x+reye.x)/2.),
                            cvip::round((leye.y+reye.y)/2.));

        // interocular distance
        double d = cvip::Image::dist(leye, reye);

        // constants that will define where we should extract the rectangle from
        double kt = 0.6;    // top
        double kb = 1.5;   // bottom
        double kl = 1.05;    // left
        double kr = 1.05;    // right

        // coumpute bounding box
        int top = midEyes.y-cvip::round(kt*d);
        int bottom = midEyes.y+cvip::round(kb*d);
        int left =  midEyes.x-cvip::round(kl*d);
        int right =  midEyes.x+cvip::round(kr*d);

        if (top <= 0) top = 0;
        if (bottom >= h-1) bottom = h-1;
        if (left <= 0) left = 0;
        if (right >= w-1) right = w-1;

        return cvip::Rect(left, top, right-left, bottom-top);
    }
    else if (patchType == "body")
    {
        cvip::Rect faceRect = imutil::rectFromLandmarks(pts, "face", sz);
        cv::Point2f faceCenter = faceRect.centerPoint();

        // constants that define bounding box for body
        double k3 = 3.6775;
        double k4 = 3.5892;
        double k2 = 3.0304;
        double k1 = 1.9790;

        double d = faceRect.height/2.;

        int fx = faceCenter.x;
        int fy = faceCenter.y;

        int top = fy-cvip::round(k1*d);
        int bottom = fy+cvip::round(k2*d);

        d = faceRect.width/2.;
        int left = fx-cvip::round(k3*d);
        int right = fx+cvip::round(k4*d);

        if (top <= 0) top = 0;
        if (bottom >= h-1) bottom = h-1;
        if (left <= 0) left = 0;
        if (right >= w-1) right = w-1;

        return cvip::Rect(left, top, right-left, bottom-top);
    }
    else if (patchType == "leye" || patchType == "reye" || patchType == "mouth")
    {
        cv::Point2f pt = imutil::pointFromLandmarks(pts, patchType);

        cv::Point2f leye = imutil::pointFromLandmarks(pts, "leye");
        cv::Point2f reye = imutil::pointFromLandmarks(pts, "reye");

        double d = cvip::Image::dist(leye, reye);

        if (patchType == "leye" || patchType == "reye")
            d = d*0.9;

        int x1 = pt.x-d/2;
        int x2 = x1+d;
        int y1 = pt.y-d/2;
        int y2 = y1+d;

        if (x1 < 0 || y1 < 0 || x2 >= sz.width || y2 >= sz.height)
        {
            return cvip::Rect(0,0,d,d);
        }

        return cvip::Rect(pt.x-d/2, pt.y-d/2, d, d);
    }
    else
    {
        std::cerr << "Unknown patch type for rectangle localisation!" << std::endl;
        return cvip::Rect();
    }
}



cv::Point2f imutil::pointFromLandmarks(const cv::Mat &pts, const std::string& patchType)
{
    std::vector<cv::Point2f> tmpPts;

    if (patchType == "lec" || patchType == "leye")
    {
        //! std::vector<int> ptIds({19, 22}); // these were for intraface
        std::vector<int> ptIds({36, 39}); // these were for intraface
        for (size_t i=0; i<ptIds.size(); ++i)
        {
            cv::Point2f pt(pts.at<float>(ptIds[i], 0),pts.at<float>(ptIds[i], 1));
            tmpPts.push_back(pt);
        }
    }
    else if (patchType == "rec" || patchType == "reye")
    {
        std::vector<int> ptIds({42, 45});
        for (size_t i=0; i<ptIds.size(); ++i)
        {
            cv::Point2f pt(pts.at<float>(ptIds[i], 0),pts.at<float>(ptIds[i], 1));
            tmpPts.push_back(pt);
        }
    }
    else if (patchType == "mouth")
    {
        std::vector<int> ptIds({48, 54});
        for (size_t i=0; i<ptIds.size(); ++i)
        {
            cv::Point2f pt(pts.at<float>(ptIds[i], 0),pts.at<float>(ptIds[i], 1));
            tmpPts.push_back(pt);
        }
    }
    else
    {
        std::cerr << "Invalid patch type for point localisation!" << std::endl;
    }

    cv::Point2f ret = imutil::avgPt(tmpPts);
    /*
    float rx = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float ry = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    rx = (rx-1)*1.15;
    ry = (ry-1)*1.15;

    ret.x += rx;
    ret.y += ry;
    */
    return ret;
}




/**
 * @brief DbCreator::avgPt -- compute the average of a collection of points
 *        (e.g. used to "find" eye center from the points that surround the eye
 * @param pts
 * @return
 */
cv::Point2f imutil::avgPt(const std::vector<cv::Point2f> &pts)
{
    cv::Point2f pt(0,0);
    for (size_t i=0; i<pts.size(); ++i) {
        pt.x += pts[i].x;
        pt.y += pts[i].y;
    }

    pt.x /= pts.size();
    pt.y /= pts.size();

    return pt;
}




