#include <opencv2/opencv.hpp>
#include<bits/stdc++.h>
using namespace cv;
using namespace std;    
int main() {
	cv::Mat img = cv::imread("/home/chenjie/.ssh/start/guard.jpg"),zerobule,blue,white;
	if (img.empty())
	{
		cout<<"未找到guard.jpg"<<endl;
		return -1;
	}
	cvtColor(img,zerobule,COLOR_BGR2HSV);
    Mat core=getStructuringElement(cv::MORPH_RECT, cv::Size(3,9),Point(-1,-1));
    inRange(zerobule, Scalar(0, 0, 190), Scalar(180,50, 255), zerobule);
	//inRange(zerobule, Scalar(100, 100, 190), Scalar(140, 255, 200), blue);
	//bitwise_or(white,blue,zerobule);
	erode(zerobule,zerobule,core,Point(-1,-1),1,BORDER_CONSTANT);
	dilate(zerobule,zerobule,core,Point(-1,-1),2,BORDER_CONSTANT);
	threshold(zerobule,zerobule,0, 255, THRESH_BINARY|THRESH_OTSU);
	imshow("装甲板转化结果",zerobule);
	waitKey(0);
    vector<vector<Point>> ctv;
    findContours(zerobule,ctv, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	struct lightbar {
    RotatedRect rect;  
    Point2f endpoints[2];
 };
    vector<lightbar> lightbars;
	for(int i=0;i<ctv.size();i++){
		std::vector<cv::Point>& contour=ctv[i];
			RotatedRect rect = minAreaRect(contour);
        float ratio=max(rect.size.height,rect.size.width)/min(rect.size.height,rect.size.width);
        float area=rect.size.area();
			lightbar record;
			record.rect=rect;
			Vec4f l;
			fitLine(contour,l,DIST_L2,0,0.01,0.01);
			vector<double> pointj;
			for (auto& p:contour) {
            float t = (p.x-l[2])*l[0]+(p.y-l[3])*l[1];
            pointj.push_back(t);
        }
		int minn=min_element(pointj.begin(), pointj.end()) - pointj.begin();
			int maxn=max_element(pointj.begin(), pointj.end()) - pointj.begin();
			//cout<<endl<<minn<<" "<<maxn<<endl;
			record.endpoints[0]=contour[minn];
			record.endpoints[1]=contour[maxn];
			lightbars.push_back(record);
	}
	Mat result = img.clone();
for(int i=0;i<lightbars.size();i++){
	Point2f p1=lightbars[0].endpoints[0]; 
    Point2f p2=lightbars[1].endpoints[0]; 
    Point2f p3=lightbars[1].endpoints[1]; 
    Point2f p4=lightbars[0].endpoints[1]; 
	line(result,p1,p3,Scalar(0,255,0),2,LINE_AA);
	line(result,p3,p2,Scalar(0,255,0),2,LINE_AA);
	line(result,p2,p4,Scalar(0,255,0),2,LINE_AA);
	line(result,p4,p1,Scalar(0,255,0),2,LINE_AA); 
}
     imshow("装甲板检测结果",result);
	 waitKey(0);
	 destroyAllWindows();
	 vector<Point3f> solidpoint;
	 vector<Point2f> corners;
	 corners.push_back(lightbars[1].endpoints[0]);
	corners.push_back(lightbars[0].endpoints[1]);
	 corners.push_back(lightbars[0].endpoints[0]);
	 	 corners.push_back(lightbars[1].endpoints[1]);
	 cout<<corners<<endl;
	 solidpoint.emplace_back(0,-0.625,0.25);
	 solidpoint.emplace_back(0,0.625,0.25);
	 solidpoint.emplace_back(0,0.625,-0.25);
	 solidpoint.emplace_back(0,-0.625,-0.25);
	 Mat rvec,tvec;
	  Mat includeinsex=(Mat_<double>(3,3) << 
        928.130989,0,377.572945,
        0,930.138391,283.892859,
        0,0,1.0);
    Mat mistakechange=(Mat_<double>(1,5) <<-0.254433647,0.569431382,0.00365405229,-0.00109433818,-1.33846840);
	 solvePnP(solidpoint,corners,includeinsex,mistakechange,rvec,tvec);
	 float zlength=1.0;
	 drawFrameAxes(result,includeinsex,mistakechange,rvec,tvec,zlength);
	 imshow("装甲板检测结果",result);
	 waitKey(0);
	 destroyAllWindows();
	 return 0;
}