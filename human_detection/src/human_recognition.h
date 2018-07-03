#include "includs.h"

#define FNUM 329
//struct section
typedef struct{
	MatrixXf mat_a;
	MatrixXf mat_b;
}Mats;

typedef struct{
	pcl::PointXYZ min;
	pcl::PointXYZ max;
	pcl::PointXYZ position;
	Eigen::Matrix3f rot_mat;
	Eigen::Matrix3f rot_mat_inv;
	Eigen::Vector3f centroid;
	float width;
	float length;
	float height;
	double rsl;	
	double rsl_rev;	
	int object_flag;
	int pre_id;
	int byc_flag;
	int likelihood_flag;
	float likelihood;
	float velocity;
	int points_num;
	int track_num;
	float distance;
	float sub_likelihood;
	float track_vel;
}clusterInformation;

typedef struct{
	pcl::PointCloud<pcl::PointXYZINormal> clouds;
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudsPtr;
	Vector3f centroid;
	int size_ok;
}dividedCluster;

typedef struct{
	pcl::PointCloud<pcl::PointXYZINormal> clouds;
	float dist;
	Vector3f centroid;
	bool rm_flag;
}clusterTemporary;

typedef struct{
	vector<clusterInformation> info;
	vector<clusterTemporary> cluster;
}clusterBox;

typedef struct{
	vector<float> f1;//cloud size
	vector<float> f2; 
	vector<float> f3;//normalized moment of inertia tensor
	vector<float> f4;//slice feature
	vector<float> f5;//2D covariance
	vector<float> f6;//histogram of 2d
	vector<float> f7;//histogram of normal gradient
	vector<float> f8;//number of points
	vector<float> f9;//minimum deistance for cluster(distance to centroid)
	vector<double> f_i;
	double features[];
}shapeFeatures;

typedef struct{
	vector<double> min;
	vector<double> max;
}svmRange;

//function section
bool is_even(clusterTemporary input){
	if(!input.rm_flag) return true;
	else return false;
}

double S2D(const std::string& str){
	double tmp;
	stringstream ss;
	ss << str;
	ss >> tmp;
	return tmp;
}

void pcdWriter(int name, int label, pcl::PointCloud<pcl::PointXYZINormal> pt)
{
	pcl::PCDWriter writer;  
	std::stringstream ss;
	ss << name << ".pcd";
	cout<<"writing!!!!!!"<<endl;
	//writer.write<pcl::PointXYZ> ("clustering/check_pcd/rwrc_neg/" + ss.str (), pt, false);
	if(label>0)writer.write<pcl::PointXYZINormal> ("pcd_tmp/test_human_sample/pos/" + ss.str (), pt, false);
	else if(label<0)writer.write<pcl::PointXYZINormal> ("pcd_tmp/test_human_sample/neg/" + ss.str (), pt, false);
}
pcl::PointCloud<pcl::PointXYZINormal> normalCheck2(pcl::PointCloud<pcl::PointXYZI>::Ptr &pcl_in)
{	
	pcl::PointCloud<pcl::PointXYZINormal> pcl_pcs;
	size_t num=pcl_in->points.size();
	//pcl_pcs.points.resize(num);
	pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
	ne.setInputCloud (pcl_in);
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZI> ());
	ne.setSearchMethod (tree2);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	//ne.setRadiusSearch (1.0/14.0);
	//ne.setRadiusSearch (0.14);
	ne.setRadiusSearch (0.30);
	ne.compute (*cloud_normals);
	for(size_t i=0;i<num;i++){
		pcl::PointXYZINormal p;
		p.x=pcl_in->points[i].x;
		p.y=pcl_in->points[i].y;
		p.z=pcl_in->points[i].z;
		p.intensity=pcl_in->points[i].intensity;
		//p.intensity=0.0;
		p.normal_x=cloud_normals->points[i].normal_x;
		p.normal_y=cloud_normals->points[i].normal_y;
		p.normal_z=cloud_normals->points[i].normal_z;
		p.curvature=cloud_normals->points[i].curvature;
		//if(p.curvature>1.0)p.curvature=1.0;
		//if(p.curvature<0.0)p.curvature=0.0;
		//if(isnan(p.curvature))p.curvature=1.0;
		//else p.curvature=0.0;
		//cout<<p.curvature<<endl;
		//if(p.normal_z<0.9) pcl_pcs.points.push_back(p);

		//if(p.normal_y>1.0)p.normal_y=1.0;
		//else if(p.normal_y<-1.0)p.normal_y=-1.0;

		pcl_pcs.points.push_back(p);
	}
	return(pcl_pcs);
}
pcl::PointCloud<pcl::PointXYZI>::Ptr ptrTransform(pcl::PointCloud<pcl::PointXYZI> pcl_in){
	pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);

	output->width=1;
	output->height=pcl_in.points.size();
	output->points.resize(output->width*output->height);
	for(size_t i=0;i<pcl_in.points.size();i++)
	{
		output->points[i].x=pcl_in.points[i].x;
		output->points[i].y=pcl_in.points[i].y;
		output->points[i].z=pcl_in.points[i].z;
		output->points[i].intensity=pcl_in.points[i].intensity;
	}
	return output;
}
pcl::PointCloud<pcl::PointXYZINormal>::Ptr ptrTransform(pcl::PointCloud<pcl::PointXYZINormal> pcl_in){
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr output(new pcl::PointCloud<pcl::PointXYZINormal>);

	output->width=1;
	output->height=pcl_in.points.size();
	output->points.resize(output->width*output->height);
	for(size_t i=0;i<pcl_in.points.size();i++)
	{
		output->points[i].x=pcl_in.points[i].x;
		output->points[i].y=pcl_in.points[i].y;
		output->points[i].z=pcl_in.points[i].z;
		output->points[i].intensity=pcl_in.points[i].intensity;
		output->points[i].normal_x=pcl_in.points[i].normal_x;
		output->points[i].normal_y=pcl_in.points[i].normal_y;
		output->points[i].normal_z=pcl_in.points[i].normal_z;
		output->points[i].curvature=pcl_in.points[i].curvature;
	}
	return output;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr convertXN2XI(pcl::PointCloud<pcl::PointXYZINormal> pcl_in){
	pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);

	output->width=1;
	output->height=pcl_in.points.size();
	output->points.resize(output->width*output->height);
	for(size_t i=0;i<pcl_in.points.size();i++)
	{
		output->points[i].x=pcl_in.points[i].x;
		output->points[i].y=pcl_in.points[i].y;
		output->points[i].z=pcl_in.points[i].z;
		output->points[i].intensity=pcl_in.points[i].intensity;
	}
	return output;
}
Vector3f calculateCentroid(pcl::PointCloud<pcl::PointXYZINormal> pt)
{
	////calculate_centroid_section
	Vector3f centroid=Vector3f::Zero(3);
	for(size_t i=0;i<pt.points.size();i++){
		centroid[0]+=pt.points[i].x;
		centroid[1]+=pt.points[i].y;
		centroid[2]+=pt.points[i].z;
	}
	centroid[0]/=(float)pt.points.size();
	centroid[1]/=(float)pt.points.size();
	centroid[2]/=(float)pt.points.size();

	return centroid;
}

////////////////////////////////////////////////////////////////////////
pcl::PointXYZ calculateCentroid2(pcl::PointCloud<pcl::PointXYZI> pt)
{
	////calculate_centroid_section
	pcl::PointXYZ centroid;
	float num=(float)pt.points.size();
	for(size_t i=0;i<(size_t)num;i++){
		centroid.x+=pt.points[i].x;
		centroid.y+=pt.points[i].y;
		centroid.z+=pt.points[i].z;
	}
	centroid.x/=num;
	centroid.y/=num;
	centroid.z/=num;

	return centroid;
}
shapeFeatures mergeAllFeatures(shapeFeatures f)
{
	size_t f1=f.f1.size();
	size_t f2=f.f2.size();
	size_t f3=f.f3.size();
	size_t f4=f.f4.size();
	size_t f5=f.f5.size();
	size_t f6=f.f6.size();
	size_t f7=f.f7.size();
	size_t f8=f.f8.size();
	size_t f9=f.f9.size();
	size_t _size=f1+f2+f3+f4+f5+f6+f7+f8+f9;
	f.f_i.resize(_size);
	size_t num=0;
	for(size_t i=num;i<f.f1.size()+num;i++){
		f.f_i[i]=(double)f.f1[i-num];
	}
	num+=f.f1.size();
	for(size_t i=num;i<f.f2.size()+num;i++){
		f.f_i[i]=(double)f.f2[i-num];//3-8
	}
	num+=f.f2.size();
	for(size_t i=num;i<f.f3.size()+num;i++){
		f.f_i[i]=(double)f.f3[i-num];//9-14
	}
	num+=f.f3.size();
	for(size_t i=num;i<f.f4.size()+num;i++){
		f.f_i[i]=(double)f.f4[i-num];//15-34
	}
	num+=f.f4.size();
	for(size_t i=num;i<f.f5.size()+num;i++){
		f.f_i[i]=(double)f.f5[i-num];//35-40
	}
	num+=f.f5.size();
	for(size_t i=num;i<f.f6.size()+num;i++){
		f.f_i[i]=(double)f.f6[i-num];//41-183
	}
	num+=f.f6.size();
	for(size_t i=num;i<f.f7.size()+num;i++){
		f.f_i[i]=(double)f.f7[i-num];//184-327
	}
	num+=f.f7.size();
	for(size_t i=num;i<f.f8.size()+num;i++){
		f.f_i[i]=(double)f.f8[i-num];//42
	}
	num+=f.f8.size();
	for(size_t i=num;i<f.f9.size()+num;i++){
		f.f_i[i]=(double)f.f9[i-num];//43-186
	}
	num+=f.f9.size();
	return(f);
}
vector<float> mergeVector(vector<float> &vec1, vector<float> &vec2)
{
	vector<float> vec;
	//size_t num=vec1.size()+vec2.size();
	//vec.resize(num,0);
	for(size_t j=0;j<vec1.size();j++){
		vec.push_back(vec1[j]);
	}
	for(size_t j=0;j<vec2.size();j++){
		vec.push_back(vec1[j]);
	}
	return(vec);
}

pcl::PointCloud<pcl::PointXYZ> visualizer(Eigen::Matrix3f rot, pcl::PointXYZ min_pt, pcl::PointXYZ max_pt, pcl::PointXYZ position_pt)
{
	pcl::PointCloud<pcl::PointXYZ> output;
	Eigen::Vector3f position (position_pt.x, position_pt.y, position_pt.z);

	Eigen::Vector3f p1 (min_pt.x, min_pt.y, min_pt.z);
	Eigen::Vector3f p2 (min_pt.x, min_pt.y, max_pt.z);
	Eigen::Vector3f p3 (max_pt.x, min_pt.y, max_pt.z);
	Eigen::Vector3f p4 (max_pt.x, min_pt.y, min_pt.z);
	Eigen::Vector3f p5 (min_pt.x, max_pt.y, min_pt.z);
	Eigen::Vector3f p6 (min_pt.x, max_pt.y, max_pt.z);
	Eigen::Vector3f p7 (max_pt.x, max_pt.y, max_pt.z);
	Eigen::Vector3f p8 (max_pt.x, max_pt.y, min_pt.z);

	p1 = rot * p1 + position;
	p2 = rot * p2 + position;
	p3 = rot * p3 + position;
	p4 = rot * p4 + position;
	p5 = rot * p5 + position;
	p6 = rot * p6 + position;
	p7 = rot * p7 + position;
	p8 = rot * p8 + position;

	pcl::PointXYZ pt1 (p1 (0), p1 (1), p1 (2));
	pcl::PointXYZ pt2 (p2 (0), p2 (1), p2 (2));
	pcl::PointXYZ pt3 (p3 (0), p3 (1), p3 (2));
	pcl::PointXYZ pt4 (p4 (0), p4 (1), p4 (2));
	pcl::PointXYZ pt5 (p5 (0), p5 (1), p5 (2));
	pcl::PointXYZ pt6 (p6 (0), p6 (1), p6 (2));
	pcl::PointXYZ pt7 (p7 (0), p7 (1), p7 (2));
	pcl::PointXYZ pt8 (p8 (0), p8 (1), p8 (2));

	output.points.resize(24);
	output.points[0]=pt1;
	output.points[1]=pt2;
	output.points[2]=pt1;
	output.points[3]=pt4;
	output.points[4]=pt1;
	output.points[5]=pt5;
	output.points[6]=pt5;
	output.points[7]=pt6;
	output.points[8]=pt5;
	output.points[9]=pt8;
	output.points[10]=pt2;
	output.points[11]=pt6;
	output.points[12]=pt6;
	output.points[13]=pt7;
	output.points[14]=pt7;
	output.points[15]=pt8;
	output.points[16]=pt2;
	output.points[17]=pt3;
	output.points[18]=pt4;
	output.points[19]=pt8;
	output.points[20]=pt4;
	output.points[21]=pt3;
	output.points[22]=pt7;
	output.points[23]=pt3;

	return output;
}

