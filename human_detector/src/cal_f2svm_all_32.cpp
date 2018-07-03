#include <ros/ros.h>
#include <boost/thread.hpp>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
///////Eigen///////
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Eigenvalues"
#include "Eigen/LU"
///////Eigen///////

#include <iostream>
#include <math.h>
#include <vector>
#include <iostream>
#include <std_msgs/Int32.h>

using namespace std;
using namespace Eigen;

#include <stdio.h>
#include <sstream>
#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/features/normal_3d.h>
#include <pcl-1.7/pcl/common/common.h>
#include <pcl-1.7/pcl/common/transforms.h>
#include <pcl-1.7/pcl/features/moment_of_inertia_estimation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <omp.h>

//#define N 3
//#define PN 4000//読み込むPCDファイルの数
int PN=1000;
//#define LABEL -1//ポジティブ--> 1   ネガティブ--> -1
//boost::mutex laser_mutex;

//MODE: 32E->32, 64E->64//
//#define MODE 64
#define MODE 32
#define normal_MODE 1//1->use normal, 0->not use normal
#define remove_MODE 0//1-all point, 0->only near point
//STATICMODE: if robot moving -> 1
#define STATICMODE 0
//#define test_MODE 8//0->all, 1->size, 2->3Dcov, 3->moment, 4->block, 5->2Dcov, 6->other 3Dcov, 7->other moment, 8->other block, 9->other 2Dcov
#define STARTNUM 0
const int mode_max=16;

int cout_count=0;
float cout_tmp=0;

typedef struct{
	MatrixXf mat_a;
	MatrixXf mat_b;
}Mats;

pcl::PointXYZ cluster_minimum_pt;
pcl::PointXYZ cluster_maximum_pt;
pcl::PointXYZ cluster_position_pt;
Eigen::Matrix3f transformMatrix(pcl::PointCloud<pcl::PointXYZINormal> pcl_in) 
{ 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width=1;
	cloud->height=pcl_in.points.size();
	cloud->points.resize(cloud->width*cloud->height);
	for(size_t i=0;i<pcl_in.points.size();i++){
		cloud->points[i].x=pcl_in.points[i].x;
		cloud->points[i].y=pcl_in.points[i].y;
		cloud->points[i].z=pcl_in.points[i].z;
	}
	Eigen::Vector4f min; 
	Eigen::Vector4f max; 
	pcl::getMinMax3D (*cloud, min, max); 
	for(size_t i=0;i<pcl_in.points.size();i++){
		cloud->points[i].z=0.0;
	}

	Eigen::Matrix3f rot;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	Eigen::Matrix3f rotational_matrix_OBB;
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	rot=rotational_matrix_OBB.inverse();

	cluster_minimum_pt=min_point_OBB;
	cluster_minimum_pt.z=min[2];
	cluster_maximum_pt=max_point_OBB;
	cluster_maximum_pt.z=max[2];
	cluster_position_pt=position_OBB;

	return rot;
}

Vector3f calculateCentroid(pcl::PointCloud<pcl::PointXYZINormal> &pt)
{
	////calculate_centroid_section
	Vector3f c;
	float sum_x=0;
	float sum_y=0;
	float sum_z=0;

	for(size_t i=0;i<pt.points.size();i++){
		sum_x+=pt.points[i].x;
		sum_y+=pt.points[i].y;
		sum_z+=pt.points[i].z;
	}
	sum_x/=(float)pt.points.size();
	sum_y/=(float)pt.points.size();
	sum_z/=(float)pt.points.size();

	c[0]=sum_x;
	c[1]=sum_y;
	c[2]=sum_z;

	return c;
}
//Function of string to double
double S2D(const std::string& str){
	double tmp;
	stringstream ss;
	ss << str;
	ss >> tmp;
	return tmp;
}

pcl::PointCloud<pcl::PointXYZINormal> calRotationMatrix(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in, float rad)
{
	pcl::PointCloud<pcl::PointXYZINormal> rotate_pcl;
	MatrixXf aq_mat=MatrixXf::Zero(4,1);
	MatrixXf rot_mat=MatrixXf::Zero(4,4);
	MatrixXf rsl_mat=MatrixXf::Zero(4,1);
	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZINormal p;
		aq_mat(0,0)=pcl_in.points[i].x;
		aq_mat(1,0)=pcl_in.points[i].y;
		aq_mat(2,0)=pcl_in.points[i].z;
		aq_mat(3,0)=1.0;
		rot_mat(0,0)=cos(rad);
		rot_mat(0,1)=sin(rad);
		rot_mat(1,0)=-sin(rad);
		rot_mat(1,1)=cos(rad);
		rot_mat(2,2)=1.0;
		rot_mat(3,3)=1.0;
		rsl_mat=rot_mat*aq_mat;
		p.x=rsl_mat(0,0)/rsl_mat(3,0);
		p.y=rsl_mat(1,0)/rsl_mat(3,0);
		p.z=rsl_mat(2,0)/rsl_mat(3,0);
		p.intensity=pcl_in.points[i].intensity;
		p.curvature=pcl_in.points[i].curvature;
		p.normal_x=pcl_in.points[i].normal_x;
		p.normal_y=pcl_in.points[i].normal_y;
		p.normal_z=pcl_in.points[i].normal_z;
		rotate_pcl.points.push_back(p);
	}
	return (rotate_pcl);
}
vector<float> calMomentTensor(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in)
{
	vector<float> f_;
	Matrix3f nmit = MatrixXf::Zero(3,3);
	Matrix3f tensor = Matrix3f::Zero(3,3);

	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZ p;
		p.x=pcl_in.points[i].x;
		p.y=pcl_in.points[i].y;
		p.z=pcl_in.points[i].z;
		tensor(0,0)+=pow(p.y,2)+pow(p.z,2);
		tensor(0,1)+=-(p.y)*(p.x);
		tensor(0,2)+=-(p.z)*(p.x);
		tensor(1,0)+=-(p.x)*(p.y);
		tensor(1,1)+=pow(p.z,2)+pow(p.x,2);
		tensor(1,2)+=-(p.z)*(p.y);
		tensor(2,0)+=-(p.x)*(p.z);
		tensor(2,1)+=-(p.y)*(p.z);
		tensor(2,2)+=pow(p.x,2)+pow(p.y,2);
	}

	nmit=tensor.normalized();

	for(int i=0;i<3;i++){
		for(int j=i;j<3;j++){
			float tmp=nmit(i,j);
			f_.push_back(tmp);
		}
	}
	return(f_);
}
vector<float> cal3DCovariance(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in){
	vector<float> f_;
	//3次元共分散行列用のベクター計算
	vector<Vector3f> dev3d;//deviation(偏差)

	for(size_t i=0;i<pcl_in.points.size();i++){
		Vector3f dev;
		dev[0]=pcl_in.points[i].x;
		dev[1]=pcl_in.points[i].y;
		dev[2]=pcl_in.points[i].z;
		dev3d.push_back(dev);
	}

	size_t n_pt = pcl_in.points.size();
	MatrixXf dev1 = MatrixXf::Zero(n_pt,3);
	MatrixXf dev2 = MatrixXf::Zero(3,n_pt);
	Matrix3f cov_3d = MatrixXf::Zero(3,3);

	for(size_t i=0;i<n_pt;i++){
		dev1(i,0)=dev3d[i](0);
		dev1(i,1)=dev3d[i](1);
		dev1(i,2)=dev3d[i](2);
		dev2(0,i)=dev3d[i](0);
		dev2(1,i)=dev3d[i](1);
		dev2(2,i)=dev3d[i](2);
	}

	cov_3d=dev2*dev1/((float)n_pt-1.0);
	//MatrixXf cov_diagonal=MatrixXf::Zero(3,1);
	//cov_diagonal=cov_3d.diagonal();
	for(int i=0;i<3;i++){
		for(int j=i;j<3;j++){
			float tmp=cov_3d(i,j);
			f_.push_back(tmp);
		}
		//float dia = cov_diagonal(i,0);
		//f2_d.push_back(dia);
	}
	return(f_);
}
vector<float> cal2DCovariance(pcl::PointCloud<pcl::PointXYZ> &pcl_in)
{
	//	Vector3f a_cent = calculateCentroidPC(pcl_in);	
	vector<float> f_;
	vector<Vector3f> dev;
	for(size_t i=0;i<pcl_in.points.size();i++){
		Vector3f d;
		d[0]=pcl_in.points[i].x;
		d[1]=pcl_in.points[i].y;
		d[2]=pcl_in.points[i].z;
		dev.push_back(d);
	}
	int n_pt=pcl_in.points.size();

	MatrixXf eg_1=MatrixXf::Zero(n_pt,2);
	MatrixXf eg_2=MatrixXf::Zero(2,n_pt);
	Matrix2f cov_eg=MatrixXf::Zero(2,2);
	Matrix2f dev_eg=MatrixXf::Zero(2,2);

	for(size_t i=0;i<pcl_in.points.size();i++){
		eg_1(i,0)=dev[i](0);
		eg_1(i,1)=dev[i](2);
		eg_2(0,i)=dev[i](0);
		eg_2(1,i)=dev[i](2);
	}
	dev_eg=eg_2*eg_1;
	cov_eg=dev_eg/((float)n_pt-1);

	//MatrixXf dia=MatrixXf::Zero(2,1);
	//dia=cov_eg.diagonal();

	for(int i=0;i<2;i++){
		for(int j=i;j<2;j++){
			float tmp=cov_eg(i,j);
			if(isinf(tmp))tmp=0;
			f_.push_back(tmp);
		}
	}
	return(f_);	
}
vector<float> calBlockFeature(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in, float minmax_tmp, float height)
{
	vector<float> f_;
	int pitch = 10;		
	float cluster_block = height/pitch;
	vector<Vector2f> feature_delta;
	float delta_x;
	float delta_y;

	for(int h=0;h<pitch;h++){
		float max_x=0.0;
		float max_y=0.0;
		float min_x=0.0;
		float min_y=0.0;
		float high=cluster_block+(cluster_block*h)+minmax_tmp;
		float low=cluster_block*h+minmax_tmp;
		float sum_x=0.0;
		Vector2f delta;
		float check_count=0;
		for(size_t i=0;i<pcl_in.points.size();i++){
			if(pcl_in.points[i].z>low && high>pcl_in.points[i].z){
				check_count++;
				float max_tmp_x = pcl_in.points[i].x;
				float max_tmp_y = pcl_in.points[i].y;
				float min_tmp_x = pcl_in.points[i].x;
				float min_tmp_y = pcl_in.points[i].y;

				if(max_tmp_x>max_x) max_x=max_tmp_x;
				if(max_tmp_y>max_y) max_y=max_tmp_y;
				if(min_tmp_x<min_x) min_x=min_tmp_x;
				if(min_tmp_y<min_y) min_y=min_tmp_y;

				sum_x += fabs(pcl_in.points[i].x); 
			}
		}
		//cout<<"SUM="<<sum_x<<" ave="<<sum_x/check_count<<endl;
		//if(check_count>0){
		delta_x=max_x-min_x;
		delta_y=max_y-min_y;
		//}
		delta[0]=delta_x;
		delta[1]=delta_y;
		feature_delta.push_back(delta);	
	}
	
	for(int i=(pitch-1);i>-1;i--){
		//てことで↓ でΔのプッシュバック
		float delta_0=feature_delta[i](0);
		bool infcheck_0=isinf(delta_0);
		if(infcheck_0)delta_0=0.0;
		f_.push_back(delta_0);
		float delta_1=feature_delta[i](1);
		bool infcheck_1=isinf(delta_1);
		if(infcheck_1)delta_1=0.0;
		f_.push_back(delta_1);
	}
	return(f_);
}
//vector<double> mergeAllFeatures(vector<float> f1, vector<float> f2, vector<float> f3, vector<float> f4, vector<float> f5, vector<float>f6, vector<float> f7, vector<float> f8)
vector<double> mergeAllFeatures(vector<float> f1, vector<float> f2, vector<float> f3, vector<float> f4, vector<float> f5, vector<float>f6, vector<float>f7, vector<float>f8, vector<float>f9)
{
	vector<double> f_i;
	size_t _size=f1.size()+f2.size()+f3.size()+f4.size()+f5.size()+f6.size()+f7.size()+f8.size()+f9.size();
	f_i.resize(_size);
	size_t num=0;
	for(size_t i=num;i<f1.size()+num;i++){
		f_i[i]=(double)f1[i-num];
	}
	num+=f1.size();
	for(size_t i=num;i<f2.size()+num;i++){
		f_i[i]=(double)f2[i-num];//3-8
	}
	num+=f2.size();
	for(size_t i=num;i<f3.size()+num;i++){
		f_i[i]=(double)f3[i-num];//9-14
	}
	num+=f3.size();
	for(size_t i=num;i<f4.size()+num;i++){
		f_i[i]=(double)f4[i-num];//15-34
	}
	num+=f4.size();
	for(size_t i=num;i<f5.size()+num;i++){
		f_i[i]=(double)f5[i-num];//35-40
	}
	num+=f5.size();
	for(size_t i=num;i<f6.size()+num;i++){
		f_i[i]=(double)f6[i-num];//41-183
	}
	num+=f6.size();
	for(size_t i=num;i<f7.size()+num;i++){
		f_i[i]=(double)f7[i-num];//184-327
	}
	num+=f7.size();
	for(size_t i=num;i<f8.size()+num;i++){
		f_i[i]=(double)f8[i-num];//328
	}
	num+=f8.size();
	for(size_t i=num;i<f9.size()+num;i++){
		f_i[i]=(double)f9[i-num];//329
	}
	num+=f9.size();
	/*for(int i=0;i<41;i++){
	  features[i]=f_i[i];
	  }*/
	return(f_i);
}
/*vector<double> mergeAllFeatures(vector<float> f1, vector<float> f2, vector<float> f3, vector<float> f4, vector<float> f5)
{
	vector<double> f_i;
	f_i.resize(41);
	f_i[0]=(double)f1[0];
	f_i[1]=(double)f1[1];
	f_i[2]=(double)f1[2];
	for(size_t i=3;i<f2.size()+3;i++){
		f_i[i]=(double)f2[i-3];//3-8
	}
	for(size_t i=9;i<f3.size()+9;i++){
		f_i[i]=(double)f3[i-9];//9-14
	}
	for(size_t i=15;i<f4.size()+15;i++){
		f_i[i]=(double)f4[i-15];//15-34
	}
	for(size_t i=35;i<f5.size()+35;i++){
		f_i[i]=(double)f5[i-35];//36-40
	}
	return(f_i);
}*/
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

pcl::PointCloud<pcl::PointXYZINormal> normalCheck(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in1)
{	
	//cout<<"start normal check"<<endl;
	pcl::PointCloud<pcl::PointXYZINormal> pcl_pcs;
	pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_in (new pcl::PointCloud<pcl::PointXYZI>);
	pcl_in->points.resize(pcl_in1.points.size());
	for(size_t i=0;i<pcl_in1.points.size();i++){
		pcl_in->points[i].x=pcl_in1.points[i].x;
		pcl_in->points[i].y=pcl_in1.points[i].y;
		pcl_in->points[i].z=pcl_in1.points[i].z;
		pcl_in->points[i].intensity=pcl_in1.points[i].intensity;
	}
	if(normal_MODE==1){
		pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
		ne.setInputCloud (pcl_in);
		pcl::search::KdTree<pcl::PointXYZI>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZI> ());
		ne.setSearchMethod (tree2);
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
		ne.setRadiusSearch (0.14);
		ne.compute (*cloud_normals);
		//cout<<cloud_normals->points.size()<<endl;
		// cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
		for(size_t i=0;i<pcl_in->points.size();i++){
			pcl::PointXYZINormal p;
			p.x=pcl_in->points[i].x;
			p.y=pcl_in->points[i].y;
			p.z=pcl_in->points[i].z;
			p.intensity=pcl_in->points[i].intensity;
			p.normal_x=cloud_normals->points[i].normal_x;
			p.normal_y=cloud_normals->points[i].normal_y;
			p.normal_z=cloud_normals->points[i].normal_z;
			p.curvature=cloud_normals->points[i].curvature;
			//if(p.curvature>1.0)p.curvature=1.0;
			//else if(p.curvature<0.0)p.curvature=0.0;

			//if(p.normal_z<0.9) 
			pcl_pcs.points.push_back(p);
		}
	}else{
		for(size_t i=0;i<pcl_in->points.size();i++){
			pcl::PointXYZINormal p;
			p.x=pcl_in->points[i].x;
			p.y=pcl_in->points[i].y;
			p.z=pcl_in->points[i].z;
			p.intensity=pcl_in->points[i].intensity;
			pcl_pcs.points.push_back(p);
		}
	}
	return(pcl_pcs);
}
pcl::PointCloud<pcl::PointXYZINormal> normalCheck2(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_in)
{	
	//cout<<"start normal check"<<endl;
	pcl::PointCloud<pcl::PointXYZINormal> pcl_pcs;
	size_t num=pcl_in->points.size();
	//pcl_pcs.points.resize(num);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (pcl_in);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
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
		p.intensity=0.0;
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
Mats histogramProcess(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, int e)
{
	size_t ii=14;
	size_t jj=7;
	if(e==3){
		ii=9;
		jj=5;
	}
	//cout<<"eeeeeeeeeeeeeeeeeeee="<<e<<endl;
	Mats histograms;
	MatrixXf histogram=MatrixXf::Zero(ii,jj);
	MatrixXf histogram_normal=MatrixXf::Zero(ii,jj);
	/*
	float min_e1=0;
	float max_e1=0;
	float min_e2=-0.5;
	float max_e2=0.5;
	for(size_t i=0;i<pcl_in.points.size();i++){
		float z1=pcl_in.points[i].z;
		float z2=pcl_in.points[i].z;
		if(z1>max_e1)max_e1=z1;
		if(min_e1>z2)min_e1=z2;
	}*/
	//学習データは距離を考慮していないのでこっちで
	float min_e1=0;
	float max_e1=0;
	float min_e2=0;
	float max_e2=0;
	if(e==2){
		for(size_t i=0;i<pcl_in.points.size();i++){
			float z1=pcl_in.points[i].z;
			float z2=pcl_in.points[i].z;
			float x1=pcl_in.points[i].x;
			float x2=pcl_in.points[i].x;
			if(z1>max_e1)max_e1=z1;
			if(min_e1>z2)min_e1=z2;
			if(x1>max_e2)max_e2=x1;
			if(min_e2>x2)min_e2=x2;
		}
	}else if(e==3){
		for(size_t i=0;i<pcl_in.points.size();i++){
			float z1=pcl_in.points[i].z;
			float z2=pcl_in.points[i].z;
			float y1=pcl_in.points[i].y;
			float y2=pcl_in.points[i].y;
			if(z1>max_e1)max_e1=z1;
			if(min_e1>z2)min_e1=z2;
			if(y1>max_e2)max_e2=y1;
			if(min_e2>y2)min_e2=y2;
		}
	}
	float d_e1=max_e1-min_e1;
	float d_e2=max_e2-min_e2;
	pcl::PointCloud<pcl::PointXYZINormal> pcl_tmp;
	
	int points_num=0;
	int max_points=0;
	int min_points=0;
	float max_normal=0;
	float min_normal=0;

	for(size_t i=0;i<ii;i++){
		for(size_t j=0;j<jj;j++){
			float sum_normal=0;
			points_num=0;
			pcl_tmp.points.clear();
			pcl_tmp=pcl_in;
			pcl_in.points.clear();
			for(size_t k=0;k<pcl_tmp.points.size();k++){
				pcl::PointXYZINormal p;
				p.x=pcl_tmp.points[k].x;
				p.y=pcl_tmp.points[k].y;
				p.z=pcl_tmp.points[k].z;
				p.normal_y=pcl_tmp.points[k].normal_y;
				//p.curvature=pcl_tmp.points[k].curvature;
				float pt_e1=p.z;
				float pt_e2=p.x;
				if(e==3)pt_e2=p.y;
				bool eq1=((max_e1-d_e1/(float)ii*i)>pt_e1&&pt_e1>(max_e1-d_e1/(float)ii*(i+1)));
				bool eq2=((min_e2+d_e2/(float)jj*j)<pt_e2&&pt_e2<(min_e2+d_e2/(float)jj*(j+1)));

				if(eq1&&eq2){
					sum_normal+=p.normal_y;
					points_num++;
				}
				else pcl_in.points.push_back(p);
			}

			float avg_normal=0;
			bool sum_normal_check=(isnan(sum_normal));
			bool zero_check=(sum_normal==0||points_num==0);
			if(!sum_normal_check&&!zero_check)avg_normal=fabs(sum_normal/points_num);
			if(points_num!=0){
				if(max_points<points_num)max_points=points_num;
				if(min_points>points_num)min_points=points_num;
				if(max_normal<avg_normal)max_normal=avg_normal;
				if(min_normal>avg_normal)min_normal=avg_normal;
				histogram(i,j)=points_num;
				histogram_normal(i,j)=avg_normal;
				sum_normal=0;
			}
		}
	}
	for(size_t i=0;i<ii;i++){
		for(size_t j=0;j<jj;j++){
			histogram(i,j)=(float)(histogram(i,j)-min_points)/(float)(max_points-min_points);
			histogram_normal(i,j)=(histogram_normal(i,j)-min_normal)/(max_normal-min_normal);
		}
	}
	//histogram.normalize();
	//histogram_normal.normalize();
	histograms.mat_a=histogram;
	histograms.mat_b=histogram_normal;

	//cout<<"hist"<<endl<<histograms.mat_a<<endl<<endl;
	//cout<<"hist_normal"<<endl<<histograms.mat_b<<endl<<endl;
	return histograms;
}
//////////////////////////////////////////////////////////////////////////////////////////
vector<vector<float> > histogramBrain(pcl::PointCloud<pcl::PointXYZINormal> pcl_in)
{

	Mats histograms_e2;
	Mats histograms_e3;

	histograms_e2=histogramProcess(pcl_in,2);
	histograms_e3=histogramProcess(pcl_in,3);

	vector<vector<float> > hist_features;
	hist_features.resize(2);

	size_t ii=14;
	size_t jj=7;
	for(size_t num=0;num<2;num++){
		for(size_t i=0;i<ii;i++){
			for(size_t j=0;j<jj;j++){
				if(num==0){
					hist_features[0].push_back(histograms_e2.mat_a(i,j));
					hist_features[1].push_back(histograms_e2.mat_b(i,j));
				}
				if(num==1){
					hist_features[0].push_back(histograms_e3.mat_a(i,j));
					hist_features[1].push_back(histograms_e3.mat_b(i,j));
				}
			}
		}

		//cout<<"size0="<<hist_features[0].size()<<endl;
		//cout<<"size1="<<hist_features[1].size()<<endl;
		ii=9;
		jj=5;
	}
	return hist_features;
}
/*
MatrixXf histogramProcess(pcl::PointCloud<pcl::PointXYZINormal> in, int e)
{
	pcl::PointCloud<pcl::PointXYZINormal> pcl_in=in;
	float min_e1=0;
	float max_e1=0;
	float min_e2=0;
	float max_e2=0;
	if(e==2){
		for(size_t i=0;i<pcl_in.points.size();i++){
			float x1=pcl_in.points[i].x;
			float x2=pcl_in.points[i].x;
			float z1=pcl_in.points[i].z;
			float z2=pcl_in.points[i].z;
			if(z1>max_e1)max_e1=z1;
			if(min_e1>z2)min_e1=z2;
			if(x1>max_e2)max_e2=x1;
			if(min_e2>x2)min_e2=x2;
		}
	}else if(e==3){
		for(size_t i=0;i<pcl_in.points.size();i++){
			float z1=pcl_in.points[i].z;
			float z2=pcl_in.points[i].z;
			float y1=pcl_in.points[i].y;
			float y2=pcl_in.points[i].y;
			if(z1>max_e1)max_e1=z1;
			if(min_e1>z2)min_e1=z2;
			if(y1>max_e2)max_e2=y1;
			if(min_e2>y2)min_e2=y2;
		}
	}
	float d_e1=max_e1-min_e1;
	float d_e2=max_e2-min_e2;
	pcl::PointCloud<pcl::PointXYZINormal> pcl_tmp;
	size_t ii=14;
	size_t jj=7;
	if(e==3){
		ii=9;
		jj=5;
	}
	MatrixXf histogram=MatrixXf::Zero(ii,jj);
	//cout<<"d_e1="<<d_e1<<" d_e2="<<d_e2<<endl;
	
	int points_num=0;
	int max_points_num=0;
	int min_points_num=0;
	for(size_t i=0;i<ii;i++){
		for(size_t j=0;j<jj;j++){
			points_num=0;
			pcl_tmp.points.clear();
			pcl_tmp=pcl_in;
			pcl_in.points.clear();
			for(size_t k=0;k<pcl_tmp.points.size();k++){
				pcl::PointXYZINormal p;
				p.x=pcl_tmp.points[k].x;
				p.y=pcl_tmp.points[k].y;
				p.z=pcl_tmp.points[k].z;
				float pt_e1=p.z;
				float pt_e2=p.x;
				if(e==3)pt_e2=p.y;
				bool eq1=((max_e1-d_e1/(float)ii*i)>pt_e1&&pt_e1>(max_e1-d_e1/(float)ii*(i+1)));
				bool eq2=((min_e2+d_e2/(float)jj*j)<pt_e2&&pt_e2<(min_e2+d_e2/(float)jj*(j+1)));

				if(eq1&&eq2)points_num++;
				else pcl_in.points.push_back(p);
			}
			if(points_num>max_points_num)max_points_num=points_num;
			if(points_num<min_points_num)min_points_num=points_num;
			if(points_num!=0)histogram(i,j)=points_num;
		}
	}
	//cout<<endl<<histogram<<endl;
	for(size_t i=0;i<ii;i++){
		for(size_t j=0;j<jj;j++){
			histogram(i,j)=histogram(i,j)/(min_points_num+max_points_num);
		}
	}
	//cout<<endl<<histogram<<endl;

	return histogram;
}
vector<float> histogramBrain(pcl::PointCloud<pcl::PointXYZINormal> cluster_pcl)
{

	MatrixXf histogram_e2=MatrixXf::Zero(14,7);
	MatrixXf histogram_e3=MatrixXf::Zero(9,5);
	histogram_e2=histogramProcess(cluster_pcl, 2);
	histogram_e3=histogramProcess(cluster_pcl, 3);

	vector<float> hist_features;
	size_t ii=14;
	size_t jj=7;
	for(size_t num=0;num<2;num++){
		for(size_t i=0;i<ii;i++){
			for(size_t j=0;j<jj;j++){
				if(num==0)hist_features.push_back(histogram_e2(i,j));
				if(num==1)hist_features.push_back(histogram_e3(i,j));
			}
		}
		ii=9;
		jj=5;
	}
	for(size_t i=0;i<hist_features.size();i++){
		cout<<hist_features[i]<<" ";
	}
	cout<<endl;

	return hist_features;
}
*/
int main(int argc, char** argv)
{
	ros::init(argc, argv, "cal_feature");
	ros::NodeHandle n;
		
	int sensor_mode=32;
	//cin >> sensor_mode;

	int label=0;
	while(label!=1||label!=-1){
		//cout<<"LABEL = ";
		cin >> label;
		if(label==1||label==-1)break;
		//else cout<<"LABEL number is 1 or -1. Please reinput 1 or -1."<<endl;
	}

	int test_MODE=-1;
	//cout<<"0->all, 1->size, 2->3Dcov, 3->moment, 4->block, 5->2Dcov, 6->other 3Dcov, 7->other moment, 8->other block, 9->other 2Dcov, 10->histogram, 11->other histogram"<<endl;
	while(test_MODE>mode_max||test_MODE<0){
		//cout<<"test_mode = ";
		cin >> test_MODE;
		if(test_MODE<mode_max+1&&test_MODE>-1)break;
		//else cout<<"test_MODE is 0-11. Please reinput 0-11."<<endl;
	}

	int start_NUM=0;
	cin >> start_NUM;

	int load_NUM=1000;
	cin >> load_NUM;

	PN=load_NUM+start_NUM;

for(int ppp=start_NUM;ppp<PN;ppp++){
		//cout<<" ppp "<<ppp<<endl;

		string str;//[PN];
		char tmp[PN];
		sprintf(tmp,"%d",ppp);
		str=tmp;
		str += ".pcd";
		//cout<<"pcd="<<str<<endl;

		pcl::PointCloud<pcl::PointXYZINormal> pcl_in;
		pcl_in.points.clear();
		pcl::PointCloud<pcl::PointXYZ> cloud; (new pcl::PointCloud<pcl::PointXYZ>);
		if(pcl::io::loadPCDFile<pcl::PointXYZ> (str, cloud)==-1) exit(1);
		//pcl::io::loadPCDFile<pcl::PointXYZ> (str, cloud);

		pcl_in.points.resize(cloud.points.size());
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
		for(unsigned int j=0;j<cloud.points.size();j++){
			pcl_in.points[j].x = cloud.points[j].x;
			pcl_in.points[j].y = cloud.points[j].y;
			pcl_in.points[j].z = cloud.points[j].z;
		}

		cloud.points.clear();

		vector<float> f1;//cloud size
		vector<float> f2; vector<float> f2_d;//3D covariance
		vector<float> f3;//normalized moment of inertia tensor
		vector<float> f4;//slice feature
		vector<float> f5;//2D covariance
		vector<float> f6;//histogram 14*7+9*5=143
		vector<float> f7;//histogram of normal gradient
		vector<float> f8;//number of points
		vector<float> f9;//minimum deistance for cluster(distance to centroid)
		
		vector<double> f_i;
////////////////////////
//検証用
		vector<double> f_test;
////////////////////////
		pcl::PointCloud<pcl::PointXYZINormal> rotate_pcl;

		//calculate centroid
		Vector3f centroid_v=calculateCentroid(pcl_in);
		//distance of centroid
		float dist_to_centroid=sqrt(pow(centroid_v[0],2)+pow(centroid_v[1],2)+pow(centroid_v[2],2));
		//Detecting distance

		pcl::PointCloud<pcl::PointXYZINormal> cluster_pcl;

		Eigen::Matrix3f rot_mat=transformMatrix(pcl_in);
		for(size_t i=0;i<pcl_in.points.size();i++){
			pcl::PointXYZINormal p;//=pcl_in.points[i];
			p.x=pcl_in.points[i].x;
			p.y=pcl_in.points[i].y;
			p.z=pcl_in.points[i].z;
			Eigen::Vector3f n;
			n(0)=pcl_in.points[i].x-centroid_v(0);
			n(1)=pcl_in.points[i].y-centroid_v(1);
			n(2)=pcl_in.points[i].z-centroid_v(2);
			n=rot_mat*n;
			p.x=n(0);
			p.y=n(1);
			p.z=n(2);
			cluster_pcl.points.push_back(p);
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);	
		cloud_in->width = 1;
		cloud_in->height = cluster_pcl.points.size();
		cloud_in->points.resize (cloud_in->width * cloud_in->height);
		for(size_t i=0;i<cluster_pcl.points.size();i++){
			cloud_in->points[i].x=cluster_pcl.points[i].x;
			cloud_in->points[i].y=cluster_pcl.points[i].y;
			cloud_in->points[i].z=cluster_pcl.points[i].z;
		}

		cluster_pcl.points.clear();
		rotate_pcl=normalCheck2(cloud_in);

		//minx,miny,minz,maxx,maxy,maxz
		float minmax[6]={cluster_minimum_pt.x, cluster_minimum_pt.y, cluster_minimum_pt.z, 
												cluster_maximum_pt.x, cluster_maximum_pt.y, cluster_maximum_pt.z};
		float width=minmax[3]-minmax[0];
		float length=minmax[4]-minmax[1];
		float height=minmax[5]-minmax[2];
		
		//各軸（x,y,z）において重心から最も離れた座標を計算
		f1.push_back(width);//x
		f1.push_back(length);//y
		f1.push_back(height);//z
		//////////////////////////////////以下特徴量算出///////////////////////////////////////////////////////
		f8.push_back((float)rotate_pcl.points.size());
		float max_range=0;
		float min_range=0;
		if(sensor_mode==32){
			max_range=15.0;
			min_range=1.0;
		}else if(sensor_mode==64){
			max_range=40.0;
			min_range=3.0;
		}
		float normalized_dist=(dist_to_centroid-min_range)/(max_range-min_range);
		if(normalized_dist<0.0)normalized_dist=0.0;
		else if(normalized_dist>1.0)normalized_dist=1.0;
		//f9.push_back(normalized_dist);
		f9.push_back(dist_to_centroid);

		//if(f6[0]>340&&f7[0]>9.0) cout<<"They are Pretty Cure????"<<endl;
		f2=cal3DCovariance(rotate_pcl);
		f3=calMomentTensor(rotate_pcl);
		f4=calBlockFeature(rotate_pcl, minmax[2]-centroid_v(2), height);
		vector<vector<float> > f6_f7=histogramBrain(rotate_pcl);
		f6=f6_f7[0];
		f7=f6_f7[1];
		pcl::PointCloud<pcl::PointXYZ> upper_pcl;
		pcl::PointCloud<pcl::PointXYZ> bottom_pcl;
		for(size_t i=0;i<rotate_pcl.points.size();i++){
			pcl::PointXYZ p;
			//cout<<"z="<<rotate_pcl.points[i].z<<endl;
			if(rotate_pcl.points[i].z > 0.0){
				p.x=rotate_pcl.points[i].x;
				p.y=0.0;
				p.z=rotate_pcl.points[i].z;
				upper_pcl.points.push_back(p);
			}else{
				p.x=rotate_pcl.points[i].x;
				p.y=0.0;
				p.z=rotate_pcl.points[i].z;
				bottom_pcl.points.push_back(p);
			}
		}
		vector<float> f5_upper=cal2DCovariance(upper_pcl);
		vector<float> f5_bottom=cal2DCovariance(bottom_pcl);
		f5=mergeVector(f5_upper, f5_bottom);

		///////////////////////////////////////////////////////////////////////////////////////////////////////

		//f_i=mergeAllFeatures(f1,f2,f3,f4,f5,f8);
		//f_i=mergeAllFeatures(f1,f2,f3,f4,f5,f6);
		f_i=mergeAllFeatures(f1,f2,f3,f4,f5,f6,f7,f8,f9);
		/*if(test_MODE==12){
			f_i.resize(f_9.size());
			f_i=f9;
		}*/
		/*double features[f_i.size()];
		for(size_t i=0;i<f_i.size();i++){
			features[i]=(double)f_i[i];
		}*/

		size_t index_num=0;
		if(test_MODE==0){
			//all feature
			f_test.resize(f_i.size());
			for(size_t i=0;i<f_i.size();i++){
				f_test[i]=f_i[i];
			}
		}else if(test_MODE==1){
			//only cluster size
			f_test.resize(f1.size());
			for(size_t i=0;i<f1.size();i++){
				f_test[i]=f1[i];
			}
		}else if(test_MODE==2){
			//only 3d covariance
			f_test.resize(f2.size());
			for(size_t i=0;i<f2.size();i++){
				f_test[i]=f2[i];
			}
		}else if(test_MODE==3){
			//only moment
			f_test.resize(f3.size());
			for(size_t i=0;i<f3.size();i++){
				f_test[i]=f3[i];
			}
		}else if(test_MODE==4){
			//only 2d covariance
			f_test.resize(f4.size());
			for(size_t i=0;i<f4.size();i++){
				f_test[i]=f4[i];
			}
		}else if(test_MODE==5){
			//only block feature
			f_test.resize(f5.size());
			for(size_t i=0;i<f5.size();i++){
				f_test[i]=f5[i];
			}
		}else if(test_MODE==14){
			//without cluster size
			f_test.resize(f_i.size()-f1.size());
			size_t num=0;
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>=num+f1.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==6){
			//without 3d covariance
			f_test.resize(f_i.size()-f2.size());
			size_t num=f1.size();
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>=num+f2.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==7){
			//without moment
			f_test.resize(f_i.size()-f3.size());
			size_t num=f1.size()+f2.size();
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>=num+f3.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==8){
			//without 2d covariance
			f_test.resize(f_i.size()-f4.size());
			size_t num=f1.size()+f2.size()+f3.size();
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>=num+f4.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==9){
			//without block feature
			f_test.resize(f_i.size()-f5.size());
			size_t num=f1.size()+f2.size()+f3.size()+f4.size();
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>num+f5.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==10){
			//only histogram of points number
			f_test.resize(f6.size());
			for(size_t i=0;i<f6.size();i++){
				f_test[i]=f6[i];
			}
		}else if(test_MODE==11){
			//without histogram of points number
			f_test.resize(f_i.size()-f6.size());
			size_t num=f1.size()+f2.size()+f3.size()+f4.size()+f5.size();
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>=num+f6.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==12){
			//only histogram of normal
			f_test.resize(f7.size());
			for(size_t i=0;i<f7.size();i++){
				f_test[i]=f7[i];
			}
		}else if(test_MODE==13){
			//without histogram of normal
			f_test.resize(f_i.size()-f7.size());
			size_t num=f1.size()+f2.size()+f3.size()+f4.size()+f5.size()+f6.size();
			for(size_t i=0;i<f_i.size();i++){
				if(i<num||i>=num+f7.size()){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==15){
			//without size&moment
			f_test.resize(f_i.size()-f1.size()-f3.size());
			//size_t num=f1.size()+f2.size()+f3.size()+f4.size()+f5.size()+f6.size();
			size_t num1=0;
			size_t num2=f1.size()+f2.size()+f3.size();
			for(size_t i=0;i<f_i.size();i++){
				if((i<num1||i>=num1+f1.size())&&(i<num2||i>=num2+f3.size())){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}else if(test_MODE==15){
			//without distance information
			f_test.resize(f_i.size()-f1.size()-f3.size()-f8.size()-f9.size());
			//size_t num=f1.size()+f2.size()+f3.size()+f4.size()+f5.size()+f6.size();
			size_t num1=0;
			size_t num2=f1.size()+f2.size()+f3.size();
			size_t num3=f1.size()+f2.size()+f3.size()+f4.size()+f5.size()+f6.size()+f7.size();
			for(size_t i=0;i<f_i.size();i++){
				if((i<num1||i>=num1+f1.size())&&(i<num2||i>=num2+f3.size())&&(i<num3||i>=num3+f8.size()+f9.size())){
					f_test[index_num]=f_i[i];
					index_num++;
				}
			}
		}




	//int label=LABEL;
	for(size_t i=0;i<f_test.size();i++){
		if(i==0)cout<<label;
		cout<<" "<<i+1<<":"<<f_test[i];
	}
	cout<<endl;

	float cout_thre=f_i[0]*f_i[1];
	
	bool f_i_flag=true;
	for(size_t i=0;i<f_test.size();i++){
		double det_inf=f_test[i];
			if(isinf(det_inf)){
			f_i_flag=false;
			//cout<<"NULL"<<endl;
		}
	}
	if(cout_tmp!=cout_thre) cout_tmp=0;

	ros::Rate loop_rate(5000);
	//while (ros::ok()) {
        //ros::spinOnce();
		//break;
		//}


	if(ppp==PN-1)exit(1);
	}
	ros::spin();

}

		/*
		f_i[0]=(double)f1[0];
		f_i[1]=(double)f1[1];
		f_i[2]=(double)f1[2];
		for(size_t i=3;i<f2.size()+3;i++){
			f_i[i]=(double)f2[i-3];//3-8
		}
		for(size_t i=9;i<f3.size()+9;i++){
			f_i[i]=(double)f3[i-9];//9-14
		}
		for(size_t i=15;i<f4.size()+15;i++){
			f_i[i]=(double)f4[i-15];//15-34
		}
		for(size_t i=35;i<f5.size()+35;i++){
			f_i[i]=(double)f5[i-35];//35-40
		}
		for(int i=0;i<41;i++){
			features[i]=f_i[i];
			//cout<<"i="<<i<<" "<<features[i]<<endl;
		}
		*/
