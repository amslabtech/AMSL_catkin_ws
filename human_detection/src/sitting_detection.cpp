/* /////////////////////////////////
sitting_detection with SVM
author:tyokota          
last update:  2014/11/09
latest update:2014/11/11
///////////////////////////////// */

/*確認用メモ*/
//まだ確認ができていない変更点は残しておいてください。確認ができたものは消してください。
//twinCheckProcess -> splitCheckProcess
//2人以上の人を含むクラスタへの対応
//makeできるかな？

/*変更希望箇所*/
//複数人が検出された場合、クラスタの色を変える。bounding boxは全体のものと、各クラスタのものを作成。
//複数クラスタの全てが人だと判定された場合は全体のbounding boxはグレーにして各クラスタのboundingを黄緑に。
//それ以外は全体のboundingは水色で各クラスタのboundingは黄緑。

/*その他メモ*/
//最終アップデート: 2014/11/11
//動作確認済みアップデート:2014/11/09
//

//ros include
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>

// PCL specific includes
#include <pcl-1.8/pcl/common/common.h>
#include <pcl-1.8/pcl/common/transforms.h>
#include <pcl-1.8/pcl/features/moment_of_inertia_estimation.h>
#include <pcl-1.8/pcl/features/normal_3d.h>  
#include <pcl-1.8/pcl/filters/extract_indices.h> 
#include <pcl-1.8/pcl/filters/voxel_grid.h> 
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/kdtree/kdtree.h>
#include <pcl-1.8/pcl/ModelCoefficients.h>  
#include <pcl-1.8/pcl/octree/octree.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/sample_consensus/model_types.h>  
#include <pcl-1.8/pcl/segmentation/extract_clusters.h> 
#include <pcl-1.8/pcl/segmentation/sac_segmentation.h>
/*
#include <pcl-1.7/pcl/gpu/octree/octree.hpp>
#include <pcl-1.7/pcl/gpu/containers/device_array.hpp>
#include <pcl-1.7/pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl-1.7/pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>
*/

//ros <-> pcl
#include <pcl_conversions/pcl_conversions.h>

//Eigen include
//#include <Eigen/SVD>
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Eigen/LU"
//#include <unsupported/Eigen/Splines>

//base include
#include <fstream>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>

//other include
#include "svm_2.h"
#include <omp.h>

//namecpace
using namespace std;
using namespace Eigen;

//MODE: 32E->32, 64E->64//
//#define MODE 64
#define MODE 32
#define normal_MODE 0//1->use normal, 0->not use normal
#define remove_MODE 1//1-all point, 0->only near point
//STATICMODE: if robot moving -> 1
#define STATICMODE 0
#define RWRC 1

//model file param
//use human1109
/*
#define COST 11.313708499
#define GAMMA 0.5
#define NU 0.019865
#define RHO 1.455746	
#define COEF0 0.0
#define RSL_TH 0.0
#define SCORE_TH 0.1
*/
//rwrc14

#define COST 1.0
#define GAMMA 2.0
#define NU 0.018128
#define RHO 0.982981	
#define COEF0 0.0
#define RSL_TH 0.0
#define SCORE_TH 0.1

size_t len_t = 100;
int spl_count=0;
int cluster_count=0;
bool rwrc=(RWRC==1);

int pcd_name=0;
size_t num;
sensor_msgs::PointCloud2 scan_pc;
sensor_msgs::PointCloud2 pc2;
sensor_msgs::PointCloud2 cloud_clustering;
sensor_msgs::PointCloud2 normal_pt;
sensor_msgs::PointCloud2 target_pt;
sensor_msgs::PointCloud2 test_pt;

//pcl::PointCloud<pcl::PointXYZINormal> rotate_pcl;

pcl::PointCloud<pcl::PointXYZ> bb_pcl;
pcl::PointCloud<pcl::PointXYZ> bb_pcl_n;
pcl::PointCloud<pcl::PointXYZ> bb_pcl_s;

vector<Vector3f> past_centroid;
vector<Vector3f> cent_tmp;
//objects human;
pcl::PCDWriter writer;  
////////////////////////////////////////////////
pcl::PointCloud<pcl::PointXYZINormal> candidate_pcl;
sensor_msgs::PointCloud2 ros_pc;
sensor_msgs::PointCloud2 forS_pt;
sensor_msgs::PointCloud2 candidate_pt;
pcl::PointCloud<pcl::PointXYZ>::Ptr pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZINormal> pcl_pc;
pcl::PointCloud<pcl::PointXYZINormal> target_pcl;
pcl::PointCloud<pcl::PointXYZINormal> test_pcl;
//pcl::PointCloud<pcl::PointXYZINormal> past_pcl;
//pcl::PointCloud<pcl::PointXYZINormal> now_pcl;
pcl::PointCloud<pcl::PointXYZI> pcl_scan;
pcl::PointCloud<pcl::PointXYZINormal> perfect_pcl;
//pcl::PointCloud<pcl::PointNormal> perfect_pcl;
pcl::PointCloud<pcl::PointXYZI> pclGround;

vector<Vector3f> remove_centroid;
vector<float> remove_size;

//for masada
std_msgs::Int32 human_number;
float nearest_distance=5.0;
std_msgs::Float32 nearest_callback;
//nearest_callback.data=5.0;
//

bool callback_flag = false;
bool pastCheckProcessFlag=false;
bool pastCheckStartFlag=false;
bool pastCentroidUpdateFlag=false;
sensor_msgs::PointCloud flag_pt;
clock_t tStart;// = clock();
void pc2Callback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pc2=*msgs;
}
void pclCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs,pcl_scan);
	callback_flag = true;
	//	cout<<"recieve pointcloud. flag->true "<<endl;
}
void distanceCallback(const std_msgs::Float32::Ptr &msgs)
{
	nearest_callback=*msgs;
	//cout<<"nearest="<<nearest_callback.data<<endl;
	//if(nearest_callback.data<2.0)nearest_callback.data=5.0;
}
void perfectCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs, perfect_pcl);
}
void pclCallback2(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs,pclGround);
}
void pcdWriter(int name, pcl::PointCloud<pcl::PointXYZ>& pt)
{
	std::stringstream ss;
	ss << name << ".pcd";
	writer.write<pcl::PointXYZ> ("clustering/check_pcd/" + ss.str (), pt, false);
}
Vector3f calculateCentroid(pcl::PointCloud<pcl::PointXYZI>& pt)
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
Vector3f calculateCentroid(sensor_msgs::PointCloud& pt)
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


/////////////////////////////////////////
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
Eigen::Matrix3f transformMatrix(pcl::PointCloud<pcl::PointXYZI> pcl_in) 
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

	//cluster_minimum_pt=min_point_OBB;
	//cluster_minimum_pt.z=min[2];
	//cluster_maximum_pt=max_point_OBB;
	//cluster_maximum_pt.z=max[2];
	//cluster_position_pt=position_OBB;

	return rot;
}
Eigen::Matrix3f transformMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in) 
{ 

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	cloud=cloud_in;

	Eigen::Matrix3f rot;
	Eigen::Vector4f min; 
	Eigen::Vector4f max; 
	//pcl::getMinMax3D (*cloud, min, max); 
	for(size_t i=0;i<cloud->points.size();i++){
		cloud->points[i].z=0.0;
	}
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	Eigen::Matrix3f rotational_matrix_OBB;
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	rot=rotational_matrix_OBB.inverse();
	//Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
	return rot;
}
////////////////////////////////////////



//SVM Evaluation
int times=0;
#define f_num 35
//sitting_detection
//svm_model *model = svm_load_model("/home/amsl/AMSL_ros_pkg/pcl17_pkg/pcl17_sitting_detection/svm_model/human1109.model",times);
//rwrc14
svm_model *model = svm_load_model("/home/amsl/AMSL_ros_pkg/rwrc15/rwrc15_human_detection/classification/svm_model/rwrc14_sitting.model",times);
double minval[f_num];//min
double maxval[f_num];//max

double svmClassifier(double features[f_num])
{
	//tStart = clock();
	if(1>times){
		//sitting_detection
		//ifstream ifs("/home/amsl/AMSL_ros_pkg/pcl17_pkg/pcl17_sitting_detection/svm_model/human1109.range");
		//rwrc14
		ifstream ifs("/home/amsl/AMSL_ros_pkg/rwrc15/rwrc15_human_detection/classification/svm_model/rwrc14_sitting.range");
		string buf;
		int n=0;
		while(ifs && getline(ifs, buf)){
			istringstream is (buf);
			string st[4];
			is >> st[0] >> st[1] >> st[2];
			//through heading word.
			if(st[0]!="x"&&st[0]!="-1"&&st[1]!="1"){
				minval[n]=S2D(st[1]);
				maxval[n]=S2D(st[2]);
				n++;
			}
		}
		//Setting SVM params.
		model->param.C = COST;
		model->param.degree = RHO;
		model->param.gamma = GAMMA;
		//model->param.coef0 = COEF0;
		model->param.nu = NU;
	}
	tStart = clock();
	svm_node data[f_num];
	for(int j=0;j<f_num;j++){
		data[j].index=j+1; data[j].value = features[j];
	}
	//scaling process [-1,+1] *Using something of model generation range.
	for(int j=0;j<f_num;j++){
		data[j].value = (2 * (data[j].value - minval[j]) / (maxval[j] - minval[j])) - 1;
	}
	//Adjust Once beyond the range [-1,+1].
	for (int j=0;j<f_num;j++){
		if (data[j].value > 1){
			data[j].value = 1;
		}
		if (data[j].value < -1){
			data[j].value = -1;
		}
	}	
	//svm_predict is calculate dec_value.
	double result_dec = 0;
	result_dec = svm_predict_dec(model, data);
	if(times==0) times=1;
	return (result_dec);

}

void forTracker(sensor_msgs::PointCloud poin)
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZI>);	
	cloud_in->width = 1;
	cloud_in->height = poin.points.size()+1;
	cloud_in->points.resize (cloud_in->width * cloud_in->height);
	for(int i=1;i<(int)poin.points.size()+1;i++)
	{
		cloud_in->points[0].x=0.0;
		cloud_in->points[0].y=0.0;
		cloud_in->points[0].z=0.0;
		cloud_in->points[i].x = poin.points[i-1].x;
		cloud_in->points[i].y = poin.points[i-1].y;
		cloud_in->points[i].z = poin.points[i-1].z;
	}
	pcl::VoxelGrid<pcl::PointXYZI> vgt;  
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);  
	vgt.setInputCloud (cloud_in);  
	//Down sampling. x*y*z to one point
	vgt.setLeafSize (0.05f, 0.05f, 0.05f);
	vgt.filter (*cloud_filtered);
}

sensor_msgs::PointCloud calRotationMatrix(sensor_msgs::PointCloud &pt_in, float rad)
{
	sensor_msgs::PointCloud rotate_pts;
	MatrixXf aq_mat=MatrixXf::Zero(4,1);
	MatrixXf rot_mat=MatrixXf::Zero(4,4);
	MatrixXf rsl_mat=MatrixXf::Zero(4,1);
	for(size_t i=0;i<pt_in.points.size();i++){
		geometry_msgs::Point32 rotate_g;
		aq_mat(0,0)=pt_in.points[i].x;
		aq_mat(1,0)=pt_in.points[i].y;
		aq_mat(2,0)=pt_in.points[i].z;
		aq_mat(3,0)=1.0;
		rot_mat(0,0)=cos(rad);
		rot_mat(0,1)=sin(rad);
		rot_mat(1,0)=-sin(rad);
		rot_mat(1,1)=cos(rad);
		rot_mat(2,2)=1.0;
		rot_mat(3,3)=1.0;
		rsl_mat=rot_mat*aq_mat;
		rotate_g.x=rsl_mat(0,0)/rsl_mat(3,0);
		rotate_g.y=rsl_mat(1,0)/rsl_mat(3,0);
		rotate_g.z=rsl_mat(2,0)/rsl_mat(3,0);
		rotate_pts.points.push_back(rotate_g);

	}
	return (rotate_pts);
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
	vector<Vector3f> dev3d;//deviation

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
			if(isinf(tmp))tmp=0.0;
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
		float delta_0=feature_delta[i](0);
		bool infcheck_0=isinf(delta_0);
		if(infcheck_0)delta_0=0.0;
		f_.push_back(delta_0);
		float delta_1=feature_delta[i](1);
		bool infcheck_1=isinf(delta_1);
		if(infcheck_1)delta_1=0.0;
		f_.push_back(delta_1);

		//float delta_0=feature_delta[i](0);
		//f_.push_back(delta_0);
		//float delta_1=feature_delta[i](1);
		//f_.push_back(delta_1);
	}
	return(f_);
}
vector<double> mergeAllFeatures(vector<float> f1, vector<float> f2, vector<float> f3, vector<float> f4, vector<float> f5, vector<float>f6, vector<float> f7)
{
	vector<double> f_i;
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
	/*for(int i=0;i<f_num;i++){
	  features[i]=f_i[i];
	  }*/
	return(f_i);
}
vector<double> mergeAllFeatures(vector<float> f1, vector<float> f2, vector<float> f3, vector<float> f4, vector<float> f5)
{
	vector<double> f_i;
	f_i.resize(f_num);
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
	/*for(int i=0;i<f_num;i++){
	  features[i]=f_i[i];
	  }*/
	return(f_i);
}
vector<double> mergeAllFeatures(vector<float> f1, vector<float> f2, vector<float> f4, vector<float> f5)
{
	vector<double> f_i;
	f_i.resize(f_num);
	f_i[0]=(double)f1[0];
	f_i[1]=(double)f1[1];
	f_i[2]=(double)f1[2];
	for(size_t i=3;i<f2.size()+3;i++){
		f_i[i]=(double)f2[i-3];//3-8
	}
	//for(size_t i=9;i<f3.size()+9;i++){
	//	f_i[i]=(double)f3[i-9];//9-14
	//}
	for(size_t i=9;i<f4.size()+9;i++){
		f_i[i]=(double)f4[i-9];//15-34
	}
	for(size_t i=29;i<f5.size()+29;i++){
		f_i[i]=(double)f5[i-29];//34-40
	}
	/*for(int i=0;i<f_num;i++){
	  features[i]=f_i[i];
	  }*/
	return(f_i);
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
void visualizer(Eigen::Matrix3f rot, pcl::PointXYZ min_pt, pcl::PointXYZ max_pt, pcl::PointXYZ position_pt)
{
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

	bb_pcl.points.push_back(pt1);
	bb_pcl.points.push_back(pt2);
	bb_pcl.points.push_back(pt1);
	bb_pcl.points.push_back(pt4);
	bb_pcl.points.push_back(pt1);
	bb_pcl.points.push_back(pt5);
	bb_pcl.points.push_back(pt5);
	bb_pcl.points.push_back(pt6);
	bb_pcl.points.push_back(pt5);
	bb_pcl.points.push_back(pt8);
	bb_pcl.points.push_back(pt2);
	bb_pcl.points.push_back(pt6);
	bb_pcl.points.push_back(pt6);
	bb_pcl.points.push_back(pt7);
	bb_pcl.points.push_back(pt7);
	bb_pcl.points.push_back(pt8);
	bb_pcl.points.push_back(pt2);
	bb_pcl.points.push_back(pt3);
	bb_pcl.points.push_back(pt4);
	bb_pcl.points.push_back(pt8);
	bb_pcl.points.push_back(pt4);
	bb_pcl.points.push_back(pt3);
	bb_pcl.points.push_back(pt7);
	bb_pcl.points.push_back(pt3);
}

void visualizer(Eigen::Matrix3f rot, pcl::PointXYZ min_pt, pcl::PointXYZ max_pt, pcl::PointXYZ position_pt, int kind)
{
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

	if(kind==1){
		bb_pcl.points.push_back(pt1);
		bb_pcl.points.push_back(pt2);
		bb_pcl.points.push_back(pt1);
		bb_pcl.points.push_back(pt4);
		bb_pcl.points.push_back(pt1);
		bb_pcl.points.push_back(pt5);
		bb_pcl.points.push_back(pt5);
		bb_pcl.points.push_back(pt6);
		bb_pcl.points.push_back(pt5);
		bb_pcl.points.push_back(pt8);
		bb_pcl.points.push_back(pt2);
		bb_pcl.points.push_back(pt6);
		bb_pcl.points.push_back(pt6);
		bb_pcl.points.push_back(pt7);
		bb_pcl.points.push_back(pt7);
		bb_pcl.points.push_back(pt8);
		bb_pcl.points.push_back(pt2);
		bb_pcl.points.push_back(pt3);
		bb_pcl.points.push_back(pt4);
		bb_pcl.points.push_back(pt8);
		bb_pcl.points.push_back(pt4);
		bb_pcl.points.push_back(pt3);
		bb_pcl.points.push_back(pt7);
		bb_pcl.points.push_back(pt3);
	}else if(kind==2){
		bb_pcl_s.points.push_back(pt1);
		bb_pcl_s.points.push_back(pt2);
		bb_pcl_s.points.push_back(pt1);
		bb_pcl_s.points.push_back(pt4);
		bb_pcl_s.points.push_back(pt1);
		bb_pcl_s.points.push_back(pt5);
		bb_pcl_s.points.push_back(pt5);
		bb_pcl_s.points.push_back(pt6);
		bb_pcl_s.points.push_back(pt5);
		bb_pcl_s.points.push_back(pt8);
		bb_pcl_s.points.push_back(pt2);
		bb_pcl_s.points.push_back(pt6);
		bb_pcl_s.points.push_back(pt6);
		bb_pcl_s.points.push_back(pt7);
		bb_pcl_s.points.push_back(pt7);
		bb_pcl_s.points.push_back(pt8);
		bb_pcl_s.points.push_back(pt2);
		bb_pcl_s.points.push_back(pt3);
		bb_pcl_s.points.push_back(pt4);
		bb_pcl_s.points.push_back(pt8);
		bb_pcl_s.points.push_back(pt4);
		bb_pcl_s.points.push_back(pt3);
		bb_pcl_s.points.push_back(pt7);
		bb_pcl_s.points.push_back(pt3);
	}else{
		bb_pcl_n.points.push_back(pt1);
		bb_pcl_n.points.push_back(pt2);
		bb_pcl_n.points.push_back(pt1);
		bb_pcl_n.points.push_back(pt4);
		bb_pcl_n.points.push_back(pt1);
		bb_pcl_n.points.push_back(pt5);
		bb_pcl_n.points.push_back(pt5);
		bb_pcl_n.points.push_back(pt6);
		bb_pcl_n.points.push_back(pt5);
		bb_pcl_n.points.push_back(pt8);
		bb_pcl_n.points.push_back(pt2);
		bb_pcl_n.points.push_back(pt6);
		bb_pcl_n.points.push_back(pt6);
		bb_pcl_n.points.push_back(pt7);
		bb_pcl_n.points.push_back(pt7);
		bb_pcl_n.points.push_back(pt8);
		bb_pcl_n.points.push_back(pt2);
		bb_pcl_n.points.push_back(pt3);
		bb_pcl_n.points.push_back(pt4);
		bb_pcl_n.points.push_back(pt8);
		bb_pcl_n.points.push_back(pt4);
		bb_pcl_n.points.push_back(pt3);
		bb_pcl_n.points.push_back(pt7);
		bb_pcl_n.points.push_back(pt3);
	}
}

pcl::PointCloud<pcl::PointXYZINormal> normalCheck(pcl::PointCloud<pcl::PointXYZI>::Ptr &pcl_in)
{	
	//cout<<"start normal check"<<endl;
	pcl::PointCloud<pcl::PointXYZINormal> pcl_pcs;
	size_t num=pcl_in->points.size();
	//pcl_pcs.points.resize(num);
	if(normal_MODE==1){
		pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
		ne.setInputCloud (pcl_in);
		pcl::search::KdTree<pcl::PointXYZI>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZI> ());
		ne.setSearchMethod (tree2);
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
		ne.setRadiusSearch (0.20);
		ne.compute (*cloud_normals);
		for(size_t i=0;i<num;i++){
			pcl::PointXYZINormal p;
			p.x=pcl_in->points[i].x;
			p.y=pcl_in->points[i].y;
			p.z=pcl_in->points[i].z;
			p.intensity=pcl_in->points[i].intensity;
			p.normal_x=cloud_normals->points[i].normal_x;
			p.normal_y=cloud_normals->points[i].normal_y;
			p.normal_z=cloud_normals->points[i].normal_z;
			p.curvature=cloud_normals->points[i].curvature;
			if(p.curvature>1.0)p.curvature=1.0;
			if(p.curvature<0.0)p.curvature=0.0;
			//if(p.normal_z<0.9) pcl_pcs.points.push_back(p);
			pcl_pcs.points.push_back(p);
		}
	}else{
		for(size_t i=0;i<num;i++){
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
vector<double> twinSplitFeature(pcl::PointCloud<pcl::PointXYZI> &pcl_in)
{
	vector<float> f1, f2, f3, f4, f5, f6, f7, f8;
	vector<double> f_i;

	pcl::PointCloud<pcl::PointXYZINormal> rotate_pcl;
	Vector3f centroid_v=calculateCentroid(pcl_in);

	Eigen::Matrix3f rot_matrix=transformMatrix(pcl_in);

	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZINormal p;//=pcl_in.points[i];
		Eigen::Vector3f n;
		n(0)=pcl_in.points[i].x-centroid_v(0);
		n(1)=pcl_in.points[i].y-centroid_v(1);
		n(2)=pcl_in.points[i].z-centroid_v(2);
		n=rot_matrix*n;
		p.x=n(0);
		p.y=n(1);
		p.z=n(2);
		p.intensity=pcl_in.points[i].intensity;
		rotate_pcl.points.push_back(p);
	}

	int num[6]={};
	float minmax[6]={};//minx,miny,minz,maxx,maxy,maxz

	for(size_t i=0;i<rotate_pcl.points.size();i++){
		float x=rotate_pcl.points[i].x;
		float y=rotate_pcl.points[i].y;
		float z=rotate_pcl.points[i].z;

		if(minmax[0]>x){minmax[0]=x;num[0]=i;}
		if(minmax[1]>y){minmax[1]=y;num[1]=i;}
		if(minmax[2]>z){minmax[2]=z;num[2]=i;}
		if(x>minmax[3]){minmax[3]=x;num[3]=i;}
		if(y>minmax[4]){minmax[4]=y;num[4]=i;}
		if(z>minmax[5]){minmax[5]=z;num[5]=i;}
	}

	float width=minmax[3]-minmax[0];
	float length=minmax[4]-minmax[1];
	float height=minmax[5]-minmax[2];
	int size_comp=0;
	if(rwrc){
		if(height>0.7&&height<1.3)size_comp++;
		if(width>0.4&&width<1.6)size_comp++;
		if(length>0.3&&length<1.6)size_comp++;
	}else{
		//if(height>1.2&&height<2.0)size_comp++;
		//if(width>0.2&&width<1.5)size_comp++;
		//if(length>0.2&&length<1.5)size_comp++;
		if(height>0.9&&height<1.7)size_comp++;
		if(width>0.3&&width<0.8)size_comp++;
		if(length>0.1&&length<0.7)size_comp++;

	}

	if(size_comp==3){

		f1.push_back(width);//x
		f1.push_back(length);//y
		f1.push_back(height);//z

		f2=cal3DCovariance(rotate_pcl);
		f3=calMomentTensor(rotate_pcl);
		f4=calBlockFeature(rotate_pcl, minmax[2], height);

		pcl::PointCloud<pcl::PointXYZ> upper_pcl;
		pcl::PointCloud<pcl::PointXYZ> bottom_pcl;
		for(size_t i=0;i<rotate_pcl.points.size();i++){
			pcl::PointXYZ p;
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

		f6.push_back((float)pcl_in.points.size());
		float f7_tmp=0.0;
		f7.push_back(f7_tmp);

		//f_i=mergeAllFeatures(f1,f2,f3,f4,f5);
		f_i=mergeAllFeatures(f1,f2,f4,f5);
		return(f_i);
	}else{
		for(size_t i=0;i<f_num;i++){
			f_i.push_back(0.0);//=0.0;
		}
		return(f_i);
	}
}
//vector<pcl::PointCloud<pcl::PointXYZINormal> > splitCheckProcess(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, float min_x, float cluster_width)
vector<pcl::PointCloud<pcl::PointXYZI> > splitCheckProcess(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, float min_max[6], float cluster_width)
{
	//twinsplit → PCAを施したクラスタを重心点で分割している。
	//理想はクラスタの高さ方向の山を数えて、その数に分割し、各rslを計算する。
	//妥協案はクラスタのbounding boxの中心点で分割する。
	//いずれにせよ重心点での分割はやめよう。
	//また、現在は左半分、右半分に分けているが複数個できるようループ処理にしましょう。
	//bounding boxのwidthで山の数を決める。
	//widthは基本は0.8mで1人とする。0.8ごとにカウントする。
	cout<<"split process start"<<endl;
	bool humanFlag=false;
	size_t split_number=(size_t)(cluster_width/0.8)+1;
	cout<<"this cluster include "<<split_number<<" cluster"<<endl;
	//pcl::PointCloud<pcl::PointXYZI> pcl_sp[split_number];
	//vector<pcl::PointCloud<pcl::PointXYZI> > pcl_sp[split_number];
	vector<pcl::PointCloud<pcl::PointXYZI> > pcl_sp;
	pcl_sp.resize(split_number);
	float rsl_sp[split_number];//各要素はrslの結果。
	float width=cluster_width/(float)split_number;//分割方法は要検討。単純に分割するのではなく、一人ひとりに分割できるようにするべき。分割部分はオーバラップするか、消すか。
	for(size_t i=0;i<6;i++){
		cout<<"minmax="<<min_max[i]<<endl;
	}
	float start_position=min_max[0];
	float rsl_thresh=0.0;
	int human_number=0;
	int negative_number=0;

	for(size_t i=0;i<split_number;i++){
		float end_position=start_position+width;
		cout<<"start position="<<start_position<<", width="<<width<<", end position="<<end_position<<endl;
		for(size_t j=0;j<pcl_in.points.size();j++){
				//pcl::PointXYZINormal p;
				pcl::PointXYZI p;
				p.x=pcl_in.points[j].x;
				p.y=pcl_in.points[j].y;
				p.z=pcl_in.points[j].z;
				p.intensity=pcl_in.points[j].intensity;
				if(p.x>=start_position&&p.x<=end_position)pcl_sp[i].points.push_back(p);
		}
		start_position=end_position;
		size_t point_number=pcl_sp[i].points.size();
		cout<<"pcl_sp["<<i<<"].points.size()="<<point_number<<endl;
		if(point_number<40){
			//クラスタの含む点群数が少ない場合はネガティブにする。
			cout<<"cluster include points is too few. rsl = -1.0"<<endl;
			rsl_sp[i]=-1.0;
			negative_number++;
			continue;
		}else{
			vector<double> feature_vec=twinSplitFeature(pcl_sp[i]);
			double features[f_num];
			for(int j=0;j<f_num;j++){
				features[i]=feature_vec[j];
			}
			double rsl_tmp=svmClassifier(features);
			rsl_sp[i]=(float)rsl_tmp;
			if(rsl_sp[i]>rsl_thresh){
				human_number++;
				for(size_t j=0;j<pcl_sp[i].points.size();j++){
				//pcl::PointXYZINormal p;
					pcl_sp[i].points[j].intensity=1.0;
				}
			}else{
				negative_number++;
				for(size_t j=0;j<pcl_sp[i].points.size();j++){
					pcl_sp[i].points[j].intensity=-1.0;
				}
			}
			cout<<"rsl_tmp="<<rsl_tmp<<endl;
		}
	}
	cout<<"this cluster include "<<human_number<<" people."<<endl;
	return pcl_sp;
	//if(human_number!=0) return 1;
	//else return -1;
}
double twinCheckProcess(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in)
{
	//twinsplit → PCAを施したクラスタを重心点で分割している。
	//理想はクラスタの高さ方向の山を数えて、その数に分割し、各rslを計算する。
	//妥協案はクラスタのbounding boxの中心点で分割する。
	//いずれにせよ重心点での分割はやめよう。
	//また、現在は左半分、右半分に分けているが複数個できるようループ処理にしましょう。
	cout<<"twin split process start"<<endl;
	pcl::PointCloud<pcl::PointXYZI> pcl_l, pcl_r;
	bool humanFlag=false;
	for(size_t i=0;i<pcl_in.points.size();i++){
		if(pcl_in.points[i].x<0){
			pcl::PointXYZI p;
			p.x=pcl_in.points[i].x;
			p.y=pcl_in.points[i].y;
			p.z=pcl_in.points[i].z;
			p.intensity=pcl_in.points[i].intensity;
			pcl_l.points.push_back(p);
		}else{
			pcl::PointXYZI p;
			p.x=pcl_in.points[i].x;
			p.y=pcl_in.points[i].y;
			p.z=pcl_in.points[i].z;
			p.intensity=pcl_in.points[i].intensity;
			pcl_r.points.push_back(p);
		}
	}
	bool size_l=pcl_l.points.size()<40;
	bool size_r=pcl_r.points.size()<40;
	if(size_r&&size_l){
		cout<<"cluster size is too small"<<endl;
		return(-1);
	}else{

		vector<double> left_f=twinSplitFeature(pcl_l);
		vector<double> right_f=twinSplitFeature(pcl_r);

		double left_features[f_num];
		double right_features[f_num];
		for(int i=0;i<f_num;i++){
			left_features[i]=left_f[i];
		}
		for(int i=0;i<f_num;i++){
			right_features[i]=right_f[i];
		}
		double rsl_l=0.0;
		double rsl_r=0.0;
		if(size_l&&!size_r){		
			rsl_l=-1.0;//svmClassifier(left_features);
			rsl_r=svmClassifier(right_features);
		}else if(!size_l&&size_r){
			rsl_l=svmClassifier(left_features);
			rsl_r=-1.0;//svmClassifier(right_features);
		}else{
			rsl_l=svmClassifier(left_features);
			rsl_r=svmClassifier(right_features);
		}
		double twin_rsl_thresh=0.0;
		//Vector3f centroid_left, centroid_right;
		//centroid_left=calculateCentroid(pcl_l);
		//centroid_right=calculateCentroid(pcl_r);

		if(rsl_l>twin_rsl_thresh||rsl_r>twin_rsl_thresh)humanFlag=true;
		if(humanFlag==true){
			//visualizer(pcl_l, 1, 1, rad_tmp);
			//visualizer(pcl_r);
			return(1);
			//return(rsl_l+rsl_r);
			//ビジュアライズ用の点群と全体をハクように指示するものを書く
		}else{
			return(-1);
		}
	}

}
////////////////////////////////////////////////////////////
int pastCheckProcess(Vector3f &cent_in, vector<Vector3f> &cent_past)
{
	//pastCheckProcessFlag=true;
	//cout<<"process start"<<endl;
	int checkValue=(-1);
	if(cent_past.size()!=0){
		float tmp_dist=10000.0;  
		size_t id_;
		for(size_t i=0;i<cent_past.size();i++){
			float distance_=sqrt(pow(cent_in[0]-cent_past[i](0),2)+pow(cent_in[1]-cent_past[i](1),2));
			//float distance_=sqrt(pow(cent_past[i](0)-cent_in[0],2)+pow(cent_past[i](1)-cent_in[1],2));
			//float distance_=sqrt(pow(cent_in[0]-cent_past[i](0),2)+pow(cent_in[1]-cent_past[i](1),2)+pow(cent_in[2]-cent_past[i](2),2));
			//float distance_=sqrt(pow(cent_in[0]-past_centroid[i](0),2)+pow(cent_in[1]-past_centroid[i](1),2)+pow(cent_in[2]-past_centroid[i](2),2));
			//cout<<distance_<<endl;
			if(tmp_dist>distance_){
				tmp_dist=distance_;
				id_=i;
			}
		}
		if(tmp_dist<0.35){
			//cout<<"change rsl -> 1.0 "<<"dist="<<tmp_dist<<endl;
			//rsl_thresh=-100.0;
			cent_tmp.push_back(cent_in);
			//cent_past[id_](2)=10000.0;
			checkValue=1.0;
		}
	}
	//cout<<"process end"<<endl;
	return(checkValue);
}
void pastCentroidUpdate()
{
	cout<<"update start"<<endl;
	size_t num=cent_tmp.size();
	past_centroid.clear();
	past_centroid.resize(0);
	cout<<"num="<<num<<endl;
	for(int i=0;i<(int)num;i++){
		Vector3f p;
		p[0]=cent_tmp[i](0);
		p[1]=cent_tmp[i](1);
		p[2]=cent_tmp[i](2);
		past_centroid.push_back(p);

	}
	cent_tmp.clear();
	cent_tmp.resize(0);
	cout<<"update end"<<endl;
}
///////////////////////////////////////////////////////////
void calFeatures(pcl::PointCloud<pcl::PointXYZI> &pcl_in)
{
	//vector<Vector3f> cent_tmp;//now centroid temporary

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZI>);

	cloud_in->width = 1;
	cloud_in->height = pcl_in.points.size();
	cloud_in->points.resize (cloud_in->width * cloud_in->height);
	for(int i=0;i<(int)pcl_in.points.size();i++)
	{
		cloud_in->points[i].x = pcl_in.points[i].x;
		cloud_in->points[i].y = pcl_in.points[i].y;
		cloud_in->points[i].z = pcl_in.points[i].z;
		cloud_in->points[i].intensity = pcl_in.points[i].intensity;
	}

	pcl::VoxelGrid<pcl::PointXYZI> vg;  
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);  
	//*cloud_filtered=*cloud_in;
	vg.setInputCloud (cloud_in);  
	//Down sampling. x*y*z to one point
	vg.setLeafSize (0.05f, 0.05f, 0.05f);
	vg.filter (*cloud_filtered);

	//////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tmp (new pcl::PointCloud<pcl::PointXYZI>);  
	cloud_tmp->points.resize(cloud_filtered->points.size());
	*cloud_tmp=*cloud_filtered;	

	for(int i=0;i<(int)cloud_filtered->points.size();i++){
		cloud_filtered->points[i].z  = 0.0;
	}

	std::vector<pcl::PointIndices> cluster_indices;

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
	tree->setInputCloud (cloud_filtered);
	pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
	ec.setClusterTolerance (0.10);//Points clearance for clustering
	ec.setMinClusterSize (25);//Minimum include points of the cluster
	ec.setMaxClusterSize (1000);//Maximum points of the cluster //32Eなら近くても500点いかない。400でも大丈夫かな。
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);
	int cluster_number = 0;
	int target_number=0;
	int color=0;

	*cloud_filtered=*cloud_tmp;
	vector<pcl::PointCloud<pcl::PointXYZI>::Ptr >clusters;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
	for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
		for(std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++){
			cloud_cluster->points.push_back (cloud_filtered->points[*pit]);cloud_cluster->width = cloud_cluster->points.size ();
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = true;
		}
		pcl::toROSMsg(*cloud_cluster, cloud_clustering);//変換したものを使ってない？
		cluster_count++;
		////////////////////////////////////////////////////////////////
		vector<float> f1;//cloud size
		vector<float> f2; vector<float> f2_d;//3D covariance
		vector<float> f3;//normalized moment of inertia tensor
		vector<float> f4;//slice feature
		vector<float> f5;//2D covariance
		vector<float> f6;//number of points
		vector<float> f7;//minimum deistance for cluster(distance to centroid)
		vector<float> f8;
		//vector<float> f9;
		//vector<float> f10;
		vector<double> f_i;
		double features[f_num];
		pcl::PointCloud<pcl::PointXYZINormal> rotate_pcl;

		//calculate centroid
		Vector3f centroid_v=calculateCentroid(*cloud_cluster);
		//distance of centroid
		//float dist_to_centroid=sqrt(pow(centroid_v[0],2)+pow(centroid_v[1],2)+pow(centroid_v[2],2));
		float dist_to_centroid=sqrt(pow(centroid_v[0],2)+pow(centroid_v[1],2));
		//Detecting distance
		float distanceThresh;
		if(MODE==64){
			distanceThresh=60.0;
		}else if(MODE==32){
			distanceThresh=12.0;
		}
		if(dist_to_centroid>distanceThresh){
			cloud_cluster->points.clear();
			continue;//設定した距離よりも遠い場合，次のループに移行
		}

		pcl::PointCloud<pcl::PointXYZINormal> cluster_pcl;
		cluster_pcl=normalCheck(cloud_cluster);

		Eigen::Matrix3f rot_mat=transformMatrix(cluster_pcl);
		for(size_t i=0;i<cluster_pcl.points.size();i++){
			pcl::PointXYZINormal p=cluster_pcl.points[i];
			Eigen::Vector3f n;
			n(0)=cluster_pcl.points[i].x-centroid_v(0);
			n(1)=cluster_pcl.points[i].y-centroid_v(1);
			n(2)=cluster_pcl.points[i].z-centroid_v(2);
			n=rot_mat*n;
			p.x=n(0);
			p.y=n(1);
			p.z=n(2);
			rotate_pcl.points.push_back(p);
		}

		float minmax[6]={cluster_minimum_pt.x, cluster_minimum_pt.y, cluster_minimum_pt.z, cluster_maximum_pt.x, cluster_maximum_pt.y, cluster_maximum_pt.z};//minx,miny,minz,maxx,maxy,maxz
		float width=minmax[3]-minmax[0];
		float length=minmax[4]-minmax[1];
		float height=minmax[5]-minmax[2];
		int size_comp=0;
		if(rwrc){
			if(height>0.7&&height<1.3)size_comp++;
			if(width>0.4&&width<1.6)size_comp++;
			if(length>0.3&&length<1.6)size_comp++;
		}else{
			if(height>0.9&&height<1.7)size_comp++;
			if(width>0.3&&width<1.6)size_comp++;
			if(length>0.1&&length<0.7)size_comp++;
		}
		//if(height>1.2&&height<2.0)size_comp++;
		//if(width>0.1&&width<1.5)size_comp++;
		//if(length>0.1&&length<1.5)size_comp++;
		//cout<<"size_comp="<<size_comp<<endl;
		//cout<<"height:"<<height<<" width:"<<width<<" length:"<<length<<endl;
		if(size_comp!=3){
			cloud_cluster->points.clear();
			rotate_pcl.points.clear();
			continue;
		}else{
			if(width>length)remove_size.push_back(width);
			else remove_size.push_back(length);
		}
		//各軸（x,y,z）において重心から最も離れた座標を計算
		f1.push_back(width);//x
		f1.push_back(length);//y
		f1.push_back(height);//z
		//////////////////////////////////以下特徴量算出///////////////////////////////////////////////////////
		f6.push_back((float)cluster_pcl.points.size());
		f7.push_back(dist_to_centroid);

//#pragma omp parallel
//#pragma omp sections
//		{
//#pragma omp section
//			{
				f2=cal3DCovariance(rotate_pcl);
//			}
//#pragma omp section
//			{
				f3=calMomentTensor(rotate_pcl);
//			}
//#pragma omp section
//			{
				f4=calBlockFeature(rotate_pcl, minmax[2]-centroid_v(2), height);
//			}
//#pragma omp section
//			{
				pcl::PointCloud<pcl::PointXYZ> upper_pcl;
				pcl::PointCloud<pcl::PointXYZ> bottom_pcl;
				for(size_t i=0;i<rotate_pcl.points.size();i++){
					pcl::PointXYZ p;
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
//			}
//		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////

		//f_i=mergeAllFeatures(f1,f2,f3,f4,f5);
		f_i=mergeAllFeatures(f1,f2,f4,f5);
		///////////////////つかわないとくちょうりょうがあればここ
		//size_t nofeature=f_num-f3.size();
		//size_t f_index=0;
		//features.resize(nofeature);
		for(size_t i=0;i<f_i.size();i++){
			//if(i<9||i>14){
			features[i]=f_i[i];
			//	f_index++;
			//}
		}
		/////////////////////////////////////////////////////

		double rsl=0.0;
		//vector<pcl::PointCloud<pcl::PointXYZINormal> > split_cluster;
		vector<pcl::PointCloud<pcl::PointXYZI> > split_cluster;
		if(width<0.8)rsl=svmClassifier(features);
		else{
			size_t cluster_num=(int)(width/0.8)+1;
			//split_cluster.resize(cluster_num);//コピーならリサイズいらないかも
			split_cluster=splitCheckProcess(rotate_pcl, minmax, width);
			rsl=twinCheckProcess(rotate_pcl);
			for(size_t i=0;i<split_cluster.size();i++){
				for(size_t j=0;j<split_cluster[i].points.size();j++){
					pcl::PointXYZINormal p;
					Vector3f p_tmp;
					p_tmp(0)=split_cluster[i].points[j].x;
					p_tmp(1)=split_cluster[i].points[j].y;
					p_tmp(2)=split_cluster[i].points[j].z;
					Vector3f p_rop=rot_mat.inverse()*p_tmp;
					//p.x=split_cluster[i].points[j].x+centroid_v[0];
					//p.y=split_cluster[i].points[j].y+centroid_v[1];
					//p.z=split_cluster[i].points[j].z+centroid_v[2];
					p.x=p_rop(0)+centroid_v[0];
					p.y=p_rop(1)+centroid_v[1];
					p.z=p_rop(2)+centroid_v[2];
					p.intensity=split_cluster[i].points[j].intensity;
					p.curvature=-1.0;//split_cluster[i].points[i].curvature;
					p.normal_x=0.0;
					p.normal_y=0.0;
					p.normal_z=i;
					test_pcl.points.push_back(p);		
				}
			}

		}
		double rsl_thresh=0.0;
		int objectKind=0;
		int twinFlag=0;
		cout<<"rsl="<<rsl<<endl;
		//bool rsl_flag=false;
		//if(f7[0]<10.0) cout<<"rsl="<<rsl<<" dist="<<f7[0]<<endl;
		//cout<<"---------rsl="<<rsl<<endl;
		//cout<<"height:"<<height<<" width:"<<width<<" length:"<<length<<endl;

		bool pushBackFlag=false;

		if(pastCheckStartFlag==true){
			int checkValue=pastCheckProcess(centroid_v, past_centroid);
			if(rsl<rsl_thresh){
				if(rsl>-4.0)rsl=(double)checkValue;
				pushBackFlag=true;
			}
			pastCentroidUpdateFlag=true;
		}
		remove_centroid.push_back(centroid_v);

		///////////////////////////////warning section&normal estimation///////////////////////////////////////
		bool warning=false;
		if(rsl>rsl_thresh){
			int bar_num[10]={};
			//int bar_numy[10]={};
			int bar_count=0;
			bool warn_flag=false;
			//bar is short??
			//cout<<"x"<<endl;
			for(int i=0;i<10;i++){
				int bar=(int)(f4[i*2]*30);
				bar_num[i]=bar;
				for(int j=0;j<bar;j++){
					//cout<<"=";
				}
				//cout<<endl;
				if(4>bar) bar_count++;
			}
			//cout<<"y"<<endl;
			/*for(int i=1;i<11;i++){
			  int bar=(int)(f4[i*2-1]*30);
			  bar_numy[i]=bar;
			  for(int j=0;j<bar;j++){
			//cout<<"=";
			}
			//cout<<endl;
			//if(4>bar) bar_count++;
			}*/

			if(bar_count>6) {warn_flag=true; warning=true;}
			if(warn_flag==true){
				cout<<"warning_1!!!"<<endl;
				//rsl=-1.0;
			}
			bar_count=0;
			warn_flag=false;
			//bar is same long??
			for(int i=0;i<10;i++){
				for(int j=0;j<10;j++){
					int m=bar_num[i]-bar_num[j];
					if(m==0) bar_count++;
				}
				if(bar_count>5){ warn_flag=true; warning=true; bar_count=0;}
				else bar_count=0;
			}
			if(warn_flag==true){
				cout<<"warning_2!!!"<<endl;
				//rsl=-1.0;
			}
		}
		//////////////////////////////////////////////////////////////

		//objectKind
		//0: negative, 1: human, 2: split, 3: ...
		if(rsl>rsl_thresh)objectKind=1;
		else objectKind=0;
		if(rsl>rsl_thresh){
			target_number++;
			if(pastCheckStartFlag==false) past_centroid.push_back(centroid_v);
			if(pushBackFlag==false) cent_tmp.push_back(centroid_v);
			pcl::PointXYZINormal target_p;
			target_p.x=centroid_v[0];
			target_p.y=centroid_v[1];
			target_p.z=centroid_v[2];
			target_pcl.points.push_back(target_p);
		}
		if(rsl>rsl_thresh){
			for(size_t i=0;i<cluster_pcl.points.size();i++){
				pcl::PointXYZINormal p;
				p.x=cluster_pcl.points[i].x;
				p.y=cluster_pcl.points[i].y;
				p.z=cluster_pcl.points[i].z;
				p.intensity=cluster_pcl.points[i].intensity;
				p.curvature=(float)color;//cluster_pcl.points[i].curvature;
				p.normal_x=cluster_pcl.points[i].normal_x;
				p.normal_y=cluster_pcl.points[i].normal_y;
				p.normal_z=cluster_pcl.points[i].normal_z;
				candidate_pcl.points.push_back(p);		
			}
			color+=1;
		}
		//visualize用　バウンディングボックス
		//visualizer(rotate_pcl, objectKind, twinFlag, rad_tmp, centroid_v, num);
		//visualizer(rot_mat.inverse(), cluster_minimum_pt, cluster_maximum_pt, cluster_position_pt);
		visualizer(rot_mat.inverse(), cluster_minimum_pt, cluster_maximum_pt, cluster_position_pt, objectKind);

		cluster_number++;
		pcd_name++;
		//pcdWriter(pcd_name,*cloud_cluster);
		cloud_cluster->points.clear();
		//rotate_pcl.points.clear();
	}

	int human_counter=0;
	if(nearest_callback.data!=0){
		nearest_distance=nearest_callback.data;
	}
	cout<<"nearest_distance="<<nearest_distance<<", nearest_callback="<<nearest_callback.data<<endl;
	for(size_t i=0;i<target_pcl.points.size();i++){
		float dist=sqrt(pow(target_pcl.points[i].x,2)+pow(target_pcl.points[i].y,2));
		if(dist<nearest_distance)human_counter++;
	}
	human_number.data=human_counter;
	cout<<"human_counter="<<human_counter<<endl;
	cout<<"all cluster="<<cluster_number<<", positive="<<target_number<<", negative="<<cluster_number-target_number<<endl;
	//return 0;
}
int calculateRemoveArea(pcl::PointXYZ &p, vector<Vector3f> &centroid, vector<float> &area_size, float pub_range)
{
	float distTmp=10000.0;
	std_msgs::Int32 size_id;
	size_id.data=0;
	if(centroid.size()!=0){
		for(size_t i=0;i<centroid.size();i++){

			float distanceRobot=sqrt(pow(p.x,2)+pow(p.y,2));
			if(remove_MODE==1){
				if(2.1>distanceRobot){
					//cout<<"can't detect area"<<endl;
					//size_id.data=100000;
					return(0);
					break;			
					//continue;
				}
			}else{
				if(2.1>distanceRobot||distanceRobot>pub_range){
					//cout<<"can't detect area"<<endl;
					//size_id.data=100000;
					return(0);
					break;			
					//continue;
				}
			}
			float distanceToP=sqrt(pow(p.x-centroid[i](0),2)+pow(p.y-centroid[i](1),2));
			if(distTmp>distanceToP){
				distTmp=distanceToP;
				size_id.data=(int)i;
			}

		}
		if(distTmp<area_size[size_id.data]){
			//cout<<"remove point"<<endl;
			return(0);
		}else{
			//cout<<"else"<<endl;
			return(1);
		}

	}else{
		float distanceRobot=sqrt(pow(p.x,2)+pow(p.y,2));
		if(remove_MODE==1){
			if(2.1>distanceRobot){
				//cout<<"can't detect area"<<endl;
				return(0);
				//continue;
			}else{
				return(1);
			}
		}else{
			if(2.1>distanceRobot||distanceRobot>pub_range){
				//cout<<"can't detect area"<<endl;
				return(0);
				//continue;
			}else{
				return(1);
			}
		}
		cout<<"number of cluster = zero"<<endl;
	}
}
int calculateRemoveArea(pcl::PointXYZINormal &p, vector<Vector3f> &centroid, vector<float> &area_size, float pub_range)
{
	float distTmp=10000.0;
	std_msgs::Int32 size_id;
	size_id.data=0;
	if(centroid.size()!=0){
		for(size_t i=0;i<centroid.size();i++){

			float distanceRobot=sqrt(pow(p.x,2)+pow(p.y,2));
			if(remove_MODE==1){
				if(2.1>distanceRobot){
					//cout<<"can't detect area"<<endl;
					//size_id.data=100000;
					return(0);
					break;			
					//continue;
				}
			}else{
				if(2.1>distanceRobot||distanceRobot>pub_range){
					//cout<<"can't detect area"<<endl;
					//size_id.data=100000;
					return(0);
					break;			
					//continue;
				}
			}
			float distanceToP=sqrt(pow(p.x-centroid[i](0),2)+pow(p.y-centroid[i](1),2));
			if(distTmp>distanceToP){
				distTmp=distanceToP;
				size_id.data=(int)i;
			}

		}
		if(distTmp<area_size[size_id.data]){
			//cout<<"remove point"<<endl;
			return(0);
		}else{
			//cout<<"else"<<endl;
			return(1);
		}

	}else{
		float distanceRobot=sqrt(pow(p.x,2)+pow(p.y,2));
		if(remove_MODE==1){
			if(2.1>distanceRobot){
				//cout<<"can't detect area"<<endl;
				return(0);
				//continue;
			}else{
				return(1);
			}
		}else{
			if(2.1>distanceRobot||distanceRobot>pub_range){
				//cout<<"can't detect area"<<endl;
				return(0);
				//continue;
			}else{
				return(1);
			}
		}
		cout<<"number of cluster = zero"<<endl;
	}
}
pcl::PointCloud<pcl::PointXYZINormal> pcl_removedPoints;
void removeHumanCluster(pcl::PointCloud<pcl::PointNormal> &pcl_in, vector<Vector3f> &centroid, vector<float> remove_area_size, float pub_range)
{
	//cout<<"cluster size="<<cluster_cent.size()<<endl;
	//cout<<"remove size="<<area_size.size()<<endl;
	size_t num=pcl_in.points.size();
	pcl_removedPoints.resize(num);
#pragma omp parallel for
	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZ p;
		p.x=pcl_in.points[i].x;
		p.y=pcl_in.points[i].y;
		p.z=pcl_in.points[i].z;
		int judge=calculateRemoveArea(p, centroid, remove_area_size, pub_range);
		if(judge==1){
			pcl_removedPoints[i].x=pcl_in.points[i].x;
			pcl_removedPoints[i].y=pcl_in.points[i].y;
			pcl_removedPoints[i].z=pcl_in.points[i].z;
			pcl_removedPoints[i].intensity=0.0;//pcl_in.points[i].intensity;
			pcl_removedPoints[i].curvature=pcl_in.points[i].curvature;
			pcl_removedPoints[i].normal_x=pcl_in.points[i].normal_x;
			pcl_removedPoints[i].normal_y=pcl_in.points[i].normal_y;
			pcl_removedPoints[i].normal_z=pcl_in.points[i].normal_z;
		}else{
			pcl_removedPoints[i].x=0.0;
			pcl_removedPoints[i].y=0.0;
			pcl_removedPoints[i].z=0.0;
			pcl_removedPoints[i].intensity=0.0;
			pcl_removedPoints[i].curvature=0.0;
			pcl_removedPoints[i].normal_x=0.0;
			pcl_removedPoints[i].normal_y=0.0;
			pcl_removedPoints[i].normal_z=0.0;
		}
		//pcl_removedPoints.points.push_back(p);
		//pcl_removedPoints.points.push_back(p);
	}
}
void removeHumanCluster(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in, vector<Vector3f> &centroid, vector<float> remove_area_size, float pub_range)
{
	size_t num=pcl_in.points.size();
	pcl_removedPoints.resize(num);

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
	for(size_t i=0;i<num;i++){
		pcl::PointXYZ p;
		p.x=pcl_in.points[i].x;
		p.y=pcl_in.points[i].y;
		p.z=pcl_in.points[i].z;
		int judge=calculateRemoveArea(p, centroid, remove_area_size, pub_range);
		if(judge==1){
			pcl_removedPoints[i].x=pcl_in.points[i].x;
			pcl_removedPoints[i].y=pcl_in.points[i].y;
			pcl_removedPoints[i].z=pcl_in.points[i].z;
			pcl_removedPoints[i].intensity=pcl_in.points[i].intensity;
			pcl_removedPoints[i].curvature=pcl_in.points[i].curvature;
			pcl_removedPoints[i].normal_x=pcl_in.points[i].normal_x;
			pcl_removedPoints[i].normal_y=pcl_in.points[i].normal_y;
			pcl_removedPoints[i].normal_z=pcl_in.points[i].normal_z;
		}else{
			pcl_removedPoints[i].x=0.0;
			pcl_removedPoints[i].y=0.0;
			pcl_removedPoints[i].z=0.0;
			pcl_removedPoints[i].intensity=0.0;
			pcl_removedPoints[i].curvature=0.0;
			pcl_removedPoints[i].normal_x=0.0;
			pcl_removedPoints[i].normal_y=0.0;
			pcl_removedPoints[i].normal_z=0.0;
		}
	}
}
void removeHumanCluster2(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in, vector<Vector3f> &centroid, vector<float> remove_area_size, float pub_range)
{
	size_t num=pcl_in.points.size();
	for(size_t i=0;i<num;i++){
		pcl::PointXYZINormal p;
		p.x=pcl_in.points[i].x;
		p.y=pcl_in.points[i].y;
		p.z=pcl_in.points[i].z;
		int judge=calculateRemoveArea(p, centroid, remove_area_size, pub_range);
		if(judge==1){
			p.intensity=pcl_in.points[i].intensity;
			p.curvature=pcl_in.points[i].curvature;
			p.normal_x=pcl_in.points[i].normal_x;
			p.normal_y=pcl_in.points[i].normal_y;
			p.normal_z=pcl_in.points[i].normal_z;
		}else{
			p.x=0.0;
			p.y=0.0;
			p.z=0.0;
			p.intensity=0.0;
			p.curvature=0.0;
			p.normal_x=0.0;
			p.normal_y=0.0;
			p.normal_z=0.0;
		}
		pcl_removedPoints.points.push_back(p);
	}
}
////////////////////////////////////main/////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	ros::init(argc, argv, "sitting_detection");
	ros::NodeHandle nh,n;

	////subscriber
	ros::Subscriber pc2_sub = n.subscribe("/velodyne_index/velodyne_points", 10, pc2Callback);
	//ros::Subscriber perfect_sub = n.subscribe("/perfect_velodyne/estimateXYZINormal", 10, perfectCallback);
	ros::Subscriber perfect_sub = n.subscribe("/perfect_tyokota/normal", 10, perfectCallback);
	ros::Subscriber pcl_object_sub = n.subscribe("/rm_ground", 10, pclCallback);
	ros::Subscriber pcl_ground_sub = n.subscribe("/ground", 10, pclCallback2);
	ros::Subscriber distance_sub = n.subscribe("/nearest_dist", 10, distanceCallback);
	////publisher
	ros::Publisher pc2_pub = n.advertise<sensor_msgs::PointCloud2>("/sitting_detection/velodyne_points",10);
	ros::Publisher forS_pub = n.advertise<sensor_msgs::PointCloud2>("/sitting_detection/forS",10);
	ros::Publisher candidate_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/sitting_detection/candidate_pt",10);
	ros::Publisher normal_pub = n.advertise<sensor_msgs::PointCloud2>("/sitting_detection/normal",10);
	ros::Publisher target_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/sitting_detection/forPotential",10);
	ros::Publisher test_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/sitting_detection/test",10);
	ros::Publisher bd_line_positive_pub = n.advertise<visualization_msgs::Marker>("/sitting_detection/Marker/positive",10);
	ros::Publisher bd_line_warning_pub = n.advertise<visualization_msgs::Marker>("/sitting_detection/Marker/semi_positive",10);
	ros::Publisher bd_line_negative_pub = n.advertise<visualization_msgs::Marker>("/sitting_detection/Marker/negative",10);
	ros::Publisher bd_line_twin_pub = n.advertise<visualization_msgs::Marker>("/sitting_detection/Marker/twin",10);
	ros::Publisher bb_line_pub = n.advertise<visualization_msgs::Marker>("/sitting_detection/Marker/bb_line",10);
	ros::Publisher bb_line_n_pub = n.advertise<visualization_msgs::Marker>("/sitting_detection/Marker/bb_line_n",10);

	ros::Publisher human_number_pub = n.advertise<std_msgs::Int32>("/sitting_detection/human_number",10);
	ros::Rate loop_rate(10);

	float pubRange=0;
	if(remove_MODE!=1){
		cout<<"publishRange(Do you want to display what meter the points)= ";
		cin >> pubRange;
	}
	while (ros::ok()){

		if(callback_flag == true){
			callback_flag=false;

			clock_t startTime;//, endTime;
			startTime=clock();
			calFeatures(pcl_scan);
			//forPotential();
			//forTracker(candidate);
			if(perfect_pcl.points.size()){
				//removeHumanCluster(perfect_pcl, remove_centroid, remove_size, pubRange);
				removeHumanCluster2(perfect_pcl, remove_centroid, remove_size, pubRange);
			}
			remove_centroid.clear();
			remove_size.clear();
			if(pastCentroidUpdateFlag==true){
				pastCentroidUpdate();
			}
			if(past_centroid.size()!=0){
				pastCheckStartFlag=true;
			}else{
				pastCheckStartFlag=false;
				pastCentroidUpdateFlag=false;
			}
			//////////////////////////

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if(candidate_pcl.points.size()==0){
				pcl::PointXYZINormal p;
				p.x=100000.0;
				p.y=100000.0;
				p.z=0.0;
				candidate_pcl.points.push_back(p);
			}

			if(pcl_pc.points.size()==0){
				pcl::PointXYZINormal p;
				p.x=100000.0;
				p.y=100000.0;
				p.z=100000.0;
				pcl_pc.points.push_back(p);
			}

			visualization_msgs::Marker bb_line;
			bb_line.header.frame_id="/velodyne";
			bb_line.header.stamp=ros::Time::now();
			bb_line.ns="bounding_box";
			bb_line.action=visualization_msgs::Marker::ADD;
			bb_line.pose.orientation.w=1.0;
			bb_line.id=0;
			bb_line.type=visualization_msgs::Marker::LINE_LIST;
			bb_line.scale.x=0.05;
			bb_line.color.r=0.0;
			bb_line.color.g=1.0;
			bb_line.color.b=1.0;
			bb_line.color.a=1.0;
			for(size_t i=0;i<bb_pcl.points.size();i++){
				geometry_msgs::Point p;
				p.x=bb_pcl.points[i].x;
				p.y=bb_pcl.points[i].y;
				p.z=bb_pcl.points[i].z;
				bb_line.points.push_back(p);
			}
			bb_line_pub.publish(bb_line);
			bb_pcl.points.clear();

			visualization_msgs::Marker bb_line_n;
			bb_line_n.header.frame_id="/velodyne";
			bb_line_n.header.stamp=ros::Time::now();
			bb_line_n.ns="bounding_box";
			bb_line_n.action=visualization_msgs::Marker::ADD;
			bb_line_n.pose.orientation.w=1.0;
			bb_line_n.id=1;
			bb_line_n.type=visualization_msgs::Marker::LINE_LIST;
			bb_line_n.scale.x=0.05;
			bb_line_n.color.r=0.5;
			bb_line_n.color.g=0.5;
			bb_line_n.color.b=0.5;
			bb_line_n.color.a=1.0;
			for(size_t i=0;i<bb_pcl_n.points.size();i++){
				geometry_msgs::Point p;
				p.x=bb_pcl_n.points[i].x;
				p.y=bb_pcl_n.points[i].y;
				p.z=bb_pcl_n.points[i].z;
				bb_line_n.points.push_back(p);
			}
			bb_line_n_pub.publish(bb_line_n);
			bb_pcl_n.points.clear();

			pc2.header.frame_id="/velodyne";
			pc2.header.stamp=ros::Time::now();
			pc2_pub.publish(pc2);

			pcl::toROSMsg(pcl_removedPoints,forS_pt);
			pcl_removedPoints.points.clear();
			forS_pt.header.frame_id="/velodyne";
			forS_pt.header.stamp=ros::Time::now();
			forS_pub.publish(forS_pt);

			pcl::toROSMsg(pcl_pc,normal_pt);
			pcl_pc.points.clear();
			normal_pt.header.frame_id="/velodyne";
			normal_pt.header.stamp=ros::Time::now();
			normal_pub.publish(normal_pt);

			pcl::toROSMsg(candidate_pcl, candidate_pt);
			candidate_pcl.points.clear();
			candidate_pt.header.frame_id="/velodyne";
			candidate_pt.header.stamp=ros::Time::now();
			candidate_pt_pub.publish(candidate_pt);


			pcl::toROSMsg(target_pcl,target_pt);
			target_pcl.points.clear();
			target_pt.header.frame_id="/velodyne";
			target_pt.header.stamp=ros::Time::now();
			target_pt_pub.publish(target_pt);

			pcl::toROSMsg(test_pcl,test_pt);
			test_pcl.points.clear();
			test_pt.header.frame_id="/velodyne";
			test_pt.header.stamp=ros::Time::now();
			test_pt_pub.publish(test_pt);

			human_number_pub.publish(human_number);
			//normal_pt.points.clear();
			//pcl_scan.points.clear();	
			cout<<setw(40)<<"==========================All time taken:"<<setw(4)<<(double)(clock()-startTime)/CLOCKS_PER_SEC<<"[sec]"<<"=========================="<<endl<<endl;
		}
		//pc2.data.clear();
		ros::spinOnce();
		loop_rate.sleep();	
	}
}
//std::stringstream ss;
//ss << pcd_name << ".pcd";
//writer.write<pcl::PointXYZ> ("clustering/check_roc/pos/" + ss.str (), *cloud_cluster, false);

