////////////////////////////////////
//human_detection with SVM        //
//author:tyokota                  //
//last update:  2014/02/17        //
//latest update:2014/09/03        //
//chenged subscribe message type. //
//PointCloud->pcl                 //
////////////////////////////////////

//MODE param
#define rmMODE 0
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

// PCL specific includes
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/sample_consensus/model_types.h>  
#include <pcl-1.8/pcl/segmentation/sac_segmentation.h>
#include <pcl-1.8/pcl/ModelCoefficients.h>  
#include <pcl-1.8/pcl/features/normal_3d.h>  
#include <pcl-1.8/pcl/filters/extract_indices.h> 
#include <pcl-1.8/pcl/filters/voxel_grid.h> 
#include <pcl-1.8/pcl/kdtree/kdtree.h>
#include <pcl-1.8/pcl/segmentation/extract_clusters.h> 
#include <pcl-1.8/pcl/features/normal_3d.h>
#include <pcl-1.8/pcl/octree/octree.h>
#include <pcl-1.8/pcl/filters/bilateral.h>
#include <pcl_conversions/pcl_conversions.h>
//#include <velodyne_pointcloud/point_types.h>

//Eigen include
//#include <Eigen/SVD>
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
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
#include <omp.h>

//1200rpm
//const float VR = 0.5034f;
//600rpm
//const float VR = 0.839f;
const float VR = 0.496777163904;

//namecpace
using namespace std;
using namespace Eigen;

size_t num;
const float dist_th=12.0;

sensor_msgs::PointCloud2 ros_pc;
sensor_msgs::PointCloud2 ros_pc2;
sensor_msgs::PointCloud2 obj_pc;
sensor_msgs::PointCloud2 curb_pc;
sensor_msgs::PointCloud2 test_pc;
pcl::PointCloud<pcl::PointXYZINormal> pcl_pc;
pcl::PointCloud<pcl::PointXYZINormal> pcl_scan;
pcl::PointCloud<pcl::PointXYZI> pcl_scan2;

bool callback_flag=false;
clock_t tStart;
pcl::PointCloud<pcl::PointXYZINormal> rmzero(pcl::PointCloud<pcl::PointXYZINormal> in)
{
	pcl::PointCloud<pcl::PointXYZINormal> out;
	for(size_t i=0;i<in.points.size();i++){
		bool zero=(in.points[i].x==0.0&&in.points[i].y==0.0);
		if(!zero){
			pcl::PointXYZINormal p;
			p.x=in.points[i].x;
			p.y=in.points[i].y;
			p.z=in.points[i].z;
			p.normal_x=in.points[i].normal_x;
			p.normal_y=in.points[i].normal_y;
			p.normal_z=in.points[i].normal_z;
			p.intensity=in.points[i].intensity;
			p.curvature=in.points[i].curvature;
			out.points.push_back(p);
		}
	}
	return out;
}
void pc2Callback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	//pcl::fromROSMsg(*msgs,pcl_scan);
	ros_pc=*msgs;
	callback_flag=true;
}

///////////////////////////////////////////////////////////
void process1(pcl::PointCloud<pcl::PointXYZI> &pcl_in)
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZI>);
	cloud_in->width = 1;
	cloud_in->height = pcl_in.points.size();
	cloud_in->points.resize (cloud_in->width * cloud_in->height);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i=0;i<(int)pcl_in.points.size();i++)
	{
		cloud_in->points[i].x = pcl_in.points[i].x;
		cloud_in->points[i].y = pcl_in.points[i].y;
		cloud_in->points[i].z = pcl_in.points[i].z;
		cloud_in->points[i].intensity = pcl_in.points[i].intensity;
	}

	pcl::VoxelGrid<pcl::PointXYZI> vg;  
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);  
	vg.setInputCloud (cloud_in);  
	//Down sampling. x*y*z to one point
	vg.setLeafSize (0.05f, 0.05f, 0.05f);
	vg.filter (*cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
	tree->setInputCloud (cloud_filtered);
	pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
	ec.setClusterTolerance (0.10);//Points clearance for clustering
	ec.setMinClusterSize (60);//Minimum include points of the cluster
	ec.setMaxClusterSize (1000);//Maximum points of the cluster //32Eなら近くても500点いかない。400でも大丈夫かな。
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);

	vector<pcl::PointCloud<pcl::PointXYZI>::Ptr >clusters;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
	for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
		for(std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++){
			cloud_cluster->points.push_back (cloud_filtered->points[*pit]);cloud_cluster->width = cloud_cluster->points.size ();
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = true;
		}
	}

}
static inline int getPointCloud2FieldsIndex (const sensor_msgs::PointCloud2 &cloud, const std::string &field_name)
{
	for (size_t d = 0; d < cloud.fields.size (); ++d)
		if (cloud.fields[d].name == field_name)
			return (d);
	return (-1);
}
pcl::PointCloud<pcl::PointXYZINormal> linkage(pcl::PointCloud<pcl::PointXYZINormal> pcl_in)
{
	pcl::PointCloud<pcl::PointXYZINormal> pcl_out;
	size_t num=pcl_in.points.size();

	pcl::PointXYZINormal p;
	for(size_t i=1;i<num;i++){
		if(i%32==0)i++;
		p=pcl_in.points[i-1];
		pcl_out.points.push_back(p);
		p=pcl_in.points[i];
		pcl_out.points.push_back(p);
	}
	for(size_t i=32;i<num;i++){
		p=pcl_in.points[i-32];
		pcl_out.points.push_back(p);
		p=pcl_in.points[i];
		pcl_out.points.push_back(p);
	}
	return pcl_out;
}
pcl::PointCloud<pcl::PointXYZINormal> output;
pcl::PointCloud<pcl::PointXYZINormal> linker_oc;
static inline void convert(sensor_msgs::PointCloud2 &input)
{
	size_t num=input.height*input.width*VR;
	output.resize(num);
	int x_idx = getPointCloud2FieldsIndex (input, "x");
	int y_idx = getPointCloud2FieldsIndex (input, "y");
	int z_idx = getPointCloud2FieldsIndex (input, "z");
	int i_idx = getPointCloud2FieldsIndex (input, "intensity");
	int r_idx = getPointCloud2FieldsIndex (input, "ring");
	int x_offset = input.fields[x_idx].offset;
	int y_offset = input.fields[y_idx].offset;
	int z_offset = input.fields[z_idx].offset;
	int i_offset = input.fields[i_idx].offset;
	int r_offset = input.fields[r_idx].offset;
	uint16_t rings[num];

	size_t j=0;
	for(size_t i=0;i<num;i++){
		memcpy(&output.points[i].x, &ros_pc.data[(i+j)*input.point_step+x_offset], sizeof(float));
		memcpy(&output.points[i].y, &ros_pc.data[(i+j)*input.point_step+y_offset], sizeof(float));
		memcpy(&output.points[i].z, &ros_pc.data[(i+j)*input.point_step+z_offset], sizeof(float));
		memcpy(&output.points[i].intensity, &ros_pc.data[(i+j)*input.point_step+i_offset], sizeof(float));
		memcpy(&rings[i], &ros_pc.data[(i+j)*input.point_step+r_offset], sizeof(float));
		output.points[i].normal_x=(float)rings[i];//ring
		output.points[i].curvature=(float)i;//index
		//if(i<=63)output.points[i].z=0.0;
		if(i%32==15){
			j=-j;
		}else if(i%32==31){
			j=0;
		}else{
			j+=1;
		}
	}
	for(size_t i=0;i<num-32;i++){
		float x1=output.points[i].x;
		float y1=output.points[i].y;
		//float z1=output.points[i].z;
		float x2=output.points[i+32].x;
		float y2=output.points[i+32].y;
		//float z2=output.points[i+32].z;

		//float dist_diff=sqrt(pow(x2-x1,2)+pow(y2-y1,2)+pow(z2-z1,2));
		float dist_diff=sqrt(pow(x2-x1,2)+pow(y2-y1,2));
		if(dist_diff>0.2){
			output.points[i].normal_y=1.0;
			output.points[i+32].normal_y=1.0;
			pcl::PointXYZINormal p;
			p=output.points[i];
			linker_oc.points.push_back(p);
			p=output.points[i+32];
			linker_oc.points.push_back(p);
		}
	}

	if(rmMODE==1){
		pcl::PointCloud<pcl::PointXYZINormal> outtmp=rmzero(output);
		output.points.clear();
		output=outtmp;
	}
}
pcl::PointCloud<pcl::PointXYZINormal> gaussianFilter(pcl::PointCloud<pcl::PointXYZINormal> pcl_in)
{
	pcl::PointCloud<pcl::PointXYZINormal> pcl_out;
	pcl_out.points.resize(pcl_in.points.size());
	//size_t masterIndex=0;

	//#pragma omp parallel for
	for(size_t ii=0;ii<32;ii++){
		const int patch_size=4;
		const float sigma=2.0;
		const float sigma_r=16.0;
		size_t num=pcl_in.points.size();
		for(size_t i=0;i<num;i++){
			if(pcl_in.points[i].normal_x==ii){
				float sum=0;
				float sum_w=0;
				int occCheck=0;
				float cent_x=pcl_in.points[i].x;
				float cent_y=pcl_in.points[i].y;
				int tmp_j=0;
				
				int occ_num=0;
				for(int j=-patch_size;j<=patch_size;j++){
					if(cent_x==0.0&&cent_y==0){
						occ_num++;
						continue;
					}else{
						float patch_i=i+(j*32);
						if(patch_i<0){
							j=-1;
							continue;
						}else if(patch_i>=num){
							patch_i=ii+tmp_j*32;
							tmp_j++;
						}
						if(pcl_in.points[patch_i].normal_y)occ_num++;
					}
				}

				for(int j=-patch_size;j<=patch_size;j++){
					if(cent_x==0.0&&cent_y==0){
						continue;
					}else{
						float patch_i=i+(j*32);
						if(patch_i<0){
							j=-1;
							continue;
						}else if(patch_i>num-patch_size){
							j=patch_size;
							continue;
						}
						if(occ_num>0||i>num-32){
							float dz=sqrt(pow(pcl_in.points[i].z-pcl_in.points[patch_i].z,2));
							if(dz>0.1){
								continue;
							}
						}

						float x=cent_x-pcl_in.points[patch_i].x;
						float y=cent_y-pcl_in.points[patch_i].y;
						//float weight=exp(-(x*x+y*y)/(2.0*sigma*sigma))/(2.0*M_PI*sigma*sigma);//gaussianFilter
						float b_weight=exp(-(cent_x*cent_x+cent_y*cent_y)/(2.0*sigma*sigma))*exp(-(x*x+y*y)/(2.0*sigma_r*sigma_r));//subspecific Filter
						sum+=b_weight*pcl_in.points[patch_i].z;
						sum_w+=b_weight;
					}
				}
				pcl_out.points[i].x=pcl_in.points[i].x;
				pcl_out.points[i].y=pcl_in.points[i].y;

				bool infc=isinf(sum/sum_w);
				bool nanc=isnan(sum/sum_w);

				if(infc||nanc)pcl_out.points[i].z=pcl_in.points[i].z;
				else if(occCheck==0)pcl_out.points[i].z=sum/sum_w;
				else pcl_out.points[i].z=pcl_in.points[i].z;

				pcl_out.points[i].normal_x=pcl_in.points[i].normal_x;
				pcl_out.points[i].normal_y=pcl_in.points[i].normal_y;
				pcl_out.points[i].normal_z=pcl_in.points[i].normal_z;
				pcl_out.points[i].intensity=pcl_in.points[i].normal_z;
				pcl_out.points[i].curvature=pcl_in.points[i].curvature;
				occCheck=0;
			}else{
				continue;
			}
		}
	}
	return pcl_out;
}
//edgeDetection() is doesn't work
pcl::PointCloud<pcl::PointXYZINormal> edgeDetection(pcl::PointCloud<pcl::PointXYZINormal> pcl_in)
{
	pcl::PointCloud<pcl::PointXYZINormal> pcl_out;
	//pcl_out.points.resize(pcl_in.points.size());
	//size_t masterIndex=0;
	for(size_t ii=0;ii<32;ii++){
		size_t num=pcl_in.points.size();

		//float r=2.0;
		for(size_t i=0;i<num;i++){
			if(pcl_in.points[i].normal_x==ii&&i>32&&i<num-32){
				float x=pcl_in.points[i].x;
				float y=pcl_in.points[i].y;
				float z=pcl_in.points[i].z;
				//float x_p=pcl_in.points[i-32].x;
				float y_p=pcl_in.points[i-32].y;
				float z_p=pcl_in.points[i-32].z;
				//float x_n=pcl_in.points[i+32].x;
				float y_n=pcl_in.points[i+32].y;
				float z_n=pcl_in.points[i+32].z;
				float h1=fabs(z-z_p);
				float h2=fabs(z_n-z);
				float dz=fabs(y_n-2.0*y+y_p)/h1/h2;
				if(dz<50){
					pcl::PointXYZINormal p;
					p.x=x;
					p.y=y;
					p.z=z;
					p.curvature=dz;
					p.intensity=(float)i;//index
					pcl_out.points.push_back(p);
				}
			}	
		}
	}
	return pcl_out;
}
////////////////////////////////////main/////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	ros::init(argc, argv, "index_objects");
	ros::NodeHandle nh,n;

	////subscriber
	ros::Subscriber pc2_sub = n.subscribe("/velodyne_points", 1, pc2Callback);
	////publisher
	//ros::Publisher ros_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/test_sample/velodyne_points",1);
	ros::Publisher obj_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/velodyne_index/object_points",1);
	//ros::Publisher curb_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/test_sample/curb_points",1);
	//ros::Publisher test_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/test_sample/test_points",1);
	//ros::Publisher link_oc_pub = n.advertise<visualization_msgs::Marker>("/test_sample/link_oc",1);

	//ros::Rate loop_rate(20);
	ros::Rate loop_rate(10);

	while (ros::ok()){
		if(callback_flag==true){
			callback_flag=false;
			//clock_t startTime=clock();
			convert(ros_pc);
			pcl::PointCloud<pcl::PointXYZINormal> all_link;
			pcl::PointCloud<pcl::PointXYZINormal> pcl_object;
			//pcl::PointCloud<pcl::PointXYZINormal> pcl_curb;
			pcl::PointCloud<pcl::PointXYZINormal> pcl_test;
			//pcl::PointCloud<pcl::PointXYZINormal> pcl_test_tmp;

			//all_link=linkage(output);
			//denoise process
			pcl_test=gaussianFilter(output);
			all_link=linkage(pcl_test);
			
			//visualization_msgs::Marker link_oc;
			//link_oc.header.frame_id="/velodyne";
			//link_oc.header.stamp=ros::Time::now();
			//link_oc.ns="linker";
			//link_oc.action=visualization_msgs::Marker::ADD;
			//link_oc.pose.orientation.w=1.0;
			//link_oc.id=0;
			//link_oc.type=visualization_msgs::Marker::LINE_LIST;
			//link_oc.scale.x=0.003;
			float rad_pre=0.0;
			///////// visualize link->include nan or inf. must check them.
			for(size_t i=0;i<all_link.points.size();i++){
				geometry_msgs::Point p;
				//std_msgs::ColorRGBA c;
				bool zero_link1=(all_link.points[i].x==0&&all_link.points[i].y==0);
				bool zero_link2=(all_link.points[i+1].x==0&&all_link.points[i+1].y==0);

				if(!zero_link1&&!zero_link2){
					float a1=all_link.points[i].x;
					float b1=all_link.points[i].y;
					float c1=all_link.points[i].z;
					float a2=all_link.points[i+1].x;
					float b2=all_link.points[i+1].y;
					float c2=all_link.points[i+1].z;

					float rad=-atan2(c2-c1, sqrt(pow(a2-a1,2)+pow(b2-b1,2)));
					//float dd=sqrt(pow(a2-a1,2)+pow(b2-b1,2));

					rad=fabs(rad)/(M_PI/2.0);
					if(rad>1.0)rad=1.0;
					else if(rad<0.0)rad=0.0;

					//p.x=all_link.points[i].x;
					//p.y=all_link.points[i].y;
					//p.z=all_link.points[i].z;
					//link_oc.points.push_back(p);
					//c.a=1.0;
					//if(rad<0.1){
					//	c.a=0.1;
					//	c.r=0.0;
					//	c.g=1.0;
					//	c.b=0.0;
					//}else if(0.1<=rad&&rad<0.3){
					//	//if(sqrt(pow(p.x-all_link.points)))
					//	c.a=0.1;
					//	c.r=0.0;
					//	c.g=1.0;
					//	c.b=1-(0.3-rad)/0.2;
					//}else if(0.3<=rad&&rad<0.5){
					//	//c.a=0.1;
					//	c.r=0.0;
					//	c.g=(0.5-rad)/0.2;
					//	c.b=1.0;
					//}else if(0.5<=rad&&rad<0.7){
					//	c.r=1-(0.7-rad)/0.2;
					//	c.g=0.0;
					//	c.b=1.0;
					//}else if(0.7<=rad&&rad<0.9){
					//	c.r=1.0;
					//	c.g=0.0;
					//	c.b=(0.9-rad)/0.2;
					//}else{
					//	c.r=1.0;
					//	c.g=0.0;
					//	c.b=0.0;
					//}
					//link_oc.colors.push_back(c);
					//p.x=all_link.points[i+1].x;
					//p.y=all_link.points[i+1].y;
					//p.z=all_link.points[i+1].z;
					//link_oc.points.push_back(p);
					//link_oc.colors.push_back(c);				

					bool occ=(all_link.points[i].normal_y==0&&all_link.points[i+1].normal_y==0);
					//bool hc=(all_link.points[i].z<-0.7&&all_link.points[i+1].z<-0.7);
					bool last_num=(i<num-32);
					if(rad>=(0.3-rad_pre)&&occ&&last_num){
						rad_pre=0.1;
						pcl::PointXYZINormal pcl_p;
						pcl_p.x=all_link.points[i].x;
						pcl_p.y=all_link.points[i].y;
						pcl_p.z=all_link.points[i].z;
						pcl_p.curvature=rad;
						if(1.5>sqrt(pow(pcl_p.x,2)+pow(pcl_p.y,2))){
							i++;
							rad_pre=0.0;
							continue;
						}
						pcl_object.points.push_back(pcl_p);

						pcl_p.x=all_link.points[i+1].x;
						pcl_p.y=all_link.points[i+1].y;
						pcl_p.z=all_link.points[i+1].z;
						pcl_p.curvature=rad;
						pcl_object.points.push_back(pcl_p);
					}else{
						rad_pre=0.0;
					}
				}
				i++;
			}
			//link_oc_pub.publish(link_oc);
			//linker_oc.points.clear();

			//cout<<"test_sample="<<output.points.size()<<" test_points="<<pcl_test.points.size()<<" pcl_object="<<pcl_object.points.size()<<endl;

			//process1(pcl_scan);
			//pcl::toROSMsg(output,ros_pc2);
			//output.points.clear();
			//ros_pc2.header.frame_id="/velodyne";
			//ros_pc2.header.stamp=ros::Time::now();
			//ros_pc_pub.publish(ros_pc2);

			pcl::toROSMsg(pcl_object,obj_pc);
			pcl_object.points.clear();
			obj_pc.header.frame_id="/velodyne";
			obj_pc.header.stamp=ros::Time::now();
			obj_pc_pub.publish(obj_pc);

			//pcl::toROSMsg(pcl_curb,curb_pc);
			//pcl_curb.points.clear();
			//curb_pc.header.frame_id="/velodyne";
			//curb_pc.header.stamp=ros::Time::now();
			//curb_pc_pub.publish(curb_pc);

			//pcl::toROSMsg(pcl_test,test_pc);
			//pcl_test.points.clear();
			//test_pc.header.frame_id="/velodyne";
			//test_pc.header.stamp=ros::Time::now();
			//test_pc_pub.publish(test_pc);


			//cout<<setw(40)<<"==========================All time taken:"<<setw(4)<<(double)(clock()-startTime)/CLOCKS_PER_SEC<<"[sec]"<<"=========================="<<endl<<endl;

		}
		ros::spinOnce();
		loop_rate.sleep();	
		}
	}
