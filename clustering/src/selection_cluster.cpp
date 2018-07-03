#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Core>
#include <boost/thread.hpp>
using namespace std;
using namespace Eigen;

typedef pcl::PointXYZINormal PointA;
typedef pcl::PointCloud<PointA>  CloudA;
typedef pcl::PointCloud<PointA>::Ptr  CloudAPtr;

bool centroid_flag = false;
bool point_flag = false;

ros::Publisher select_pub;
ros::Publisher select_centroid_pub;

CloudAPtr tmpcr_cloud (new CloudA);
CloudAPtr tmppt_cloud (new CloudA);

inline void Copy_point(PointA &input,PointA &output)
{
	output.x = input.x;
	output.y = input.y;
	output.z = input.z;
	output.normal_x = input.normal_x;
	output.normal_y = input.normal_y;
	output.normal_z = input.normal_z;
	output.intensity = input.intensity;
	output.curvature = input.curvature;
}

inline void pubPointCloud2 (ros::Publisher& pub,
                            const CloudA cloud,
                            const char* frame_id,
                            ros::Time& time)
{
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg (cloud,output);
    output.header.frame_id = frame_id;
    output.header.stamp = time;
    pub.publish (output);
}

void Selection_centroid(CloudAPtr centroid,CloudA &tmp_cloud)
{
	size_t centroid_size = centroid->points.size();
	for(size_t i =0;i<centroid_size;i++){
		// if(centroid->points[i].normal_z >0.3){
		// if(centroid->points[i].normal_z >-0.3){
		if(centroid->points[i].normal_z >-0.3){
			if(centroid->points[i].normal_x*centroid->points[i].normal_y > 0.02 && centroid->points[i].normal_x*centroid->points[i].normal_y < 0.20){
				tmp_cloud.push_back(centroid->points[i]);
			}
		}
	}

	// size_t cluster_size = tmp_cloud.points.size();
	// cluster->points.resize(cluster_size);
	// for(size_t i = 0;i<cluster_size;i++){
	// 	Copy_point(cluster->points[i],tmp_cloud.points[i]);
	// }

}
void Selection_cluster(CloudAPtr centroid,CloudAPtr cluster,CloudA &tmp_cloud)
{
	size_t centroid_size = centroid->points.size();
	for(size_t i =0;i<centroid_size;i++){
		// if(centroid->points[i].normal_z >0.3){
		// if(centroid->points[i].normal_z >-0.3){
		if(centroid->points[i].normal_z >-0.3){
			if(centroid->points[i].normal_x*centroid->points[i].normal_y > 0.02 && centroid->points[i].normal_x*centroid->points[i].normal_y < 0.20){
				for(size_t j = centroid->points[i].intensity;j< (centroid->points[i].intensity + centroid->points[i].curvature);j++){
					tmp_cloud.push_back(cluster->points[j]);
				}
			}
		}
	}

	// size_t cluster_size = tmp_cloud.points.size();
	// cluster->points.resize(cluster_size);
	// for(size_t i = 0;i<cluster_size;i++){
	// 	Copy_point(cluster->points[i],tmp_cloud.points[i]);
	// }

}

void Centroid_Callback(const sensor_msgs::PointCloud2::Ptr &msg)
{
	pcl::fromROSMsg(*msg,*tmpcr_cloud);
	centroid_flag = true;
}

void Point_Callback(const sensor_msgs::PointCloud2::Ptr &msg)
{
	pcl::fromROSMsg(*msg,*tmppt_cloud);
	point_flag = true;
}

int main (int argc, char** argv)
{
	ros::init(argc, argv, "selection_cluster");
	ros::NodeHandle nh,n;
  
	ros::Subscriber centroid_sub=n.subscribe("/cluster/centroid", 1, Centroid_Callback);
	ros::Subscriber points_sub=n.subscribe("/cluster/points", 1, Point_Callback);

    select_pub = nh.advertise<sensor_msgs::PointCloud2> ("/cluster/select", 1);
    select_centroid_pub = nh.advertise<sensor_msgs::PointCloud2> ("/cluster/select/centroid", 1);

    ros::Rate loop_rate(40);
    while (ros::ok()){
		if(centroid_flag&&point_flag){
			CloudA ros_pt;
			CloudA ros_centroid;
			Selection_cluster(tmpcr_cloud,tmppt_cloud,ros_pt);
			Selection_centroid(tmpcr_cloud,ros_centroid);

			ros::Time time = ros::Time::now();
			pubPointCloud2(select_pub,ros_pt,"/velodyne",time);
			pubPointCloud2(select_centroid_pub,ros_centroid,"/velodyne",time);
			centroid_flag = false;
			point_flag = false;
		}
        ros::spinOnce();
        loop_rate.sleep();
    }

  return (0);
}
