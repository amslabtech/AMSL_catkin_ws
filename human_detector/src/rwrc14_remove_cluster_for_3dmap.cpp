#include "includs.h"

// #define MODE 0
#define MODE 1

#define THRESH_DIST 0.8//1.8
sensor_msgs::PointCloud2 original_pt;
sensor_msgs::PointCloud2 removed_pt;

pcl::PointCloud<pcl::PointXYZINormal> pcl_org;
pcl::PointCloud<pcl::PointXYZINormal> pcl_centroid;

////////////////////////////////////////////////

bool callback_flag = false;
bool callback_flag1 = false;
bool callback_flag2 = false;

void originalCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
				//cout<<"aaaa"<<endl;
	original_pt=*msgs;
	pcl::fromROSMsg(*msgs,pcl_org);
	callback_flag1 = true;
}
void objectCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
				//cout<<"bbbbb"<<endl;
	pcl::fromROSMsg(*msgs,pcl_centroid);
	callback_flag2 = true;
}
pcl::PointCloud<pcl::PointXYZINormal> rmCluster(pcl::PointCloud<pcl::PointXYZINormal>& pcl_in,
												pcl::PointCloud<pcl::PointXYZINormal>& centroid)
{
	pcl::PointCloud<pcl::PointXYZINormal> output;
	output.points.clear();

	for(size_t i=0;i<pcl_in.points.size();i++){
		//pcl::PointXYZINormal p=pcl_in.points[i];
		pcl::PointXYZINormal p;
		p.x=pcl_in.points[i].x;
		p.y=pcl_in.points[i].y;
		p.z=pcl_in.points[i].z;
		p.normal_x=pcl_in.points[i].normal_x;
		p.normal_y=pcl_in.points[i].normal_y;
		p.normal_z=pcl_in.points[i].normal_z;
		p.curvature=pcl_in.points[i].curvature;
		//p.intensity=0;
		p.intensity=pcl_in.points[i].intensity;
		float min_dist=1000.0;
		int min_id=0;
		float dead_line=sqrt(p.x*p.x+p.y*p.y);
		if(THRESH_DIST>dead_line){
			p.x=0;
			p.y=0;
			p.z=0;
		}else{
			for(size_t j=0;j<centroid.points.size();j++){
				float dist=sqrt(pow(p.x-centroid.points[j].x,2)+pow(p.y-centroid.points[j].y,2));
				if(min_dist>dist){
					min_dist=dist;
					min_id=j;
				}
				if((centroid.points[min_id].normal_x/1.0)>min_dist){
					break;
				}
			}
			if((centroid.points[min_id].normal_x/1.0)>min_dist){
				p.x=0;
				p.y=0;
				p.z=0;
			}
		}
		output.points.push_back(p);
	}
	return output;
}
pcl::PointCloud<pcl::PointXYZINormal> rmCluster0(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, pcl::PointCloud<pcl::PointXYZINormal> centroid)
{
	pcl::PointCloud<pcl::PointXYZINormal> output;
	output.points.clear();

	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZINormal p;//p=pcl_in.points[i];
		p.x=pcl_in.points[i].x;
		p.y=pcl_in.points[i].y;
		p.z=pcl_in.points[i].z;
		p.normal_x=pcl_in.points[i].normal_x;
		p.normal_y=pcl_in.points[i].normal_y;
		p.normal_z=pcl_in.points[i].normal_z;
		p.curvature=pcl_in.points[i].curvature;
		//p.intensity=0;
		p.intensity=pcl_in.points[i].intensity;
		float min_dist=1000.0;
		int min_id=0;
		float dead_line=sqrt(p.x*p.x+p.y*p.y);
		if(THRESH_DIST>dead_line){
			p.x=0;
			p.y=0;
			p.z=0;
		}
		output.points.push_back(p);
	}
	return output;
}
/*{{{*/
// pcl::PointCloud<pcl::PointXYZINormal> rmCluster0(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, pcl::PointCloud<pcl::PointXYZINormal> centroid)
// {
	// pcl::PointCloud<pcl::PointXYZINormal> output;

	// for(size_t i=0;i<pcl_in.points.size();i++){
		// pcl::PointXYZINormal p=pcl_in.points[i];
		// float min_dist=1000.0;
		// int min_id=0;
		// float dead_line=sqrt(p.x*p.x+p.y*p.y);
		// if(dead_line>2.0){
			// for(size_t j=0;j<centroid.points.size();j++){
				// float dist=sqrt(pow(p.x-centroid.points[j].x,2)+pow(p.y-centroid.points[j].y,2));
				// if(min_dist>dist){
					// min_dist=dist;
					// min_id=j;
				// }
				// if((centroid.points[min_id].normal_x/1.0)>min_dist){
					// break;
				// }
			// }
			// if(min_dist>(centroid.points[min_id].normal_x/1.0)){
				// output.points.push_back(p);
			// }
		// }
	// }
	// return output;
// }
/*}}}*/
////////////////////////////////////main/////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	ros::init(argc, argv, "remove_cluster");
	ros::NodeHandle nh,n;

	////subscriber////
	ros::Subscriber original_sub;
	if(MODE==0)original_sub = n.subscribe("/velodyne_points", 1, originalCallback);
	// if(MODE==0)original_sub = n.subscribe("human_recognition/velodyne_points", 1, originalCallback);
	// if(MODE==0)original_sub = n.subscribe("/velodyne_index/velodyne_points", 1, originalCallback);
	if(MODE==1)original_sub = n.subscribe("/perfect_velodyne/normal", 1, originalCallback);
	ros::Subscriber centroid_sub = n.subscribe("/human_recognition/positive_position", 1, objectCallback);
	////publisher////
	ros::Publisher original_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/rm_cluster/subscribe_points",1);
	ros::Publisher removed_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/rm_cluster/removed_points",1);

	//cout<<"start"<<endl;	
	ros::Rate loop_rate(20); // 20
	while (ros::ok()){
		if(callback_flag1&&callback_flag2)callback_flag=true;
		if(callback_flag){
			// clock_t start;
			// start=clock();
			callback_flag=false;
			callback_flag1=false;
			callback_flag2=false;

			pcl::PointCloud<pcl::PointXYZINormal> removed_pcl;
			// if(pcl_org.points.size()!=0&&pcl_centroid.points.size()!=0)
			if(pcl_org.points.size()!=0&&pcl_centroid.points.size()!=0)
				removed_pcl=rmCluster(pcl_org, pcl_centroid);
			else if(pcl_centroid.points.size()==0)
				removed_pcl=rmCluster0(pcl_org, pcl_centroid);
				// removed_pcl=pcl_org;

			original_pt.header.frame_id="/velodyne";
			original_pt.header.stamp=ros::Time::now();
			original_pt_pub.publish(original_pt);

			pcl::toROSMsg(removed_pcl,removed_pt);
			removed_pcl.points.clear();
			removed_pt.header.frame_id="/velodyne";
			removed_pt.header.stamp=ros::Time::now();
			removed_pt_pub.publish(removed_pt);
			// std::cout<<(double)(clock()-start)/CLOCKS_PER_SEC<<std::endl;
			//cout<<"end"<<endl;	

		}
		ros::spinOnce();
		loop_rate.sleep();	
	}
	cout<<"END program........"<<endl;
	return 0;
}
