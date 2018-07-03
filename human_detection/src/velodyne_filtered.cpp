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
#include "human_recognition.h"

//1200rpm
//const float VR = 0.5034f;
//under 0.5
const float VR = 1.0;
//600rpm
//const float VR = 0.839f;

const size_t start_end=32;

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
pcl::PointCloud<pcl::PointXYZINormal> pcl_scan3;
pcl::PointCloud<pcl::PointXYZI> pcl_scan2;

bool callback_flag=false;
bool callback_flag2=false;
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
	pcl::fromROSMsg(*msgs,pcl_scan);
	//ros_pc=*msgs;
	callback_flag=true;
}
void pc2Callback2(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs,pcl_scan3);
	//ros_pc=*msgs;
	callback_flag2=true;
}
pcl::PointCloud<pcl::PointXYZINormal> linkage(pcl::PointCloud<pcl::PointXYZINormal> pcl_in)
{
	pcl::PointCloud<pcl::PointXYZINormal> pcl_out;
	size_t num=pcl_in.points.size();

	for(size_t i=1;i<num;i++){
		if(i%32==0)i++;
		pcl::PointXYZINormal p;
		p=pcl_in.points[i-1];
		pcl_out.points.push_back(p);
		p=pcl_in.points[i];
		pcl_out.points.push_back(p);
	}
	for(size_t i=32;i<num;i++){
		pcl::PointXYZINormal p;
		p=pcl_in.points[i-32];
		pcl_out.points.push_back(p);
		p=pcl_in.points[i];
		pcl_out.points.push_back(p);
	}
	return pcl_out;
}
////////////////////////////////////main/////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	ros::init(argc, argv, "velodyne_filtered");
	ros::NodeHandle nh,n;

	////subscriber
	//ros::Subscriber pc2_sub = n.subscribe("velodyne_index/velodyne_points", 1, pc2Callback);
	ros::Subscriber pc2_sub = n.subscribe("rm_ground/original_points", 1, pc2Callback);
	ros::Subscriber pc2_sub2 = n.subscribe("rm_ground", 1, pc2Callback2);
	////publisher
	ros::Publisher ros_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/velodyne_filtered/velodyne_points",1);
	ros::Publisher obj_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/velodyne_filtered/object_points",1);
	ros::Publisher curb_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/velodyne_filtered/curb_points",1);
	ros::Publisher test_pc_pub = n.advertise<sensor_msgs::PointCloud2>("/velodyne_filtered/filtered_points",1);
	ros::Publisher link_oc_pub = n.advertise<visualization_msgs::Marker>("/velodyne_filtered/link_marker",1);

	ros::Rate loop_rate(20);

	while (ros::ok()){
		if(callback_flag){
			callback_flag=false;
			//callback_flag2=false;
			clock_t startTime=clock();
			pcl::PointCloud<pcl::PointXYZINormal> all_link;
			pcl::PointCloud<pcl::PointXYZINormal> pcl_object;
			pcl::PointCloud<pcl::PointXYZINormal> pcl_curb;
			pcl::PointCloud<pcl::PointXYZINormal> pcl_test;
			pcl::PointCloud<pcl::PointXYZINormal> pcl_test_tmp;

			//all_link=linkage(output);
			//denoise process
			//cout<<"pcl_scan="<<pcl_scan.points.size()<<endl;
			for(size_t i=0;i<pcl_scan.points.size()-32;i++){
				float x1=pcl_scan.points[i].x;
				float y1=pcl_scan.points[i].y;
				//float z1=pcl_scan.points[i].z;
				float x2=pcl_scan.points[i+32].x;
				float y2=pcl_scan.points[i+32].y;
				//float z2=pcl_scan.points[i+32].z;
				bool zero_check=((x1==0&&y1==0)||(x2==0&&y2==0));
				bool oc_check=(pcl_scan.points[i].curvature>0.0);
				bool dist_check=(sqrt(pow(x1,2)+pow(y1,2))>15.0&&sqrt(pow(x2,2)+pow(y2,2))>15.0);
				//float dist_diff=sqrt(pow(x2-x1,2)+pow(y2-y1,2)+pow(z2-z1,2));
				float dist_diff=sqrt(pow(x2-x1,2)+pow(y2-y1,2));
				if(dist_diff>0.25&&!zero_check&&!dist_check){
					pcl_scan.points[i].curvature=1.0;
					pcl_scan.points[i+32].curvature=1.0;
				}else if(oc_check){
					pcl_scan.points[i].curvature=1.0;
					pcl_scan.points[i+32].curvature=0.0;
				}else{
					pcl_scan.points[i].curvature=0.0;
					pcl_scan.points[i+32].curvature=0.0;
				}

			}

			// pcl_test=gaussianFilter(pcl_scan);
			// all_link=linkage(pcl_test);
			all_link=linkage(pcl_scan);

			visualization_msgs::Marker link_oc;
			link_oc.header.frame_id="/velodyne";
			link_oc.header.stamp=ros::Time::now();
			link_oc.ns="linker";
			link_oc.action=visualization_msgs::Marker::ADD;
			link_oc.pose.orientation.w=1.0;
			link_oc.id=0;
			link_oc.type=visualization_msgs::Marker::LINE_LIST;
			link_oc.scale.x=0.01;
			float rad_pre=0.0;
			///////// visualize link->include nan or inf. must check them.
			for(size_t i=0;i<all_link.points.size();i++){
				//cout<<"all points="<<all_link.points.size()<<endl;
				geometry_msgs::Point p;
				std_msgs::ColorRGBA c;
				bool zero_link1=(all_link.points[i].x==0&&all_link.points[i].y==0);
				bool zero_link2=(all_link.points[i+1].x==0&&all_link.points[i+1].y==0);
				//bool occ_link=(all_link.points[i].curvature==1||all_link.points[i+1].curvature==1);
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

					p.x=all_link.points[i].x;
					p.y=all_link.points[i].y;
					p.z=all_link.points[i].z;
					link_oc.points.push_back(p);
					c.a=0.5;
					if(rad<0.1){
						//c.a=0.1;
						c.r=0.0;
						c.g=1.0;
						c.b=0.0;
					}else if(0.1<=rad&&rad<0.3){
						//if(sqrt(pow(p.x-all_link.points)))
						//c.a=0.1;
						c.r=0.0;
						c.g=1.0;
						c.b=1-(0.3-rad)/0.2;
					}else if(0.3<=rad&&rad<0.5){
						//c.a=0.1;
						c.r=0.0;
						c.g=(0.5-rad)/0.2;
						c.b=1.0;
					}else if(0.5<=rad&&rad<0.7){
						c.a=1.0;
						c.r=1-(0.7-rad)/0.2;
						c.g=0.0;
						c.b=1.0;
					}else if(0.7<=rad&&rad<0.9){
						c.a=1.0;
						c.r=1.0;
						c.g=0.0;
						c.b=(0.9-rad)/0.2;
					}else{
						c.a=1.0;
						c.r=1.0;
						c.g=0.0;
						c.b=0.0;
					}
					link_oc.colors.push_back(c);
					p.x=all_link.points[i+1].x;
					p.y=all_link.points[i+1].y;
					p.z=all_link.points[i+1].z;
					link_oc.points.push_back(p);
					link_oc.colors.push_back(c);				


					bool occ=(all_link.points[i].curvature==0&&all_link.points[i+1].curvature==0);
					//bool hc=(all_link.points[i].z<-0.7&&all_link.points[i+1].z<-0.7);
					bool first_num=(all_link.points[i].normal_y<32);
					bool last_num=(all_link.points[i].normal_y>num-32);
					bool nearing=(all_link.points[i].normal_x<5);
					float obj_distance=sqrt(pow(all_link.points[i].x,2)+pow(all_link.points[i].y,2));
					bool head_cut=(all_link.points[i].z<1.0);
					bool distance_cut=(obj_distance<2.0||obj_distance>16.0);

					/*if(all_link.points[i].curvature>0.0&&obj_distance<20.0&&head_cut){
					  pcl::PointXYZINormal pcl_p;
					  pcl_p.x=all_link.points[i].x;
					  pcl_p.y=all_link.points[i].y;
					  pcl_p.z=all_link.points[i].z;
					  pcl_p.normal_x=all_link.points[i].normal_x;
					  pcl_p.normal_y=all_link.points[i].normal_y;
					  pcl_p.normal_z=-1.0;
					  pcl_p.intensity=all_link.points[i].intensity;
					  pcl_p.curvature=all_link.points[i].curvature;
					//これコメントアウトしてね
					pcl_object.points.push_back(pcl_p);
					//ここまで
					}*/

					float weight=(obj_distance/10.0)+1.0;
					float weight2=0.0;
					if(nearing)weight2=0.1;
					float weight3=0.0;
					if(all_link.points[i].normal_x==0)weight3=0.2;
					else if(all_link.points[i].normal_x==1)weight3=0.1;
					else if(all_link.points[i].normal_x<7&&all_link.points[i].normal_x>1)weight3=0.05;
					float rad_weight=rad*weight-weight2-weight3;
					bool limit=(i<all_link.points.size()/2);
					// if(rad_weight>=(0.3-rad_pre)&&occ&&head_cut&&limit&&distance_cut){
					if(rad_weight>=(0.3-rad_pre)&&head_cut&&occ&&limit){
						rad_pre=0.1;
						pcl::PointXYZINormal pcl_p;
						pcl_p.x=all_link.points[i].x;
						pcl_p.y=all_link.points[i].y;
						pcl_p.z=all_link.points[i].z;
						pcl_p.normal_x=all_link.points[i].normal_x;
						pcl_p.normal_y=all_link.points[i].normal_y;
						pcl_p.normal_z=rad;
						pcl_p.intensity=all_link.points[i].intensity;
						pcl_p.curvature=all_link.points[i].curvature;

						// if(1.0>sqrt(pow(pcl_p.x,2)+pow(pcl_p.y,2))||15.0<sqrt(pow(pcl_p.x,2)+pow(pcl_p.y,2))){
						if(15.0<sqrt(pow(pcl_p.x,2)+pow(pcl_p.y,2))){
							i++;
							rad_pre=0.0;
							continue;
						}
						pcl_object.points.push_back(pcl_p);

						pcl_p.x=all_link.points[i+1].x;
						pcl_p.y=all_link.points[i+1].y;
						pcl_p.z=all_link.points[i+1].z;
						pcl_p.normal_x=all_link.points[i+1].normal_x;
						pcl_p.normal_y=all_link.points[i+1].normal_y;
						pcl_p.normal_z=rad;
						pcl_p.intensity=all_link.points[i+1].intensity;
						pcl_p.curvature=all_link.points[i+1].curvature;
						pcl_object.points.push_back(pcl_p);


					}else{
						rad_pre=0.0;
					}
				}
				i++;
			}
			link_oc_pub.publish(link_oc);
			link_oc.points.clear();
			link_oc.colors.clear();

			for(size_t i=0;i<pcl_scan3.points.size();i++){
				pcl::PointXYZINormal p;
				p.x=pcl_scan3.points[i].x;
				p.y=pcl_scan3.points[i].y;
				p.z=pcl_scan3.points[i].z;
				p.normal_x=pcl_scan3.points[i].normal_x;
				p.normal_y=pcl_scan3.points[i].normal_y;
				p.normal_z=pcl_scan3.points[i].normal_z;
				p.intensity=pcl_scan3.points[i].intensity;
				p.curvature=pcl_scan3.points[i].curvature;
				pcl_object.points.push_back(p);

			}
			cout<<"pub size="<<pcl_object.points.size()<<endl;
			pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZINormal>);
			cloud_in=ptrTransform(pcl_object);
			pcl::VoxelGrid<pcl::PointXYZINormal> vg;  
			pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZINormal>);  

			// *cloud_filtered=*cloud_in;
			vg.setInputCloud (cloud_in);  
			vg.setLeafSize (0.05f, 0.05f, 0.05f);
			vg.filter (*cloud_filtered);
			cout<<"down sumple="<<cloud_filtered->points.size()<<endl;

			pcl::toROSMsg(*cloud_filtered,obj_pc);
			pcl_object.points.clear();
			obj_pc.header.frame_id="/velodyne";
			obj_pc.header.stamp=ros::Time::now();
			obj_pc_pub.publish(obj_pc);

			pcl::toROSMsg(pcl_curb,curb_pc);
			pcl_curb.points.clear();
			curb_pc.header.frame_id="/velodyne";
			curb_pc.header.stamp=ros::Time::now();
			curb_pc_pub.publish(curb_pc);

			pcl::toROSMsg(pcl_test,test_pc);
			pcl_test.points.clear();
			test_pc.header.frame_id="/velodyne";
			test_pc.header.stamp=ros::Time::now();
			test_pc_pub.publish(test_pc);


			cout<<setw(40)<<"==========================All time taken:"<<setw(4)<<(double)(clock()-startTime)/CLOCKS_PER_SEC<<"[sec]"<<"=========================="<<endl<<endl;

		}
		ros::spinOnce();
		// loop_rate.sleep();	
	}
}
