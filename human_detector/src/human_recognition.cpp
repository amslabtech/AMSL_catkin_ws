//human_recognition//
//author: tyokota
//update: 2015/01/07

#include "human_recognition.h"
// MODE: 32E->32, 64E->64 //
// #define MODE 64
#define MODE 32

//svm param//////////////////////
// all_feature6_2 param
#define COST 4.0
#define GAMMA 0.0078125
#define NU 0.028061
#define RHO 1.240640
//////////////////////////////////
bool WRITEMODE=false;
// bool WRITEMODE=true;
bool func_check=false;
double loop_count=1.0;
double avg_time=0.0;
int pcd_pos=0;
int pcd_neg=0;
sensor_msgs::PointCloud2 original_pt;
sensor_msgs::PointCloud2 removed_pt;
sensor_msgs::PointCloud2 positive_pt;
sensor_msgs::PointCloud2 negative_pt;
sensor_msgs::PointCloud2 position_pt;
pcl::PointCloud<pcl::PointXYZINormal> pcl_org;
pcl::PointCloud<pcl::PointXYZINormal> pcl_scan;
pcl::PointCloud<pcl::PointXYZINormal> perfect_pcl;
////////////////////////////////////////////////

bool callback_flag = false;
bool callback_flag1 = false;
bool callback_flag2 = false;
bool callback_flag3 = false;

void originalCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	original_pt=*msgs;
	pcl::fromROSMsg(*msgs,pcl_org);
	callback_flag1=true;
}
void objectCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs,pcl_scan);
	callback_flag2=true;
}
void perfectCallback(const sensor_msgs::PointCloud2::Ptr &msgs)
{
	pcl::fromROSMsg(*msgs, perfect_pcl);
	callback_flag3=true;
}
clusterInformation transformMatrix(pcl::PointCloud<pcl::PointXYZINormal> pcl_in) 
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
	clusterInformation output;
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	feature_extractor.getOBB(output.min, output.max, output.position, output.rot_mat);
	output.min.z=min[2];
	output.max.z=max[2];
	output.width=output.max.x-output.min.x;
	output.length=output.max.y-output.min.y;
	output.height=output.max.z-output.min.z;
	output.rot_mat_inv=output.rot_mat;
	output.rot_mat=output.rot_mat.inverse();
	return output;
}
//SVM Evaluation
int times=0;
const int f_num=329;
//svm_model *model=svm_load_model("/home/amsl/AMSL_ros_pkg/human_detection/classification/svm_model/all_feature6_2.model", times);
// svm_model *model=svm_load_model("/home/ubuntu/AMSL_ros_pkg/object-recognition/human_detection/svm_model/all_feature6_2.model", times);
svm_model *model=svm_load_model("/home/amsl/AMSL_ros_pkg/object-recognition/human_detection/svm_model/all_feature6_2.model", times);
svmRange svm_range;
int svmLoadRange()
{
	cout<<"load range file"<<endl;
	svm_range.min.resize(f_num);
	svm_range.max.resize(f_num);
	// ifstream ifs("/home/amsl/AMSL_ros_pkg/human_detection/classification/svm_model/all_feature6.range");
	// ifstream ifs("/home/ubuntu/AMSL_ros_pkg/object-recognition/human_detection/svm_model/all_feature6.range");
	ifstream ifs("/home/amsl/AMSL_ros_pkg/object-recognition/human_detection/svm_model/all_feature6.range");
	string buf;
	int n=0;
	while(ifs && getline(ifs, buf)){
		istringstream is(buf);
		string st[4];
		is >> st[0] >> st[1] >> st[2];
		if(st[0]!="x"&&st[0]!="-1"&&st[1]!="1"){
			svm_range.min[n]=S2D(st[1]);
			svm_range.max[n]=S2D(st[2]);
			n++;
		}
	}
	model->param.C=COST;
	model->param.gamma=GAMMA;
	model->param.degree=RHO;
	model->param.nu=NU;
	return 1;
}
double svmClassifier(shapeFeatures features)
{
	svm_node data[f_num];
	for(int j=0;j<f_num;j++){
		data[j].index=j+1;
		data[j].value=features.f_i[j];
	}
	for(int j=0;j<f_num;j++)
		data[j].value=(2*(data[j].value-svm_range.min[j])/(svm_range.max[j]-svm_range.min[j]))-1;
	for (int j=0;j<f_num;j++){
		if(data[j].value>1)data[j].value=1;
		if(data[j].value<-1)data[j].value=-1;
	}	
	double result_dec=svm_predict_dec(model, data);
	if(!times)times=1;
	return (result_dec);
}
vector<float> calMomentTensor(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in)
{
	vector<float> f_;
	Matrix3f nmit=MatrixXf::Zero(3,3);
	Matrix3f tensor=Matrix3f::Zero(3,3);

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

	size_t n_pt=pcl_in.points.size();
	MatrixXf dev1=MatrixXf::Zero(n_pt,3);
	MatrixXf dev2=MatrixXf::Zero(3,n_pt);
	Matrix3f cov_3d=MatrixXf::Zero(3,3);

	for(size_t i=0;i<n_pt;i++){
		dev1(i,0)=dev3d[i](0);
		dev1(i,1)=dev3d[i](1);
		dev1(i,2)=dev3d[i](2);
		dev2(0,i)=dev3d[i](0);
		dev2(1,i)=dev3d[i](1);
		dev2(2,i)=dev3d[i](2);
	}
	cov_3d=dev2*dev1/((float)n_pt-1.0);
	for(int i=0;i<3;i++){
		for(int j=i;j<3;j++){
			float tmp=cov_3d(i,j);
			f_.push_back(tmp);
		}
	}
	return(f_);
}
vector<float> cal2DCovariance(pcl::PointCloud<pcl::PointXYZ> &pcl_in)
{
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
	float cluster_block=height/pitch;
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
			if(pcl_in.points[i].z>low&&high>pcl_in.points[i].z){
				check_count++;
				float max_tmp_x=pcl_in.points[i].x;
				float max_tmp_y=pcl_in.points[i].y;
				float min_tmp_x=pcl_in.points[i].x;
				float min_tmp_y=pcl_in.points[i].y;

				if(max_tmp_x>max_x)max_x=max_tmp_x;
				if(max_tmp_y>max_y)max_y=max_tmp_y;
				if(min_tmp_x<min_x)min_x=min_tmp_x;
				if(min_tmp_y<min_y)min_y=min_tmp_y;

				sum_x+=fabs(pcl_in.points[i].x); 
			}
		}
		delta_x=max_x-min_x;
		delta_y=max_y-min_y;

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
	}
	return(f_);
}
///////////////////////////////////////////////////////////////////////////////////////////
Mats histogramProcess(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, int e)
{
	size_t ii=14;
	size_t jj=7;
	if(e==3){
		ii=9;
		jj=5;
	}

	Mats histograms;
	MatrixXf histogram=MatrixXf::Zero(ii,jj);
	MatrixXf histogram_normal=MatrixXf::Zero(ii,jj);

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
				float pt_e1=p.z;
				float pt_e2=p.x;
				if(e==3)pt_e2=p.y;
				bool eq1=((max_e1-d_e1/(float)ii*i)>pt_e1&&pt_e1>(max_e1-d_e1/(float)ii*(i+1)));
				bool eq2=((min_e2+d_e2/(float)jj*j)<pt_e2&&pt_e2<(min_e2+d_e2/(float)jj*(j+1)));

				if(eq1&&eq2){
					points_num++;
					sum_normal+=p.normal_y;
				}
				else pcl_in.points.push_back(p);
			}

			float avg_normal=0;
			bool sum_normal_check=(isnan(sum_normal));
			bool zero_check=(!sum_normal||!points_num);
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
	//cout<<"histogram normal"<<endl<<histogram_normal<<endl<<endl;
	//cout<<"histogram geo"<<endl<<histogram<<endl<<endl;
	histograms.mat_a=histogram;
	histograms.mat_b=histogram_normal;

	return histograms;
}
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
				// main plane
				if(num==0){
					hist_features[0].push_back(histograms_e2.mat_a(i,j));
					hist_features[1].push_back(histograms_e2.mat_b(i,j));
				}
				// second plane
				if(num==1){
					hist_features[0].push_back(histograms_e3.mat_a(i,j));
					hist_features[1].push_back(histograms_e3.mat_b(i,j));
				}
			}
		}
		ii=9;
		jj=5;
	}
	return hist_features;
}

bool previous_cluster=false;
vector<clusterInformation> all_cluster_info;
vector<clusterInformation> next_cluster_info;
clusterInformation previousResults(vector<clusterInformation> pre_info, clusterInformation now_info)
{
	clusterInformation output;
	float min_dist=1000.0;
	float dist=0;
	float x_dist=0;
	float y_dist=0;
	size_t id=0;
	for(size_t i=0;i<pre_info.size();i++){
		x_dist=pow(now_info.centroid[0]-pre_info[i].centroid[0],2);
		y_dist=pow(now_info.centroid[1]-pre_info[i].centroid[1],2);
		dist=sqrt(x_dist+y_dist);
		if(min_dist>dist){
			min_dist=dist;
			id=i;
		}
	}
	//if(0.26>min_dist)pre_info[id].pre_id=id;
	//else pre_info[id].pre_id=-1;
	output.likelihood=pre_info[id].likelihood;
	output.likelihood_flag=pre_info[id].likelihood_flag;
	// output.pre_id=pre_info[id].pre_id;
	output.pre_id=now_info.pre_id;
	output.rsl=now_info.rsl;
	float same_dist=0.26;//0.26
	if(MODE==64)same_dist=0.52;
	bool same_cluster=(same_dist>min_dist);
	// bool diff_cluster=(min_dist>0.5);
	bool diff_cluster=(!same_cluster);
	bool double_up=(now_info.rsl>0.0&&pre_info[id].rsl>0.0);
	bool double_down=(0.0>now_info.rsl&&0.0>pre_info[id].rsl);
	//bool absol=(output.likelihood_flag!=0);

	float gain=0;
	if(same_cluster&&pre_info[id].track_num!=0){
		output.sub_likelihood=pre_info[id].sub_likelihood+(float)pre_info[id].object_flag;
		output.track_num=pre_info[id].track_num+1;
		// #pragma omp critical
		// cout<<"sub_likelihood="<<output.sub_likelihood<<endl;
		// #pragma omp critical
		// cout<<"num="<<output.track_num<<endl;
		// if(output.track_num>10&&gain>0.9)gain=1;
		gain=output.sub_likelihood/(float)output.track_num;
		// #pragma omp critical
		// cout<<"           gain="<<gain<<endl;

	}else if(same_cluster&&pre_info[id].track_num==0){
		output.track_num=1;
		output.sub_likelihood=1;
	}
	output.track_vel=gain;
	// cout<<"min_dist="<<min_dist<<" double_flag= "<<double_up<<" now= "<<now_info.rsl<<" pre "<<pre_info[id].rsl<<" likelihood= "<<output.likelihood<<endl;
	// if(!absol){
	if(output.likelihood_flag==1)output.likelihood+=5;
	if(same_cluster&&double_up){
		output.likelihood+=3;
		output.rsl=pre_info[id].rsl;
		// cout<<"pre_info[id].pre_id"<<pre_info[id].pre_id<<endl;
		// output.pre_id=pre_info[id].pre_id;
	}
	if(same_cluster&&double_down){
		output.likelihood-=1;
		output.rsl=pre_info[id].rsl;
		// output.pre_id=pre_info[id].pre_id;
	}
	if(diff_cluster){
		output.likelihood=0;
		output.pre_id=now_info.pre_id;
		output.track_num=0;
		output.sub_likelihood=0;
		// output.pre_id=0;
	}
	if(same_cluster){
		output.pre_id=pre_info[id].pre_id;
		// output.track_num=pre_info[id].track_num+1;
		// output.sub_likelihood=(float)pre_info[id].object_flag+output.sub_likelihood;
	}
	if(output.likelihood>10)output.likelihood_flag=1;
	bool absol=(output.likelihood_flag==1&&same_cluster);
	// cout<<"absol="<<absol<<", same="<<same_cluster<<", likelihood="<<output.likelihood<<endl;
	if(absol){
		output.likelihood_flag=1;
		// output.pre_id=pre_info[id].pre_id;
	}
	else output.likelihood_flag=0;

	return output;
}
vector<dividedCluster> dividedProcess(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, Vector3f centroid) 
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
	for(size_t i=0;i<pcl_in.points.size();i++)cloud->points[i].z=0.0;

	clusterInformation cluster_info;

	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	feature_extractor.getOBB(cluster_info.min, cluster_info.max, cluster_info.position, cluster_info.rot_mat);
	cluster_info.min.z=min[2];
	cluster_info.max.z=max[2];
	cluster_info.width=cluster_info.max.x-cluster_info.min.x;
	cluster_info.length=cluster_info.max.y-cluster_info.min.y;
	cluster_info.height=cluster_info.max.z-cluster_info.min.z;
	cluster_info.rot_mat_inv=cluster_info.rot_mat;
	cluster_info.rot_mat=cluster_info.rot_mat.inverse();
/////////////////////////////calRot////////////////////////////////////	
		

	vector<dividedCluster> div_cluster;
	pcl::PointCloud<pcl::PointXYZINormal> cluster_pcl;
	cluster_pcl.points.resize(pcl_in.points.size());
	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZINormal p=pcl_in.points[i];
		Eigen::Vector3f n;
		n << p.x-centroid[0], p.y-centroid[1], p.z-centroid[2];
		n=cluster_info.rot_mat*n;
		cluster_pcl.points[i].x=n[0];
		cluster_pcl.points[i].y=n[1];
		cluster_pcl.points[i].z=n[2];
		cluster_pcl.points[i].intensity=p.intensity;
		cluster_pcl.points[i].normal_x=p.normal_x;
		cluster_pcl.points[i].normal_y=p.normal_y;
		cluster_pcl.points[i].normal_z=p.normal_z;
		cluster_pcl.points[i].curvature=p.curvature;
	}
	float div_width=0.7;
	int div_num=((cluster_info.width/div_width)+0.5);
	if(div_num==0)div_num=1;
	div_cluster.resize(div_num);

	int size_comp=0;
	if(cluster_info.height>0.6&&cluster_info.height<1.8)size_comp++;
	if(cluster_info.width>0.2&&cluster_info.width<2.4)size_comp++;
	if(cluster_info.length>0.1&&cluster_info.length<1.6)size_comp++;

	if(size_comp!=3){
		div_cluster.resize(1);
		div_cluster[0].clouds=pcl_in;
		div_cluster[0].cloudsPtr=ptrTransform(div_cluster[0].clouds);
		div_cluster[0].centroid=centroid;
		div_cluster[0].size_ok=0;
		return div_cluster;
	}else{
		float start_position=cluster_info.min.x;
		for(int i=0;i<div_num;i++){
			float end_position=start_position+div_width;
			for(size_t j=0;j<cluster_pcl.points.size();j++){
					pcl::PointXYZINormal p=cluster_pcl.points[j];
					if(p.x>=start_position&&p.x<=end_position){
						Vector3f p_rop;//=cluster_info.rot_mat_inv*p;
						p_rop << p.x, p.y, p.z;
						p_rop=cluster_info.rot_mat_inv*p_rop;
						p.x=p_rop[0]+centroid[0];
						p.y=p_rop[1]+centroid[1];
						p.z=p_rop[2]+centroid[2];
						div_cluster[i].clouds.points.push_back(p);
					}
			}
			start_position=end_position;
		}

		for(size_t i=0;i<div_cluster.size();i++){
			div_cluster[i].cloudsPtr=ptrTransform(div_cluster[i].clouds);
			Vector3f centroid_v=calculateCentroid(*div_cluster[i].cloudsPtr);
			div_cluster[i].centroid=centroid_v;
			div_cluster[i].size_ok=1;
		}
		return div_cluster;
	}
}
///////////////////////////////////////////////////////////////////////
vector<clusterTemporary> calFeatures(pcl::PointCloud<pcl::PointXYZINormal> &pcl_in)
{
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZINormal>);
	cloud_in=ptrTransform(pcl_in);
	pcl::VoxelGrid<pcl::PointXYZINormal> vg;  
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZINormal>);  

	// *cloud_filtered=*cloud_in;
	vg.setInputCloud (cloud_in);  
	if(MODE==32)vg.setLeafSize (0.07f, 0.07f, 0.07f);
	else if(MODE==64)vg.setLeafSize (0.1f, 0.1f, 0.1f);
	vg.filter (*cloud_filtered);

	//////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_tmp (new pcl::PointCloud<pcl::PointXYZINormal>);  
	cloud_tmp->points.resize(cloud_filtered->points.size());
	*cloud_tmp=*cloud_filtered;	

	for(int i=0;i<(int)cloud_filtered->points.size();i++){
		cloud_filtered->points[i].z  = 0.0;
	}
	std::vector<pcl::PointIndices> cluster_indices;
	double clustering_start=omp_get_wtime();
	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZINormal>);
	tree->setInputCloud (cloud_filtered);
	pcl::EuclideanClusterExtraction<pcl::PointXYZINormal> ec;
	ec.setClusterTolerance (0.15);//Points clearance for clustering
	if(MODE==32){
		ec.setMinClusterSize (20);//Minimum include points of the cluster
		ec.setMaxClusterSize (1600);//Maximum points of the cluster //32Eなら近くても500点いかない。400でも大丈夫かな。
	}else if(MODE==64){
		ec.setMinClusterSize (30);//Minimum include points of the cluster
		ec.setMaxClusterSize (2400);//Maximum points of the cluster //32Eなら近くても500点いかない。400でも大丈夫かな。
	}else if(MODE==64){
	}
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);
	*cloud_filtered=*cloud_tmp;
	vector<clusterTemporary> clusters;
	double clustering_end=omp_get_wtime();
	cout<<"clustering time="<<clustering_end-clustering_start<<endl;
	clusterBox allReturn;

	int c_count=0;
	double nomp_time=0.0;
	double start_time=0.0;
	double end_time=0.0;
	double real_start=0.0;
	double real_end=0.0;
	real_start=omp_get_wtime();
	// cout<<"cluster_indices.size()="<<cluster_indices.size()<<endl;
	// int indices_num=cluster_indices.size();
#pragma omp parallel num_threads((int)cluster_indices.size()) reduction(+:c_count,nomp_time) private(start_time,end_time)
// #pragma omp parallel num_threads(32) reduction(+:c_count,nomp_time) private(start_time,end_time)
{
	#pragma omp for nowait
	for(size_t iii=0;iii<cluster_indices.size();iii++){
		start_time=omp_get_wtime();
		pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZINormal>);
		cloud_cluster->points.resize(cluster_indices[iii].indices.size());
		for(size_t jjj=0;jjj<cluster_indices[iii].indices.size();jjj++){
			int p_num=cluster_indices[iii].indices[jjj];
			cloud_cluster->points[jjj]=cloud_filtered->points[p_num];
			// cloud_cluster->points.push_back(cloud_filtered->points[cluster_indices[iii].indices[jjj]]);
			cloud_cluster->width = p_num;
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = false;
		}
		Vector3f centroid_v=calculateCentroid(*cloud_cluster);
		float dist_to_centroid=sqrt(pow(centroid_v[0],2)+pow(centroid_v[1],2));
		//Detecting distance
		float distanceThresh;
		if(MODE==64)distanceThresh=40.0;
		else if(MODE==32)distanceThresh=16.0;
		if(dist_to_centroid>distanceThresh)continue;

		vector<dividedCluster> div_cluster;
		div_cluster=dividedProcess(*cloud_cluster, centroid_v);
		vector<clusterTemporary> private_clusters;
		for(size_t i=0;i<div_cluster.size();i++){
			clusterTemporary clusterT;
			clusterT.clouds=*div_cluster[i].cloudsPtr;
			clusterT.dist=dist_to_centroid;
			clusterT.centroid=div_cluster[i].centroid;
			if(clusterT.clouds.points.size()>20&&div_cluster[i].size_ok!=0){
				private_clusters.push_back(clusterT);
				c_count++;
			}
		}
		
		#pragma omp critical
		clusters.insert(clusters.end(), private_clusters.begin(), private_clusters.end());

		end_time=omp_get_wtime();
		// double time=end_time-start_time;
		// cout<<time<<endl;
		nomp_time+=end_time-start_time;
	}

}
	next_cluster_info.resize(clusters.size());

	real_end=omp_get_wtime();
	
	cout<<"===== end clustering ====="<<endl;
	cout<<"clustering time (omp: OFF) = "<<nomp_time<<endl;
	cout<<"clustering time (omp: ON ) = "<<real_end-real_start<<endl;

	double all_time=nomp_time;
	double all_time_omp=real_end-real_start;
	nomp_time=0.0;
	real_start=omp_get_wtime();
// #pragma omp parallel num_threads(c_count) reduction(+:nomp_time) private(start_time,end_time)
#pragma omp parallel num_threads(32) reduction(+:nomp_time) private(start_time,end_time)
{
	#pragma omp for nowait
	for(size_t it=0;it<clusters.size();it++){
		start_time=omp_get_wtime();
		shapeFeatures features;

		pcl::PointCloud<pcl::PointXYZINormal> cluster_pcl=clusters[it].clouds;
		clusterInformation cluster_info;
		cluster_info=transformMatrix(cluster_pcl);
		cluster_info.centroid=clusters[it].centroid;
		cluster_pcl.points.resize(clusters[it].clouds.points.size());
		for(size_t i=0;i<clusters[it].clouds.points.size();i++){
			pcl::PointXYZINormal p=clusters[it].clouds.points[i];
			Eigen::Vector3f n;
			n(0)=p.x-cluster_info.centroid(0);
			n(1)=p.y-cluster_info.centroid(1);
			n(2)=p.z-cluster_info.centroid(2);
			n=cluster_info.rot_mat*n;
			cluster_pcl.points[i].x=n(0);
			cluster_pcl.points[i].y=n(1);
			cluster_pcl.points[i].z=n(2);
		}
		float minmax[6]={cluster_info.min.x, cluster_info.min.y, cluster_info.min.z, 
								cluster_info.max.x, cluster_info.max.y, cluster_info.max.z};
		features.f1.resize(3);
		features.f1[0]=cluster_info.width;
		features.f1[1]=cluster_info.length;
		features.f1[2]=cluster_info.height;
		features.f8.resize(1);
		features.f8[0]=(float)cluster_pcl.points.size();
		features.f9.resize(1);
		features.f9[0]=clusters[it].dist;
		if(MODE==64){
			float normalized_dist=(features.f9[0]-4)/(50-4);
			if(normalized_dist<0.0)normalized_dist=0.0;
			else if(normalized_dist>1.0)normalized_dist=1.0;
			features.f9[0]=normalized_dist*15;
		}

		pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud (new pcl::PointCloud<pcl::PointXYZI>);
		pcl_cloud=convertXN2XI(cluster_pcl);
		pcl::PointCloud<pcl::PointXYZINormal> rotate_pcl;
		rotate_pcl=normalCheck2(pcl_cloud);

		features.f2=cal3DCovariance(rotate_pcl);
		features.f3=calMomentTensor(rotate_pcl);
		features.f4=calBlockFeature(rotate_pcl, minmax[2]-clusters[it].centroid[2], cluster_info.height);

		vector<vector<float> > f6_f7=histogramBrain(rotate_pcl);

		features.f6=f6_f7[0];
		features.f7=f6_f7[1];

		pcl::PointCloud<pcl::PointXYZ> upper_pcl;
		pcl::PointCloud<pcl::PointXYZ> bottom_pcl;
		int upper_num=0;
		int bottom_num=0;
		for(size_t i=0;i<rotate_pcl.points.size();i++){
			if(rotate_pcl.points[i].z>0.0)upper_num++;
			else bottom_num++;
		}
		upper_pcl.points.resize(upper_num);
		bottom_pcl.points.resize(bottom_num);
		upper_num=0;
		bottom_num=0;
		for(size_t i=0;i<rotate_pcl.points.size();i++){
			if(rotate_pcl.points[i].z>0.0){
				upper_pcl.points[upper_num].x=rotate_pcl.points[i].x;
				upper_pcl.points[upper_num].y=0.0;
				upper_pcl.points[upper_num].z=rotate_pcl.points[i].z;
				upper_num++;
			}else{
				bottom_pcl.points[bottom_num].x=rotate_pcl.points[i].x;
				bottom_pcl.points[bottom_num].y=0.0;
				bottom_pcl.points[bottom_num].z=rotate_pcl.points[i].z;
				bottom_num++;
			}
		}
		vector<float> f5_upper=cal2DCovariance(upper_pcl);
		vector<float> f5_bottom=cal2DCovariance(bottom_pcl);
		features.f5=mergeVector(f5_upper, f5_bottom);

		features=mergeAllFeatures(features);
		
		double rsl=0.0;
		double rsl_thresh=0.0;
		if(cluster_info.width<1.0)rsl=svmClassifier(features);
		cluster_info.rsl=rsl;
		cluster_info.pre_id=it+1;
		if(all_cluster_info.size()==0){
			cluster_info.likelihood=0;
			cluster_info.track_num=0;
			cluster_info.track_vel=0;
			cluster_info.sub_likelihood=0;
			next_cluster_info[it]=cluster_info;
		}else{
			clusterInformation tmp_info;
			tmp_info=previousResults(all_cluster_info, cluster_info);
			cluster_info.pre_id=tmp_info.pre_id;
			cluster_info.rsl_rev=(cluster_info.rsl+tmp_info.rsl)/2.0+(float)tmp_info.likelihood_flag;
			cluster_info.likelihood=tmp_info.likelihood;
			cluster_info.likelihood_flag=tmp_info.likelihood_flag;
			cluster_info.sub_likelihood=tmp_info.sub_likelihood;
			cluster_info.track_num=tmp_info.track_num;
			cluster_info.track_vel=tmp_info.track_vel;
		}
		if(cluster_info.rsl_rev>rsl_thresh)cluster_info.object_flag=1;
		else cluster_info.object_flag=0;
		if(cluster_info.likelihood_flag)cluster_info.object_flag=1;
		clusterInformation _info=cluster_info;
		_info.rsl=_info.rsl_rev;
		next_cluster_info[it]=_info;

		bool writing_delay=(!(int)loop_count%20);
		if(WRITEMODE&&cluster_info.object_flag){
			pcdWriter(pcd_pos,1,clusters[it].clouds);
			pcd_pos++;
		}else if(WRITEMODE&&!cluster_info.object_flag&&writing_delay){
			float distance=sqrt(pow(cluster_info.centroid[0],2)+pow(cluster_info.centroid[1],2));
			if(distance>2.1){
				pcdWriter(pcd_neg,-1,clusters[it].clouds);
				pcd_neg++;
			}
		}

		end_time=omp_get_wtime();
		nomp_time+=end_time-start_time;
	}
}//parallel end
	real_end=omp_get_wtime();
	all_time+=nomp_time;	

	cout<<endl<<"===== end calFeature ====="<<endl;
	cout<<"calFeature time (omp: OFF) = "<<nomp_time<<endl;
	cout<<"calFeature time (omp: ON ) = "<<real_end-real_start<<endl;

	cout<<endl<<"===== end keyProcess ====="<<endl;
	cout<<"keyProcess time (omp: OFF) = "<<all_time<<endl;
	cout<<"keyProcess time (omp: ON ) = "<<all_time_omp+real_end-real_start<<endl;
	avg_time+=(all_time_omp+real_end-real_start);
	double avg_time_=avg_time/loop_count;
	// avg_time/=loop_count;
	loop_count+=1.0;
	cout<<endl;
	cout<<"<Key Process avarage time> = "<<avg_time_<<endl;

	return clusters;
}
pcl::PointCloud<pcl::PointXYZINormal> rmCluster(pcl::PointCloud<pcl::PointXYZINormal> pcl_in, vector<Vector3f> centroid, vector<float> rm_size)
{
	pcl::PointCloud<pcl::PointXYZINormal> output;
	for(size_t i=0;i<pcl_in.points.size();i++){
		pcl::PointXYZINormal p=pcl_in.points[i];
		float min_dist=1000.0;
		int min_id=0;
		for(size_t j=0;j<centroid.size();j++){
			float dist=sqrt(pow(p.x-centroid[j](0),2)+pow(p.y-centroid[j](1),2));
			if(min_dist>dist){
				min_dist=dist;
				min_id=j;
			}
			if((rm_size[min_id]/1.2)>min_dist)break;
		}
		if((rm_size[min_id]/1.2)>min_dist){
			p.x=0;
			p.y=0;
			p.z=0;
		}
		output.points.push_back(p);
	}
	return output;
}
////////////////////////////////////main/////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	ros::init(argc, argv, "human_recognition");
	ros::NodeHandle nh,n;

	////subscriber////
	ros::Subscriber original_sub;
	// if(MODE==32)original_sub = n.subscribe("/velodyne_index/velodyne_points", 1, originalCallback);
	// if(MODE==32)original_sub = n.subscribe("/velodyne_index/v_points", 1, originalCallback);
	if(MODE==32)original_sub = n.subscribe("/velodyne_index/velodyne_points", 1, originalCallback);
	if(MODE==64)original_sub = n.subscribe("/velodyne_points", 1, originalCallback);
	// original_sub = n.subscribe("/velodyne_points", 1, originalCallback);
	//perfect_velodyne
	//ros::Subscriber original_sub = n.subscribe("/perfect_velodyne/normal", 10, originalCallback);
	ros::Subscriber object_sub = n.subscribe("/rm_ground", 1, objectCallback);
	// ros::Subscriber object_sub = n.subscribe("/velodyne_filtered/object_points", 1, objectCallback);
	//ros::Subscriber pcl_object_sub = n.subscribe("/velodyne_index/object_points", 10, pclCallback);
	////publisher////
	ros::Publisher original_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/human_recognition/velodyne_points",1);
	ros::Publisher removed_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/human_recognition/removed_points",1);
	ros::Publisher positive_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/human_recognition/positive_pt",1);
	ros::Publisher negative_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/human_recognition/negative_pt",1);
	ros::Publisher positive_position_pt_pub = n.advertise<sensor_msgs::PointCloud2>("/human_recognition/positive_position",1);
	ros::Publisher bb_pos_pub = n.advertise<visualization_msgs::Marker>("/human_recognition/bounding_box/positive",1);
	ros::Publisher bb_neg_pub = n.advertise<visualization_msgs::Marker>("/human_recognition/bounding_box/negative",1);
	ros::Publisher test_pub = n.advertise<visualization_msgs::MarkerArray>("/human_detection/likelihood",1);

	ros::Rate loop_rate(20);
	
	int svmRangeLoadCheck=0;
	while(!svmRangeLoadCheck){
		svmRangeLoadCheck=svmLoadRange();
	}
	int total_cluster=0;
	int total_positive=0;
	int total_negative=0;

	while (ros::ok()){
		if(callback_flag1&&callback_flag2)callback_flag=true;
		else if(func_check)callback_flag=true;
		if(callback_flag){
			callback_flag=false;
			callback_flag1=false;
			callback_flag2=false;
			callback_flag3=false;

			double start_t=omp_get_wtime();

			vector<clusterTemporary> clusters=calFeatures(pcl_scan);
			double start_t2=omp_get_wtime();
			all_cluster_info.resize(next_cluster_info.size());
			all_cluster_info=next_cluster_info;
			next_cluster_info.resize(0);
			vector<clusterInformation>().swap(next_cluster_info);

			int positive_number=0;
			int negative_number=0;
			pcl::PointCloud<pcl::PointXYZINormal> positive_pcl;
			pcl::PointCloud<pcl::PointXYZINormal> negative_pcl;
			for(size_t it=0;it<clusters.size();it++){
				int c_num=0;
				for(size_t i=0;i<clusters[it].clouds.points.size();i++){
					pcl::PointXYZINormal p;
					p.x=clusters[it].clouds.points[i].x;
					p.y=clusters[it].clouds.points[i].y;
					p.z=clusters[it].clouds.points[i].z;
					p.intensity=clusters[it].clouds.points[i].intensity;
					p.intensity=(float)positive_number;
					p.normal_x=all_cluster_info[it].rsl_rev;
					p.normal_y=all_cluster_info[it].likelihood;
					p.normal_z=all_cluster_info[it].likelihood_flag;
					p.curvature=all_cluster_info[it].track_vel;
					// p.curvature=all_cluster_info[it].pre_id;
					//p.normal_x=clusters[i].clouds.points[i].normal_x;
					//p.normal_y=clusters[i].clouds.points[i].normal_y;
					//p.normal_z=clusters[i].clouds.points[i].normal_z;
					//p.curvature=clusters[i].clouds.points[i].curvature;
					if(all_cluster_info[it].object_flag==1)positive_pcl.points.push_back(p);
					else if(all_cluster_info[it].object_flag==0)negative_pcl.points.push_back(p);
					c_num++;
					
				}
				if(all_cluster_info[it].object_flag)positive_number++;
				else if(all_cluster_info[it].object_flag==0)negative_number++;
				all_cluster_info[it].points_num=c_num;
			}

			pcl::PointXYZINormal delete_pcl;
			delete_pcl.x=10000.0;
			delete_pcl.y=10000.0;
			delete_pcl.z=10000.0;
			if(!positive_pcl.points.size())positive_pcl.points.push_back(delete_pcl);
			if(!negative_pcl.points.size())negative_pcl.points.push_back(delete_pcl);
			


			visualization_msgs::MarkerArray test_all;
			// test_all.clear();
			visualization_msgs::Marker test;
			test.header.frame_id="/velodyne";
			test.header.stamp=ros::Time::now();
			test.ns="test";
			test.action=visualization_msgs::Marker::ADD;
			test.pose.orientation.w=1.0;
			test.type=visualization_msgs::Marker::TEXT_VIEW_FACING;
			test.scale.x=0.5;
			test.scale.y=0.5;
			test.scale.z=0.5;
			test.color.r=0.0;
			test.color.g=1.0;
			test.color.b=1.0;
			test.color.a=1.0;
	
			for(size_t i=0;i<all_cluster_info.size();i++){
				test.id=i;
				test.pose.position.x=all_cluster_info[i].centroid[0];
				test.pose.position.y=all_cluster_info[i].centroid[1];
				test.pose.position.z=all_cluster_info[i].centroid[2]+1.0;


				// float val=all_cluster_info[i].rsl;
				float val=all_cluster_info[i].track_vel;
				// float val=all_cluster_info[i].width;
				// float val=all_cluster_info[i].points_num;
				if(0.001>val)val=0;
				stringstream ss;
				ss<<setprecision(3)<<val;
				string ts=ss.str(); 
				test.text= ts;
				test.lifetime=ros::Duration(0.07);
				test_all.markers.push_back(test);
				
			}
			test_pub.publish(test_all);


			vector<Vector3f> remove_centroid;
			vector<float> remove_size;
			pcl::PointCloud<pcl::PointNormal> position_pcl;
			pcl::PointCloud<pcl::PointXYZI> bb_pos_pcl;
			pcl::PointCloud<pcl::PointXYZI> bb_neg_pcl;
			pcl::PointCloud<pcl::PointXYZI> bb_div_pcl;

			for(size_t it=0;it<all_cluster_info.size();it++){
				pcl::PointCloud<pcl::PointXYZ> viz_tmp;
				viz_tmp=visualizer(all_cluster_info[it].rot_mat_inv, all_cluster_info[it].min, all_cluster_info[it].max, all_cluster_info[it].position);
				if(all_cluster_info[it].object_flag==0){
					for(size_t i=0;i<viz_tmp.points.size();i++){
						pcl::PointXYZI p;
						p.x=viz_tmp.points[i].x;
						p.y=viz_tmp.points[i].y;
						p.z=viz_tmp.points[i].z;
						p.intensity=all_cluster_info[it].object_flag;
						bb_neg_pcl.points.push_back(p);
					}
				}else if(all_cluster_info[it].object_flag==1){
					for(size_t i=0;i<viz_tmp.points.size();i++){
						pcl::PointXYZI p;
						p.x=viz_tmp.points[i].x;
						p.y=viz_tmp.points[i].y;
						p.z=viz_tmp.points[i].z;
						p.intensity=all_cluster_info[it].object_flag;
						bb_pos_pcl.points.push_back(p);
					}
					remove_centroid.push_back(all_cluster_info[it].centroid);
					remove_size.push_back(all_cluster_info[it].width);
					pcl::PointNormal p;
					p.x=all_cluster_info[it].centroid[0];
					p.y=all_cluster_info[it].centroid[1];
					p.z=all_cluster_info[it].centroid[2];
					p.curvature=all_cluster_info[it].width;
					position_pcl.points.push_back(p);
				}else if(all_cluster_info[it].object_flag==2){
					for(size_t i=0;i<viz_tmp.points.size();i++){
						pcl::PointXYZI p;
						p.x=viz_tmp.points[i].x;
						p.y=viz_tmp.points[i].y;
						p.z=viz_tmp.points[i].z;
						p.intensity=all_cluster_info[it].object_flag;
						bb_div_pcl.points.push_back(p);
					}
				}
			}



			pcl::PointCloud<pcl::PointXYZINormal> removed_pcl;
			if(pcl_org.points.size()!=0&&remove_centroid.size()!=0)
				removed_pcl=rmCluster(pcl_org, remove_centroid, remove_size);
			else removed_pcl=pcl_org;
			remove_centroid.clear();
			remove_size.clear();

			visualization_msgs::Marker bb_pos;
			bb_pos.header.frame_id="/velodyne";
			bb_pos.header.stamp=ros::Time::now();
			bb_pos.ns="bounding_box";
			bb_pos.action=visualization_msgs::Marker::ADD;
			bb_pos.pose.orientation.w=1.0;
			bb_pos.id=0;
			bb_pos.type=visualization_msgs::Marker::LINE_LIST;
			bb_pos.scale.x=0.05;
			bb_pos.color.r=0.0;
			bb_pos.color.g=1.0;
			bb_pos.color.b=1.0;
			bb_pos.color.a=1.0;
			for(size_t i=0;i<bb_pos_pcl.points.size();i++){
				geometry_msgs::Point p;
				p.x=bb_pos_pcl.points[i].x;
				p.y=bb_pos_pcl.points[i].y;
				p.z=bb_pos_pcl.points[i].z;
				bb_pos.points.push_back(p);
			}
			bb_pos_pub.publish(bb_pos);
			bb_pos_pcl.points.clear();

			visualization_msgs::Marker bb_neg;
			bb_neg.header.frame_id="/velodyne";
			bb_neg.header.stamp=ros::Time::now();
			bb_neg.ns="bounding_box";
			bb_neg.action=visualization_msgs::Marker::ADD;
			bb_neg.pose.orientation.w=1.0;
			bb_neg.id=1;
			bb_neg.type=visualization_msgs::Marker::LINE_LIST;
			bb_neg.scale.x=0.04;
			bb_neg.color.r=0.4;
			bb_neg.color.g=0.4;
			bb_neg.color.b=0.4;
			bb_neg.color.a=1.0;
			for(size_t i=0;i<bb_neg_pcl.points.size();i++){
				geometry_msgs::Point p;
				p.x=bb_neg_pcl.points[i].x;
				p.y=bb_neg_pcl.points[i].y;
				p.z=bb_neg_pcl.points[i].z;
				bb_neg.points.push_back(p);
			}
			bb_neg_pub.publish(bb_neg);
			bb_neg_pcl.points.clear();



			original_pt.header.frame_id="/velodyne";
			original_pt.header.stamp=ros::Time::now();
			original_pt_pub.publish(original_pt);

			pcl::toROSMsg(removed_pcl,removed_pt);
			removed_pcl.points.clear();
			removed_pt.header.frame_id="/velodyne";
			removed_pt.header.stamp=ros::Time::now();
			removed_pt_pub.publish(removed_pt);

			pcl::toROSMsg(positive_pcl, positive_pt);
			positive_pcl.points.clear();
			positive_pt.header.frame_id="/velodyne";
			positive_pt.header.stamp=ros::Time::now();
			positive_pt_pub.publish(positive_pt);

			pcl::toROSMsg(negative_pcl, negative_pt);
			negative_pcl.points.clear();
			negative_pt.header.frame_id="/velodyne";
			negative_pt.header.stamp=ros::Time::now();
			negative_pt_pub.publish(negative_pt);

			pcl::toROSMsg(position_pcl,position_pt);
			position_pcl.points.clear();
			position_pt.header.frame_id="/velodyne";
			position_pt.header.stamp=ros::Time::now();
			positive_position_pt_pub.publish(position_pt);


			double end_t=omp_get_wtime();
			cout<<"All cluster = "<<positive_number+negative_number<<", positive cluster = "<<positive_number<<", negative cluster = "<<negative_number<<endl;
			total_cluster+=(positive_number+negative_number);
			total_positive+=positive_number;
			total_negative+=negative_number;
			cout<<"Total cluster = "<<total_cluster<<", Total positive cluster = "<<total_positive<<", Total negative cluster = "<<total_negative<<endl;
			cout<<"===============withoutF taken:"<<setw(10)<<end_t-start_t2<<"[sec]"<<"==============="<<endl<<endl;
			cout<<"===============All time taken:"<<setw(10)<<end_t-start_t<<"[sec]"<<"==============="<<endl<<endl;
		}
		ros::spinOnce();
		// loop_rate.sleep();	
	}
	cout<<"END program........"<<endl;
	return 0;
}
