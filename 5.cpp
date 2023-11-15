// 5.cpp : Defines the entry point for the console application.
//


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>


float alpha=0.01;
int N=10000;

int n_layer=4;
int nunits[]={4,6,5,3};


class layer{
public:
	int n_out;
	float*output;
	float*delta_output;

	float**w,*b;
	float**delta_w,*delta_b;

	void creat(int nin,int nout);
	void compute_output(float*input,int n_in);
	void compute_delta_input(float*input,float*delta_input,int n_in);
	void compute_delta_wb(float*input,int n_in);
	void update_wb(int n_in);
};


void layer::creat(int nin,int nout)
{
	n_out=nout;
	output=new float[n_out];
	delta_output=new float[n_out];

	w=new float*[n_out];
	delta_w=new float*[n_out];
	for(int i=0;i<n_out;i++)
	{
		w[i]=new float[nin];
		delta_w[i]=new float[nin];
	}
	b=new float[n_out];
	delta_b=new float[n_out];


	srand(unsigned (time(NULL)));
	for(int i=0;i<n_out;i++)
	{
		for(int j=0;j<nin;j++)
		{
			w[i][j]=0.1*(2*(float)rand()/65535.0-1);
		}
		b[i]=0;
	}
}
float sigmoid(float x)
{

	return 1/(1+exp(-x));

}
void layer::compute_output(float*input,int n_in)
{
	float s;
	for(int i=0;i<n_out;i++)
	{
		s=0;
		for(int j=0;j<n_in;j++)
		{
			s+=input[j]*w[i][j];
		}
		output[i]=sigmoid(s+b[i]);
	}
}

void layer::compute_delta_input(float*input,float*delta_input,int n_in)
{
	float s;
	int i;
	for(int j=0;j<n_in;j++)
	{
		s=0;
		for( i=0;i<n_out;i++)
		{
			s+=delta_output[i]*w[i][j];
		}
		delta_input[j]=s*input[i]*(1-input[i]);
	}
}

void layer::compute_delta_wb(float*input,int n_in)
{
	for(int i=0;i<n_out;i++)
	{
		for(int j=0;j<n_in;j++)
		{
			delta_w[i][j]=delta_output[i]*input[j];
		}
		delta_b[i]=delta_output[i];
	}
}

void layer::update_wb(int n_in)
{
	for(int i=0;i<n_out;i++)
	{
		for(int j=0;j<n_in;j++)
		{
			w[i][j]-=alpha*delta_w[i][j];
		}
		b[i]-=alpha*delta_b[i];
	}
}



void forward(float*x,int nx,layer*Net)
{
	Net[0].compute_output(x,nx);

	for(int i=1;i<n_layer;i++)
	{
		Net[i].compute_output(Net[i-1].output,Net[i-1].n_out);
	}
}

void compute_delta_O(layer output_layer,float*label)
{
	for(int i=0;i<output_layer.n_out;i++)
	{
		output_layer.delta_output[i]=(output_layer.output[i]-label[i])*output_layer.output[i]*(1-output_layer.output[i]);
	}
}

void backward(layer *Net)
{
	for(int i=n_layer-1;i>0;i--)
	{
		Net[i].compute_delta_input(Net[i-1].output,Net[i-1].delta_output,Net[i-1].n_out);
	}
}

void compute_gradient(float*x,int nx,layer*Net)
{

	Net[0].compute_delta_wb(x,nx);
	for(int i=1;i<n_layer;i++)
	{
		Net[i].compute_delta_wb(Net[i-1].output,Net[i-1].n_out);
	}
}
void update_parameter(int nx,layer*Net)
{
	Net[0].update_wb(nx);
	for(int i=1;i<n_layer;i++)
	{
		Net[i].update_wb(Net[i-1].n_out);
	}
}



float compute_error(float*x,int nx,layer*Net,float*label)
{
	forward(x,nx,Net);
	float s=0;
	for(int i=0;i<Net[n_layer-1].n_out;i++)
	{
		s+=(Net[n_layer-1].output[i]-label[i])*(Net[n_layer-1].output[i]-label[i]);
	}

	return s;
}




int classification(float*x,int nx,layer*Net)
{
	forward(x,nx,Net);

	int index=0;
	float max=Net[n_layer-1].output[0];
	for(int i=1;i<Net[n_layer-1].n_out;i++)
	{
		if(Net[n_layer-1].output[i]>max)
		{
			max=Net[n_layer-1].output[i];
			index=i;
			
		}
	}


	return index;
}

void test(float**test_x,float**test_label,int n_test,int nx,layer*Net)
{
	float max;
	int lab,pre;
	float accuracy=0;
	for(int i=0;i<n_test;i++)
	{
		max=test_label[i][0];
		lab=0;
		for(int j=1;j<Net[n_layer-1].n_out;j++)
		{
			if(test_label[i][j]>max)
			{
				max=test_label[i][j];
				lab=j;
			}
		}
		

		pre=classification(test_x[i],nx,Net);

		if(pre==lab)
		{
			accuracy++;
		}

		printf("predict=%d   label=%d\n",pre,lab);
	}

	printf("accuracy=%f\n",accuracy/(float)n_test);
}


void read_data(int &n_train,int &n_test,float**&train_x,float**&train_label,float**&test_x,float**&test_label,int &nx,int&ny)
{
	nx=4;
	ny=3;

	n_train=90;
	n_test=60;

	int label,i,j;

	train_x=new float*[n_train];
	train_label=new float*[n_train];
	for(i=0;i<n_train;i++)
	{
		train_x[i]=new float[nx];
		train_label[i]=new float[ny];
	}

	test_x=new float*[n_test];
	test_label=new float*[n_test];
	for(i=0;i<n_test;i++)
	{
		test_x[i]=new float[nx];
		test_label[i]=new float[ny];
	}
	
	FILE*data;
	data=fopen("iris.data","r+");


	int train_index,test_index;
	train_index=0;
	test_index=0;
	for(i=0;i<30;i++)
	{
		for(j=0;j<nx;j++)
		{
			fscanf(data,"%f ",&train_x[train_index][j]);
		}
		fscanf(data,"%d ",&label);
		if(label==0)
		{
			train_label[train_index][0]=1;
			train_label[train_index][1]=0;
			train_label[train_index][2]=0;
		}
		if(label==1)
		{
			train_label[train_index][0]=0;
			train_label[train_index][1]=1;
			train_label[train_index][2]=0;
		}
		if(label==2)
		{
			train_label[train_index][0]=0;
			train_label[train_index][1]=0;
			train_label[train_index][2]=1;
		}
		
		train_index++;
	}

	for(i=0;i<20;i++)
	{
		for(int j=0;j<nx;j++)
		{
			fscanf(data,"%f ",&test_x[test_index][j]);
		}
		fscanf(data,"%d ",&label);
		if(label==0)
		{
			test_label[test_index][0]=1;
			test_label[test_index][1]=0;
			test_label[test_index][2]=0;
		}
		if(label==1)
		{
			test_label[test_index][0]=0;
			test_label[test_index][1]=1;
			test_label[test_index][2]=0;
		}
		if(label==2)
		{
			test_label[test_index][0]=0;
			test_label[test_index][1]=0;
			test_label[test_index][2]=1;
		}
		test_index++;
	}

	for(i=0;i<30;i++)
	{
		for(int j=0;j<nx;j++)
		{
			fscanf(data,"%f ",&train_x[train_index][j]);
		}
		fscanf(data,"%d ",&label);
		if(label==0)
		{
			train_label[train_index][0]=1;
			train_label[train_index][1]=0;
			train_label[train_index][2]=0;
		}
		if(label==1)
		{
			train_label[train_index][0]=0;
			train_label[train_index][1]=1;
			train_label[train_index][2]=0;
		}
		if(label==2)
		{
			train_label[train_index][0]=0;
			train_label[train_index][1]=0;
			train_label[train_index][2]=1;
		}
		
		train_index++;
	}

	for(i=0;i<20;i++)
	{
		for(int j=0;j<nx;j++)
		{
			fscanf(data,"%f ",&test_x[test_index][j]);
		}
		fscanf(data,"%d ",&label);
		if(label==0)
		{
			test_label[test_index][0]=1;
			test_label[test_index][1]=0;
			test_label[test_index][2]=0;
		}
		if(label==1)
		{
			test_label[test_index][0]=0;
			test_label[test_index][1]=1;
			test_label[test_index][2]=0;
		}
		if(label==2)
		{
			test_label[test_index][0]=0;
			test_label[test_index][1]=0;
			test_label[test_index][2]=1;
		}
		test_index++;
		
	}

	for(i=0;i<30;i++)
	{
		for(int j=0;j<nx;j++)
		{
			fscanf(data,"%f ",&train_x[train_index][j]);
		}
		fscanf(data,"%d ",&label);
		if(label==0)
		{
			train_label[train_index][0]=1;
			train_label[train_index][1]=0;
			train_label[train_index][2]=0;
		}
		if(label==1)
		{
			train_label[train_index][0]=0;
			train_label[train_index][1]=1;
			train_label[train_index][2]=0;
		}
		if(label==2)
		{
			train_label[train_index][0]=0;
			train_label[train_index][1]=0;
			train_label[train_index][2]=1;
		}
		train_index++;
		
	}

	for(i=0;i<20;i++)
	{
		for(int j=0;j<nx;j++)
		{
			fscanf(data,"%f ",&test_x[test_index][j]);
		}
		fscanf(data,"%d ",&label);
		if(label==0)
		{
			test_label[test_index][0]=1;
			test_label[test_index][1]=0;
			test_label[test_index][2]=0;
		}
		if(label==1)
		{
			test_label[test_index][0]=0;
			test_label[test_index][1]=1;
			test_label[test_index][2]=0;
		}
		if(label==2)
		{
			test_label[test_index][0]=0;
			test_label[test_index][1]=0;
			test_label[test_index][2]=1;
		}
		test_index++;
	}
	fclose(data);

}

int main()
{
	float**train_x,**train_label;
	float**test_x,**test_label;
	layer*Net;


	int nx,ny,n_train,n_test,i,j;

	read_data(n_train,n_test,train_x,train_label,test_x,test_label,nx,ny);
	nunits[n_layer-1]=ny;

	Net=new layer[n_layer];
	Net[0].creat(nx,nunits[0]);
	for(i=1;i<n_layer;i++)
	{
		Net[i].creat(nunits[i-1],nunits[i]);
	}

	float s;

	for(i=0;i<N;i++)
	{
		s=0;
		for(int j=0;j<n_train;j++)
		{
			forward(train_x[j],nx,Net);
			compute_delta_O(Net[n_layer-1],train_label[j]);
			backward(Net);
			compute_gradient(train_x[j],nx,Net);
			update_parameter(nx,Net);
			s=compute_error(train_x[j],nx,Net,train_label[j]);
		}
		printf("error=%f\n",s/(float)n_train);
	}

	test(test_x,test_label,n_test,nx,Net);
	return 0;

}
