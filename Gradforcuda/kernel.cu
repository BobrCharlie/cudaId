#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
#define pic 3.14159265359
#include<ctime>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <windows.h>

using namespace cv;


void setDev()
{
	cudaSetDevice(0);
}

void setDev(int i)
{
	cudaSetDevice(i);
}

void delDev()
{
	cudaDeviceReset();
}

__global__ void GradKernel(char *image, char *cont, unsigned int sizex, int sizey, int rg) //360/8 x2+y2=r2 r and x
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int maxim = 0; int maxint;
	/*if (y != 0 && y != sizey && x != 0 && x != sizex)
	{

		if (image[(y - 1)*sizex + x - 1] - image[y*sizex + x] > maxim)
		{
			maxim = image[(y - 1)*sizex + x - 1] - image[y*sizex + x];
			maxint = pic+pic/2+pic/4;
		}

		if (image[(y - 1)*sizex + x] - image[y*sizex + x] > maxim)
		{
			maxim = image[(y - 1)*sizex + x] - image[y*sizex + x];
			maxint = 0;
		}

		if (image[(y - 1)*sizex + x + 1] - image[y*sizex + x] > maxim)
		{
			maxim = image[(y - 1)*sizex + x + 1] - image[y*sizex + x];
			maxint = pic/4;
		}

		if (image[y*sizex + x - 1] - image[y*sizex + x] > maxim)
		{
			maxim = image[y*sizex + x - 1] - image[y*sizex + x];
			maxint = pic/2;
		}

		if (image[y*sizex + x + 1] - image[y*sizex + x] > maxim)
		{
			maxim = image[y*sizex + x + 1] - image[y*sizex + x];
			maxint = pic/2+pic/4;
		}

		if (image[(y + 1)*sizex + x - 1] - image[y*sizex + x] > maxim)
		{
			maxim = image[(y + 1)*sizex + x - 1] - image[y*sizex + x];
			maxint = pic;
		}
		if (image[(y + 1)*sizex + x] - image[y*sizex + x] > maxim)
		{
			maxim = image[(y + 1)*sizex + x] - image[y*sizex + x];
			maxint = pic+pic/4;
		}

		if (image[(y + 1)*sizex + x + 1] - image[y*sizex + x] > maxim)
		{
			maxim = image[(y + 1)*sizex + x + 1] - image[y*sizex + x];
			maxint = pic+pic/2;
		}
	}*/
	//cont[3 * (y*sizex + x)] = maxim;
	//cont[3 * (y*sizex + x) + 1] = sinf(maxint) * 255;
	//cont[3 * (y*sizex + x) + 2] = image[y*sizex + x];
	if (y > rg && y < sizey-rg && x > rg && x < sizex-rg)
	{
		float xk = x - rg;
		float y1 = sqrt((float)(xk*xk + rg*rg));
		float y2 = -y1;
		while (xk < x)
		{
			float xf = x - 0.5;
			int xi = int(xf);
			int yfromx;
			yfromx = int((xf - xk) * (y1 - y) / (xk - x) + y);
			while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xf > (x - rg)))
			{
				xf = xf - 0.5;
				xi = int(xf);
				yfromx = int((xf - xk)*(y1 - y) / (xk - x) + y);
			}
			if (xf == x - rg)
				if (image[yfromx*sizex + xi] > cont[3 * y*sizex + x])
				{
					cont[3 * y*sizex + x] = image[yfromx*sizex + xi] - image[y*sizex + x];
				}

			xk += 0.5;
			float y1 = sqrtf((float)(xk*xk + rg*rg));
			float y2 = -y1;
		}
	}
	/*for (int t = y - rg; t < y + rg + 1; t++)
	{

	float xf = x - 0.5;
	int xi = int(xf)-1;
	int yfromx;
	yfromx = int((xf - x + rg)*(y - t) / rg + t);
	while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xf > (x - rg))) //
	{
	xf = xf - 0.5;
	xi = int(xf)-1;
	if (t>y)
	yfromx = int((xf - x + rg)*(y - t) / rg + t);
	else if (t<y)
	yfromx = int((xf - x + rg)*(y - t) / rg + t) + 1;
	else yfromx = y;
	}
	if (xf == x - rg)
	if (image[yfromx*sizex + xi] > cont[y*sizex + x])
	{
	cont[y*sizex + x] = image[yfromx*sizex + xi] - image[y*sizex + x];
	}

	xf = x + 0.5;
	xi = int(xf) + 1;
	yfromx = int((xf - x - rg)*(y - t) / (-rg) + t) + 1;
	while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xf < x + rg))
	{
	xf = xf + 0.5;
	xi = int(xf) + 1;
	if (t>y)
	yfromx = int((xf - x + rg)*(y - t) / rg + t);
	else if (t<y)
	yfromx = int((xf - x + rg)*(y - t) / rg + t) + 1;
	else yfromx = y;
	}
	if (xf == x + rg)
	if (image[yfromx*sizex + xi] > cont[y*sizex + x])
	{
	cont[y*sizex + x] = image[yfromx*sizex + xi] - image[y*sizex + x];
	}

	/*xf = x - 0.5;
	xi = int(xf);
	yfromx = (xf - t)*(-rg) / (x - t) + y - rg;
	while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xf > x - rg))
	{
	xf = xf - 0.5;
	xi = int(xf);
	}
	if (xf == x - rg)
	if ((image[yfromx*sizex + xi] > cont[y*sizex + x]))
	{
	cont[y*sizex + x] = image[yfromx*sizex + xi] - image[y*sizex + x];
	}

	xf = x + 0.5;
	xi = int(xf) + 1;
	yfromx = (xf - t)*rg / (x - t) + y + rg;
	while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xf < x + rg))
	{
	xf = xf + 0.5;
	xi = int(xf) + 1;
	}
	if (xf == x + rg)
	if (image[yfromx*sizex + xi] > cont[y*sizex + x])
	{
	cont[y*sizex + x] = image[yfromx*sizex + xi] - image[y*sizex + x];
	}
	}*/
}

void findGrad(char *image, char *cont, unsigned int sizex, unsigned int sizey, int rg)
{
	char *gray_d;
	char *cont_d;
	cudaMalloc((void**)&gray_d, sizex * sizey * sizeof(char));
	cudaMalloc((void**)&cont_d, sizex * sizey * 3 * sizeof(char));

	cudaMemcpy(gray_d, image, sizex * sizey * sizeof(char), cudaMemcpyHostToDevice);

	GradKernel KERNEL_ARGS2(dim3(sizey), dim3(sizex)) (gray_d, cont_d, sizex, sizey, rg);
	cudaDeviceSynchronize();
	cudaMemcpy(cont, cont_d, sizex * sizey * 3 * sizeof(char), cudaMemcpyDeviceToHost);

	cudaFree(cont_d);
	cudaFree(gray_d);
}
int main()
{
	VideoCapture capture(0);
	Mat gray, image, cont;
	int width = 640;
	int height = 480;
	int rg;
	std::cout << "gradient radius: "; std::cin >> rg;
	setDev(0);
	cont = Mat::Mat(Size(width, height), CV_8UC3);
	time_t t1;
	while (1)
	{
		capture >> image;
		//imshow("image", image);		//CV_8UC3; Vec3b;
		cvtColor(image, gray, CV_BGR2GRAY);
		//imshow("gray", gray); 		//Canny(gray, cont, 50, 100);
		t1 = clock();
		findGrad((char*)gray.data, (char*)cont.data, width, height, rg);
		std::cout << clock() - t1 << std::endl;
		imshow("cont1", cont);
		if (waitKey(33) == 27)
		{
			delDev();
			imwrite("constMy.png", cont);
			return 1;
		}

		//	imshow("cont2", cont);
	}
	delDev();
	return 1;
}

/*
ParallelDevice::ParallelDevice()
{
cudaSetDevice(0);
}

ParallelDevice::ParallelDevice(int i)
{
cudaSetDevice(i);
}

ParallelDevice::~ParallelDevice()
{
cudaDeviceReset();
}*/

/*	cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
goto Error;
}*/

/*	cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}*/

/*
Error:
cudaFree(dev_c);
cudaFree(dev_a);
cudaFree(dev_b);

return cudaStatus; */

/*int main()
{
const int size = 100; int i, j;
int a[size][size];
int b[size][size];
int c[size][size];

for (i = 0; i < size; i++)
for (j = 0; j < size; j++)
{
a[i][j] = rand()%1000;
b[i][j] = rand()%1000;
}


for (i = 0; i < size; i++)
for (j = 0; j < size; j++)
{
c[i][j] = 1;
}

// Add vectors in parallel.
cudaError_t cudaStatus = addWithCuda(*c, *a, *b, size);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "addWithCuda failed!");
return 1;
}

// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaStatus = cudaDeviceReset();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceReset failed!");
return 1;
}

return 0;
}*/