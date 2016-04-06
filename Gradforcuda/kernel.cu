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
	//cont[3 * (y*sizex + x)] = 0;
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
	//cont[3 * (y*sizex + x) + 1] = abs(sin(float(maxint))) * 255;
	//cont[3 * (y*sizex + x) + 2] = image[y*sizex + x];
	if (y > rg && y < sizey-rg && x > rg && x < sizex-rg)
	{








		float xk = -rg;
		int y1 = int(sqrt(float((rg*rg - xk*xk))));
		int y2 = -y1;
		while (xk < rg)
		{
			float xf = x - 0.5; //II
			int xi = int(xf);
			int yfromx;
			yfromx = int(y + y1*(1-1/xf));
			while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xi > (x - rg)))
			{
				xf -= 0.5;
				xi = int(xf);
				yfromx = int(y + y1*(1-1/xf)); //yfromx = int((xf - x2)*(y2 - y1) / (x2 - x1) + y2);
			}
			if (xf == x - rg)
				if (image[yfromx*sizex + xi] > cont[3 * (y*sizex + x)])
				{
					//cont[3 * (y*sizex + x)] = image[yfromx*sizex + xi] - image[y*sizex + x];
					cont[3 * (y*sizex + x) + 1] = sin(float(pic/2 + pic / 6 * (1 - y1))) * 255;
				}
			
			xf = x - 0.5; // III
			xi = int(xf);
			yfromx = int(y + y2*(1-1/xf));
			while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xi > (x - rg)))
			{
				xf -= 0.5;
				xi = int(xf);
				yfromx = int(y + y2*(1-1/xf)); //yfromx = int((xf - xk)*(y1 - y) / (xk - x) + y);
			}
			if (xf == x - rg)
				if (image[yfromx*sizex + xi] > cont[3 * (y*sizex + x)])
				{
					//cont[3 * (y*sizex + x)] = image[yfromx*sizex + xi] - image[y*sizex + x];
					cont[3 * (y*sizex + x) + 1] = sin(float(pic + pic / 6 * (1 - y1))) * 255;
				}

			xk += 0.5;
			y1 = int(sqrt(float((rg*rg - xk*xk))));
			y2 = -y1;
		}

		xk = 0;
		y1 = int(sqrt(float((rg*rg - xk*xk))));
		y2 = -y1;
		while (xk < rg)
		{
			float xf = x + 0.5; //I
			int xi = int(xf);
			int yfromx = int(y + y1*(1-1/xf));
			while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xi < (x + rg)))
			{
				xf += 0.5;
				xi = int(xf);
				yfromx = int(y + y1*(1-1/xf)); //yfromx = int((xf - xk)*(y1 - y) / (xk - x) + y);
			}
			if (xf == x - rg)
				if (image[yfromx*sizex + xi] > cont[3 * (y*sizex + x)])
				{
					//cont[3 * (y*sizex + x)] = image[yfromx*sizex + xi] - image[y*sizex + x];
					cont[3 * (y*sizex + x) + 1] = sin(float(pic / 6 * (1 - y1))) * 255;
				}

			xf = x + 0.5; // IV
			xi = int(xf);
			yfromx = int(y + y2*(1-1/xf));
			while ((image[yfromx*sizex + xi] > image[y*sizex + x]) && (xi < (x + rg)))
			{
				xf += 0.5;
				xi = int(xf);
				yfromx = int(y + y2*(1-1/xf)); //yfromx = int((xf - xk)*(y1 - y) / (xk - x) + y);
			}
			if (xf == x + rg)
				if (image[yfromx*sizex + xi] > cont[3 * (y*sizex + x)])
				{
					//cont[3 * (y*sizex + x)] = image[yfromx*sizex + xi] - image[y*sizex + x];
					cont[3 * (y*sizex + x) + 1] = sin(float(3* pic / 2 + pic / 6 * (1 - y1))) * 255;
				}

			xk += 0.5;
			y1 = int(sqrt(float((rg*rg - xk*xk))));
			y2 = -y1;
		}
	}

	//cont[3 * (y*sizex + x) + 2] = image[y*sizex + x];





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
	
	int **sobelx = new int*[rg];
	for (int i=0; i < rg; i++) sobelx[i] = new int[rg];

	for (int i=0; i < rg; i++)
		for (int j; j < rg; j++)
			;

	int **sobely = new int*[rg];
	for (int i; i < rg; i++) sobely[i] = new int[rg];
	
	for (int i = 0; i < floor(rg/2); i++)
		for (int j=0; j < floor(rg/2); j++)
			;

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
