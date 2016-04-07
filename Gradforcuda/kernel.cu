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

int sobelx[5][5];
int sobely[5][5];

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

__global__ void GradKernel(char *image, char *cont, unsigned int sizex, int sizey, int rg, int *sobelx, int *sobely) //360/8 x2+y2=r2 r and x
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int maxim = 0; int maxint;
	cont[3 * (y*sizex + x)] = 0;
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
	
	

	if (y > 5 && y < sizey - 5 && x > 5 && x < sizex - 5)
	{
		int sumx1[5], sumy1[5], sumx, sumy;

		sumx = 0;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
				sumx += image[(y - rg + i)*sizex + (x - rg + j)] * sobelx[i * 5 + j];
		}

		sumy = 0;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
				sumy += image[(y - rg + i)*sizex + (x - rg + j)] * sobely[i * 5 + j];
		}
		cont[3 * (y*sizex + x)] = sumx;
		cont[3 * (y*sizex + x) + 1] = sumy;
		if (sumx>0 && sumy>0)
			cont[3 * (y*sizex + x) + 2] = abs(sin(float(sumy/sqrt(float(sumy*sumy+sumx*sumx))))) * 255;
		else if (sumx>0 && sumy<0)
			cont[3 * (y*sizex + x) + 2] = abs(sin(float(pic / 2 + sumy / sqrt(float(sumy*sumy + sumx*sumx))))) * 255;
		else if (sumx<0 && sumy<0)
			cont[3 * (y*sizex + x) + 2] = abs(sin(float(pic + sumy / sqrt(float(sumy*sumy + sumx*sumx))))) * 255;
		else
			cont[3 * (y*sizex + x) + 2] = abs(sin(float(pic + pic / 2 + sumy / sqrt(float(sumy*sumy + sumx*sumx))))) * 255;


	}





		/*
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
					cont[3 * (y*sizex + x) + 1] = sin(float(pic + pic/2 + (rg - y1) / rg)) * 255;
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
					cont[3 * (y*sizex + x) + 1] = sin(float(pic + (rg - y1) / rg)) * 255;
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
					cont[3 * (y*sizex + x) + 1] = sin(float((rg - y1)/rg)) * 255;
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
					cont[3 * (y*sizex + x) + 1] = sin(float(pic / 2 + (rg - y1) / rg)) * 255;
				}

			xk += 0.5;
			y1 = int(sqrt(float((rg*rg - xk*xk))));
			y2 = -y1;
		}
	}*/

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
	int *sobelx_d;
	int *sobely_d;
	cudaMalloc((void**)&gray_d, sizex * sizey * sizeof(char));
	cudaMalloc((void**)&cont_d, sizex * sizey * 3 * sizeof(char));
	cudaMalloc((void**)&sobelx_d, 25 * sizeof(int));
	cudaMalloc((void**)&sobely_d, 25 * sizeof(int));

	cudaMemcpy(gray_d, image, sizex * sizey * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(sobelx_d, &sobelx, 25 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sobely_d, &sobely, 25 * sizeof(int), cudaMemcpyHostToDevice);

	GradKernel KERNEL_ARGS2(dim3(sizey), dim3(sizex)) (gray_d, cont_d, sizex, sizey, rg, sobelx_d, sobely_d);
	cudaDeviceSynchronize();
	cudaMemcpy(cont, cont_d, sizex * sizey * 3 * sizeof(char), cudaMemcpyDeviceToHost);

	cudaFree(cont_d);
	cudaFree(gray_d);
}

void initsobel()
{
	sobelx[0][0] = -1; sobelx[0][1] = -1; sobelx[0][2] = 0; sobelx[0][3] = 1; sobelx[0][4] = 1;
	sobelx[1][0] = -2; sobelx[1][1] = -2; sobelx[1][2] = 0; sobelx[1][3] = 2; sobelx[1][4] = 2;
	sobelx[2][0] = -3; sobelx[2][1] = -6; sobelx[2][2] = 0; sobelx[2][3] = 6; sobelx[2][4] = 3;
	sobelx[3][0] = -2; sobelx[3][1] = -2; sobelx[3][2] = 0; sobelx[3][3] = 2; sobelx[3][4] = 2;
	sobelx[4][0] = -1; sobelx[4][1] = -1; sobelx[4][2] = 0; sobelx[4][3] = 1; sobelx[4][4] = 1;


	sobely[0][0] = -1; sobely[0][1] = -2; sobely[0][2] = -3; sobely[0][3] = -2; sobely[0][4] = -1;
	sobely[1][0] = -1; sobely[1][1] = -2; sobely[1][2] = -6; sobely[1][3] = -2; sobely[1][4] = -1;
	sobely[2][0] = 0; sobely[2][1] = 0; sobely[2][2] = 0; sobely[2][3] = 0; sobely[2][4] = 0;
	sobely[3][0] = -1; sobely[3][1] = -2; sobely[3][2] = 6; sobely[3][3] = 2; sobely[3][4] = 2;
	sobely[4][0] = -1; sobely[4][1] = -2; sobely[4][2] = 3; sobely[4][3] = 2; sobely[4][4] = 2;
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
	initsobel();
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
