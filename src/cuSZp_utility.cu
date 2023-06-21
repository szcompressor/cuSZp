//
// Created by Yafan Huang on 5/31/22.
//     Copied from SZx.
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "cuSZp_utility.h"

/*Macro Definition for Processing Data*/
// #define SZ_SCES 0  //successful
#define RW_SCES 0
#define RW_FERR 1
#define RW_TERR 2
#define LITTLE_ENDIAN_SYSTEM 0
#define QCAT_BUFS 64

/*Global Varaibles for Processing Data*/
int dataEndianType_Yafan = 0;
int sysEndianType_Yafan = 0; //0 means little endian, 1 means big endian

typedef union lint32
{
	int ivalue;
	unsigned int uivalue;
	unsigned char byte[4];
} lint32;

typedef union llfloat
{
    float value;
    unsigned int ivalue;
    unsigned char byte[4];
} llfloat;

/** ************************************************************************
 * @brief Reverse 4-bit-length unsigned char array.
 * 
 * @param   data[4]         4-bit-length unsigned char array.
 * *********************************************************************** */
void symTransForm_4Bytes(unsigned char data[4])
{
        unsigned char tmp = data[0];
        data[0] = data[3];
        data[3] = tmp;

        tmp = data[1];
        data[1] = data[2];
        data[2] = tmp;
}

/** ************************************************************************
 * @brief Read byte data from path to source binary format file.
 *        Usually used for decompressing data from input file.
 *        Variables byteLength and status can be obtained through this function.       
 * 
 * @param   srcFilePath     input source file path
 * @param   byteLength      the length of byte array
 * @param   status          data processing states (macro definitions) 
 * 
 * @return  byteBuf         unsigned char array with length byteLength
 * *********************************************************************** */
unsigned char *readByteData_Yafan(char *srcFilePath, size_t *byteLength, int *status)
{
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        *status = RW_FERR;
        return 0;
    }
	fseek(pFile, 0, SEEK_END);
    *byteLength = ftell(pFile);
    fclose(pFile);
    
    unsigned char *byteBuf = ( unsigned char *)malloc((*byteLength)*sizeof(unsigned char)); //sizeof(char)==1
    
    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        *status = RW_FERR;
        return 0;
    }
    fread(byteBuf, 1, *byteLength, pFile);
    fclose(pFile);
    *status = RW_SCES;
    return byteBuf;
}

/** ************************************************************************
 * @brief Read float data from path to source binary format file in endian systems.
 *        Usually used for compressing data from input file.
 *        Variables nbEle and status can be obtained through this function. 
 * 
 * @param   srcFilePath     input source file path
 * @param   nbEle           the length of float array
 * @param   status          data processing states (macro definitions) 
 * 
 * @return  daBuf           float array with length nbEle
 * *********************************************************************** */
float *readFloatData_systemEndian_Yafan(char *srcFilePath, size_t *nbEle, int *status)
{
	size_t inSize;
	FILE *pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 1\n");
        *status = RW_FERR;
        return NULL;
    }
	fseek(pFile, 0, SEEK_END);
    inSize = ftell(pFile);
    *nbEle = inSize/4; 
    fclose(pFile);
    
    if(inSize<=0)
    {
		printf("Error: input file is wrong!\n");
		*status = RW_FERR;
	}
    
    float *daBuf = (float *)malloc(inSize);
    
    pFile = fopen(srcFilePath, "rb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 2\n");
        *status = RW_FERR;
        return NULL;
    }
    fread(daBuf, 4, *nbEle, pFile);
    fclose(pFile);
    *status = RW_SCES;
    return daBuf;
}

/** ************************************************************************
 * @brief Read float data from path to source binary format file.
 *        Usually used for compressing data from input file.
 *        Variables nbEle and status can be obtained through this function. 
 * 
 * @param   srcFilePath     input source file path
 * @param   nbEle           the length of float array
 * @param   status          data processing states (macro definitions) 
 * 
 * @return  daBuf           float array with length nbEle
 * *********************************************************************** */
float *readFloatData_Yafan(char *srcFilePath, size_t *nbEle, int *status)
{
	int state = RW_SCES;
	if(dataEndianType_Yafan==sysEndianType_Yafan)
	{
		float *daBuf = readFloatData_systemEndian_Yafan(srcFilePath, nbEle, &state);
		*status = state;
		return daBuf;
	}
	else
	{
		size_t i,j;
		
		size_t byteLength;
		unsigned char* bytes = readByteData_Yafan(srcFilePath, &byteLength, &state);
		if(state == RW_FERR)
		{
			*status = RW_FERR;
			return NULL;
		}
		float *daBuf = (float *)malloc(byteLength);
		*nbEle = byteLength/4;
		
		llfloat buf;
		for(i = 0;i<*nbEle;i++)
		{
			j = i*4;
			memcpy(buf.byte, bytes+j, 4);
			symTransForm_4Bytes(buf.byte);
			daBuf[i] = buf.value;
		}
		free(bytes);
		return daBuf;
	}
}

/** ************************************************************************
 * @brief Write byte data to binary format file.
 *        Usually used for writing compressed data.
 *        Variable status can be obtained/switched through this function. 
 * 
 * @param   bytes           unsigned char array (compressed data)
 * @param   byteLength      the length of unsigned char array
 * @param   tgtFilePath     output file path
 * @param   status          data processing states (macro definitions) 
 * *********************************************************************** */
void writeByteData_Yafan(unsigned char *bytes, size_t byteLength, char *tgtFilePath, int *status)
{
	FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL)
    {
        printf("Failed to open input file. 3\n");
        *status = RW_FERR;
        return;
    }
    
    fwrite(bytes, 1, byteLength, pFile); //write outSize bytes
    fclose(pFile);
    *status = RW_SCES;
}

/** ************************************************************************
 * @brief Write float data to binary format file.
 *        Usually used for writing decompressed (reconstructed) data.
 *        Variable status can be obtained/switched through this function. 
 * 
 * @param   bytes           unsigned char array (compressed data)
 * @param   nbEle           the length of float array
 * @param   tgtFilePath     output file path
 * @param   status          data processing states (macro definitions) 
 * *********************************************************************** */
void writeFloatData_inBytes_Yafan(float *data, size_t nbEle, char* tgtFilePath, int *status)
{
	size_t i = 0; 
	int state = RW_SCES;
	llfloat buf;
	unsigned char* bytes = (unsigned char*)malloc(nbEle*sizeof(float));
	for(i=0;i<nbEle;i++)
	{
		buf.value = data[i];
		bytes[i*4+0] = buf.byte[0];
		bytes[i*4+1] = buf.byte[1];
		bytes[i*4+2] = buf.byte[2];
		bytes[i*4+3] = buf.byte[3];					
	}

	size_t byteLength = nbEle*sizeof(float);
	writeByteData_Yafan(bytes, byteLength, tgtFilePath, &state);
	free(bytes);
	*status = state;
}

// void convertIntArrayToBytes(int* states, size_t stateLength, unsigned char* bytes)
// {
// 	lint32 ls;
// 	size_t index = 0;
// 	size_t i;
// 	if(sysEndianType_Yafan==dataEndianType_Yafan)
// 	{
// 		for(i=0;i<stateLength;i++)
// 		{
// 			index = i << 2; //==i*4
// 			ls.ivalue = states[i];
// 			bytes[index] = ls.byte[0];
// 			bytes[index+1] = ls.byte[1];
// 			bytes[index+2] = ls.byte[2];
// 			bytes[index+3] = ls.byte[3];
// 		}		
// 	}
// 	else
// 	{
// 		for(i=0;i<stateLength;i++)
// 		{
// 			index = i << 2; //==i*4
// 			ls.ivalue = states[i];
// 			bytes[index] = ls.byte[3];
// 			bytes[index+1] = ls.byte[2];
// 			bytes[index+2] = ls.byte[1];
// 			bytes[index+3] = ls.byte[0];
// 		}			
// 	}
// }

// void writeIntData_inBytes(int *states, size_t stateLength, char *tgtFilePath, int *status)
// {
// 	int state = SZ_SCES;
// 	size_t byteLength = stateLength*4;
// 	unsigned char* bytes = (unsigned char*)malloc(byteLength*sizeof(char));
// 	convertIntArrayToBytes(states, stateLength, bytes);
// 	writeByteData_Yafan(bytes, byteLength, tgtFilePath, &state);
// 	free(bytes);
// 	*status = state;
// }


/** ************************************************************************
 * @brief Calculate SSIM in a small fraction of a 3D data file.
 *        A subfunction used in computeSSIM().
 * 
 * @param   data            original float array
 * @param   other           other (reconstructed) float array
 * @param   size1           3d-ssim setting.
 * @param   size0           3d-ssim setting.
 * @param   offset0         3d-ssim setting.
 * @param   offset1         3d-ssim setting.
 * @param   offset2         3d-ssim setting.
 * @param   windowSize0     3d-ssim setting.
 * @param   windowSize1     3d-ssim setting.
 * @param   windowSize2     3d-ssim setting.
 * 
 * @return  ssim            ssim value of the current small fraction data
 * *********************************************************************** */
double SSIM_3d_calcWindow_float(float* data, float* other, size_t size1, size_t size0, int offset0, int offset1, int offset2, int windowSize0, int windowSize1, int windowSize2) {
    int i0,i1,i2,index;
    int np=0; //Number of points
    float xMin=data[offset0+size0*(offset1+size1*offset2)];
    float xMax=data[offset0+size0*(offset1+size1*offset2)];
    float yMin=other[offset0+size0*(offset1+size1*offset2)];
    float yMax=other[offset0+size0*(offset1+size1*offset2)];
    double xSum=0;
    double ySum=0;
    for(i2=offset2; i2<offset2+windowSize2; i2++) {
        for(i1=offset1; i1<offset1+windowSize1; i1++) {
            for(i0=offset0; i0<offset0+windowSize0; i0++) {
                np++;
                index=i0+size0*(i1+size1*i2);
                if(xMin>data[index])
                    xMin=data[index];
                if(xMax<data[index])
                    xMax=data[index];
                if(yMin>other[index])
                    yMin=other[index];
                if(yMax<other[index])
                    yMax=other[index];
                xSum+=data[index];
                ySum+=other[index];
            }
        }
    }
    double xMean=xSum/np;
    double yMean=ySum/np;
    double var_x = 0, var_y = 0, var_xy = 0;
    for(i2=offset2; i2<offset2+windowSize2; i2++) {
        for(i1=offset1; i1<offset1+windowSize1; i1++) {
            for(i0=offset0; i0<offset0+windowSize0; i0++) {
                index=i0+size0*(i1+size1*i2);
                var_x += (data[index] - xMean)*(data[index] - xMean);
                var_y += (other[index] - yMean)*(other[index] - yMean);
                var_xy += (data[index] - xMean)*(other[index] - yMean);
            }
        }
    }
    var_x /= np;
    var_y /= np;
    var_xy /= np;
    double xSigma=sqrt(var_x);
    double ySigma=sqrt(var_y);
    double xyCov = var_xy;
    double c1,c2;
    if(xMax-xMin==0) {
		/*K1==0.01, K2==0.03*/
        c1=0.01*0.01;
        c2=0.03*0.03;
    } else {
        c1=0.01*0.01*(xMax-xMin)*(xMax-xMin);
        c2=0.03*0.03*(xMax-xMin)*(xMax-xMin);
    }
    double c3=c2/2;
    double luminance=(2*xMean*yMean+c1)/(xMean*xMean+yMean*yMean+c1);
    double contrast=(2*xSigma*ySigma+c2)/(xSigma*xSigma+ySigma*ySigma+c2);
    double structure=(xyCov+c3)/(xSigma*ySigma+c3);
    double ssim=luminance*contrast*structure;
    return ssim;
}

/** ************************************************************************
 * @brief Calculate SSIM between 3D original and decompressed (reconstructed) data.
 *        API for computing SSIM.
 * 
 * @param   oriData         original float array
 * @param   decData         decompressed (reconstructed) float array
 * @param   size2           the 1st dim of 3D data.
 * @param   size1           the 2nd dim of 3D data.
 * @param   size0           the 3rd dim of 3D data. (the fastest dim)
 * 
 * @return  ssimSum/nw      final ssim value between oriData and decData
 * *********************************************************************** */
double computeSSIM(float* oriData, float* decData, size_t size2, size_t size1, size_t size0)
{
	int windowSize0=7;
	int windowSize1=7;
	int windowSize2=7;
	int windowShift0=2;
	int windowShift1=2;
	int windowShift2=2;
    int offset0,offset1,offset2;
    int nw=0; //Number of windows
    double ssimSum=0;
    int offsetInc0,offsetInc1,offsetInc2;
    if(windowSize0>size0) {
        printf("ERROR: windowSize0 = %d > %zu\n", windowSize0, size0);
    }
    if(windowSize1>size1) {
        printf("ERROR: windowSize1 = %d > %zu\n", windowSize1, size1);
    }
    if(windowSize2>size2) {
        printf("ERROR: windowSize2 = %d > %zu\n", windowSize2, size2);
    }
    //offsetInc0=windowSize0/2;
    //offsetInc1=windowSize1/2;
    //offsetInc2=windowSize2/2;
    offsetInc0=windowShift0;
    offsetInc1=windowShift1;
    offsetInc2=windowShift2;
    for(offset2=0; offset2+windowSize2<=size2; offset2+=offsetInc2) { //MOVING WINDOW
        for(offset1=0; offset1+windowSize1<=size1; offset1+=offsetInc1) { //MOVING WINDOW
            for(offset0=0; offset0+windowSize0<=size0; offset0+=offsetInc0) { //MOVING WINDOW
                nw++;
                ssimSum+=SSIM_3d_calcWindow_float(oriData, decData, size1, size0, offset0, offset1, offset2, windowSize0, windowSize1, windowSize2);
            }
        }
    }
    return ssimSum/nw;
}


/** ************************************************************************
 * @brief Calculate PSNR between 3D original and decompressed (reconstructed) data.
 *        API for computing PSNR.
 * 
 * @param   nbEle           the length of float array
 * @param   ori_data        original float array
 * @param   dec_data        decompressed (reconstructed) float array
 * 
 * @return  result          6-length double array, which contains:
 *                              0. *Mean Square Error (MSE)*
 *                              1. *Value Range (Max-Min)*
 *                              2. *Peak Signal-to-noise Ratio (PSNR)*
 *                              3. Squared Error
 *                              4. Normalized Squared Error
 *                              5. Normalized Squared MSE
 * *********************************************************************** */
double *computePSNR(size_t nbEle, float *ori_data, float *data) {
    size_t i = 0;
    double Max = 0, Min = 0, diffMax = 0;
    Max = ori_data[0];
    Min = ori_data[0];
    diffMax = data[0] > ori_data[0] ? data[0] - ori_data[0] : ori_data[0] - data[0];

    //diffMax = fabs(data[0] - ori_data[0]);
    double sum1 = 0, sum2 = 0, sum22 = 0;

    for (i = 0; i < nbEle; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
        sum22 += data[i] * data[i];
    }
    double mean1 = sum1 / nbEle;
    double mean2 = sum2 / nbEle;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double maxpw_relerr = 0;
    for (i = 0; i < nbEle; i++) {
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];

        float err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr)
                maxpw_relerr = relerr;
        }

        if (diffMax < err)
            diffMax = err;
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / nbEle);
    double std2 = sqrt(sum4 / nbEle);
    double ee = prodSum / nbEle;
    double acEff = ee / std1 / std2;

    double mse = sum / nbEle;
    double range = Max - Min;
    double psnr = 20 * log10(range) - 10 * log10(mse);
    double normErr = sqrt(sum);
    double normErr_norm = normErr / sqrt(sum22);
    double nrmse = sqrt(mse) / range;
    double *result = (double *) malloc(sizeof(double) * 6);
    result[0] = mse;
    result[1] = range;
    result[2] = psnr;
    result[3] = normErr;
    result[4] = normErr_norm;
    result[5] = nrmse;

    return result;
}