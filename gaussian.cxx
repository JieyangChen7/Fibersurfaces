#include <cmath>
#include <math.h>
#include <iostream>
#include <sstream>
#include <deque>
#include <stdlib.h>
#include <vtkXMLImageDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>
#include "omp.h"
#include <bits/stdc++.h>

namespace bg = boost::geometry;
  
typedef bg::model::d2::point_xy<double> point_t;
typedef bg::model::polygon<point_t> polygon_t;

double helperErrorFunc(double z) {
  return sqrt(M_PI/2.0) * erf(z/sqrt(2));
}

double computeLineIntegralGaussianDensity(double z2min, double z2max, double z1min, double z1max, double z2, double z1) {

  double i1 = (-0.5)*(1/(2*M_PI))*helperErrorFunc(z1)*(helperErrorFunc(z2max)-helperErrorFunc(z2min));
  double i2 = (0.5)*(1/(2*M_PI))*helperErrorFunc(z2)*(helperErrorFunc(z1max)-helperErrorFunc(z1min));
  double lineIntegral = i1 + i2;
//  std::cout << "The line integral val is : " << lineIntegral << std::endl;
  return lineIntegral;
}

double computeIntersectionIntegrationWithGreensTheorem(polygon_t result, double v1mean, double v1std, double v2mean, double v2std) {
  double polyIntegral = 0.0;
  std::vector<double> x, y;
  for(auto it = boost::begin(bg::exterior_ring(result)); it != boost::end(bg::exterior_ring(result)); ++it) {
    x.push_back(bg::get<0>(*it)); 
    y.push_back(bg::get<1>(*it)); 
  }

  for(int i = 0; i < x.size() - 1; i++) {
    double xNew1 = (x[i] - v1mean)/v1std; 
    double xNew2 = (x[i+1] - v1mean)/v1std; 
    double yNew1 = (y[i] - v2mean)/v2std; 
    double yNew2 = (y[i+1] - v2mean)/v2std; 
    polyIntegral = polyIntegral + computeLineIntegralGaussianDensity(xNew1, xNew2, yNew1, yNew2, xNew1, yNew1);
  }
  polyIntegral = polyIntegral*(-1.0);
  return polyIntegral;
}

double computeGaussianIntegralOverp1(double v1low, double v2low, double v1high, double v2high, double v1mean, double v1std, double v2mean, double v2std) {
  double polyIntegral = 0.0;
  double v1x = v1low;
  double v1y = v2low;

  double v2x = v1high;
  double v2y = v2low;
  
  double xNew1 = (v1x - v1mean)/v1std;
  double xNew2 = (v2x - v1mean)/v1std;
  double yNew1 = (v1y - v2mean)/v2std;
  double yNew2 = (v2y - v2mean)/v2std;

  polyIntegral = polyIntegral + computeLineIntegralGaussianDensity(xNew1, xNew2, yNew1, yNew2, xNew1, yNew1);
  
  v1x = v1high;
  v1y = v2low;

  v2x = v1high;
  v2y = v2high;
  
  xNew1 = (v1x - v1mean)/v1std;
  xNew2 = (v2x - v1mean)/v1std;
  yNew1 = (v1y - v2mean)/v2std;
  yNew2 = (v2y - v2mean)/v2std;

  polyIntegral = polyIntegral + computeLineIntegralGaussianDensity(xNew1, xNew2, yNew1, yNew2, xNew1, yNew1);
  
  v1x = v1high;
  v1y = v2high;

  v2x = v1low;
  v2y = v2high;
  
  xNew1 = (v1x - v1mean)/v1std;
  xNew2 = (v2x - v1mean)/v1std;
  yNew1 = (v1y - v2mean)/v2std;
  yNew2 = (v2y - v2mean)/v2std;

  polyIntegral = polyIntegral + computeLineIntegralGaussianDensity(xNew1, xNew2, yNew1, yNew2, xNew1, yNew1);
  
  v1x = v1low;
  v1y = v2high;

  v2x = v1low;
  v2y = v2low;
  
  xNew1 = (v1x - v1mean)/v1std;
  xNew2 = (v2x - v1mean)/v1std;
  yNew1 = (v1y - v2mean)/v2std;
  yNew2 = (v2y - v2mean)/v2std;

  polyIntegral = polyIntegral + computeLineIntegralGaussianDensity(xNew1, xNew2, yNew1, yNew2, xNew1, yNew1);
  return polyIntegral;
}


int median(int l, int r) {
  int n = r - l + 1;
  n = ((n+1)/2) - 1;
  return n + l;
}

double interquartile_range(double* arr, int len) {
  std::sort(arr, arr+len);
  int mid_index = median(0, len); 
  double q1 = arr[median(0, mid_index)];
  double q3 = arr[median(mid_index+1, len)];
  return (q3 - q1);
}

double standard_dev(double* arr, int len) {
  double sum = 0.0;
  for(int i = 0; i < len; i++) {
    sum += arr[i];
  }
  double mean = sum/(len*1.0);
  double variance = 0.0;
  for(int i = 0; i < len; i++) {
    variance += pow(arr[i] - mean, 2.0);
  }
  variance = variance/(len*1.0);
  return sqrt(variance);
}

double computeSilvermansBandwidth(double *arr, int len) {
  double iqr = interquartile_range(arr, len);
  double stddev = standard_dev(arr, len);
  double iqr_val = (1.0/1.34)*iqr;
  double spread = std::min(stddev, iqr_val);
  double quant1 = 1.0;
  double quant2 = (1.0/(2.0*sqrt(M_PI)));
  double bandwidth; 
  double coeff1 = (pow(quant1, (-2.0/5.0)))*(pow(quant2, (1.0/5.0)));
  double coeff2 = (((3.0/8.0)*pow( M_PI, (-1.0/2.0)))*pow(pow(spread, -5.0), (-1.0/5.0)));
  double coeff3 = pow(len, (-1.0/5.0));
  bandwidth = coeff1*coeff2*coeff3; 
  return 0.01;  
}


int main(int argc, char* argv[])
{
//  std::cout << "Executing Gaussian version" << std::endl;
  omp_set_num_threads(atoi(argv[5]));

  polygon_t fsCP;
  boost::geometry::read_wkt(
        "POLYGON((5.09007307 28.76663592, 4.75749559 64.9236978, 5.54 55.88443233, 5.54 28.76663592, 5.09007307 28.76663592))", fsCP);

  std::string inputPath = argv[1];
  std::string inputFilename1 = argv[2];
  std::string inputFilename2 = argv[3];
  int nMembers = atoi(argv[4]);
  // Read the file

  float xmin = -5.0;
  float xmax = 5.0;
  float ymin = -5.0;
  float ymax = 5.0;
  float zmin = -5.0;
  float zmax = 5.0;

  int xdim, ydim, zdim;
  xdim = ydim = zdim = 64;
  int nPts = xdim*ydim*zdim;
  std::cout << "The npts: " << nPts << std::endl;

  float xsp, ysp, zsp;
  xsp = (xmax - xmin)/((xdim-1)*1.0);
  ysp = (ymax - ymin)/((ydim-1)*1.0);
  zsp = (zmax - zmin)/((zdim-1)*1.0);

  float* xcoords = (float*)malloc(sizeof(float)*xdim);
  float* ycoords = (float*)malloc(sizeof(float)*ydim);
  float* zcoords = (float*)malloc(sizeof(float)*zdim);

  #pragma omp parallel for 
  for(int i = 0; i < xdim; i++) {
    if( i == 0) 
      std::cout << "Get number of threads: " << omp_get_num_threads() << std::endl;
    xcoords[i] = xmin + i*xsp;
  } 

  for(int i = 0; i < ydim; i++) {
    ycoords[i] = ymin + i*ysp;
  } 

  for(int i = 0; i < zdim; i++) {
    zcoords[i] = zmin + i*zsp;
  } 

  double* insideProb = (double*)malloc(sizeof(double)*nPts); 
  double* outsideProb = (double*)malloc(sizeof(double)*nPts); 

  double* var1 = (double*)malloc(sizeof(double)*nPts*nMembers); 
  double* var2 = (double*)malloc(sizeof(double)*nPts*nMembers); 

  double gtFS[nPts] = {0};
  double diff_arr[nPts]; 


  auto reader1 = vtkSmartPointer<vtkXMLImageDataReader>::New();
  auto reader2 = vtkSmartPointer<vtkXMLImageDataReader>::New();
  auto ensemble1 = vtkSmartPointer<vtkImageData>::New(); // vtkImageData 
  auto ensemble2 = vtkSmartPointer<vtkImageData>::New(); // vtkImageData 
  for(int m = 0; m < nMembers; m++) {
    std::stringstream input1;
    input1 << inputPath << inputFilename1 << "/" <<  inputFilename1 << "_" << m <<".vti";
  
    reader1->SetFileName(input1.str().c_str());
    reader1->Update();
    ensemble1 = reader1->GetOutput(); // vtkImageData 
    auto field1 = ensemble1->GetPointData()->GetArray(0);
    std::stringstream input2;
    input2 << inputPath << inputFilename2 << "/" <<  inputFilename2 << "_" << m <<".vti";
    reader2->SetFileName(input2.str().c_str());
    reader2->Update();
    ensemble2 = reader2->GetOutput(); // vtkImageData 
    auto field2 = ensemble2->GetPointData()->GetArray(0);
    for(int k = 0; k < zdim; k++) {
      for(int j = 0; j < ydim; j++) {
        for(int i = 0; i < xdim; i++) {
          int index = k*xdim*ydim + j*xdim + i;
          double d[3];
          field1->GetTuple(index, d); 
          var1[(m*nPts)+index] = d[0];
          field2->GetTuple(index, d); 
          var2[(m*nPts)+index] = d[0];
        }
      } 
    }
  }
  
  // Create sphere and tangle functions
  double* func1 = (double*)malloc(sizeof(double)*nPts); 
  double* func2 = (double*)malloc(sizeof(double)*nPts); 
    
  for(int k = 0; k < zdim; k++) {
    for(int j = 0; j < ydim; j++) {
      for(int i = 0; i < xdim; i++) {
        int index = k*xdim*ydim + j*xdim + i;
        func1[index] = (xcoords[i]*xcoords[i]) + (ycoords[j]*ycoords[j]) + (zcoords[k]*zcoords[k]);
        func2[index] = pow(xcoords[i], 4.0) - (5*pow(xcoords[i], 2.0)) + pow(ycoords[j], 4.0) - (5*pow(ycoords[j], 2.0)) + pow(zcoords[k], 4.0) - (5*pow(zcoords[k], 2.0)) + 40;
      }
    } 
  }

  // Populate the gtFS array with 1s 
  int c = 0;
  for(int k = 0; k < zdim; k++) {
    for(int j = 0; j < ydim; j++) {
      for(int i = 0; i < xdim; i++) {
        int index = k*xdim*ydim + j*xdim + i;
        point_t p(func1[index], func2[index]);
        if(boost::geometry::within(p, fsCP)) {
          gtFS[index] = 1;
          c++;
        }
      }
    } 
  }

  std::cout << "C : " << c << std::endl;

  // compute histogram - inside outside probabilities

  double* minVar1 = (double*)malloc(sizeof(double)*nPts);
  double* maxVar1 = (double*)malloc(sizeof(double)*nPts);
  double* minVar2 = (double*)malloc(sizeof(double)*nPts);
  double* maxVar2 = (double*)malloc(sizeof(double)*nPts);

  for(int i = 0; i < nPts; i++) {
    minVar1[i] = var1[i];
    maxVar1[i] = var1[i];
    minVar2[i] = var2[i];
    maxVar2[i] = var2[i];

    for(int m = 0; m < nMembers; m++) {
      if(var1[(m*nPts)+i] < minVar1[i]) {
        minVar1[i] = var1[(m*nPts)+i];
      }
      if(var1[(m*nPts)+i] > maxVar1[i]) {
        maxVar1[i] = var1[(m*nPts)+i];
      }
      if(var2[(m*nPts)+i] < minVar2[i]) {
        minVar2[i] = var2[(m*nPts)+i];
      }
      if(var2[(m*nPts)+i] > maxVar2[i]) {
        maxVar2[i] = var2[(m*nPts)+i];
      }
    }
  }

  double* inP = (double*)malloc(sizeof(double)*nPts);
  double* outP = (double*)malloc(sizeof(double)*nPts);

  double sumIA = 0.0;
  #pragma omp parallel for
  for(int k = 0; k < zdim; k++) {
    for(int j = 0; j < ydim; j++) {
      for(int i = 0; i < xdim; i++) {
        int index = k*xdim*ydim + j*xdim + i;
    //    double v1[nMembers], v2[nMembers];
    //    for(int m = 0; m < nMembers; m++) {
    //      v1[m] = var1[(m*nPts)+index];
    //      v2[m] = var2[(m*nPts)+index];
    //    } 
  //      double band_v1 = computeSilvermansBandwidth(v1, nMembers);
  //      double band_v2 = computeSilvermansBandwidth(v2, nMembers);
        double band_v1 = 0.01;
        double band_v2 = 0.01;
   

        double xlow = minVar1[index] - (band_v1/2.0);
        double xhigh = maxVar1[index] + (band_v1/2.0);
        double ylow = minVar2[index] - (band_v2/2.0);
        double yhigh = maxVar2[index] + (band_v2/2.0);

        polygon_t test_poly; 
        std::vector<point_t> points;
        points.push_back(point_t(xlow, ylow));
        points.push_back(point_t(xlow, yhigh));
        points.push_back(point_t(xhigh, yhigh));
        points.push_back(point_t(xhigh, ylow));
        points.push_back(point_t(xlow, ylow));
        boost::geometry::assign_points(test_poly, points);
     
        double insideProb = 0.0;
        double outsideProb = 0.0;
        if(boost::geometry::intersects(test_poly, fsCP)) {
          for(int l = 0; l < nMembers; l++) {
            for(int m = 0; m < nMembers; m++) { 
              band_v1 = 0.01;
              band_v2 = 0.01;
               
              double v1mean = var1[(nPts*l)+index];
              double v2mean = var2[(nPts*m)+index];

              double v1low =  v1mean - (band_v1/2.0);
              double v1high =  v1mean + (band_v1/2.0);
              double v2low =  v2mean - (band_v2/2.0);
              double v2high =  v2mean + (band_v2/2.0);

              double v1std = band_v1/2.0; 
              double v2std = band_v2/2.0; 
           
              polygon_t inside_poly; 
              std::vector<point_t> points_in;
              points_in.push_back(point_t(v1low, v2low));
              points_in.push_back(point_t(v1low, v2high));
              points_in.push_back(point_t(v1high, v2high));
              points_in.push_back(point_t(v1high, v2low));
              points_in.push_back(point_t(v1low, v2low));
              boost::geometry::assign_points(inside_poly, points_in);
              if(boost::geometry::intersects(inside_poly, fsCP)) {
                std::deque<polygon_t> overlap;
                boost::geometry::intersection(fsCP, inside_poly, overlap);
                double overlap_a;
                double area_poly = boost::geometry::area(inside_poly);
                double intersectedArea = 0.0;
                double pIntegral = 1.0;
                BOOST_FOREACH(polygon_t const& result, overlap)
                {
                    intersectedArea = computeIntersectionIntegrationWithGreensTheorem(result, v1mean, v1std, v2mean, v2std);
                    sumIA += intersectedArea;
                    pIntegral = computeGaussianIntegralOverp1(v1low, v2low, v1high, v2high, v1mean, v1std, v2mean, v2std);
                    insideProb = insideProb + (intersectedArea/pIntegral); 
                }
                // Gaussian changes start here
              }
            }
          } 
          insideProb = insideProb/(nMembers*nMembers);
        }

        outsideProb = 1.0 - insideProb;
        inP[index] = insideProb; 
        outP[index] = outsideProb; 
      }
    } 
  }
 
  std::cout << "The intersection area sum: " << sumIA << std::endl;
  // compute the norm
  double sq_sum = 0;
  for(int i = 0; i < nPts; i++){
    diff_arr[i] = inP[i] - gtFS[i];
    sq_sum += pow(diff_arr[i], 2.0);
  }
  
  float l2_norm = sqrt(sq_sum);
  std::cout << "L2 norm: " << l2_norm;
  // What is the value?
  //free(var1);
  //free(var2);

  return 0;
}



