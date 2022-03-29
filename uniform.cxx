#include <iostream>
#include <sstream>
#include <chrono>
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

namespace bg = boost::geometry;
  
typedef bg::model::d2::point_xy<double> point_t;
typedef bg::model::polygon<point_t> polygon_t;

int main(int argc, char* argv[])
{
//  omp_set_num_threads(atoi(argv[6]));

  polygon_t fsCP;
  boost::geometry::read_wkt(
        "POLYGON((5.09007307 28.76663592, 4.75749559 64.9236978, 5.54 55.88443233, 5.54 28.76663592, 5.09007307 28.76663592))", fsCP);

  std::string inputPath = argv[1];
  std::string inputFilename1 = argv[2];
  std::string inputFilename2 = argv[3];
  int nfiles = atoi(argv[4]);
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

  float xsp, ysp, zsp;
  xsp = (xmax - xmin)/((xdim-1)*1.0);
  ysp = (ymax - ymin)/((ydim-1)*1.0);
  zsp = (zmax - zmin)/((zdim-1)*1.0);

  float* xcoords = (float*)malloc(sizeof(float)*xdim);
  float* ycoords = (float*)malloc(sizeof(float)*ydim);
  float* zcoords = (float*)malloc(sizeof(float)*zdim);
 
  #pragma omp parallel for 
  for(int i = 0; i < xdim; i++) {
    xcoords[i] = xmin + i*xsp;
  } 

  for(int i = 0; i < ydim; i++) {
    ycoords[i] = ymin + i*ysp;
  } 

  for(int i = 0; i < zdim; i++) {
    zcoords[i] = zmin + i*zsp;
  } 

  int nMembers = 40;
  int nBins = atoi(argv[5]);
  double* insideProb = (double*)malloc(sizeof(double)*nPts); 
  double* outsideProb = (double*)malloc(sizeof(double)*nPts); 

  double* var1 = (double*)malloc(sizeof(double)*nPts*nMembers); 
  double* var2 = (double*)malloc(sizeof(double)*nPts*nMembers); 

  double gtFS[nPts] = {0};
  double diff_arr[nPts]; 
  /* Construct ground truth Fiber Surface */

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
  
  int intersection = 0;
  
  for(int t = 0; t < 5; t++) { 
  for(int b = 5; b <= 100; b+=5) {
  nBins = b;
  int thread_count = pow(2, t*1.0);
  omp_set_num_threads(thread_count);
  auto start_c = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int k = 0; k < zdim; k++) {
    for(int j = 0; j < ydim; j++) {
      for(int i = 0; i < xdim; i++) {
        int index = k*xdim*ydim + j*xdim + i;
        
        double v1[nMembers], v2[nMembers];
        double min_v1 = var1[index];
        double max_v1 = var1[index];
        double min_v2 = var2[index];
        double max_v2 = var2[index];
        for(int m = 0; m < nMembers; m++) {
          v1[m] = var1[(m*nPts)+index];
          v2[m] = var2[(m*nPts)+index];
          if(var1[(m*nPts)+index] < min_v1) {
            min_v1 = var1[(m*nPts)+index];
          }
          if(var1[(m*nPts)+index] > max_v1) {
            max_v1 = var1[(m*nPts)+index];
          }
          if(var2[(m*nPts)+index] < min_v2) {
            min_v2 = var2[(m*nPts)+index];
          }
          if(var2[(m*nPts)+index] > max_v2) {
            max_v2 = var2[(m*nPts)+index];
          }
        } 
        double band_v1 = (max_v1 - min_v1) / nBins*1.0;
        double band_v2 = (max_v2 - min_v2) / nBins*1.0;

        double hist_v1[nBins], hist_v2[nBins];
        double intercepts_v1[nBins+1], intercepts_v2[nBins+1];

        for(int h = 0; h < nBins+1; h++) { 
          intercepts_v1[h] = min_v1 + h*band_v1;
          intercepts_v2[h] = min_v2 + h*band_v2;
        }
        
        for(int h = 0; h < nBins; h++) {
          hist_v1[h] = 0;
          hist_v2[h] = 0;
          for(int m = 0; m < nMembers; m++) {
            if(v1[m] > intercepts_v1[h] && v1[m] < intercepts_v1[h+1]) {
              hist_v1[h] += 1;
            } 
            if(v2[m] > intercepts_v2[h] && v2[m] < intercepts_v2[h+1]) {
              hist_v2[h] += 1;
            } 
          }
          hist_v1[h] = hist_v1[h]/(nMembers*1.0);
          hist_v2[h] = hist_v2[h]/(nMembers*1.0);
        }

        double xlow = minVar1[index];
        double xhigh = maxVar1[index];
        double ylow = minVar2[index];
        double yhigh = maxVar2[index];
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
          for(int l = 0; l < nBins; l++) {
            for(int m = 0; m < nBins; m++) { 
              double v1low = intercepts_v1[l];
              double v1high = intercepts_v1[l+1];
              double v2low = intercepts_v2[m];
              double v2high = intercepts_v2[m+1];

              polygon_t inside_poly; 
              std::vector<point_t> points_in;
              points_in.push_back(point_t(v1low, v2low));
              points_in.push_back(point_t(v1low, v2high));
              points_in.push_back(point_t(v1high, v2high));
              points_in.push_back(point_t(v1high, v2low));
              points_in.push_back(point_t(v1low, v2low));
              boost::geometry::assign_points(inside_poly, points_in);
              if(boost::geometry::intersects(inside_poly, fsCP)) {
                intersection += 1;
                std::deque<polygon_t> overlap;
                boost::geometry::intersection(fsCP, inside_poly, overlap);
                double area_poly = boost::geometry::area(inside_poly);
                BOOST_FOREACH(polygon_t const& p, overlap)
                {
                   double intersectionArea = boost::geometry::area(p);
                   insideProb = insideProb + ((intersectionArea/area_poly)*hist_v1[l]*hist_v2[m]); 
                }
              }
            }
          }
        }

        outsideProb = 1.0 - insideProb;
        inP[index] = insideProb; 
        outP[index] = outsideProb; 
      }
    } 
  }
 
  auto end_c = std::chrono::high_resolution_clock::now();
  auto elapsed_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - start_c);
  double sq_sum = 0;
  for(int i = 0; i < nPts; i++){
    diff_arr[i] = inP[i] - (gtFS[i]*1.0);
    sq_sum += diff_arr[i]*diff_arr[i];
  }
  
  float l1_norm = sqrt(sq_sum);
  std::cout << nBins << ", " << thread_count << ", " << elapsed_c.count()/1000.0 << ", " << l1_norm << std::endl; 
 
  }  // THREADS
  }  // bins
  
  int expected = 93922;

  // compute the norm
  // What is the value?
  free(var1);
  free(var2);
  return 0;
}



/*
 
  polygon_t green, blue;
  boost::geometry::read_wkt(
        "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", green);
  boost::geometry::read_wkt(
        "POLYGON((0 0.5, 0 1.5, 1 1.5, 1 0.5, 0 0.5))", blue);

  std::deque<polygon_t> output;
  bg::intersection(green, blue, output);

  int i = 0;
  std::cout << "green && blue:" << std::endl;
  BOOST_FOREACH(polygon_t const& p, output)
  {
    std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;
  }
  
*/
