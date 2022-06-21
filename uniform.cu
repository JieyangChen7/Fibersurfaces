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

#define ENTER 0
#define EXIT 1
#define PARALLEL 2

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}


// Modified based on GPU-optimized code in here: https://stackoverflow.com/questions/6781153/gpu-friendly-2d-line-segment-intersection-algorithm
// Function return if there is an intersection
// intersectionPoint return the actual intersection point
__device__ bool edge_intersetion(double2& p1, double2& p2, double2& p3, double2& p4, double2& intersectionPoint) {
  double2 A = p2;
  double2 a = make_double2(p2.x - p1.x, p2.y - p1.y);
  double2 B = p4;
  double2 b = make_double2(p4.x - p3.x, p4.y - p3.y);
  double2 c = make_double2(B.x - A.x, B.y - A.y);
  double2 b_perp = make_double2(-b.y, b.x);

  double numerator = b_perp.x * c.x + b_perp.y * c.y;
  double denominator = b_perp.x * a.x + b_perp.y * a.y;
  bool isParallel = (denominator == 0.0f);

  double quotient = numerator / denominator;
  intersectionPoint = make_double2(quotient * a.x + A.x, quotient * a.y + A.y);

  bool output = (!isParallel && 
                intersectionPoint.x >= min(p1.x, p2.x) && 
                intersectionPoint.x >= min(p3.x, p4.x) &&
                intersectionPoint.x <= max(p1.x, p2.x) && 
                intersectionPoint.x <= max(p3.x, p4.x) &&
                intersectionPoint.y >= min(p1.y, p2.y) && 
                intersectionPoint.y >= min(p3.y, p4.y) &&
                intersectionPoint.y <= max(p1.y, p2.y) && 
                intersectionPoint.y <= max(p3.y, p4.y));
  return output;
}

// Cyrus Beck algorithm
// Determine if edge p3->p4 is entering the polygon through edge p1->p2
// This assuming inside of the polygon is always on the right-hand side i.e., points of polygon are stored in clockwise.
// Note: this does not check edge intersection
__device__ int edge_direction(double2& p1, double2& p2, double2& p3, double2& p4) {
  // Get an arbitrary vector perpendicular to edge p1->p2 pointing to its right-hand side (inside of polygon)
  // Reference: https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector
  double2 perpendicular_vec = make_double2(p2.y - p1.y, -(p2.x - p1.x));

  // Calculate the dot product between the perpendicular_vec with edge p3->p4
  double2 edge_34 = make_double2(p4.x - p3.x, p4.y - p3.y);
  double dot_product = perpendicular_vec.x * edge_34.x + perpendicular_vec.y * edge_34.y;
  if (dot_product > 0) return ENTER;
  if (dot_product < 0) return EXIT;
  else return PARALLEL; // dot_product == 0
}


// Ray casting algorithm
// Reference: https://en.wikipedia.org/wiki/Point_in_polygon
// check if poly2 is inside poly1
template <int nPoint>
__device__ bool is_inside(double2 * poly1, double2 * poly2) {

  // int index = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < nPoint; i++) {
    for (int j = 0; j < nPoint; j++) {
      double2 intersectionPoint;
      if (edge_intersetion(poly1[i], poly1[(i+1)%nPoint], poly2[j], poly2[(j+1)%nPoint], intersectionPoint)) {
        // if(index == 186394) printf("is_inside::edge_intersetion - true\n");
        return false;
      }
    }
  }

  for (int i = 0; i < nPoint; i++) {
    // cast ray for each point on poly2
    // Assuming x10000 is enough for the ray to cast to the outside of poly1
    // This needs to be replaced with a better approach
    double2 ray_start = make_double2(poly2[i].x, poly2[i].y);
    double2 ray_end = make_double2(poly2[i].x * (10000), poly2[i].y * (10000));
    int num_intersects = 0;
    for (int j = 0; j < nPoint - 1; j++) {
      double2 intersectionPoint;
      if (edge_intersetion(poly1[j], poly1[(j+1)%nPoint], ray_start, ray_end, intersectionPoint)) {
        num_intersects++;
      }
    }

    // if(index == 186394) printf("is_inside::num_intersects: %d\n", num_intersects);

    if (num_intersects % 2 == 0) {
      // intersect even times --> outside
      return false;
    }
  }

  return true;
}

template <int nPoint>
__device__ bool test_intersect(double2 * poly1, double2 * poly2) {
  for (int i = 0; i < nPoint; i++) {
    for (int j = 0; j < nPoint; j++) {
      double2 intersectionPoint;
      if (edge_intersetion(poly1[i], poly1[(i+1)%nPoint], poly2[j], poly2[(j+1)%nPoint], intersectionPoint)) {
        return true;
      }
    }
  }
  return is_inside<nPoint>(poly1, poly2) || is_inside<nPoint>(poly2, poly1);
}

__device__ double interpolation_ratio(double2& p1, double2& p2, double2& interpo_point) {
  if (p2.x - p1.x != 0) {
    return (interpo_point.x - p1.x) / (p2.x - p1.x);
  } else if (p2.y - p1.y != 0) {
    return (interpo_point.y - p1.y) / (p2.y - p1.y);
  } else {
    return 0;
  }
}



// Sort intersections on each edge so that we can follow the right order
template <int nPoint>
__device__ void sort_intersections(double2 intersectionPoint_list[(nPoint)*(nPoint)],
                                   uint8_t intersectionPoint_id_poly[nPoint][nPoint],
                                   uint8_t intersectionPoint_count_poly[nPoint],
                                   double2 poly[nPoint], bool debug) {
  double ratio_list[nPoint];
  bool mask[nPoint];
  uint8_t sorted_id[nPoint];
  
  for (int i = 0; i < nPoint; i++) {
    double2 start_p = poly[i];
    double2 end_p = poly[(i+1)%nPoint];
    // For each edge, calculate the interpolation ratio of each intersections
    for (int j = 0; j < intersectionPoint_count_poly[i]; j++) {
      uint8_t id = intersectionPoint_id_poly[i][j];
      double2 intersect_p = intersectionPoint_list[id];
      double ratio = interpolation_ratio(start_p, end_p, intersect_p);
      ratio_list[j] = ratio;
      mask[j] = true;
    }
    // if (debug) {
    //   printf("ratio_list of point %d: ", i);
    //   for (int d = 0; d < intersectionPoint_count_poly[i]; d++) {
    //     printf("%f ", ratio_list[d]);
    //   }
    //   printf("\n");
    // }
    // Sort intersection id by interpolation ratio. Eariler intersections first.
    // Can be replaced with Sorting Network (maybe faster)
    for (int j = 0; j < intersectionPoint_count_poly[i]; j++) {
      double min_ratio = 1.0;
      uint8_t min_id = 0;
      uint8_t min_idx = 0;
      // Find j-th smallest intersection
      for (int k = 0; k < intersectionPoint_count_poly[i]; k++) {
        if (mask[k] && min_ratio > ratio_list[k]) {
          min_ratio = ratio_list[k];
          min_id = intersectionPoint_id_poly[i][k];
          min_idx = k;
        }
      }
      sorted_id[j] = min_id;
      mask[min_idx] = false; // should not select again
    }

    // if (debug) {
    //   printf("sorted_id of point %d: ", i);
    //   for (int d = 0; d < intersectionPoint_count_poly[i]; d++) {
    //     printf("%d ", sorted_id[d]);
    //   }
    //   printf("\n");
    // }

    // Copy sorted result back
    for (int j = 0; j < intersectionPoint_count_poly[i]; j++) {
      intersectionPoint_id_poly[i][j] = sorted_id[j];
    }
  }
}


// following points of a polygon starting from the intersection and stop at the next intersection
// Function returns the stopping intersection id (not recorded)
// Starting intersection is recorded
template <int nPoint>
__device__ uint8_t follow_poly(uint8_t start_id, double2 intersectionPoint_list[(nPoint)*(nPoint)],
                               double2 poly[nPoint], 
                               uint8_t intersectionPoint_count_poly[nPoint],
                               uint8_t intersectionPoint_id_poly[nPoint][nPoint],
                               double2 output[nPoint*2+(nPoint)*(nPoint)], int& write_idx, bool debug) {

  bool start_record = false;
  // record intersection start_id
  output[write_idx] = intersectionPoint_list[start_id];
  write_idx++;
  // printf("index %d write_idx %d\n", index, write_idx);
  // if (debug) printf("record intersection: %f %f\n", intersectionPoint_list[start_id].x, intersectionPoint_list[start_id].y);
  // search from beginning of poly (could be faster if we use some kind of indexing)
  int count = 0;
  int i = 0;
  while (count < nPoint) { // for each edge of poly

    if (start_record) {
      // record point i on polygon
      output[write_idx] = poly[i];
      write_idx++;
      // if (debug) printf("record ploy: %f %f intersection count: %d\n", poly[i].x, poly[i].y, intersectionPoint_count_poly[i]);
    }
    if(intersectionPoint_count_poly[i] > 0) { // check if there is any intersections on this edge
      for (int j = 0; j < intersectionPoint_count_poly[i]; j++) { // check each intersection on this edge
        uint8_t id = intersectionPoint_id_poly[i][j];
        if (start_record) {
          // after we starting following poly, we should stop once hitting another intersection
          // if (debug) printf("follow_poly::found stopping intersection id = %d (%d/%d)\n", id, j, intersectionPoint_count_poly[i]);
          return id;
        }
        if (id == start_id) { // found the start intersetion id
          // if (debug) printf("follow_poly::found starting intersection id = %d (%d/%d)\n", id, j, intersectionPoint_count_poly[i]);
          start_record = true;
        }
      }
    }
    i++;
    i = i % nPoint;
    count++;
    // printf("index %d count %d while loop %d/%d \n", index, count, i, nPoint);
  }
  // printf("Error\n");
  return 0; // should not reach here.
}

// Weiler Atherton algorithm
// Reference[1]: https://www.geeksforgeeks.org/weiler-atherton-polygon-clipping-algorithm/
// Reference[2]: https://www.cs.drexel.edu/~david/Classes/CS430/HWs/p214-weiler.pdf
// Assuming poly1 is the clipping polygon and poly2 is the clipped polygon
// Function returns the number of output polygons
// nPointOverlap return a list of which each entry is the number of points of each output polyon ((nPoint)*(nPoint)/2 == max number of entering intersection)
// overlap return a list of points of all output polyons grouped by each polygon (nPoint*2+(nPoint)*(nPoint) == total number of points)
template <int nPoint>
__device__ uint8_t find_intersection(double2 poly1[nPoint], double2 poly2[nPoint], 
                                 uint8_t nPointOverlap[(nPoint)*(nPoint)/2], 
                                 double2 overlap[nPoint*2+(nPoint)*(nPoint)], bool debug) {
  
  if (is_inside<nPoint>(poly1, poly2)) {
    // if poly2 is inside poly1
    // if (debug) printf("poly2 is inside poly1\n");
    nPointOverlap[0] = nPoint;
    for (int i = 0; i < nPoint; i++) {
      overlap[i] = poly2[i];
    }
    return 1; // just one polygon
  }

  if (is_inside<nPoint>(poly2, poly1)) {
    // if poly1 is inside poly2
    // if (debug) printf("poly1 is inside poly2\n");
    nPointOverlap[0] = nPoint;
    for (int i = 0; i < nPoint; i++) {
      overlap[i] = poly1[i];
    }
    return 1; // just one polygon
  }

  // all intersection points (their index in the array is the intersection id)
  double2 intersectionPoint_list[(nPoint)*(nPoint)];

  // the following use unit8 to reduce register usage

  // direction of each intersection
  uint8_t intersectionPoint_direction[(nPoint)*(nPoint)];
  // total number of intersection
  uint8_t intersectionPoint_count_total = 0;

  // total number of intersection on each edge of poly1
  uint8_t intersectionPoint_count_poly1[nPoint];
  // total number of intersection on each edge of poly2
  uint8_t intersectionPoint_count_poly2[nPoint];

  // list of ids of intersections on each edge of poly1
  uint8_t intersectionPoint_id_poly1[nPoint][nPoint];
  // list of ids of intersections on each edge of poly1
  uint8_t intersectionPoint_id_poly2[nPoint][nPoint];
  for (int i = 0; i < nPoint; i++) {
    intersectionPoint_count_poly1[i] = 0;
    intersectionPoint_count_poly2[i] = 0;
  }

  // Find all intersections and initialize above data structures
  for (int i = 0; i < nPoint; i++) {
    for (int j = 0; j < nPoint; j++) {
      double2 intersectionPoint;
      if (edge_intersetion(poly1[i], poly1[(i+1)%nPoint], poly2[j], poly2[(j+1)%nPoint], intersectionPoint)) {
        intersectionPoint_list[intersectionPoint_count_total] = intersectionPoint;
        // Check the direction of edges on clipped polygon (poly2) against edges on the clipping polygon (poly1)
        intersectionPoint_direction[intersectionPoint_count_total] = 
          edge_direction(poly1[i], poly1[(i+1)%nPoint], poly2[j], poly2[(j+1)%nPoint]);
        intersectionPoint_id_poly1[i][intersectionPoint_count_poly1[i]] = intersectionPoint_count_total;
        intersectionPoint_id_poly2[j][intersectionPoint_count_poly2[j]] = intersectionPoint_count_total;
        intersectionPoint_count_total++;
        intersectionPoint_count_poly1[i]++;
        intersectionPoint_count_poly2[j]++;
      }
    }
  }

  sort_intersections<nPoint>(intersectionPoint_list, intersectionPoint_id_poly1,
                             intersectionPoint_count_poly1, poly1, debug);
  sort_intersections<nPoint>(intersectionPoint_list, intersectionPoint_id_poly2,
                             intersectionPoint_count_poly2, poly2, debug);


  // if (debug) {
  //   printf("intersectionPoint_list: ");
  //   for (int i = 0; i < intersectionPoint_count_total; i++) {
  //     printf("(%f %f), ", intersectionPoint_list[i].x, intersectionPoint_list[i].y);
  //   }
  //   printf("\n");
  //   printf("intersectionPoint_direction: ");
  //   for (int i = 0; i < intersectionPoint_count_total; i++) {
  //     printf("(%u), ", intersectionPoint_direction[i]);
  //   }
  //   printf("\n");
  //   for (int i = 0; i < nPoint; i++) {
  //     printf("intersectionPoint_id_poly1[%d]: ", i);
  //     for (int j = 0; j < intersectionPoint_count_poly1[i]; j++) {
  //       printf("(%u), ", intersectionPoint_id_poly1[i][j]);
  //     }
  //     printf("\n");
  //   }
  //   for (int i = 0; i < nPoint; i++) {
  //     printf("intersectionPoint_id_poly2[%d]: ", i);
  //     for (int j = 0; j < intersectionPoint_count_poly2[i]; j++) {
  //       printf("(%d), ", intersectionPoint_id_poly2[i][j]);
  //     }
  //     printf("\n");
  //   }
  // }

  // if (debug) printf("intersectionPoint_count_total: %d\n", intersectionPoint_count_total);

  int total_num_overlap_polygons = 0;
  int write_idx = 0; // output write index
  // Begin finding intersecting polygons
  for (int i = 0; i < intersectionPoint_count_total; i++) {
    // start from each entering intersection
    if (intersectionPoint_direction[i] == ENTER) {
      // if (debug) printf("found ENTER (id = %d): %f %f\n", i, intersectionPoint_list[i].x, intersectionPoint_list[i].y);
      uint8_t start_id = i; // starting intersetion id of current interseting polygon      
      uint8_t current_intersection_id = start_id; // current intersection will keep changing as we follow poly1 or poly2
      int write_idx_start = write_idx; // for calculating total number of points in this polygon
      do {
        if (intersectionPoint_direction[current_intersection_id] == ENTER) {
          // follow clipped polygon list (poly2) if we are entering
          current_intersection_id = follow_poly<nPoint>(current_intersection_id, intersectionPoint_list, poly2, 
                                                        intersectionPoint_count_poly2,
                                                        intersectionPoint_id_poly2, overlap, write_idx, debug);
          // if (debug) {
          //   for (int w = 0; w < write_idx; w++)
          //     printf("follow poly2: overlap: %f %f\n", overlap[w].x, overlap[w].y);
          //   printf("current_intersection_id: %d\n", current_intersection_id);
          // }
        } else if (intersectionPoint_direction[current_intersection_id] == EXIT) {
          // follow clipping polygon list (poly1) if we are exiting
          current_intersection_id = follow_poly<nPoint>(current_intersection_id, intersectionPoint_list, poly1, 
                                                        intersectionPoint_count_poly1,
                                                        intersectionPoint_id_poly1, overlap, write_idx, debug);
          // if (debug) {
          //   for (int w = 0; w < write_idx; w++)
          //     printf("follow poly1: overlap: %f %f\n", overlap[w].x, overlap[w].y);
          //   printf("current_intersection_id: %d\n", current_intersection_id);
          // }
        } else { // PARALLEL
          // maybe it shouldn't reach here. (not so sure)
        }
      } while (current_intersection_id != start_id); // done when we circle back to the starting intersetion
      nPointOverlap[total_num_overlap_polygons++] = write_idx - write_idx_start;
    }
  }
  return total_num_overlap_polygons;
}

// Shoelace formula
__device__ double calc_area(int nPoint, double2 * poly) {
  double result = 0.0;

  int j = nPoint - 1;
  for (int i = 0; i < nPoint; i++)
  {
      result += (poly[j].x + poly[i].x) * (poly[j].y - poly[i].y);
      j = i;  // j is previous vertex to i
  }

  // Return absolute value
  return fabs(result / 2.0);
}

template <int nMembers, int nBins>
__global__ void gpu_kernel(size_t nPts, double* var1, double* var2, 
                           double* minVar1,  double* maxVar1, double* minVar2, double* maxVar2, 
                           double*inP, double*outP) {
  const int nDim = 2;
  const int nPoint = 5;

  // use double2 type as a point in 2D space
  double2 fsCP[nPoint] = {make_double2(5.09007307, 28.76663592), 
                                make_double2(4.75749559, 64.9236978),
                                make_double2(5.54, 55.88443233),
                                make_double2(5.54, 28.76663592),
                                make_double2(5.09007307, 28.76663592)};

  __shared__ double v1[nMembers], v2[nMembers];
  __shared__ double hist_v1[nBins], hist_v2[nBins];
  __shared__ double intercepts_v1[nBins+1], intercepts_v2[nBins+1];

  int index0 = blockIdx.x;
  for (int index = index0; index < nPts; index += gridDim.x) {
    for(int m = threadIdx.x; m < nMembers; m += blockDim.x) { 
    // for(int m = 0; m < nMembers; m++) {
      v1[m] = var1[(m*nPts)+index];
      v2[m] = var2[(m*nPts)+index];
    }
    __syncthreads();

    double min_v1 = v1[0];
    double max_v1 = v1[0];
    double min_v2 = v2[0];
    double max_v2 = v2[0];

    // This could be improved by using register shuffle instructions
    for(int m = 1; m < nMembers; m++) {
      if(v1[m] < min_v1) {
        min_v1 = v1[m];
      }
      if(v1[m] > max_v1) {
        max_v1 = v1[m];
      }
      if(v2[m] < min_v2) {
        min_v2 = v2[m];
      }
      if(v2[m] > max_v2) {
        max_v2 = v2[m];
      }
    } 
    double band_v1 = (max_v1 - min_v1) / nBins*1.0;
    double band_v2 = (max_v2 - min_v2) / nBins*1.0;

    

    for(int h = threadIdx.x; h < nBins+1; h += blockDim.x) { 
    // for(int h = 0; h < nBins+1; h++) { 
      intercepts_v1[h] = min_v1 + h*band_v1;
      intercepts_v2[h] = min_v2 + h*band_v2;
    }

    __syncthreads();
    
    // for(int h = 0; h < nBins; h++) {
    for(int h = threadIdx.x; h < nBins; h += blockDim.x) { 
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

    double2 test_poly[nPoint] = {make_double2(xlow, ylow), 
                                make_double2(xlow, yhigh),
                                make_double2(xhigh, yhigh),
                                make_double2(xhigh, ylow),
                                make_double2(xlow, ylow)};

    double insideProb = 0.0;
    double outsideProb = 0.0;

    __syncthreads();
    
    if(test_intersect<nPoint>(test_poly, fsCP)) {
        for (int l = 0; l < nBins; l++) {
          for (int m = threadIdx.x; m < nBins; m+= blockDim.x) { 

          bool debug = false;

          double v1low = intercepts_v1[l];
          double v1high = intercepts_v1[l+1];
          double v2low = intercepts_v2[m];
          double v2high = intercepts_v2[m+1];

          double2 inside_poly[nPoint] = {make_double2(v1low, v2low),
                                        make_double2(v1low, v2high),
                                        make_double2(v1high, v2high),
                                        make_double2(v1high, v2low),
                                        make_double2(v1low, v2low)};

          // if (debug) printf("inside_poly: (%f %f), (%f %f), (%f %f), (%f %f), (%f %f)\n",
          //                               v1low, v2low, v1low, v2high, v1high, v2high, v1high, v2low, v1low, v2low);
          if(test_intersect<nPoint>(inside_poly, fsCP)) { 
            // if (debug) printf("%d %d inside_poly test_intersect true\n", l, m);
            uint8_t nPointOverlap[(nPoint)*(nPoint)/2];
            double2 overlap[nPoint*2+(nPoint)*(nPoint)];
            uint8_t npolyOverlap = 0;
            
            npolyOverlap = find_intersection<nPoint>(inside_poly, fsCP, nPointOverlap, overlap, debug);
            double area_poly = calc_area(nPoint, inside_poly);
            // if (debug) printf("inside_poly area: %f\n", area_poly);
            int offset = 0;
            for (int i = 0; i < npolyOverlap; i++)
            {
               double intersectionArea = calc_area(nPointOverlap[i], overlap+offset);
               // if (debug) printf("%d %d overlap area: %f %f\n", l, m, intersectionArea, area_poly);
               insideProb = insideProb + ((intersectionArea/area_poly)*hist_v1[l]*hist_v2[m]); 
               offset += nPointOverlap[i];
            }
          }
        }
      }
    }
    
    // Multiple thread may write to the same inP, so we need atomic
    atomicAdd(&inP[index], insideProb);
    outsideProb = 1.0 - insideProb;
    // outP[index] = outsideProb; // seems we are not using outP
  }
}

template <int nBins, int nMembers>
void launch_gpu_kernel(int nPts, double* var1, double* var2, double* minVar1, 
                      double* maxVar1, double* minVar2, double* maxVar2, 
                      double* diff_arr, double* gtFS) {

  double* inP = (double*)malloc(sizeof(double)*nPts);
  double* outP = (double*)malloc(sizeof(double)*nPts);
  // prepare GPU buffers
  double* var1_d = NULL, *var2_d = NULL, *inP_d = NULL, *outP_d = NULL;
  double* minVar1_d = NULL, *maxVar1_d = NULL, *minVar2_d = NULL, *maxVar2_d = NULL;

  gpuErrchk(cudaMalloc((void**)&var1_d, sizeof(double)*nPts*nMembers));
  gpuErrchk(cudaMalloc((void**)&var2_d, sizeof(double)*nPts*nMembers));
  gpuErrchk(cudaMalloc((void**)&minVar1_d, sizeof(double)*nPts));
  gpuErrchk(cudaMalloc((void**)&maxVar1_d, sizeof(double)*nPts));
  gpuErrchk(cudaMalloc((void**)&minVar2_d, sizeof(double)*nPts));
  gpuErrchk(cudaMalloc((void**)&maxVar2_d, sizeof(double)*nPts))
  gpuErrchk(cudaMalloc((void**)&inP_d, sizeof(double)*nPts));
  gpuErrchk(cudaMalloc((void**)&outP_d, sizeof(double)*nPts));

  gpuErrchk(cudaMemcpy(minVar1_d, minVar1, sizeof(double)*nPts, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(maxVar1_d, maxVar1, sizeof(double)*nPts, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(minVar2_d, minVar2, sizeof(double)*nPts, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(maxVar2_d, maxVar2, sizeof(double)*nPts, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(var1_d, var1, sizeof(double)*nPts*nMembers, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(var2_d, var2, sizeof(double)*nPts*nMembers, cudaMemcpyDefault));
  gpuErrchk(cudaDeviceSynchronize());

  
  int blockSize = nBins;
  int numBlock = nPts;

  auto start_c = std::chrono::high_resolution_clock::now();
  gpu_kernel<nMembers, nBins><<<numBlock, blockSize>>>(nPts, var1_d, var2_d, minVar1_d, maxVar1_d, minVar2_d, maxVar2_d, inP_d, outP_d);
  gpuErrchk(cudaDeviceSynchronize());
  auto end_c = std::chrono::high_resolution_clock::now();
  auto elapsed_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - start_c);

  gpuErrchk(cudaMemcpy(inP, inP_d, sizeof(double)*nPts, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(outP, outP_d, sizeof(double)*nPts, cudaMemcpyDefault));
  gpuErrchk(cudaDeviceSynchronize());
  double sq_sum = 0;
  for(int i = 0; i < nPts; i++){
    // printf("inP[%d]: %f\n", i, inP[i]);
    diff_arr[i] = inP[i] - (gtFS[i]*1.0);
    sq_sum += diff_arr[i]*diff_arr[i];
  }
  
  float l1_norm = sqrt(sq_sum);
  std::cout << nBins << ", " << elapsed_c.count()/1000.0 << ", " << l1_norm << std::endl; 
  delete[] inP;
  delete[] outP;

  gpuErrchk(cudaFree(var1_d));
  gpuErrchk(cudaFree(var2_d));
  gpuErrchk(cudaFree(minVar1_d));
  gpuErrchk(cudaFree(maxVar1_d));
  gpuErrchk(cudaFree(minVar2_d));
  gpuErrchk(cudaFree(maxVar2_d));
  gpuErrchk(cudaFree(inP_d));
  gpuErrchk(cudaFree(outP_d));
}

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

  const int nMembers = 40;
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


  

  launch_gpu_kernel<5, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<10, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<15, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<20, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<25, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<30, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<45, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<50, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<55, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<60, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<65, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<70, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<75, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<80, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<85, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<90, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<95, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
  launch_gpu_kernel<100, nMembers>(nPts, var1, var2, minVar1, maxVar1, minVar2, maxVar2, diff_arr, gtFS);
   
  
  
  // int expected = 93922;

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
