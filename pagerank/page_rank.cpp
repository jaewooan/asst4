#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  
  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:
  */
  // initialization: see example code above
  
  int* numEdgesLeavingFrom =  (int*)malloc(sizeof(int) * numNodes); // edge -> node
  int* numEdgesReachingTo =  (int*)malloc(sizeof(int) * numNodes); // edge -> node
  double* local_diff = (double*)malloc(sizeof(double) * omp_get_num_threads());
  double* score_old = (double*)malloc(sizeof(double) * numNodes);
  int scoreFromNodeNoOutLike = 0;

  #pragma omp parallel for schedule(guided)
  for (int vi = 0; vi < numNodes; ++vi) {
    score_old[vi] = solution[vi];
    if(vi < numNodes - 1){
      numEdgesLeavingFrom[vi] =  g->outgoing_starts[vi+1] - g->outgoing_starts[vi];
      numEdgesReachingTo[vi] =  g->incoming_starts[vi+1] - g->incoming_starts[vi];
    }
    else{
      numEdgesLeavingFrom[vi] =  num_edges(g) - g->outgoing_starts[vi];
      numEdgesReachingTo[vi] =  num_edges(g) - g->incoming_starts[vi];
    }
  }

  bool converged = false;
  while (!converged) {
    // initialize for no outgoing edges numEdge    
    scoreFromNodeNoOutLike = 0;
    double global_diff = 0;
    for (int iThread = 0; iThread < omp_get_num_threads(); iThread++){
      printf("nThread: %d\n", iThread);
      local_diff[iThread] = 0;
    }

    #pragma omp parallel for schedule(guided)
    for (int vi = 0; vi < numNodes; ++vi) {
      // sum over all nodes v in graph with no outgoing edges
      scoreFromNodeNoOutLike += damping * score_old[vi] / numNodes;
      solution[vi] = 0;
    }
    
    #pragma omp parallel for schedule(guided)
    for (int vi = 0; vi < numNodes; ++vi) {
      for(int iEdge = 0; iEdge < numEdgesReachingTo[vi]; iEdge++) {
        // compute score_new[vi] for all nodes vi:
        int vj = g->incoming_edges[g->incoming_starts[vi] + iEdge];
        solution[vi] += score_old[vj] / numEdgesLeavingFrom[vj];
        solution[vi] = (damping * solution[vi]) + (1.0 - damping) / numNodes;
        solution[vi] += scoreFromNodeNoOutLike;
      }
      local_diff[omp_get_thread_num()] += std::fabs(solution[vi] - score_old[vi]);
    }

    for (int iThread = 0; iThread < omp_get_num_threads(); iThread++){
      global_diff += local_diff[iThread];
    }

    // compute how much per-node scores have changed
    // quit once algorithm has converged
    converged = (global_diff < convergence);

    #pragma omp parallel for schedule(guided)
    for (int vi = 0; vi < numNodes; ++vi) {
      score_old[vi] = solution[vi];
    }
  }  
}
