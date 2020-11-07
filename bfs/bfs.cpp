#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    int new_distance = distances[frontier->vertices[0]] + 1;;
    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {
        int node = frontier->vertices[i];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
        int* local_outgoing = (int*)malloc(sizeof(int) * (end_edge - start_edge));

        // attempt to add all neighbors to the new frontier
        int local_count = 0;
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = new_distance;
                local_outgoing[local_count] = outgoing;    
	  	        local_count++;	
            }
        }

        if (local_count > 0){
            int old_index = __sync_fetch_and_add(&new_frontier->count, local_count);
            for(int neighbor = 0; neighbor < local_count; neighbor++){
                new_frontier->vertices[old_index + neighbor] = local_outgoing[neighbor];
            }
        }
        free(local_outgoing);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(guided)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_doxn_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// Take one step of "bottom-up" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
    bool* node_unvisited)
{
    int nTotThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        nTotThreads = omp_get_num_threads();
    }

    bool* is_new_frontier = (bool*)malloc(sizeof(bool) * g->num_nodes);
    int new_distance = distances[frontier->vertices[0]] + 1;
    int* num_threads = (int*)malloc(sizeof(int)* g->num_nodes*nTotThreads);
    int* nCount = (int*)malloc(sizeof(int)* nTotThreads);

    for(int iThread = 0; iThread < nTotThreads; iThread++){
        nCount[iThread] = 0;
    }

    #pragma omp parallel for
    for (int node=0; node<g->num_nodes; node++) {
        if(node_unvisited[node]){
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];

            // attempt to add all neighbors to the new frontier            
            int local_visited = NOT_VISITED_MARKER;
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int up_node = g->incoming_edges[neighbor];
                if(!node_unvisited[up_node]){
                    int iThread = omp_get_thread_num();
                    distances[node] = new_distance;
                    num_threads[iThread*g->num_nodes + nCount[iThread]] = node;
                    nCount[iThread]++;
                    break;
                }
            }
        }
    }

    #pragma omp parallel for schedule(guided)
    for(int iThread = 0; iThread < nTotThreads; iThread++){
        int index = __sync_fetch_and_add(&new_frontier->count, nCount[iThread]);
        for(int i = 0; i < nCount[iThread]; i++){
	    int node = num_threads[iThread*g->num_nodes + i];
	    node_unvisited[node] = false;
            new_frontier->vertices[index + i] = node;
        }
    }
    free(is_new_frontier);
    free(num_threads);
    free(nCount);
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    bool* node_unvisited = (bool*)malloc(sizeof(bool) * graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(guided)
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        node_unvisited[i] = true;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    node_unvisited[ROOT_NODE_ID] = false;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, node_unvisited);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(node_unvisited);
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    bool* node_unvisited = (bool*)malloc(sizeof(bool) * graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        node_unvisited[i] = true;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    node_unvisited[ROOT_NODE_ID] = false;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, node_unvisited);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(node_unvisited);
}
