#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#include <sys/unistd.h>

#define VECSIZE 1
#define ITERATIONS 100

typedef struct {
    double val;
    int   rank;
} max_cell;

MPI_Status status;

double When()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

void mergeMaxVectors(max_cell max[], max_cell temp[]) {
    int i;
    for (i = 0; i < VECSIZE; i++) {
        if (temp[i].val > max[i].val) {
            max[i].val = temp[i].val;
            max[i].rank = temp[i].rank;
        }
    }
}

// Broadcast value to all nodes
void broadcastMaxVector(max_cell* max, int rank, int num_dim) {
    int not_participating = pow(2.0, num_dim-1) - 1;
    int bit_mask = pow(2.0, num_dim - 1);
    int cur_dim;

    // printf("num_dim: %d\n", num_dim);

    for (cur_dim = 0; cur_dim < num_dim; cur_dim++) {
        if ((rank & not_participating) == 0) {
            if ((rank & bit_mask) == 0) {
                int dest_id = rank ^ bit_mask;
                MPI_Send(max, VECSIZE, MPI_DOUBLE_INT, dest_id, 0, MPI_COMM_WORLD);// Send
            }
            else {
                int src_id = rank ^ bit_mask;
                MPI_Recv(max, VECSIZE, MPI_DOUBLE_INT, src_id, 0, MPI_COMM_WORLD, &status);// Receive

                // int i;
                // for(i = 0; i < VECSIZE; i++) {
                //     printf("proc %d sent to proc %d [%d] = %f from %d\n", src_id, rank, i, max[i].val, max[i].rank);
                // }
            }
        }
        not_participating >>= 1;
        bit_mask >>= 1;
    }

    return;
}

// Reduce values to one node
void reduceMaxVector(max_cell* max, int rank, int num_dim) {
    int not_participating = 0;
    int bit_mask = 1;
    max_cell* temp = malloc(sizeof(max_cell) * (VECSIZE + 1));
    int cur_dim;

    for(cur_dim = 0; cur_dim < num_dim; cur_dim++) {
        if ((rank & not_participating) == 0) {
            if ((rank & bit_mask) != 0) {
                int dest_id = rank ^ bit_mask;
                MPI_Send(max, VECSIZE, MPI_DOUBLE_INT, dest_id, 0, MPI_COMM_WORLD);// Send
            } else {
                int src_id = rank ^ bit_mask;
                int i;

                MPI_Recv(temp, VECSIZE, MPI_DOUBLE_INT, src_id, 0, MPI_COMM_WORLD, &status);// Receive

                // for(i = 0; i < VECSIZE; i++) {
                //     printf("receive: proc %d [%d] = %f from %d\n", rank, i, max[i].val, src_id);
                // }

                // printf("REDUCE: MPI task %d got %f from %d\n", rank, new_value, src_id);
                mergeMaxVectors(max, temp);

                // for(i = 0; i < VECSIZE; i++) {
                //     printf("receive: proc %d [%d] = %f from %d\n", rank, i, temp[i].val, src_id);
                // }
            }
        }
        not_participating = not_participating ^ bit_mask;
        bit_mask <<=1;
    }
    return;
}

int main(int argc, char *argv[])
{
        int iproc, nproc, i, iter;
        char host[255], message[55];
        setvbuf(stdout, NULL, _IONBF, 0);

        // printf("Starting\n");

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

        int num_dim = (int) log2(nproc);

        // gethostname(host,253);
        // printf("I am proc %d of %d running on %s\n", iproc, nproc,host);
        // each process has an array of VECSIZE double: ain[VECSIZE]
        double init_vals[VECSIZE];
        
        int  ind[VECSIZE];
        max_cell* max = malloc(sizeof(max_cell) * (VECSIZE + 1));
        int my_id, root = 0;

        MPI_Comm_rank(MPI_COMM_WORLD, &my_id);


        // if (my_id == 0) {
        //     printf("Running benchmark.c on %d machines with arrays of size %d a total of %d times\n", nproc, VECSIZE, ITERATIONS);
        // }

        // Start time here
        srand(my_id+5);
        double start = When();
        for(iter = 0; iter < ITERATIONS; iter++) {
            for(i = 0; i < VECSIZE; i++) {
                double num = rand();
                init_vals[i] = num;
                max[i].val = num;
                max[i].rank = my_id;
                // printf("init proc %d [%d]=%f\n", my_id, i, init_vals[i]);
            }
            reduceMaxVector(max, my_id, num_dim);
            // At this point, the answer resides on process root
            // if (my_id == root) {
            //     /* read ranks out */
            //     for (i=0; i<VECSIZE; ++i) {
            //         printf("root: max[%d] = %f from %d\n", i, max[i].val, max[i].rank);
            //         // aout[i] = out[i].val;
            //         // ind[i] = out[i].rank;
            //     }
            // }
            // Now broadcast this max vector to everyone else.
            broadcastMaxVector(max, my_id, num_dim);
            // for(i = 0; i < VECSIZE; i++) {
            //     printf("final proc %d [%d] = %f from %d\n", my_id, i, max[i].val, max[i].rank);
            // }
        }
        MPI_Finalize();
        double end = When();
        if(my_id == root) {
            printf("%f",end-start);
        }
}










