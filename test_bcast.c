#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

MPI_Status status;

// Broadcast value to all nodes
float broadcast (int num_dim, int id, float value) {
    int not_participating = pow(2.0, num_dim-1) - 1;
    int bit_mask = pow(2.0, num_dim - 1);
    float new_value = value;
    int cur_dim;

    // printf("num_dim: %d\n", num_dim);

    for (cur_dim = 0; cur_dim < num_dim; cur_dim++) {
        if ((id & not_participating) == 0) {
            if ((id & bit_mask) == 0) {
                int dest_id = id ^ bit_mask;
                MPI_Send(&new_value, 1, MPI_FLOAT, dest_id, 0, MPI_COMM_WORLD);// Send
            }
            else {
                int src_id = id ^ bit_mask;
                MPI_Recv(&new_value, 1, MPI_FLOAT, src_id, 0, MPI_COMM_WORLD, &status);// Receive
            }
        }
        not_participating >>= 1;
        bit_mask >>= 1;
    }

    return new_value;
}

// Reduce values to one node
float reduceSum(int num_dim, int rank, float value)
{
    int not_participating = 0;
    int bit_mask = 1;
    float sum = value;
    float new_value;
    int cur_dim;

    for(cur_dim = 0; cur_dim < num_dim; cur_dim++) {
		if ((rank & not_participating) == 0) {
		    if ((rank & bit_mask) != 0) {
				int dest_id = rank ^ bit_mask;
				MPI_Send(&sum, 1, MPI_FLOAT, dest_id, 0, MPI_COMM_WORLD);// Send
		    } else {
				int src_id = rank ^ bit_mask;
				MPI_Recv(&new_value, 1, MPI_FLOAT, src_id, 0, MPI_COMM_WORLD, &status);// Receive
				// printf("REDUCE: MPI task %d got %f from %d\n", rank, new_value, src_id);
	            sum += new_value;
	        }
	    }
		not_participating = not_participating ^ bit_mask;
	    bit_mask <<=1;
    }
    return sum;
}

int main (int argc, char *argv[]) {
	int id;
	int num_tasks;
	float value;
	char host[255];
	setvbuf(stdout, NULL, _IONBF, 0);

	/* Obtain number of tasks and task ID */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	int num_dim = (int) (log(num_tasks) / log(2));

	gethostname(host, 253);
	printf("BEGIN: MPI task %d of %d running on %s\n", id, num_tasks, host);

	value = reduceSum(num_dim, id, id);	

	printf("REDUCE: MPI task %d of %d got %f\n", id, num_tasks, value);

	value = broadcast(num_dim, id, value);

	printf("BROADCAST: MPI task %d of %d got %f\n", id, num_tasks, value);
	
	MPI_Finalize();
	exit(0);
}
















