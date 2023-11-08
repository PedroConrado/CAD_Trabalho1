#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>

#define MAX_COORD 100.0

int manhattan_distance(int x1, int y1, int z1, int x2, int y2, int z2) {
    return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2);
}

double euclidean_distance(int x1, int y1, int z1, int x2, int y2, int z2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <N> <seed> <t>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int t = atoi(argv[3]);

    int rank, size, workers;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide o trabalho entre os processos MPI
    int points_per_process = N * N / (size-1);
    int extra_points = N * N % (size-1);

    int start_point, end_point;
    if ((rank-1) < extra_points) {
    // Processos com rank < extra_points recebem pontos extras
        start_point = (rank-1) * (points_per_process + extra_points);
        end_point = start_point + points_per_process + extra_points;
    } else {
        // Processos sem pontos extras
        start_point = (rank-1) * points_per_process + extra_points;
        end_point = start_point + points_per_process;
    }

    int *x = (int *)malloc(N*N * sizeof(int));
    int *y = (int *)malloc(N*N * sizeof(int));
    int *z = (int *)malloc(N*N * sizeof(int));

    int min_manhattan = INT_MAX, max_manhattan = 0;
    double min_euclidean = DBL_MAX, max_euclidean = 0.0;
    int sum_min_manhattan = 0, sum_max_manhattan = 0;
    double sum_min_euclidean = 0.0, sum_max_euclidean = 0.0;

    if(rank == 0){
        srand(seed); 
        for(int i = 0; i < N*N; i++){
            x[i] = rand() % (int)MAX_COORD;
            y[i] = rand() % (int)MAX_COORD;
            z[i] = rand() % (int)MAX_COORD;
        }
        for (int i = 1; i < size; i++){
            MPI_Send(&(x[0]), N*N, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&(y[0]), N*N, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&(z[0]), N*N, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

    }else{
        int i = 0;

        MPI_Recv(&(x[0]), N*N, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&(y[0]), N*N, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&(z[0]), N*N, MPI_INT, i, 1, MPI_COMM_WORLD, &status);

        for (int ij = start_point; ij < end_point; ij++) {
            int local_min_manhattan = INT_MAX;
            int local_max_manhattan = 0;
            double local_min_euclidean = DBL_MAX;
            double local_max_euclidean = 0.0;
            for (int k = ij+1; k < N*N; k++) {
                
                int manhattan_dist = manhattan_distance(x[ij], y[ij], z[ij], x[k], y[k], z[k]);
                double euclidean_dist = euclidean_distance(x[ij], y[ij], z[ij], x[k], y[k], z[k]);

                if (manhattan_dist < local_min_manhattan) { 
                    local_min_manhattan = manhattan_dist;
                }
                if (manhattan_dist > local_max_manhattan) {
                    local_max_manhattan = manhattan_dist;
                }

                if (euclidean_dist < local_min_euclidean) {
                    local_min_euclidean = euclidean_dist;
                }
                if (euclidean_dist > local_max_euclidean) {
                    local_max_euclidean = euclidean_dist;
                }
            }
            
            if (local_min_manhattan < min_manhattan)
                min_manhattan = local_min_manhattan;
            if (local_max_manhattan > max_manhattan)
                max_manhattan = local_max_manhattan;

            if (local_min_euclidean < min_euclidean)
                min_euclidean = local_min_euclidean;
            if (local_max_euclidean > max_euclidean)
                max_euclidean = local_max_euclidean;
                
            if (local_min_manhattan != INT_MAX && local_min_euclidean != DBL_MAX){
                sum_max_manhattan += local_max_manhattan;
                sum_max_euclidean += local_max_euclidean;
                sum_min_manhattan += local_min_manhattan;
                sum_min_euclidean += local_min_euclidean;
            }
        }
    }

    int global_sum_min_manhattan, global_sum_max_manhattan;
    double global_sum_min_euclidean, global_sum_max_euclidean;
    int global_min_manhattan, global_max_manhattan;
    double global_min_euclidean, global_max_euclidean;

    MPI_Reduce(&min_manhattan, &global_min_manhattan, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_manhattan, &global_max_manhattan, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min_euclidean, &global_min_euclidean, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_euclidean, &global_max_euclidean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_min_manhattan, &global_sum_min_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_manhattan, &global_sum_max_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_min_euclidean, &global_sum_min_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_euclidean, &global_sum_max_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, global_sum_min_manhattan, global_max_manhattan, global_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, global_sum_min_euclidean, global_max_euclidean, global_sum_max_euclidean);

    }

    free(x);
    free(y);
    free(z);

    MPI_Finalize();

    return 0;
}
