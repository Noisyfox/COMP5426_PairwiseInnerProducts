#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void usage(char const* const message, ...)
{
	va_list arg_ptr;
	va_start(arg_ptr, message);

	vfprintf(stderr, message, arg_ptr);

	va_end(arg_ptr);

	fprintf(stderr, "\n");

	fprintf(stderr, "Please run the application in the following format:\n");
	fprintf(stderr, "mpirun -np [proc num] inner_product [M] [N]\n");
	fprintf(stderr, "Required: [N] %% [proc num] = 0; [proc num] %% 2 = 1\n");
	fprintf(stderr, "\n");
}

/*
 * Create a matrix with n rows and m columns
 */
float** create_matrix(int n, int m)
{
	float* flat = malloc(n * m * sizeof(float));
	float** matrix = malloc(n * sizeof(float*));
	int i;

	memset(flat, 0, n * m * sizeof(float));

	for (i = 0; i < n; i++)
	{
		matrix[i] = &flat[i * m];
	}

	return matrix;
}

float product(float* row1, float* row2, int m)
{
	float result = 0;
	int i;

	for (i = 0; i < m; i++)
	{
		result += row1[i] * row2[i];
	}

	return result;
}

int product_in_block(float** block, int rows, int m, float* results)
{
	int i, j, k = 0;

	for (i = 0; i < rows - 1; i++)
	{
		for (j = i + 1; j < rows; j++)
		{
			results[k++] = product(block[i], block[j], m);
		}
	}

	return k;
}

int product_among_blocks(float** left, int left_rows, float** right, int right_rows, int m, float* results)
{
	int i, j, k = 0;

	for(i = 0; i < left_rows; i++)
	{
		for(j = 0; j < right_rows; j++)
		{
			results[k++] = product(left[i], right[j], m);
		}
	}

	return k;
}

void print_results(float* results, int result_count, int n)
{
	int i;
	int j = n - 1;
	int k = n - 2;

	for (i = 0; i < result_count; i++)
	{
		printf("%f ", results[i]);
		j--;
		if (j == 0)
		{
			j = k;
			k--;
			printf("\n");
		}
	}
}

int main(int argc, char** argv)
{
	int my_id, num_procs, prev_id, next_id;
	MPI_Request *requests = NULL;
	MPI_Status *statuses = NULL;
	int *indices = NULL;
	int count = 0;
	int is_master, has_parallism;
	int m, n;
	float** full_matrix = NULL;
	int rows_per_block;
	int i, j, k;
	float** block_left = NULL;
	float** block_right = NULL;
	int final_results_count = 0;
	float* final_results = NULL;
	float* final_results_serial = NULL;
	int processor_results_count = 0;
	float* processor_results = NULL;
	int processor_results_index = 0;

	// Init MPI and process id
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

	is_master = my_id == 0;
	has_parallism = num_procs > 1;
	prev_id = is_master ? num_procs - 1 : my_id - 1;
	next_id = (my_id + 1) % num_procs;

	if (num_procs % 2 != 1)
	{
		if (is_master)
		{
			usage("Process number must be odd!");
		}
		goto _exit;
	}

	// Read user input
	if (argc < 3)
	{
		if (is_master)
		{
			usage("Too few arguments (2 required, %d provided).", argc - 1);
		}
		goto _exit;
	}
	m = atoi(argv[1]);
	n = atoi(argv[2]);

	if (n % num_procs != 0)
	{
		if (is_master)
		{
			usage("Illegal arguments.");
		}
		goto _exit;
	}

	rows_per_block = n / num_procs;
	block_left = create_matrix(rows_per_block, m);
	block_right = create_matrix(rows_per_block, m);
	requests = (MPI_Request*)malloc((num_procs - 1) * sizeof(MPI_Request));
	statuses = (MPI_Status*)malloc((num_procs - 1) * sizeof(MPI_Status));
	indices = (int*)malloc((num_procs - 1) * sizeof(int));
	for(i = 0; i < num_procs - 1; i++)
	{
		requests[i] = MPI_REQUEST_NULL;
	}

#define pp(process) ((process) > my_id ? (process) - 1 : (process))

	if (is_master)
	{
		// Generate matrix
		full_matrix = create_matrix(n, m);
		k = 1;
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < m; j++)
			{
				full_matrix[i][j] = k;  // Here we should use random, but for debugging, just use a counter
				k++;
			}
		}

		final_results_count = n * (n - 1) / 2;

		if (has_parallism)
		{
			final_results = malloc(final_results_count * sizeof(float));

			// Send left blocks to other processors
			memcpy(block_left[0], full_matrix[0], rows_per_block * m * sizeof(float));
			for (i = 1; i < num_procs; i++)
			{
				MPI_Isend(full_matrix[i * rows_per_block], rows_per_block * m, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[pp(i)]);
			}
			MPI_Waitall(num_procs - 1, requests, statuses);

			// Send right blocks to other processors
			memcpy(block_right[0], full_matrix[rows_per_block], rows_per_block * m * sizeof(float));
			for (i = 1; i < num_procs; i++)
			{
				j = (1 + 1) % num_procs;
				MPI_Isend(full_matrix[j * rows_per_block], rows_per_block * m, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[pp(i)]);
			}
		}
	}
	else
	{
		// Receive initial left block
		MPI_Recv(block_left[0], rows_per_block * m, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &statuses[pp(0)]);
		// Receive initial right block
		MPI_Recv(block_right[0], rows_per_block * m, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &statuses[pp(0)]);
	}

	if (has_parallism)
	{
		if (is_master)
		{
			printf("Performing parallism computation...\n");
		}

		processor_results_count = n * (n - 1) / 2 / num_procs;
		processor_results = malloc(processor_results_count * sizeof(float));

		// First calculate the products inside left blocks
		processor_results_index += product_in_block(block_left, rows_per_block, m, &processor_results[processor_results_index]);

		// Here starts the computation iters
		while (1)
		{
			// Calculate among left and right
			processor_results_index += product_among_blocks(block_left, rows_per_block, block_right, rows_per_block, m, &processor_results[processor_results_index]);

			if (processor_results_index < processor_results_count)
			{
				// Shift right blocks
//				if (requests[pp(next_id)] != MPI_REQUEST_NULL)
//				{
					MPI_Wait(&requests[pp(next_id)], &statuses[pp(next_id)]);
//				}
				MPI_Isend(block_right[0], rows_per_block * m, MPI_FLOAT, next_id, 0, MPI_COMM_WORLD, &requests[pp(next_id)]);
				MPI_Recv(block_right[0], rows_per_block * m, MPI_FLOAT, prev_id, MPI_ANY_TAG, MPI_COMM_WORLD, &statuses[pp(prev_id)]);
			}
			else
			{
				break;
			}
		}

		// Merge results to master
		if (is_master)
		{
			printf("Merging results...\n");
			
			// TODO: receive results and merge
		}
		else
		{
			// Send results to master processor
//			if (requests[pp(0)] != MPI_REQUEST_NULL)
//			{
				MPI_Wait(&requests[pp(0)], &statuses[pp(0)]);
//			}
			MPI_Send(processor_results, processor_results_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
	}

	if (is_master)
	{
		if (has_parallism)
		{
			printf("Done.\n");
			// Print out results
			printf("Results from parallism computation:\n");
			print_results(final_results, final_results_count, n);

			printf("\n");
		}

		// Do serial computation

		printf("Performing serial computation...\n");
		final_results_serial = malloc(final_results_count * sizeof(float));
		product_in_block(full_matrix, n, m, final_results_serial);
		printf("Done.\n");

		printf("Results from serial computation:\n");
		// Print out results from serial computation
		print_results(final_results_serial, final_results_count, n);
	}

_exit:
	MPI_Finalize();
	return 0;
}
