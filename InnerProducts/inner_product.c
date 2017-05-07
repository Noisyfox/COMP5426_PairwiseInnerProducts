#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <mpi.h>

void usage(char const* const message, ...)
{
	va_list arg_ptr;
	va_start(arg_ptr, message);

	vfprintf(stderr, message, arg_ptr);

	va_end(arg_ptr);

	printf("\n");

	printf("Please run the application in the following format:\n");
	printf("mpirun -np [proc num] inner_product [M] [N]\n");
	printf("Required: [N] %% [proc num] = 0; [proc num] %% 2 = 1\n");
	printf("\n");
}

int main(int argc, char** argv)
{
	int my_id, num_procs, prev_id, next_id;
	int is_master;
	int m, n;

	// Init MPI and process id
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

	is_master = my_id == 0;
	prev_id = is_master ? num_procs - 1 : my_id - 1;
	next_id = (my_id + 1) % num_procs;

	if(num_procs % 2 != 1)
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

	// Generate matrix

	
_exit:
	MPI_Finalize();
	return 0;
}
