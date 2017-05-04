inner_product: InnerProducts/inner_product.c
	mpicc -lm -o $@ $^

clean:
	rm inner_product
