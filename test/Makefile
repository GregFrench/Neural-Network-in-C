CC = gcc
CFLAGS = -ansi -Wall

solution_test: solution_test.o ../solution.o
	$(CC) $(CFLAGS) -o solution_test solution_test.o ../solution.o -lm

solution_test.o: solution_test.c
	$(CC) $(CFLAGS) -c solution_test.c

all: solution_test

clean:
	rm solution_test solution_test.o