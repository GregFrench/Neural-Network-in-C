CC = gcc
CFLAGS = -ansi -Wall

network_test: network_test.o ../network.o ../misc_functions.o test.o
	$(CC) $(CFLAGS) -o network_test network_test.o ../network.o ../misc_functions.o test.o -lm

network_test.o: network_test.c
	$(CC) $(CFLAGS) -c network_test.c

misc_functions_test: misc_functions_test.o ../misc_functions.o ../network.o test.o
	$(CC) $(CFLAGS) -o misc_functions_test misc_functions_test.o ../misc_functions.o ../network.o test.o -lm

misc_functions_test.o: misc_functions_test.c
	$(CC) $(CFLAGS) -c misc_functions_test.c

test.o: test.c
	$(CC) $(CFLAGS) -c test.c

all: network_test misc_functions_test

clean:
	rm network_test network_test.o misc_functions_test misc_functions_test.o test.o