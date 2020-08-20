CC = gcc
CFLAGS = -ansi -Wall

network_test: network_test.o ../network.o ../misc_functions.o
	$(CC) $(CFLAGS) -o network_test network_test.o ../network.o ../misc_functions.o -lm

network_test.o: network_test.c
	$(CC) $(CFLAGS) -c network_test.c

misc_functions_test: misc_functions_test.o ../misc_functions.o
	$(CC) $(CFLAGS) -o misc_functions_test misc_functions_test.o ../misc_functions.o -lm

misc_functions_test.o: misc_functions_test.c
	$(CC) $(CFLAGS) -c misc_functions_test.c

all: network_test misc_functions_test

clean:
	rm network_test network_test.o misc_functions_test misc_functions_test.o