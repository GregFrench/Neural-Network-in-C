#define EPSILON 0.001
#define FALSE 0
#define TRUE 1
#define PI 3.14159265358979323846

double box_muller();
double ** create_double_matrix(int m, int n);
double ** dot2D(double ** matrix, double ** matrix2, int m, int n);
double ** dotMatrixByScalar(double ** matrix, int scalar);
int is_approx(double x, double y);
double ** randn(int x, int y);
double rand_double(double a, double b);
int * set_sizes(int sizes[], int size);
double *** set_biases(int sizes[], int size);
double *** set_weights(int sizes[], int size);
void shuffle(double **** array, size_t n);
double sigmoid(double z);
double ** sigmoid_prime(double ** z, int m);
double ** transpose(double ** matrix, int m, int n);
