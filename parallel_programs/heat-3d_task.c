/* Include benchmark-specific header. */
#include "heat-3d.h"
double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n,
   double A[ n][n][n],
   double B[ n][n][n])
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (double) (i + j + (n-k))* 10 / (n);
}

static
void print_array(int n,
   double A[ n][n][n])
{
  int i, j, k;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(stderr, "\n");
         fprintf(stderr, "%0.2lf ", A[i][j][k]);
      }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_heat_3d(int tsteps,
        int n,
        double A[n][n][n],
        double B[n][n][n])
{
    int t;
    // Тайлинг по (i, j): одна задача считает блок BS×BS по двум координатам.
    // Это уменьшает накладные расходы по сравнению с "задача на каждую (i,j)".
    const int BS = 16;
    // Создаём команду потоков один раз на всю функцию (дорого пересоздавать потоки на каждом t).
    #pragma omp parallel
    {
        // Только один поток порождает задачи, остальные в это время будут выполнять их.
        // Это предотвращает дублирование задач (если бы task создавали все потоки).
        #pragma omp single
        {
            for (t = 1; t <= tsteps; ++t) {
                // taskgroup = "барьер" для задач:
                // гарантируем, что ВСЕ задачи A->B завершены, прежде чем начнём B->A.
                // Без этого возможна гонка: часть потоков начнёт читать B, пока другая часть ещё пишет B.
                #pragma omp taskgroup
                {
                    int ii, jj;
                    for (ii = 1; ii < n - 1; ii += BS) {
                        for (jj = 1; jj < n - 1; jj += BS) {

                            int i_end = ii + BS;
                            int j_end = jj + BS;
                            if (i_end > n - 1) i_end = n - 1;
                            if (j_end > n - 1) j_end = n - 1;
                            // firstprivate фиксирует значения ii/jj/i_end/j_end на момент создания задачи.
                            // Иначе задача могла бы увидеть уже изменившиеся ii/jj из внешних циклов (классическая ошибка с task).
                            #pragma omp task firstprivate(ii, jj, i_end, j_end) shared(A, B, n)
                            {
                                int i, j, k;
                                for (i = ii; i < i_end; ++i) {
                                    for (j = jj; j < j_end; ++j) {
                                        // Внутренний цикл по k оставляем самым глубоким,
                                        // потому что A[i][j][k] в памяти хранится подряд по k (лучше кэш/векторизация).
                                        for (k = 1; k < n - 1; ++k) {
                                            B[i][j][k] =
                                                0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                              + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                              + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                              + A[i][j][k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Аналогично: ждём завершения всех задач B->A перед переходом к следующему шагу времени.
                #pragma omp taskgroup
                {
                    int ii, jj;
                    for (ii = 1; ii < n - 1; ii += BS) {
                        for (jj = 1; jj < n - 1; jj += BS) {

                            int i_end = ii + BS;
                            int j_end = jj + BS;
                            if (i_end > n - 1) i_end = n - 1;
                            if (j_end > n - 1) j_end = n - 1;

                            #pragma omp task firstprivate(ii, jj, i_end, j_end) shared(A, B, n)
                            {
                                int i, j, k;
                                for (i = ii; i < i_end; ++i) {
                                    for (j = jj; j < j_end; ++j) {
                                        for (k = 1; k < n - 1; ++k) {
                                            A[i][j][k] =
                                                0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                              + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                              + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                              + B[i][j][k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}




int main(int argc, char** argv)
{

  int n = N;
  int tsteps = TSTEPS;

  double (*A)[n][n][n]; A = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
  double (*B)[n][n][n]; B = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));

  init_array (n, *A, *B);

  bench_timer_start();

  kernel_heat_3d (tsteps, n, *A, *B);

  bench_timer_stop();
  bench_timer_print();

  if (argc > 42 && ! strcmp(argv[0], "")) print_array(n, *A);

  free((void*)A);
  free((void*)B);

  return 0;
}


