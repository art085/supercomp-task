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
    int t, i, j, k;


    // Создаём команду потоков один раз (pragma omp parallel снаружи t-цикла),
    // чтобы не платить накладные расходы на создание/уничтожение потоков на каждом шаге времени.
    // Это особенно важно при больших tsteps.
    #pragma omp parallel private(t, i, j, k)
    {
        for (t = 1; t <= tsteps; ++t) {
            // omp for делит итерации цикла между потоками.
            // collapse(2) "склеивает" два внешних цикла (i и j) в один большой диапазон итераций,
            // чтобы:
            //   1) было больше независимых итераций (лучше балансировка),
            //   2) проще равномерно раздать работу потокам.
            // Внутренний цикл k оставляем внутри одной итерации (i,j) — так доступ по памяти лучше (contiguous по k).
            #pragma omp for collapse(2) schedule(static)
            for (i = 1; i < n - 1; ++i) {
                for (j = 1; j < n - 1; ++j) {
                    for (k = 1; k < n - 1; ++k) {
                        B[i][j][k] = 0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                     + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                     + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                     + A[i][j][k];
                    }
                }
            }

                                     + A[i][j][k];
                    }
                }
            }

            // schedule(static) раздаёт каждому потоку заранее фиксированный блок итераций.
            // Это минимизирует overhead планировщика и обычно хорошо подходит для stencil,
            // потому что все итерации примерно одинаковой "тяжести" (равномерная нагрузка).
            // Если бы нагрузка была неравномерной, тогда можно было бы пробовать dynamic/guided.
            #pragma omp for collapse(2) schedule(static)
            for (i = 1; i < n - 1; ++i) {
                for (j = 1; j < n - 1; ++j) {
                    for (k = 1; k < n - 1; ++k) {
                        A[i][j][k] = 0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                     + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                     + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                     + B[i][j][k];
                    }
                }
            }
        }
    }
}

// Ключевой момент корректности:
//   1) В свипе A->B все потоки ЧИТАЮТ из A и ПИШУТ в разные элементы B.
//      Это безопасно: нет конфликтов записи, A используется как read-only.
//   2) В конце omp for есть НЕЯВНЫЙ барьер (если не стоит nowait).
//      Поэтому второй свип (B->A) начинается только после того,
//      как ВСЕ потоки полностью завершили вычисление B.
//      Иначе была бы гонка: часть потоков начала бы читать B, пока другие ещё пишут B.
//
// Аналогично для свипа B->A: он полностью завершается перед переходом к следующему t.

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

