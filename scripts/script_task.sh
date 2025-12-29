THREADS=(1 2 4 8 16 32)

echo "-------"
echo "  GCC  "
echo "-------"
gcc -fopenmp heat-3d.c -o heat_task_gcc

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  # Убираем привязку
  unset OMP_PROC_BIND
  unset OMP_PLACES
  ./heat_task_gcc
done
