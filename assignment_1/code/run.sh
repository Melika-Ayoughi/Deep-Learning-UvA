#!/bin/sh
#PBS -lwalltime=24:00:00 #PBS -lnodes=2:ppn=12 #PBS -lmem=250GB
module load eb
module load Python/2.7.14-foss-2017b module load cuDNN/7.0.5-CUDA-9.0.176 module load OpenMPI/2.1.1-GCC-6.4.0-2.28 module load NCCL
export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA- 9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH
rm machine_file
cat $PBS_NODEFILE > machine_file
sed 's/$/ slots=4/' machine_file > machinefile uniq machinefile | cat > machine_file
rm machinefile
OMP_NUM_THREADS=12 HOROVOD_FUSION_THRESHOLD=33554432 mpirun -np 8 --map-by ppr:4:node -hostfile machine_file -x NCCL_P2P_DISABLE=0 --rank-by core --display-map -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS python ~/horovod/examples/tensorflow_mnist.py --variable_update horovod --num_intra_threads 4 --num_inter_threads 3 --horovod_device cpu
