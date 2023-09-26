nvcc --cubin -O3 --use_fast_math -arch sm_70 -o reconstruct_sm_70.cubin reconstruct.cu
nvcc --cubin -O3 --use_fast_math -arch sm_75 -o reconstruct_sm_75.cubin reconstruct.cu
nvcc --cubin -O3 --use_fast_math -arch sm_80 -o reconstruct_sm_80.cubin reconstruct.cu
nvcc --cubin -O3 --use_fast_math -arch sm_70 -o kick_and_drift_sm_70.cubin kick_and_drift.cu
nvcc --cubin -O3 --use_fast_math -arch sm_75 -o kick_and_drift_sm_75.cubin kick_and_drift.cu
nvcc --cubin -O3 --use_fast_math -arch sm_80 -o kick_and_drift_sm_80.cubin kick_and_drift.cu