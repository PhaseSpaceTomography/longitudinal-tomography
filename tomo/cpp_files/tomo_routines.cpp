
// Back projection using flattened arrays
extern "C" double * back_project(double *  weights,					// inn/out
							 	 const int ** __restrict__ flat_points,	// inn
							 	 const double *  flat_profiles,			// inn
							 	 const int nparts, const int nprofs){	// inn
#pragma omp parallel for	
	for (int i = 0; i < nparts; i++)
		for (int j = 0; j < nprofs; j++)
			 weights[i] += flat_profiles[flat_points[i][j]];
}	

// Projections using flattened arrays
extern "C" double * project(double *  flat_rec,					// inn/out
							const int ** __restrict__ flat_points,	// inn
							const double *  __restrict__ weights,	// inn
							const int nparts, const int nprofs){	// inn
	for (int i = 0; i < nparts; i++)
		for (int j = 0; j < nprofs; j++)
			flat_rec[flat_points[i][j]] += weights[i];
}