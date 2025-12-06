#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

// CUDA kernel for particle mover
__global__ void mover_PC_kernel(
    FPpart* x, FPpart* y, FPpart* z,
    FPpart* u, FPpart* v, FPpart* w,
    FPinterp* q,
    FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat,
    FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat,
    FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat,
    int nxn, int nyn, int nzn,
    FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
    int NiterMover, long nop,
    FPfield xStart, FPfield yStart, FPfield zStart,
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    FPpart Lx, FPpart Ly, FPpart Lz,
    bool PERIODICX, bool PERIODICY, bool PERIODICZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= nop) return;
    
    // Local field variables
    FPfield Exl, Eyl, Ezl, Bxl, Byl, Bzl;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    int ix, iy, iz;
    
    // Intermediate variables
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // Store initial position
    xptilde = x[i];
    yptilde = y[i];
    zptilde = z[i];
    
    // Iterative mover
    for(int innter = 0; innter < NiterMover; innter++) {
        // Calculate cell indices
        ix = 2 + int((x[i] - xStart) * invdx);
        iy = 2 + int((y[i] - yStart) * invdy);
        iz = 2 + int((z[i] - zStart) * invdz);
        
        // Calculate weights
        xi[0]   = x[i] - XN_flat[(ix-1)*nyn*nzn + iy*nzn + iz];
        eta[0]  = y[i] - YN_flat[ix*nyn*nzn + (iy-1)*nzn + iz];
        zeta[0] = z[i] - ZN_flat[ix*nyn*nzn + iy*nzn + (iz-1)];
        xi[1]   = XN_flat[ix*nyn*nzn + iy*nzn + iz] - x[i];
        eta[1]  = YN_flat[ix*nyn*nzn + iy*nzn + iz] - y[i];
        zeta[1] = ZN_flat[ix*nyn*nzn + iy*nzn + iz] - z[i];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;
        
        // Initialize local fields
        Exl = Eyl = Ezl = Bxl = Byl = Bzl = 0.0;
        
        // Interpolate fields from grid to particle
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for(int kk = 0; kk < 2; kk++) {
                    int idx = (ix-ii)*nyn*nzn + (iy-jj)*nzn + (iz-kk);
                    Exl += weight[ii][jj][kk] * Ex_flat[idx];
                    Eyl += weight[ii][jj][kk] * Ey_flat[idx];
                    Ezl += weight[ii][jj][kk] * Ez_flat[idx];
                    Bxl += weight[ii][jj][kk] * Bxn_flat[idx];
                    Byl += weight[ii][jj][kk] * Byn_flat[idx];
                    Bzl += weight[ii][jj][kk] * Bzn_flat[idx];
                }
        
        // Boris mover - solve velocity and position
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl + Byl*Byl + Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        
        ut = u[i] + qomdt2*Exl;
        vt = v[i] + qomdt2*Eyl;
        wt = w[i] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        
        uptilde = (ut + qomdt2*(vt*Bzl - wt*Byl + qomdt2*udotb*Bxl)) * denom;
        vptilde = (vt + qomdt2*(wt*Bxl - ut*Bzl + qomdt2*udotb*Byl)) * denom;
        wptilde = (wt + qomdt2*(ut*Byl - vt*Bxl + qomdt2*udotb*Bzl)) * denom;
        
        // Update position for next iteration
        x[i] = xptilde + uptilde*dto2;
        y[i] = yptilde + vptilde*dto2;
        z[i] = zptilde + wptilde*dto2;
    }
    
    // Final velocity update
    u[i] = 2.0*uptilde - u[i];
    v[i] = 2.0*vptilde - v[i];
    w[i] = 2.0*wptilde - w[i];
    
    // Final position update
    x[i] = xptilde + uptilde*dt_sub_cycling;
    y[i] = yptilde + vptilde*dt_sub_cycling;
    z[i] = zptilde + wptilde*dt_sub_cycling;
    
    // Boundary conditions - X direction
    if (x[i] > Lx) {
        if (PERIODICX) {
            x[i] = x[i] - Lx;
        } else {
            u[i] = -u[i];
            x[i] = 2*Lx - x[i];
        }
    }
    if (x[i] < 0) {
        if (PERIODICX) {
            x[i] = x[i] + Lx;
        } else {
            u[i] = -u[i];
            x[i] = -x[i];
        }
    }
    
    // Boundary conditions - Y direction
    if (y[i] > Ly) {
        if (PERIODICY) {
            y[i] = y[i] - Ly;
        } else {
            v[i] = -v[i];
            y[i] = 2*Ly - y[i];
        }
    }
    if (y[i] < 0) {
        if (PERIODICY) {
            y[i] = y[i] + Ly;
        } else {
            v[i] = -v[i];
            y[i] = -y[i];
        }
    }
    
    // Boundary conditions - Z direction
    if (z[i] > Lz) {
        if (PERIODICZ) {
            z[i] = z[i] - Lz;
        } else {
            w[i] = -w[i];
            z[i] = 2*Lz - z[i];
        }
    }
    if (z[i] < 0) {
        if (PERIODICZ) {
            z[i] = z[i] + Lz;
        } else {
            w[i] = -w[i];
            z[i] = -z[i];
        }
    }
}

/** GPU particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    std::cout << "***  GPU MOVER with SUBCYCLING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    // Compute parameters
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = 0.5*dt_sub_cycling;
    FPpart qomdt2 = part->qom*dto2/param->c;
    
    long nop = part->nop;
    int nxn = grd->nxn;
    int nyn = grd->nyn;
    int nzn = grd->nzn;
    
    // Device pointers for particle data
    FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
    FPinterp *d_q;
    
    // Device pointers for field data
    FPfield *d_Ex, *d_Ey, *d_Ez;
    FPfield *d_Bxn, *d_Byn, *d_Bzn;
    FPfield *d_XN, *d_YN, *d_ZN;
    
    // Allocate particle arrays on GPU
    cudaMalloc(&d_x, nop * sizeof(FPpart));
    cudaMalloc(&d_y, nop * sizeof(FPpart));
    cudaMalloc(&d_z, nop * sizeof(FPpart));
    cudaMalloc(&d_u, nop * sizeof(FPpart));
    cudaMalloc(&d_v, nop * sizeof(FPpart));
    cudaMalloc(&d_w, nop * sizeof(FPpart));
    cudaMalloc(&d_q, nop * sizeof(FPinterp));
    
    // Allocate field arrays on GPU
    size_t field_size = nxn * nyn * nzn * sizeof(FPfield);
    cudaMalloc(&d_Ex, field_size);
    cudaMalloc(&d_Ey, field_size);
    cudaMalloc(&d_Ez, field_size);
    cudaMalloc(&d_Bxn, field_size);
    cudaMalloc(&d_Byn, field_size);
    cudaMalloc(&d_Bzn, field_size);
    cudaMalloc(&d_XN, field_size);
    cudaMalloc(&d_YN, field_size);
    cudaMalloc(&d_ZN, field_size);
    
    // Copy particle data to GPU
    cudaMemcpy(d_x, part->x, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, part->y, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, part->z, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, part->u, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, part->v, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, part->w, nop * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, part->q, nop * sizeof(FPinterp), cudaMemcpyHostToDevice);
    
    // Flatten and copy 3D field arrays to GPU
    FPfield* Ex_flat = new FPfield[nxn * nyn * nzn];
    FPfield* Ey_flat = new FPfield[nxn * nyn * nzn];
    FPfield* Ez_flat = new FPfield[nxn * nyn * nzn];
    FPfield* Bxn_flat = new FPfield[nxn * nyn * nzn];
    FPfield* Byn_flat = new FPfield[nxn * nyn * nzn];
    FPfield* Bzn_flat = new FPfield[nxn * nyn * nzn];
    FPfield* XN_flat = new FPfield[nxn * nyn * nzn];
    FPfield* YN_flat = new FPfield[nxn * nyn * nzn];
    FPfield* ZN_flat = new FPfield[nxn * nyn * nzn];
    
    for(int i = 0; i < nxn; i++)
        for(int j = 0; j < nyn; j++)
            for(int k = 0; k < nzn; k++) {
                int idx = i*nyn*nzn + j*nzn + k;
                Ex_flat[idx] = field->Ex[i][j][k];
                Ey_flat[idx] = field->Ey[i][j][k];
                Ez_flat[idx] = field->Ez[i][j][k];
                Bxn_flat[idx] = field->Bxn[i][j][k];
                Byn_flat[idx] = field->Byn[i][j][k];
                Bzn_flat[idx] = field->Bzn[i][j][k];
                XN_flat[idx] = grd->XN[i][j][k];
                YN_flat[idx] = grd->YN[i][j][k];
                ZN_flat[idx] = grd->ZN[i][j][k];
            }
    
    cudaMemcpy(d_Ex, Ex_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ey, Ey_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ez, Ez_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bxn, Bxn_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Byn, Byn_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bzn, Bzn_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_XN, XN_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_YN, YN_flat, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZN, ZN_flat, field_size, cudaMemcpyHostToDevice);
    
    // Launch kernel for each subcycle
    int blockSize = 256;
    int gridSize = (nop + blockSize - 1) / blockSize;
    
    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
        mover_PC_kernel<<<gridSize, blockSize>>>(
            d_x, d_y, d_z, d_u, d_v, d_w, d_q,
            d_Ex, d_Ey, d_Ez, d_Bxn, d_Byn, d_Bzn,
            d_XN, d_YN, d_ZN, nxn, nyn, nzn,
            dt_sub_cycling, dto2, qomdt2, part->NiterMover, nop,
            grd->xStart, grd->yStart, grd->zStart,
            grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
            grd->Lx, grd->Ly, grd->Lz,
            param->PERIODICX, param->PERIODICY, param->PERIODICZ
        );
        cudaDeviceSynchronize();
    }
    
    // Copy results back to host
    cudaMemcpy(part->x, d_x, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, d_y, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, d_z, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, d_u, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, d_v, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, d_w, nop * sizeof(FPpart), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w); cudaFree(d_q);
    cudaFree(d_Ex); cudaFree(d_Ey); cudaFree(d_Ez);
    cudaFree(d_Bxn); cudaFree(d_Byn); cudaFree(d_Bzn);
    cudaFree(d_XN); cudaFree(d_YN); cudaFree(d_ZN);
    
    // Free temporary host arrays
    delete[] Ex_flat; delete[] Ey_flat; delete[] Ez_flat;
    delete[] Bxn_flat; delete[] Byn_flat; delete[] Bzn_flat;
    delete[] XN_flat; delete[] YN_flat; delete[] ZN_flat;
    
    return 0;
}



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
