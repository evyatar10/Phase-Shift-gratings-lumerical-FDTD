/* Stub for Intel OpenMP 2021 libiomp5.so on RHEL 8 / Rocky 8 (glibc 2.28+).
 *
 * libiomp5.so bundled with Lumerical 2021R2.5 was compiled for glibc 2.17
 * (RHEL 7). On RHEL 8/Rocky 8 (glibc 2.28+) it has BIND_NOW set and its
 * PLT entry for ompt_start_tool@@VERSION fails to resolve at load time when
 * no other library provides a strong global definition. This causes a fatal
 * "undefined symbol: ompt_start_tool" crash, which cascades into the
 * "Intel MKL FATAL ERROR: Cannot load libmkl_intel_lp64.so" message.
 *
 * This stub provides a strong global ompt_start_tool (version VERSION,
 * matching libiomp5.so's version script). Returning NULL disables the OpenMP
 * Tools Interface (OMPT), which is harmless for FDTD simulations. */

void *ompt_start_tool(unsigned int omp_version, const char *runtime_version)
{
    (void)omp_version;
    (void)runtime_version;
    return 0;
}
