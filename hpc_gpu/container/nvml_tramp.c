/*
 * nvml_tramp.c — NVML compatibility shim for Lumerical FDTD on Athena R470 driver.
 *
 * Root cause: Athena DGX nodes run NVIDIA driver R470 (circa 2021). Lumerical 2026 R1's
 * GPU plugin (liblumcudaquery.so) imports two NVML symbols that were added in R510:
 *
 *   nvmlDeviceGetPcieLinkMaxSpeed  — queries PCIe slot's rated max speed
 *   nvmlDeviceGetBusType           — queries bus type (PCIe / AGP / ...)
 *
 * When the dynamic linker loads liblumcudaquery.so it fails with a fatal unresolved-symbol
 * error before the FDTD engine reaches any GPU-init code. The engine then aborts, so no
 * GPU work is done at all.
 *
 * Fix: this shared library is named libnvidia-ml.so.1 (matching the real NVML soname) and
 * placed first in LD_LIBRARY_PATH so it intercepts all NVML lookups.
 *   - For the two R470-missing symbols, it provides stubs that return NVML_ERROR_NOT_SUPPORTED.
 *   - All other NVML symbols are delegated at call time to the actual R470 NVML loaded from
 *     /.singularity.d/libs/libnvidia-ml.so.1 (Apptainer's --nv injection path).
 *
 * Build on Athena login node (no headers needed — types defined inline):
 *   mkdir -p ~/nvml_tramp
 *   gcc -O2 -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 \
 *       -o ~/nvml_tramp/libnvidia-ml.so.1 nvml_tramp.c -ldl
 *
 * Verify stubs are exported:
 *   nm -D ~/nvml_tramp/libnvidia-ml.so.1 | grep -E 'PcieLinkMaxSpeed|BusType'
 *   # Must show:  T nvmlDeviceGetBusType   T nvmlDeviceGetPcieLinkMaxSpeed
 *
 * Add to apptainer exec in job script:
 *   --bind "${HOME}/nvml_tramp:/nvml_tramp"
 * And inside the container's bash -c "..." block:
 *   export LD_LIBRARY_PATH="/nvml_tramp:$LD_LIBRARY_PATH"
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Minimal NVML type definitions (ABI-stable; no nvml.h needed) ──────────── */

typedef enum {
    NVML_SUCCESS                   = 0,
    NVML_ERROR_UNINITIALIZED       = 1,
    NVML_ERROR_INVALID_ARGUMENT    = 2,
    NVML_ERROR_NOT_SUPPORTED       = 3,
    NVML_ERROR_NO_PERMISSION       = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND           = 6,
    NVML_ERROR_DRIVER_NOT_LOADED   = 9,
    NVML_ERROR_FUNCTION_NOT_FOUND  = 13,
    NVML_ERROR_UNKNOWN             = 999,
} nvmlReturn_t;

typedef void *nvmlDevice_t;  /* opaque handle */

typedef enum {
    NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef enum {
    NVML_PCIE_UTIL_TX_BYTES = 0,
    NVML_PCIE_UTIL_RX_BYTES = 1,
} nvmlPcieUtilCounter_t;

typedef enum {
    NVML_P2P_CAPS_INDEX_READ    = 0,
    NVML_P2P_CAPS_INDEX_WRITE   = 1,
    NVML_P2P_CAPS_INDEX_NVLINK  = 2,
    NVML_P2P_CAPS_INDEX_ATOMICS = 3,
    NVML_P2P_CAPS_INDEX_PROP    = 4,
    NVML_P2P_CAPS_INDEX_LOOPBACK = 5,
    NVML_P2P_CAPS_INDEX_UNKNOWN  = 6,
} nvmlGpuP2PCapsIndex_t;

typedef enum {
    NVML_P2P_STATUS_OK                          = 0,
    NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED        = 1,
    NVML_P2P_STATUS_GPU_NOT_SUPPORTED           = 2,
    NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED  = 3,
    NVML_P2P_STATUS_DISABLED_BY_REGKEY          = 4,
    NVML_P2P_STATUS_NOT_SUPPORTED               = 5,
    NVML_P2P_STATUS_UNKNOWN                     = 6,
} nvmlGpuP2PStatus_t;

typedef enum {
    NVML_NVLINK_CAP_P2P_SUPPORTED    = 0,
    NVML_NVLINK_CAP_SYSMEM_ACCESS    = 1,
    NVML_NVLINK_CAP_P2P_ATOMICS      = 2,
    NVML_NVLINK_CAP_SYSMEM_ATOMICS   = 3,
    NVML_NVLINK_CAP_SLI_BRIDGE       = 4,
    NVML_NVLINK_CAP_VALID            = 5,
} nvmlNvLinkCapability_t;

typedef enum {
    NVML_NVLINK_DEVICE_TYPE_GPU     = 0x00,
    NVML_NVLINK_DEVICE_TYPE_IBMNPU  = 0x01,
    NVML_NVLINK_DEVICE_TYPE_SWITCH  = 0x02,
    NVML_NVLINK_DEVICE_TYPE_UNKNOWN = 0xFF,
} nvmlIntNvLinkDeviceType_t;

/* Bus type — added in R510; used in our stub */
typedef enum {
    NVML_BUS_TYPE_UNKNOWN = 0,
    NVML_BUS_TYPE_PCI     = 1,
    NVML_BUS_TYPE_PCIE    = 2,
    NVML_BUS_TYPE_FPCI    = 3,
    NVML_BUS_TYPE_AGP     = 4,
} nvmlBusType_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

/* ── Real NVML handle ───────────────────────────────────────────────────────── */

static void *real_nvml = NULL;

/* Apptainer --nv always injects host NVIDIA libs into /.singularity.d/libs/.
   Fall back to standard system paths if the container layout differs.        */
static const char * const real_paths[] = {
    "/.singularity.d/libs/libnvidia-ml.so.1",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib/libnvidia-ml.so.1",
    NULL
};

__attribute__((constructor)) static void tramp_load_real(void)
{
    for (int i = 0; real_paths[i]; i++) {
        real_nvml = dlopen(real_paths[i], RTLD_LAZY | RTLD_LOCAL);
        if (real_nvml) {
            fprintf(stderr, "[nvml_tramp] loaded real NVML from %s\n", real_paths[i]);
            return;
        }
    }
    fprintf(stderr, "[nvml_tramp] WARNING: could not find real libnvidia-ml.so.1\n");
}

/* dlsym wrapper — returns NULL without crashing if real_nvml not loaded */
static void *rsym(const char *name)
{
    if (!real_nvml) return NULL;
    return dlsym(real_nvml, name);
}

/* ── Delegate / Stub macros ─────────────────────────────────────────────────── */

/* Delegate: call the same-named function in the real R470 NVML.
   If not found (shouldn't happen for known R470 symbols), return NOT_SUPPORTED. */
#define D0(R, N) \
    R N(void) { \
        static R (*fn)(void); \
        if (!fn) fn = (R(*)(void))rsym(#N); \
        return fn ? fn() : (R)NVML_ERROR_NOT_SUPPORTED; \
    }

#define D1(R, N, T1, a1) \
    R N(T1 a1) { \
        static R (*fn)(T1); \
        if (!fn) fn = (R(*)(T1))rsym(#N); \
        return fn ? fn(a1) : (R)NVML_ERROR_NOT_SUPPORTED; \
    }

#define D2(R, N, T1, a1, T2, a2) \
    R N(T1 a1, T2 a2) { \
        static R (*fn)(T1, T2); \
        if (!fn) fn = (R(*)(T1, T2))rsym(#N); \
        return fn ? fn(a1, a2) : (R)NVML_ERROR_NOT_SUPPORTED; \
    }

#define D3(R, N, T1, a1, T2, a2, T3, a3) \
    R N(T1 a1, T2 a2, T3 a3) { \
        static R (*fn)(T1, T2, T3); \
        if (!fn) fn = (R(*)(T1, T2, T3))rsym(#N); \
        return fn ? fn(a1, a2, a3) : (R)NVML_ERROR_NOT_SUPPORTED; \
    }

#define D4(R, N, T1, a1, T2, a2, T3, a3, T4, a4) \
    R N(T1 a1, T2 a2, T3 a3, T4 a4) { \
        static R (*fn)(T1, T2, T3, T4); \
        if (!fn) fn = (R(*)(T1, T2, T3, T4))rsym(#N); \
        return fn ? fn(a1, a2, a3, a4) : (R)NVML_ERROR_NOT_SUPPORTED; \
    }

/* Stubs — symbol absent from R470 NVML; return NOT_SUPPORTED so the caller
   can degrade gracefully rather than getting a fatal linker error.          */
#define S1(R, N, T1, a1) \
    R N(T1 a1) { (void)(a1); return NVML_ERROR_NOT_SUPPORTED; }

#define S2(R, N, T1, a1, T2, a2) \
    R N(T1 a1, T2 a2) { (void)(a1); (void)(a2); return NVML_ERROR_NOT_SUPPORTED; }

#define S3(R, N, T1, a1, T2, a2, T3, a3) \
    R N(T1 a1, T2 a2, T3 a3) { \
        (void)(a1); (void)(a2); (void)(a3); \
        return NVML_ERROR_NOT_SUPPORTED; \
    }


/* ═══════════════════════════════════════════════════════════════════════════
   INIT / SHUTDOWN
   ═══════════════════════════════════════════════════════════════════════════ */

D0(nvmlReturn_t, nvmlInit)
D0(nvmlReturn_t, nvmlInit_v2)
D0(nvmlReturn_t, nvmlShutdown)

/* nvmlErrorString returns const char*, not nvmlReturn_t — handle manually */
const char *nvmlErrorString(nvmlReturn_t result)
{
    static const char *(*fn)(nvmlReturn_t);
    if (!fn) fn = (const char *(*)(nvmlReturn_t))rsym("nvmlErrorString");
    return fn ? fn(result) : "(nvml_tramp: real NVML not loaded)";
}

/* ═══════════════════════════════════════════════════════════════════════════
   SYSTEM QUERIES
   ═══════════════════════════════════════════════════════════════════════════ */

D2(nvmlReturn_t, nvmlSystemGetDriverVersion,        char *,  buf, unsigned int, len)
D2(nvmlReturn_t, nvmlSystemGetNVMLVersion,          char *,  buf, unsigned int, len)
D1(nvmlReturn_t, nvmlSystemGetCudaDriverVersion,    int *,   ver)
D2(nvmlReturn_t, nvmlSystemGetCudaDriverVersion_v2, int *,   major, int *, minor)
D3(nvmlReturn_t, nvmlSystemGetProcessName,
   unsigned int, pid, char *, name, unsigned int, length)

/* ═══════════════════════════════════════════════════════════════════════════
   DEVICE ENUMERATION
   ═══════════════════════════════════════════════════════════════════════════ */

D1(nvmlReturn_t, nvmlDeviceGetCount,            unsigned int *, cnt)
D1(nvmlReturn_t, nvmlDeviceGetCount_v2,         unsigned int *, cnt)
D2(nvmlReturn_t, nvmlDeviceGetHandleByIndex,    unsigned int,   idx,  nvmlDevice_t *, dev)
D2(nvmlReturn_t, nvmlDeviceGetHandleByIndex_v2, unsigned int,   idx,  nvmlDevice_t *, dev)
D2(nvmlReturn_t, nvmlDeviceGetHandleByUUID,     const char *,   uuid, nvmlDevice_t *, dev)
D2(nvmlReturn_t, nvmlDeviceGetHandleBySerial,   const char *,   sn,   nvmlDevice_t *, dev)
D2(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId,    const char *, id,  nvmlDevice_t *, dev)
D2(nvmlReturn_t, nvmlDeviceGetHandleByPciBusId_v2, const char *, id,  nvmlDevice_t *, dev)
D2(nvmlReturn_t, nvmlDeviceGetHandleByGpuInstanceId, unsigned int, id, nvmlDevice_t *, dev)

/* ═══════════════════════════════════════════════════════════════════════════
   DEVICE IDENTITY
   ═══════════════════════════════════════════════════════════════════════════ */

D3(nvmlReturn_t, nvmlDeviceGetName,    nvmlDevice_t, dev, char *, buf, unsigned int, len)
D3(nvmlReturn_t, nvmlDeviceGetUUID,    nvmlDevice_t, dev, char *, buf, unsigned int, len)
D3(nvmlReturn_t, nvmlDeviceGetSerial,  nvmlDevice_t, dev, char *, buf, unsigned int, len)
D2(nvmlReturn_t, nvmlDeviceGetMinorNumber,   nvmlDevice_t, dev, unsigned int *, minor)
D2(nvmlReturn_t, nvmlDeviceGetIndex,         nvmlDevice_t, dev, unsigned int *, index)

/* PCI info — Lumerical calls the _v3 variant; provide all three for safety */
D2(nvmlReturn_t, nvmlDeviceGetPciInfo,    nvmlDevice_t, dev, void *, pci)
D2(nvmlReturn_t, nvmlDeviceGetPciInfo_v2, nvmlDevice_t, dev, void *, pci)
D2(nvmlReturn_t, nvmlDeviceGetPciInfo_v3, nvmlDevice_t, dev, void *, pci)

/* Compute capability — signed ints per NVML spec */
D3(nvmlReturn_t, nvmlDeviceGetCudaComputeCapability, nvmlDevice_t, dev, int *, major, int *, minor)

/* ═══════════════════════════════════════════════════════════════════════════
   MEMORY AND PCIe TOPOLOGY
   ═══════════════════════════════════════════════════════════════════════════ */

D2(nvmlReturn_t, nvmlDeviceGetMemoryInfo,    nvmlDevice_t, dev, void *, mem)
D2(nvmlReturn_t, nvmlDeviceGetMemoryInfo_v2, nvmlDevice_t, dev, void *, mem)

D2(nvmlReturn_t, nvmlDeviceGetMaxPcieLinkWidth,       nvmlDevice_t, dev, unsigned int *, width)
D2(nvmlReturn_t, nvmlDeviceGetCurrPcieLinkWidth,      nvmlDevice_t, dev, unsigned int *, width)
D2(nvmlReturn_t, nvmlDeviceGetMaxPcieLinkGeneration,  nvmlDevice_t, dev, unsigned int *, gen)
D2(nvmlReturn_t, nvmlDeviceGetCurrPcieLinkGeneration, nvmlDevice_t, dev, unsigned int *, gen)
D3(nvmlReturn_t, nvmlDeviceGetPcieThroughput,         nvmlDevice_t, dev,
   nvmlPcieUtilCounter_t, counter, unsigned int *, val)

/* ── STUBS: symbols confirmed MISSING from Athena's R470 NVML ─────────────────
   Verified 2026-04-25 via nm on /.singularity.d/libs/libnvidia-ml.so.1 on n309.
   All three were added in R510+. Lumerical's GPU plugin (liblumcudaquery.so)
   imports them; without these stubs the dynamic linker aborts with a fatal
   unresolved-symbol error before the engine reaches any GPU init code.
   NVML_ERROR_NOT_SUPPORTED is the correct graceful-degradation return value. */

/* Added ~R510; queries PCIe slot's rated max speed. Used for topology logging. */
S2(nvmlReturn_t, nvmlDeviceGetPcieLinkMaxSpeed,
   nvmlDevice_t, dev, unsigned int *, maxSpeed)

/* Added ~R510; returns bus type (PCIe/AGP/...). All A100s are PCIe/FPCI. */
S2(nvmlReturn_t, nvmlDeviceGetBusType,
   nvmlDevice_t, dev, nvmlBusType_t *, type)

/* Added ~R510; queries GPU memory bus width in bits. R470 lacks this despite
   having nvmlDeviceGetMaxPcieLinkWidth (PCIe lanes, a different metric). */
S2(nvmlReturn_t, nvmlDeviceGetMemoryBusWidth,
   nvmlDevice_t, dev, unsigned int *, busWidth)

/* ═══════════════════════════════════════════════════════════════════════════
   THERMAL / POWER / UTILIZATION
   ═══════════════════════════════════════════════════════════════════════════ */

D3(nvmlReturn_t, nvmlDeviceGetTemperature,
   nvmlDevice_t, dev, nvmlTemperatureSensors_t, sensor, unsigned int *, temp)
D2(nvmlReturn_t, nvmlDeviceGetUtilizationRates,
   nvmlDevice_t, dev, nvmlUtilization_t *, util)
D2(nvmlReturn_t, nvmlDeviceGetPowerUsage,
   nvmlDevice_t, dev, unsigned int *, power)
D2(nvmlReturn_t, nvmlDeviceGetEnforcedPowerLimit,
   nvmlDevice_t, dev, unsigned int *, limit)
D2(nvmlReturn_t, nvmlDeviceGetPowerManagementLimit,
   nvmlDevice_t, dev, unsigned int *, limit)

/* ═══════════════════════════════════════════════════════════════════════════
   CLOCKS
   ═══════════════════════════════════════════════════════════════════════════ */

/* nvmlClockType_t is an enum (int-sized); using unsigned int is ABI-compatible */
D3(nvmlReturn_t, nvmlDeviceGetClockInfo,    nvmlDevice_t, dev, unsigned int, type, unsigned int *, clock)
D3(nvmlReturn_t, nvmlDeviceGetMaxClockInfo, nvmlDevice_t, dev, unsigned int, type, unsigned int *, clock)
D2(nvmlReturn_t, nvmlDeviceGetCurrentClocksThrottleReasons,
   nvmlDevice_t, dev, unsigned long long *, reasons)

/* ═══════════════════════════════════════════════════════════════════════════
   RUNNING PROCESSES
   ═══════════════════════════════════════════════════════════════════════════ */

D3(nvmlReturn_t, nvmlDeviceGetComputeRunningProcesses,
   nvmlDevice_t, dev, unsigned int *, infoCount, void *, infos)
D3(nvmlReturn_t, nvmlDeviceGetComputeRunningProcesses_v2,
   nvmlDevice_t, dev, unsigned int *, infoCount, void *, infos)
D3(nvmlReturn_t, nvmlDeviceGetComputeRunningProcesses_v3,
   nvmlDevice_t, dev, unsigned int *, infoCount, void *, infos)
D3(nvmlReturn_t, nvmlDeviceGetGraphicsRunningProcesses,
   nvmlDevice_t, dev, unsigned int *, infoCount, void *, infos)

/* ═══════════════════════════════════════════════════════════════════════════
   P2P / TOPOLOGY
   ═══════════════════════════════════════════════════════════════════════════ */

D4(nvmlReturn_t, nvmlDeviceGetP2PStatus,
   nvmlDevice_t, dev1, nvmlDevice_t, dev2,
   nvmlGpuP2PCapsIndex_t, capIndex, nvmlGpuP2PStatus_t *, p2pStatus)

D3(nvmlReturn_t, nvmlDeviceGetTopologyCommonAncestor,
   nvmlDevice_t, dev1, nvmlDevice_t, dev2, unsigned int *, pathInfo)

/* ═══════════════════════════════════════════════════════════════════════════
   NvLink (DGX A100 has NVLink 3.0 between GPUs)
   ═══════════════════════════════════════════════════════════════════════════ */

D3(nvmlReturn_t, nvmlDeviceGetNvLinkState,
   nvmlDevice_t, dev, unsigned int, link, unsigned int *, isActive)
D3(nvmlReturn_t, nvmlDeviceGetNvLinkVersion,
   nvmlDevice_t, dev, unsigned int, link, unsigned int *, version)
D4(nvmlReturn_t, nvmlDeviceGetNvLinkCapability,
   nvmlDevice_t, dev, unsigned int, link,
   nvmlNvLinkCapability_t, cap, unsigned int *, capResult)
D3(nvmlReturn_t, nvmlDeviceGetNvLinkRemoteDeviceType,
   nvmlDevice_t, dev, unsigned int, link, nvmlIntNvLinkDeviceType_t *, devType)
D3(nvmlReturn_t, nvmlDeviceGetNvLinkRemotePciInfo,
   nvmlDevice_t, dev, unsigned int, link, void *, pci)
D4(nvmlReturn_t, nvmlDeviceGetNvLinkErrorCounter,
   nvmlDevice_t, dev, unsigned int, link,
   unsigned int, counter, unsigned long long *, counterValue)

/* ═══════════════════════════════════════════════════════════════════════════
   MISC — additional entry points Lumerical may call
   ═══════════════════════════════════════════════════════════════════════════ */

D3(nvmlReturn_t, nvmlDeviceGetVbiosVersion,
   nvmlDevice_t, dev, char *, version, unsigned int, length)
D2(nvmlReturn_t, nvmlDeviceGetBoardId,       nvmlDevice_t, dev, unsigned int *, boardId)
D2(nvmlReturn_t, nvmlDeviceGetMultiGpuBoard, nvmlDevice_t, dev, unsigned int *, multiGpuBool)
D2(nvmlReturn_t, nvmlDeviceGetSupportedClocksThrottleReasons,
   nvmlDevice_t, dev, unsigned long long *, supportedClocksThrottleReasons)
D2(nvmlReturn_t, nvmlDeviceGetPerformanceState,
   nvmlDevice_t, dev, unsigned int *, pState)
D3(nvmlReturn_t, nvmlDeviceOnSameBoard,
   nvmlDevice_t, dev1, nvmlDevice_t, dev2, int *, onSameBoard)
