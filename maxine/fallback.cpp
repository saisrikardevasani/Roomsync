// CPU fallback diagnostics — emitted when Maxine GPU init fails.
// Exists as a separate TU so the build system can swap it out for a
// Maxine-stub that logs to telemetry in production.

#include <cstdio>

void roomsync_maxine_fallback_notify(const char* reason) {
    std::fprintf(stderr,
        "[RoomSync] Maxine GPU path unavailable (%s). "
        "Running CPU pipeline. Will retry in %ds.\n",
        reason, 30);
}
