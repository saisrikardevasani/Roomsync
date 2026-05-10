cmake_minimum_required(VERSION 3.20)
project(roomsync VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_MAXINE "Build with NVIDIA Maxine GPU support" OFF)
option(WITH_PORTAUDIO "Build with PortAudio real-time audio I/O" ON)
option(WITH_OPUS "Build with Opus codec + RED FEC" ON)
option(BUILD_TESTS "Build test binaries" ON)

# ── Find system dependencies ────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)

if(WITH_PORTAUDIO)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
endif()

if(WITH_OPUS)
    pkg_check_modules(OPUS REQUIRED opus)
endif()

find_library(SODIUM_LIB sodium REQUIRED)
find_path(SODIUM_INCLUDE sodium.h PATH_SUFFIXES sodium)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
if(DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" REQUIRED)
else()
    pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
    if(NOT ONNXRUNTIME_FOUND)
        message(WARNING "ONNX Runtime not found — AI models will be disabled at runtime")
        set(ONNXRUNTIME_INCLUDE_DIR "")
        set(ONNXRUNTIME_LIB "")
    endif()
endif()

# ── NVIDIA Maxine (optional) ──────────────────────────────────────────────────
if(WITH_MAXINE)
    if(NOT DEFINED MAXINE_SDK_PATH)
        message(FATAL_ERROR "WITH_MAXINE=ON but MAXINE_SDK_PATH is not set")
    endif()
    find_path(MAXINE_INCLUDE nvAudioEffects.h PATHS "${MAXINE_SDK_PATH}/include" REQUIRED)
    find_library(MAXINE_LIB NvAFX PATHS "${MAXINE_SDK_PATH}/lib" REQUIRED)
    add_compile_definitions(ROOMSYNC_WITH_MAXINE)
endif()

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(pipeline)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install rules ─────────────────────────────────────────────────────────────
install(TARGETS roomsync_pipeline DESTINATION lib)
install(TARGETS roomsync_test    DESTINATION bin OPTIONAL)
