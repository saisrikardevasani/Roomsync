set(PIPELINE_SOURCES
    pipeline.cpp
    aec/aec_processor.cpp
    noise/noise_suppressor.cpp
    separation/separator.cpp
    plc/plc_processor.cpp
)

if(WITH_MAXINE)
    list(APPEND PIPELINE_SOURCES
         ../maxine/maxine_pipeline.cpp
         ../maxine/fallback.cpp)
endif()

add_library(roomsync_pipeline STATIC ${PIPELINE_SOURCES})

target_include_directories(roomsync_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SODIUM_INCLUDE}
)

target_link_libraries(roomsync_pipeline PUBLIC
    ${ONNXRUNTIME_LIB}
    ${SODIUM_LIB}
)

if(WITH_PORTAUDIO)
    target_compile_definitions(roomsync_pipeline PRIVATE ROOMSYNC_WITH_PORTAUDIO)
    target_include_directories(roomsync_pipeline PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(roomsync_pipeline PUBLIC ${PORTAUDIO_LIBRARIES})
endif()

if(WITH_OPUS)
    target_compile_definitions(roomsync_pipeline PRIVATE ROOMSYNC_WITH_OPUS)
    target_include_directories(roomsync_pipeline PRIVATE ${OPUS_INCLUDE_DIRS})
    target_link_libraries(roomsync_pipeline PUBLIC ${OPUS_LIBRARIES})
endif()

if(WITH_MAXINE)
    target_include_directories(roomsync_pipeline PRIVATE ${MAXINE_INCLUDE})
    target_link_libraries(roomsync_pipeline PRIVATE ${MAXINE_LIB})
endif()

if(ONNXRUNTIME_LIB)
    target_compile_definitions(roomsync_pipeline PRIVATE ROOMSYNC_WITH_ONNXRUNTIME)
endif()

# ── CLI smoke-test binary ─────────────────────────────────────────────────────
add_executable(roomsync_test main_test.cpp)
target_link_libraries(roomsync_test PRIVATE roomsync_pipeline)
