find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Pipeline unit tests
add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE roomsync_pipeline GTest::gtest_main)
add_test(NAME PipelineTest COMMAND test_pipeline)

# Encryption unit tests
add_executable(test_encryption test_encryption.cpp)
target_link_libraries(test_encryption PRIVATE ${SODIUM_LIB} GTest::gtest_main)
target_include_directories(test_encryption PRIVATE ${SODIUM_INCLUDE})
add_test(NAME EncryptionTest COMMAND test_encryption)
