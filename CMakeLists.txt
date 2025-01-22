cmake_minimum_required(VERSION 3.14)
project("woolycore")

set(LLAMA_BUILD_TESTS Off)
set(LLAMA_BUILD_EXAMPLES Off)
set(LLAMA_BUILD_SERVER Off)
set(LLAMA_BUILD_COMMON On)

option(WOOLY_DEBUG "Enable extra debugging output while running" Off)
option(WOOLY_STATIC "Build static library instead of shared" OFF)
option(WOOLY_TESTS "Build unit tests" On)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_compile_options(-fPIC)
endif()

if(WOOLY_STATIC) 
    set(BUILD_SHARED_LIBS Off)
    set(GGML_STATIC On)

    add_library(woolycore STATIC
        bindings.cpp
    )
else()
    add_library(woolycore SHARED
        bindings.cpp
    )

    install(TARGETS woolycore LIBRARY)
endif()

if(WOOLY_DEBUG)
    add_definitions(-DWOOLY_DEBUG)
endif()


# include upstream llamacpp options
add_subdirectory("llama.cpp")

target_include_directories(woolycore PUBLIC 
    "llama.cpp"
    "llama.cpp/include"
    "llama.cpp/ggml/include"
    "llama.cpp/common"
)
target_compile_features(woolycore PRIVATE cxx_std_11) # don't bump
target_link_libraries(woolycore PRIVATE common)


# Unit testing
if(WOOLY_TESTS)
    add_subdirectory(unity)

    add_executable(test_predictions tests/test_predictions.c)
    target_include_directories(test_predictions PRIVATE 
        "llama.cpp/include"
        "llama.cpp/ggml/include"
    )

    add_executable(test_embeddings tests/test_embeddings.cpp)
    target_include_directories(test_embeddings PRIVATE 
        "llama.cpp/include"
        "llama.cpp/ggml/include"
        "llama.cpp/common"
    )
    target_compile_features(test_embeddings PRIVATE cxx_std_11)

    add_executable(test_prediction_steps tests/test_prediction_steps.c)
    target_include_directories(test_prediction_steps PRIVATE 
        "llama.cpp/include"
        "llama.cpp/ggml/include"
        "llama.cpp/common"
    )
    target_compile_features(test_prediction_steps PRIVATE cxx_std_11)

    add_executable(test_chat_formatting tests/test_chat_formatting.cpp)
    target_include_directories(test_chat_formatting PRIVATE 
        "llama.cpp/include"
        "llama.cpp/ggml/include"
        "llama.cpp/common"
    )
    target_compile_features(test_chat_formatting PRIVATE cxx_std_11)

    target_link_libraries(test_predictions PRIVATE woolycore unity)
    target_link_libraries(test_prediction_steps PRIVATE woolycore unity)
    target_link_libraries(test_embeddings PRIVATE woolycore unity common llama)
    target_link_libraries(test_chat_formatting PRIVATE woolycore unity)
endif()