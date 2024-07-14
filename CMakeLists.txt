cmake_minimum_required(VERSION 3.14)
project("woolycore")

set(LLAMA_BUILD_TESTS Off)
set(LLAMA_BUILD_EXAMPLES Off)
set(LLAMA_BUILD_SERVER Off)

# include upstream llamacpp options
add_subdirectory("llama.cpp")

add_library(woolycore SHARED
    bindings.cpp
    )

# add_library(woolycore_static STATIC
#     bindings.cpp
#     )

target_include_directories(woolycore PUBLIC "llama.cpp")
target_compile_features(woolycore PRIVATE cxx_std_11) # don't bump
target_link_libraries(woolycore PRIVATE common)

# target_include_directories(woolycore_static PUBLIC "llama.cpp")
# target_compile_features(woolycore_static PRIVATE cxx_std_11) # don't bump
# target_link_libraries(woolycore_static PRIVATE common llama)