# woolycore

A thin C wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides a surface to build FFI libraries on top of for other languages.

Upstream llama.cpp is pinned to commit 
[D5CB868](https://github.com/ggerganov/llama.cpp/releases/tag/b3889)
from Oct 6, 2024.

Supported Operating Systems: Windows, MacOS, Linux, iOS, Android 

Note: Android support is non-GPU accelerated.


## List of programming language integrations

The following projects use woolycore to create wrappers around the
[llama.cpp library](https://github.com/ggerganov/llama.cpp):

* Dart: [woolydart](https://github.com/tbogdala/woolydart)
* Rust: [woolyrust](https://github.com/tbogdala/woolyrust)


## License

MIT licensed, like the core upstream `llama.cpp` it wraps. See `LICENSE` for details.


## Features

* Basic samplers of llama.cpp, including: temp, top-k, top-p, min-p, tail free sampling, locally typical sampling, mirostat.
* Support for llama.cpp's BNF-like grammar rules for sampling.
* Ability to cache the processed prompt data in memory so that it can be reused to speed up regeneration using the exact same prompt.
* Tokenize text or just get the number of tokens for a given text string.
* Generate embeddings using models such as `nomic-ai/nomic-embed-text-v1.5-GGUF` on HuggingFace in a batched process.


## Build notes

To build, use the following commands:

```bash
cmake -B build
cmake --build build --config Release
```

This should automatically include Metal support and embed the shaders if the library is being built on MacOS. Other
platforms will only have CPU support without additional flags.

For systems that support CUDA acceleration, like Linux, you'll need to enable it using an additional compilation flag
as follows:

```bash
cmake -B build -DGGML_CUDA=On 
cmake --build build --config Release
```

On Windows, you'll need a few more flags to make sure that all the functions are exported and
available on the compiled library (And the resulting `build/Release/woolycore.dll` file will
need to be able to find the `build/bin/Release/ggml.dll` and `build/bin/Release/llama.dll` files
at runtime when deployed, so copy the three dlls to the same directory.)

```bash
cmake -B build -DGGML_CUDA=On -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE 
cmake --build build --config Release
```

### Static vs Shared

By default, *shared* versions of the library are built. This is a good approach, particularly during
development. However, you may find yourself in a position where *static* versions - ones that can
be used to compile everything into a single executable. For this, you can run the configure script
with `WOOLY_STATIC` enabled. For example, something like this:

```bash
cmake -B build -DGGML_CUDA=On -DWOOLY_STATIC=On
```

Keep in mind that the test excutables with CUDA linked in statically take about a gigabyte of storage each.


### Debugging

When loading a model with `wooly_load_model()`, setting the `silent_llama` parameter to true will attempt
to silence all output. When `silent_llama` is false, it will generate 'normal' level verbosity
log output. *However*, if you want extra debugging log output, you can configure the project
with an extra flag called `WOOLY_DEBUG` (e.g. `cmake -B build -DWOOLY_DEBUG=1`). This will
produce extra log information.


## Unit tests

The unit tests use the [Unity library](https://github.com/ThrowTheSwitch/Unity) 
and get compiled when building the main library bindings. You should see them as executables
in the `build` folder (e.g. `build/test_predictions`). Simply running the program will execute the
unit tests contained in it. 

By convention, the unit tests require an environment variable (WOOLY_TEST_MODEL_FILE) to be set 
with the path to the GGUF file for the model to use during testing.

In a unix environment, that means you can do something like this to run the unit tests:

```bash
export WOOLY_TEST_MODEL_FILE=models/example-llama-3-8b.gguf
./build/test_predictions

# note, currently the unit test uses a 2048 sized batch size targetted towards this particular model
export WOOLY_TEST_EMB_MODEL_FILE=models/nomic-embed-text-v1.5.Q8_0.gguf
./build/test_embeddings
```


## Git updates

This project uses submodules for upstream projects so make sure to update with appropriate parameters:

```bash
git pull --recurse-submodules
```


## API Changes

See the [API_CHANGES](API_CHANGES.md) document for a full listing of changes to the woolycore
API.


### Developer Notes

*   This library creates its own version of the structures to avoid failure cases where upstream
    llama.cpp structs have C++ members and cannot be wrapped automatically by tooling. While 
    inconvenient, I chose to do that over an opaque pointer and getter/setter functions.


### TODO

* Reenable some advanced sampling features again like logit biases.
* Missing calls to just tokenize text and to pull embeddings out from text.
* Maybe a dynamic LoRA layer, trained every time enough tokens fill up the context space, approx.
