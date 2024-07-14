# woolycore

A thin C wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides a surface to build FFI libraries on top of for other languages.

Upstream llama.cpp is pinned to commit 
[D7FD29F](https://github.com/ggerganov/llama.cpp/commit/d7fd29fff16456ce9c3a23fd2d09a66256b05aff)
from July 05, 2024.

At present, it is in pre-alpha development and the API is unstable. 

Supported Operating Systems: MacOS, iOS, Android (more to come!)

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


## Build notes

To build, use the following commands:

```bash
cmake -B build
cmake --build build --config Release
```

This should automatically include Metal support and embed the shaders if the library is being built on MacOS.


## Git updates

This project uses submodules for upstream projects so make sure to update with appropriate parameters:

```bash
git pull --recurse-submodules
```

### Developer Notes

*   This library creates its own version of the structures to avoid failure cases where upstream
    llama.cpp structs have C++ members and cannot be wrapped automatically by tooling. While 
    inconvenient, I chose to do that over an opaque pointer and getter/setter functions.


### TODO

* Reenable some advanced sampling features again like logit biases.
* Missing calls to just tokenize text and to pull embeddings out from text.
* Maybe a dynamic LoRA layer, trained every time enough tokens fill up the context space, approx.