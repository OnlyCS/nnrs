# NNRS - Rust Neural Network Library
NNRS is a Rust library for creating and working with feedforward neural
networks. It provides a set of tools for building and manipulating neural
networks, including creating nodes and edges, setting inputs, and firing the
network to generate outputs.

Note: this library is still in development, and is not yet ready for use.

## Installation
To use NNRS, simply add it as a dependency to your Rust project with Cargo:

```sh
cargo add nnrs
```

## Usage

For the most up-to-date usage information, clone the 
repository and run `cargo doc --open`, or view the
documentation on [docs.rs](https://docs.rs/nnrs).

## Limitations

At this moment, NNRS does not include training functionality. You
can use this library to generate outputs from pre-trained networks.

## Roadmap

- [x] Basic Parts
- [x] Generating Outputs
- [x] Serialization
- [ ] Training (NEAT?)

## License
NNRS is licensed under the AGPLv3 license. See the `LICENSE` file for more
information.
