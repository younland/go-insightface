// Package insightface provides pure-Go image processing and dynamic ONNX
// Runtime loading for SCRFD face detection and ArcFace embeddings.
//
// Model weights and the ONNX Runtime shared library are supplied by the
// application. Similarity scores returned by this package are cosine
// similarities, not probabilities.
package insightface
