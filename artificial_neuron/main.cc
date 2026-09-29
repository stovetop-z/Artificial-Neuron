#include "math/neuron.h"

#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_headers/stb_image_write.h"

#include <iostream>

int main() {
    // 1. Create stateless layer: 3 inputs -> 1 output neuron
    NeuronLayer layer(3, 1);

    // 2. Input batch: 1 sample with 3 features [1.5, -2.0, 0.5]
    auto x = Tensor::create(1, 3, 1, std::vector<float>{1.5f, 5.0f, 0.5f});

    // 3. Forward pass (builds the computation graph via shared_ptrs)
    auto out = layer.forward(x);

    std::cout << "Output value: " << out->data[0] << "\n";

    // 4. Backward pass
    out->backward();

    // 5. Inspect gradients calculated for parameters
    std::cout << "\nWeight Gradients:\n";
    for (size_t i = 0; i < layer.weights->data.size(); ++i) {
        std::cout << "dw[" << i << "] = " << layer.weights->gradient[i] << "\n";
    }

    std::cout << "\nBias Gradient:\n";
    std::cout << "db = " << layer.bias->gradient[0] << "\n";

    return 0;
}
