#ifndef NEURON_H
#define NEURON_H

#include "tensor.h"

class NeuronLayer 
{
public:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;

    NeuronLayer(size_t in_features, size_t out_features) 
    {
        // Initialize weights with small random values
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        std::vector<float> w_data(in_features * out_features);
        for(auto& val : w_data) val = dist(gen);

        weights = Tensor::create(in_features, out_features, 1, w_data, true);
        bias = Tensor::create(1, out_features, 1, true, 0.0f);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x)
    {
        auto z = Tensor::matmul(x, weights);
        auto out = Tensor::addBias(z, bias);
        return Tensor::relu(out);
    }
};
#endif
