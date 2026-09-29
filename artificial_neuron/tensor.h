#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cstdint>
#include <utility>
#include <stdexcept>

class Tensor : public std::enable_shared_from_this<Tensor> 
{
public:
    std::vector<float> data, gradient;
    size_t rows, cols, channels;
    bool requires_grad;

    std::vector<std::shared_ptr<Tensor>> previous;
    std::function<void()> backward_function;

    Tensor(size_t r, size_t c, size_t chnls, bool req_grad = false, float fill = 0.0f)
        : data(r * c * chnls, fill),
        gradient(data.size(), 0.0f),
        rows(r),
        cols(c),
        channels(chnls),
        requires_grad(req_grad)
    {
    }

    Tensor(size_t r, size_t c, size_t chnls,
           const std::vector<float>& d, bool req_grad = false)
        : data(d),
        gradient(d.size(), 0.0f),
        rows(r),
        cols(c),
        channels(chnls),
        requires_grad(req_grad)
    {
        if (data.size() != r * c * chnls) {
            throw std::invalid_argument("Tensor data size must match its dimensions");
        }
    }
    ~Tensor() = default;

    template<typename... Args>
    static std::shared_ptr<Tensor> create(Args&&... args)
    {
        return std::make_shared<Tensor>(std::forward<Args>(args)...);
    }

    /* Function: matmul
     * Two tensors are pass with rows, cols, and channels.
     * Matrix multiplication (matmul) is multiplying two tensors such that
     * t1(M, K) @ t2(K, N) = t(row, col)
     */
    static std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& t1, const std::shared_ptr<Tensor>& t2)
    {
        size_t M = t1->rows;
        size_t K = t1->cols;
        size_t N = t2->cols;

        // Basically, we pass in the two sizes and create a 
        std::shared_ptr<Tensor> out = create(M, N, 1, t1->requires_grad || t2->requires_grad);
        out->previous = {t1, t2};

        // Forward comp
        for(size_t i = 0; i < M; i++)
        {
            for(size_t j = 0; j < N; j++)
            {
                float sum = 0.0f;
                for(size_t k = 0; k < K; k++)
                {
                    sum += t1->data[i * K + k] * t2->data[k * N + j];
                }

                out->data[i * N + j] = sum;
            }
        }

        // Backward closure will capture the inputs (t1, t2) and output (out)
        out->backward_function = [t1, t2, out, M, K, N]() 
        {
            if(t1->requires_grad)
            {
                // dA = dOut @ B^T
                for(size_t i = 0; i < M; i++)
                {
                    for(size_t k = 0; k < K; k++)
                    {
                        float sum = 0.0f;
                        for(size_t j = 0; j < N; j++)
                        {
                            sum += out->gradient[i * N + j] * t2->data[k * N + j];
                        }

                        t1->gradient[i * K + k] += sum;
                    }
                }
            }
            if(t2->requires_grad)
            {
                // dB = A^T @ dOut
                for(size_t k = 0; k < K; k++)
                {
                    for(size_t j = 0; j < N; j++)
                    {
                        float sum = 0.0f;
                        for(size_t i = 0; i < M; i++)
                        {
                            sum += t1->data[i * K + k] * out->gradient[i * N + j];
                        }

                        t2->gradient[k * N + j] += sum;
                    }
                }
            }
        };

        return out;
    }

    static std::shared_ptr<Tensor> addBias(const std::shared_ptr<Tensor>& t1, const std::shared_ptr<Tensor>& t2)
    {
        size_t M = t1->rows;
        size_t N = t1->cols;

        auto out = create(M, N, 1, t1->requires_grad || t2->requires_grad);
        out->previous = {t1, t2};

        for(size_t i = 0; i < M; i++)
        {
            for(size_t j = 0; j < N; j++)
            {
                out->data[i * N + j] = t1->data[i * N + j] + t2->data[j];
            }
        }

        out->backward_function = [t1, t2, out, M, N]()
        {
            for(size_t i = 0; i < M; i++)
            {
                for(size_t j = 0; j < N; j++)
                {
                    float g = out->gradient[i * N + j];
                    if(t1->requires_grad) t1->gradient[i * N + j] += g;
                    if(t2->requires_grad) t2->gradient[j] += g;
                }
            }
        };

        return out;
    }

    static std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t)
    {
        auto out = create(t->rows, t->cols, t->channels, t->requires_grad);
        out->previous = {t};

        for(size_t i = 0; i < t->data.size(); i++)
        {
            out->data[i] = std::max(0.0f, t->data[i]);
        }

        out->backward_function = [t, out]() 
        {
            if(!t->requires_grad) return;
            for(size_t i = 0; i < t->data.size(); i++)
            {
                if(out->data[i] > 0.0f)
                {
                    t->gradient[i] += out->gradient[i];
                }
            }
        };

        return out;
    }

    void backward()
    {
        std::fill(gradient.begin(), gradient.end(), 1.0f);

        std::vector<std::shared_ptr<Tensor>> topological;
        std::unordered_set<Tensor*> visited;

        std::function<void(const std::shared_ptr<Tensor>&)> build_topo = [&](const std::shared_ptr<Tensor>& node)
        {
            if(!node || visited.count(node.get())) return;
            visited.insert(node.get());
            for(const auto& parent : node->previous)
            {
                build_topo(parent);
            }
            topological.push_back(node);
        };

        build_topo(shared_from_this());

        for(auto it = topological.rbegin(); it != topological.rend(); ++it)
        {
            if ((*it)->backward_function)
            {
                (*it)->backward_function();
            }
        }
    }
};
