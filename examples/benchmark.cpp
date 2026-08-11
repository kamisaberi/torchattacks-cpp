#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "torchattacks/fgsm.h"
#include "torchattacks/pgd.h"

int main() {
    std::cout << "=== torchattacks-cpp Benchmark Suite ===" << std::endl;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "[+] CUDA hardware acceleration detected!" << std::endl;
        device = torch::Device(torch::kCUDA, 0);
    } else {
        std::cout << "[-] CUDA not available. Running on CPU." << std::endl;
    }

    // Define dummy Convolutional Model
    auto model = std::make_shared<torch::nn::Sequential>(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
        torch::nn::Flatten(),
        torch::nn::Linear(64, 10)
    );
    model->to(device);
    model->eval();

    // Create synthetic evaluation batch (Batch size = 64, Image size = 3x224x224)
    const int batch_size = 64;
    torch::Tensor images = torch::rand({batch_size, 3, 224, 224}, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor labels = torch::randint(0, 10, {batch_size}, torch::dtype(torch::kLong).device(device));

    // Warmup CUDA engine
    std::cout << "[*] Warming up GPU kernels..." << std::endl;
    torchattacks::FGSM warmup_attack(model, 8.0f / 255.0f);
    for (int i = 0; i < 5; ++i) {
        auto tmp = warmup_attack.forward(images, labels);
    }
    if (device.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();

    // 1. Benchmark FGSM
    std::cout << "\n--- Benchmarking FGSM ---" << std::endl;
    torchattacks::FGSM fgsm(model, 8.0f / 255.0f);
    
    auto start_fgsm = std::chrono::high_resolution_clock::now();
    torch::Tensor fgsm_adv = fgsm.forward(images, labels);
    if (device.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
    auto end_fgsm = std::chrono::high_resolution_clock::now();

    double fgsm_ms = std::chrono::duration<double, std::milli>(end_fgsm - start_fgsm).count();
    std::cout << "    [+] Latency: " << fgsm_ms << " ms" << std::endl;
    std::cout << "    [+] Throughput: " << (batch_size / (fgsm_ms / 1000.0)) << " img/sec" << std::endl;

    // 2. Benchmark PGD-100 (Fused CUDA Kernel)
    std::cout << "\n--- Benchmarking PGD-100 (Fused CUDA Step) ---" << std::endl;
    torchattacks::PGD pgd100(model, 8.0f / 255.0f, 2.0f / 255.0f, 100);

    auto start_pgd = std::chrono::high_resolution_clock::now();
    torch::Tensor pgd_adv = pgd100.forward(images, labels);
    if (device.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
    auto end_pgd = std::chrono::high_resolution_clock::now();

    double pgd_ms = std::chrono::duration<double, std::milli>(end_pgd - start_pgd).count();
    std::cout << "    [+] Latency (100 steps): " << pgd_ms << " ms" << std::endl;
    std::cout << "    [+] Throughput: " << (batch_size / (pgd_ms / 1000.0)) << " img/sec" << std::endl;

    std::cout << "\n[+] Benchmark completed successfully!" << std::endl;
    return 0;
}