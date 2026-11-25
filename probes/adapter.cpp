// Query the physical device through Windows independently of the translator.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include <cwchar>

int main() {
    using Microsoft::WRL::ComPtr;
    ComPtr<ID3D12Device> device;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 description{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))) ||
        FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))) ||
        FAILED(adapter->GetDesc1(&description)) || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof(waves)))) {
        std::fwprintf(stderr, L"Could not query physical D3D12 adapter capabilities.\n");
        return 1;
    }
    std::wprintf(L"%ls | WaveOps=%d | WaveLaneCountMin=%u | WaveLaneCountMax=%u\n",
                 description.Description, waves.WaveOps, waves.WaveLaneCountMin, waves.WaveLaneCountMax);
    return 0;
}
