#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include "unitree_go2_hardware/crc32.hpp"


// ============================================================
// Test cases
// ============================================================

void test_crc32_basic() {
    
    std::cout << "========== Basic  ==========\n";
    
    uint32_t data[] = {0x12345678, 0xABCDEF00, 0xDEADBEEF};
    uint32_t len = sizeof(data) / sizeof(data[0]);
    
    std::cout << "Test data: ";
    for (uint32_t i = 0; i < len; i++) {
        std::cout << "0x" << std::hex << data[i] << " ";
    }
    std::cout << std::dec << "\n";
    
    // Test algorithms
    uint32_t crc1 = CRC32::bit_by_bit(data, len);
    uint32_t crc2 = CRC32::lookup_table(data, len);
    uint32_t crc3 = CRC32::slicing4(data, len);
    uint32_t crc4 = CRC32::slicing8(data, len);
    
    std::cout << "Bit-by-Bit:    0x" << std::hex << crc1 << "\n";
    std::cout << "Lookup Table:  0x" << std::hex << crc2 << "\n";
    std::cout << "Slicing-4:     0x" << std::hex << crc3 << "\n";
    std::cout << "Slicing-8:     0x" << std::hex << crc4 << "\n";
    
    // Verify that all methods give the same result
    if (crc1 == crc2 && crc2 == crc3 && crc3 == crc4) {
        std::cout << "✅ All methods give the same result\n";
    } else {
        std::cout << "❌ Results do not match!\n";
    }
    std::cout << std::dec << "\n";
}

void test_unitree_lowcmd() {

    using namespace unitree_go::msg::dds_;
    
    std::cout << "========== Unitree LowCmd test ==========\n";
    
    // Cr
    LowCmd_ low_cmd;
    memset(&low_cmd, 0, sizeof(low_cmd));

    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].mode() = 0x01;
        low_cmd.motor_cmd()[i].q() = 0;
        low_cmd.motor_cmd()[i].kp() = 0;
        low_cmd.motor_cmd()[i].dq() = 0;
        low_cmd.motor_cmd()[i].kd() = 0;
        low_cmd.motor_cmd()[i].tau() = 0;
    }

    std::cout << "Size of LowCmd: " << sizeof(LowCmd_) << " Bytes\n";

    // Calculate CRC (as per your original method)
    // sizeof(LowCmd_) >> 2 = number of bytes / 4 = number of uint32_t
    // -1 is because the CRC field itself is not included in the calculation

    std::cout << "Calculating CRC...\n";
    std::cout << "  Data length: " << (sizeof(LowCmd_) >> 2) - 1 << " uint32_t\n";
    
    // Method 1: Use old interface (for compatibility)
    low_cmd.crc() = crc32_core((uint32_t*)&low_cmd, (sizeof(LowCmd_) >> 2) - 1);
    std::cout << "  CRC (crc32_core):  0x" << std::hex << low_cmd.crc() << std::dec << "\n";

    // Method 2: Use new interface (recommended)
    uint32_t crc_new = CRC32::slicing8((uint32_t*)&low_cmd, (sizeof(LowCmd_) >> 2) - 1);
    std::cout << "  CRC (slicing8):    0x" << std::hex << crc_new << std::dec << "\n";
    
    if (low_cmd.crc() == crc_new) {
        std::cout << "✅ CRC calculation is correct\n";
    } else {
        std::cout << "❌ CRC does not match!\n";
    }

    // Verify CRC
    std::cout << "\nVerifying CRC...\n";
    uint32_t saved_crc = low_cmd.crc();
    low_cmd.crc() = 0;  // Clear and recalculate
    uint32_t recalc_crc = CRC32::slicing8((uint32_t*)&low_cmd, (sizeof(LowCmd_) >> 2) - 1);
    
    if (saved_crc == recalc_crc) {
        std::cout << "✅ CRC verification passed\n";
    } else {
        std::cout << "❌ CRC verification failed\n";
        std::cout << "  Original: 0x" << std::hex << saved_crc << "\n";
        std::cout << "  Recalculated: 0x" << std::hex << recalc_crc << std::dec << "\n";
    }

    low_cmd.crc() = saved_crc;  // Restore
    std::cout << "\n";
}

void test_all_methods_performance() {

    std::cout << "========== Performance Comparison Test ==========\n";

    // Simulate LowCmd size data
    constexpr size_t data_size = 576;  // Typical control command size
    uint32_t data[data_size];

    // Fill test data
    for (size_t i = 0; i < data_size; i++) {
        data[i] = i * 0x12345678 + 0xABCDEF01;
    }
    
    const int iterations = 10000;

    std::cout << "Data size: " << data_size << " words (" 
              << data_size * 4 << " bytes)\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    // Test four methods
    auto test_method = [&](const char* name, auto method) {
        // Warm up
        method(data, data_size);

        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        uint32_t result = 0;
        for (int i = 0; i < iterations; i++) {
            result = method(data, data_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_us = duration.count() / static_cast<double>(iterations);
        
        std::cout << std::left << std::setw(15) << name 
                  << " → CRC: 0x" << std::hex << std::setw(8) << std::setfill('0') << result
                  << std::dec << std::setfill(' ')
                  << "  Time: " << std::fixed << std::setprecision(3) << time_us << " μs\n";
    };
    
    test_method("Bit-by-Bit", CRC32::bit_by_bit);
    test_method("Lookup Table", CRC32::lookup_table);
    test_method("Slicing-4", CRC32::slicing4);
    test_method("Slicing-8", CRC32::slicing8);

    std::cout << "\nRecommended: CRC32::slicing8() (Fastest)\n";
    std::cout << "\n";
}

void test_edge_cases() {

    std::cout << "========== Boundary Tests ==========\n";

    // Test 1: Single uint32_t
    uint32_t single = 0x12345678;
    uint32_t crc1 = CRC32::slicing8(&single, 1);
    std::cout << "Single word: 0x" << std::hex << crc1 << std::dec << "\n";

    // Test 2: Empty data
    uint32_t crc2 = CRC32::slicing8(nullptr, 0);
    std::cout << "Empty data: 0x" << std::hex << crc2 << std::dec << "\n";

    // Test 3: All zeros
    uint32_t zeros[10] = {0};
    uint32_t crc3 = CRC32::slicing8(zeros, 10);
    std::cout << "All zeros:  0x" << std::hex << crc3 << std::dec << "\n";

    // Test 4: All ones
    uint32_t ones[10];
    memset(ones, 0xFF, sizeof(ones));
    uint32_t crc4 = CRC32::slicing8(ones, 10);
    std::cout << "All ones:   0x" << std::hex << crc4 << std::dec << "\n";
    
    std::cout << "\n";
}

// ============================================================
// Main test function
// ============================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║        Unitree CRC32 Complete Test Suite           ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    test_crc32_basic();
    test_unitree_lowcmd();
    test_all_methods_performance();
    test_edge_cases();
    
    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║                 Testing Complete!                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n";
    
    return 0;
}
