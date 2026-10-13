#ifndef CRC32_HPP
#define CRC32_HPP

#include <cstdint>
#include <array>

class CRC32 {
public:
    // Method 1: Bit-by-bit method (slow)
    static uint32_t bit_by_bit(const uint32_t* ptr, uint32_t len) {
        uint32_t crc = 0xFFFFFFFF;
        constexpr uint32_t polynomial = 0x04C11DB7;
        
        for (uint32_t i = 0; i < len; i++) {
            uint32_t xbit = 1 << 31;
            uint32_t data = ptr[i];
            
            for (uint32_t bits = 0; bits < 32; bits++) {
                if (crc & 0x80000000) {
                    crc <<= 1;
                    crc ^= polynomial;
                } else {
                    crc <<= 1;
                }
                
                if (data & xbit) crc ^= polynomial;
                xbit >>= 1;
            }
        }
        
        return crc;
    }
    
    // Method 2: Table lookup method (faster)
    static uint32_t lookup_table(const uint32_t* ptr, uint32_t len) {
        uint32_t crc = 0xFFFFFFFF;
        
        for (uint32_t i = 0; i < len; i++) {
            uint32_t word = ptr[i];
            for (int byte = 0; byte < 4; byte++) {
                uint8_t b = (word >> (24 - byte * 8)) & 0xFF;
                uint8_t index = (crc >> 24) ^ b;
                crc = (crc << 8) ^ get_table()[index];
            }
        }
        
        return crc;
    }
    
    // Method 3: Slicing-by-4 method (faster)
    static uint32_t slicing4(const uint32_t* ptr, uint32_t len) {
        uint32_t crc = 0xFFFFFFFF;
        
        for (uint32_t i = 0; i < len; i++) {
            uint32_t word = ptr[i];
            
            uint8_t b0 = (word >> 24) & 0xFF;
            uint8_t b1 = (word >> 16) & 0xFF;
            uint8_t b2 = (word >> 8) & 0xFF;
            uint8_t b3 = word & 0xFF;
            
            crc = get_table3()[(crc >> 24) ^ b0] ^
                  get_table2()[((crc >> 16) & 0xFF) ^ b1] ^
                  get_table1()[((crc >> 8) & 0xFF) ^ b2] ^
                  get_table()[(crc & 0xFF) ^ b3];
        }
        
        return crc;
    }
    
    // Method 4: Slicing-by-8 method (fastest)
    static uint32_t slicing8(const uint32_t* ptr, uint32_t len) {
        uint32_t crc = 0xFFFFFFFF;
        uint32_t i = 0;
        
        // Processing 2 words at a time
        while (i + 1 < len) {
            uint32_t word1 = ptr[i];
            uint32_t word2 = ptr[i + 1];
            
            uint8_t b0 = (word1 >> 24) & 0xFF;
            uint8_t b1 = (word1 >> 16) & 0xFF;
            uint8_t b2 = (word1 >> 8) & 0xFF;
            uint8_t b3 = word1 & 0xFF;
            uint8_t b4 = (word2 >> 24) & 0xFF;
            uint8_t b5 = (word2 >> 16) & 0xFF;
            uint8_t b6 = (word2 >> 8) & 0xFF;
            uint8_t b7 = word2 & 0xFF;
            
            crc = get_table7()[(crc >> 24) ^ b0] ^
                  get_table6()[((crc >> 16) & 0xFF) ^ b1] ^
                  get_table5()[((crc >> 8) & 0xFF) ^ b2] ^
                  get_table4()[(crc & 0xFF) ^ b3] ^
                  get_table3()[b4] ^
                  get_table2()[b5] ^
                  get_table1()[b6] ^
                  get_table()[b7];
            
            i += 2;
        }
        
        // Processing the rest
        while (i < len) {
            uint32_t word = ptr[i];
            for (int byte = 0; byte < 4; byte++) {
                uint8_t b = (word >> (24 - byte * 8)) & 0xFF;
                uint8_t index = (crc >> 24) ^ b;
                crc = (crc << 8) ^ get_table()[index];
            }
            i++;
        }
        
        return crc;
    }

private:
    static constexpr uint32_t polynomial_ = 0x04C11DB7;     // CRC-32 polynomial
    
    // Generate base table
    static constexpr std::array<uint32_t, 256> make_base_table() {
        std::array<uint32_t, 256> table{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i << 24;
            for (int j = 0; j < 8; j++) {
                if (crc & 0x80000000) {
                    crc = (crc << 1) ^ polynomial_;
                } else {
                    crc = crc << 1;
                }
            }
            table[i] = crc;
        }
        return table;
    }
    
    // Generate slice table
    template<int N>
    static constexpr std::array<uint32_t, 256> make_slice_table() {
        constexpr auto base = make_base_table();
        std::array<uint32_t, 256> table{};
        
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = base[i];
            for (int j = 0; j < N; j++) {
                uint8_t index = crc >> 24;
                crc = (crc << 8) ^ base[index];
            }
            table[i] = crc;
        }
        return table;
    }
    
    // Getters for tables
    static const std::array<uint32_t, 256>& get_table() {
        static constexpr auto table = make_base_table();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table1() {
        static constexpr auto table = make_slice_table<1>();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table2() {
        static constexpr auto table = make_slice_table<2>();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table3() {
        static constexpr auto table = make_slice_table<3>();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table4() {
        static constexpr auto table = make_slice_table<4>();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table5() {
        static constexpr auto table = make_slice_table<5>();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table6() {
        static constexpr auto table = make_slice_table<6>();
        return table;
    }
    
    static const std::array<uint32_t, 256>& get_table7() {
        static constexpr auto table = make_slice_table<7>();
        return table;
    }
};

// Compatible with old interface
inline uint32_t crc32_core(const uint32_t* ptr, uint32_t len) {
    return CRC32::slicing8(ptr, len);
}


#endif  // CRC32_HPP