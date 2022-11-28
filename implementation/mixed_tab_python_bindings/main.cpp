#include <pybind11/pybind11.h>
#include <vector>
#include <cstdint>
#include <random> 
#include <pybind11/stl.h>

namespace py = pybind11;

float some_fn(float arg1, float arg2) {
    return arg1 + arg2;
}
class polyhash

{
#ifdef DEBUG
    bool hasInit;
#endif
    uint32_t m_deg;
    uint32_t m_seed;
    std::vector<uint64_t> m_coef;
    // Large mersenne prime (2^61 - 1)
    const uint64_t m_p = 2305843009213693951;

public:

    polyhash(uint32_t deg, uint32_t seed) {
           // TODO: Remove the following 3 lines if using randomgen.h
        m_seed = seed;
        std::mt19937 rng;
        rng.seed(m_seed);
        std::uniform_int_distribution<uint64_t> dist;

        m_deg = deg;
        m_coef.resize(m_deg,0);
        for (uint32_t i = 0; i < m_deg; ++i) {
            do {
                // TODO: Swap the two lines below if using randomgen.h
                //m_coef[i] = getRandomUInt64() >> 3;
                m_coef[i] = (dist(rng)) >> 3;
            } while(m_coef[i] >= m_p);
    }
    }
    // void init() {
    //     init(2);
    // }; // 2-indep
    // void init(uint32_t deg) {
    //         // TODO: Remove the following 3 lines if using randomgen.h
    //     std::mt19937 rng;
    //     rng.seed(std::random_device()());
    //     std::uniform_int_distribution<uint64_t> dist;

    //     m_deg = deg;
    //     m_coef.resize(m_deg,0);
    //     for (uint32_t i = 0; i < m_deg; ++i) {
    //         do {
    //             // TODO: Swap the two lines below if using randomgen.h
    //             //m_coef[i] = getRandomUInt64() >> 3;
    //             m_coef[i] = (dist(rng)) >> 3;
    //         } while(m_coef[i] >= m_p);
    // }
    // };
    // uint32_t operator()(uint32_t x);

    uint32_t hash(uint32_t x) {
        __int128 h = 0;
        for (int32_t i = m_deg-1; i >= 0; --i) {
            h = h * x + m_coef[i];
            h = (h & m_p) + (h >> 61);
        }
        h = (h & m_p) + (h >> 61);
        return (uint32_t)h;
    }
    const std::vector<uint64_t> &getCoef() const { return m_coef; }
    const uint32_t &getDeg() const { return m_deg; }
};

class mixedtab
{
#ifdef DEBUG
    bool hasInit;
#endif
    // Use 4 characters + 4 derived characters.
    // TODO: Parameterize
    uint32_t c_par = 8;
    uint32_t d_par = 4;
    uint64_t mt_T1[256][8];
    uint32_t mt_T2[256][4];
    uint32_t m_seed;


public:
    mixedtab(uint32_t seed) {
        // Use a degree-20 polynomial to fill out the entries.
        m_seed = seed;
        polyhash ph(20, m_seed);

        uint32_t x = 0;
        for (int i = 0; i < c_par; ++i) {
            for (int j = 0; j < 256; ++j) {
                mt_T1[j][i] = ph.hash(x++);
                mt_T1[j][i] <<= 32;
                mt_T1[j][i] += ph.hash(x++);
                mt_T2[j][i] = ph.hash(x++);
            }
        }
    #ifdef DEBUG
        hasInit = true;
    #endif
    };
    uint64_t hash(uint64_t x) {
        uint64_t h=0; // Final hash value
        for (int i = 0; i < c_par; ++i, x >>= 8)
            // x is chopped in 4 parts of 8 bits, such that we can access a row in the mt_T1 array
            h ^= mt_T1[(uint8_t)x][i];
        uint32_t drv=h >> 32;
        for (int i = 0; i < d_par; ++i, drv >>= 8)
            h ^= mt_T2[(uint8_t)drv][i];
        return (uint32_t)h;
    };
    uint32_t operator()(uint32_t x);
};

// uint32_t mixedtab::operator()(uint32_t x)
// {
// #ifdef DEBUG
//     assert(hasInit);
// #endif
//     uint64_t h=0; // Final hash value
//     for (int i = 0; i < 4; ++i, x >>= 8)
//         h ^= mt_T1[(uint8_t)x][i];
//     uint32_t drv=h >> 32;
//     for (int i = 0; i < 4; ++i, drv >>= 8)
//         h ^= mt_T2[(uint8_t)drv][i];
//     return (uint32_t)h;
// }


PYBIND11_MODULE(pyMixedTabulation, handle) {
    handle.doc() = "This is the module docs of the pyMixedTabulation class";
    handle.def("some_fn_python_name", &some_fn);

    py::class_<polyhash>(
                        handle, "PyPolyHash"
                        )
        //.def(py::init<>())
        .def(py::init<const uint32_t, const uint32_t &>())
        .def("getCoef", &polyhash::getCoef)
        .def("getDeg", &polyhash::getDeg)
        .def("getHash", &polyhash::hash);
    py::class_<mixedtab>(
                        handle, "PyMixTab"
                        )
        .def(py::init<const uint32_t &>())
        .def("getHash", &mixedtab::hash);
}