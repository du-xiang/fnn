#include <iostream>

#include "Loader.hpp"
#include "Util.hpp"

bool loaderOri()
{
    Timer t;

    Sample sample;
	Loader loader("..\\..\\datasets\\mnist\\train.txt");

    int n = 0;
    while(n != 60000)
    {
        ++n;
        while(!loader.load(sample)){};
        sample.img.clear();
    }

    t.pause();
	std::cout << "time: " << t.elapsedTime() << "ms" << std::endl;

    return true;
}

// 大小端反转
inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8)  | uint32_t(p[3]);
}

bool loaderImpr()
{
    Timer t;

    std::vector<double> images;
    std::vector<int> labels;

    FILE *fimg = std::fopen("..\\..\\datasets\\mnist\\train-images.idx3-ubyte", "rb");
    FILE *flbl = std::fopen("..\\..\\datasets\\mnist\\train-labels.idx1-ubyte", "rb");
    if (!fimg || !flbl) return false;

    uint8_t head[16];
    if (std::fread(head, 1, 16, fimg) != 16) return false;
    if (be32(head) != 0x00000803) return false;
    int N = be32(head + 4);
    int R = be32(head + 8);
    int C = be32(head + 12);
    const int pixels = R * C;

    uint8_t lhead[8];
    if (std::fread(lhead, 1, 8, flbl) != 8) return false;
    if (be32(lhead) != 0x00000801) return false;
    if (be32(lhead + 4) != uint32_t(N)) return false;  // 判断样本数是否一致

    std::vector<uint8_t> buf(N * pixels);
    if (std::fread(buf.data(), 1, buf.size(), fimg) != buf.size()) return false;
    images.resize(buf.size());
    for (size_t i = 0; i < buf.size(); ++i)
        images[i] = static_cast<double>(buf[i]);

    labels.resize(N);
    if (std::fread(labels.data(), 1, N, flbl) != size_t(N)) return false;

    std::fclose(fimg);
    std::fclose(flbl);

    t.pause();
	std::cout << "time: " << t.elapsedTime() << "ms" << std::endl;

    return true;
}

int main()
{
    loaderOri();
    loaderImpr();
    return 0;
}