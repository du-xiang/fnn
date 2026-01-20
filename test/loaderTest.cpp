#include <iostream>
#include <cstdio>
#include <cstdint>

#include "LoaderOri.hpp"

bool loaderOri(std::vector<double>& img, std::vector<unsigned int>&lbl)
{
    Sample sample;
	Loader loader("..\\..\\datasets\\mnist\\train.txt");

    int n = 0;
    while(n != 60000)
    {
        ++n;

        loader.load(sample);
        lbl.push_back(static_cast<int>(sample.value));
        for(int i = 0; i < sample.img.size(); i++)
        {
            img.push_back(sample.img[i]);
        }

        sample.img.clear();
    }

    return true;
}

// 大小端反转
static inline uint32_t be_32(const uint8_t* p) 
{
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8)  | uint32_t(p[3]);
}

bool loaderImpr(std::vector<double>& img, std::vector<unsigned int>&lbl)
{
    FILE *fimg = std::fopen("..\\..\\datasets\\mnist\\train-images.idx3-ubyte", "rb");
    FILE *flbl = std::fopen("..\\..\\datasets\\mnist\\train-labels.idx1-ubyte", "rb");
    if (!fimg || !flbl) return false;

    uint8_t head[16];
    if (std::fread(head, 1, 16, fimg) != 16) return false;
    if (be_32(head) != 0x00000803) return false;
    int N = be_32(head + 4);
    int R = be_32(head + 8);
    int C = be_32(head + 12);
    const int pixels = R * C;

    uint8_t lhead[8];
    if (std::fread(lhead, 1, 8, flbl) != 8) return false;
    if (be_32(lhead) != 0x00000801) return false;
    if (be_32(lhead + 4) != uint32_t(N)) return false;  // 判断样本数是否一致

    std::vector<uint8_t> buf(N * pixels);
    if (std::fread(buf.data(), 1, buf.size(), fimg) != buf.size()) return false;
    img.resize(buf.size());
    for (size_t i = 0; i < buf.size(); ++i)
        img[i] = static_cast<double>(buf[i]);

    lbl.resize(N);
    std::vector<uint8_t> lblTemp(N);
    if (std::fread(lblTemp.data(), 1, N, flbl) != size_t(N)) return false;
    lbl.assign(lblTemp.begin(), lblTemp.end());

    std::fclose(fimg);
    std::fclose(flbl);

    return true;
}

int main()
{
    std::vector<double> img1, img2;
    std::vector<unsigned int> lbl1, lbl2;

    loaderImpr(img2, lbl2);
    loaderOri(img1, lbl1);

    std::cout << "lbl1.size(): " << lbl1.size() << " lbl2.size(): " << lbl2.size() << std::endl;
    std::cout << "img1.size(): " << img1.size() << " img2.size(): " << img2.size() << std::endl;

    for(int i = 0; i < lbl1.size(); i++)
    {
        //std::cout << lbl1[i] << "/" << lbl2[i] << "\t";
        if(lbl1[i] != lbl2[i])
        {
            std::cout << "lbl No." << i << " : not match" << std::endl;
            return 0;
        }
    }
    for(int i = 0; i < img1.size(); i++)
    {
        if(img1[i] != img2[i])
        {
            std::cout << "img No." << i << " : not match" << std::endl;
        }
    }
    std::cout << "matched" << std::endl;

    return 0;
}
