#ifndef Loader_H
#define Loader_H

#include <iostream>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>


static inline uint32_t be32(const uint8_t* p)
{
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8)  | uint32_t(p[3]);
}

class Loader
{
private:
	FILE *fimg;
	FILE *flbl;

public:
	int pos;
	std::vector<double> images;
	std::vector<double>::const_iterator winBeign;
	std::vector<double>::const_iterator winEnd;
	std::vector<unsigned int> labels;
	
	Loader() = delete;
	explicit Loader(const char *&imgPath, const char *&lblPath);
	~Loader();
	bool load();
};

inline Loader::Loader(const char *&imgPath, const char *&lblPath) : pos(-1)
{
	fimg = std::fopen(imgPath, "rb");
    flbl = std::fopen(lblPath, "rb");
    if (!fimg || !flbl) throw std::runtime_error("Loader: Cannot open file");

	uint8_t head[16];
    if (std::fread(head, 1, 16, fimg) != 16) throw std::runtime_error("Loader: image header data read error");
    if (be32(head) != 0x00000803) throw std::runtime_error("Loader: image dimension mismatch");
    int N = be32(head + 4);
    int R = be32(head + 8);
    int C = be32(head + 12);
    const int pixels = R * C;

    uint8_t lhead[8];
    if (std::fread(lhead, 1, 8, flbl) != 8) throw std::runtime_error("Loader: Header data read error");
    if (be32(lhead) != 0x00000801) throw std::runtime_error("Loader: label dimension mismatch");
    if (be32(lhead + 4) != uint32_t(N)) throw std::runtime_error("Loader: The sample sizes are not consistent.");  // 判断样本数是否一致

    std::vector<uint8_t> buf(N * pixels);
    if (std::fread(buf.data(), 1, buf.size(), fimg) != buf.size()) throw std::runtime_error("Loader: Data read error");
    images.resize(buf.size());
    for (size_t i = 0; i < buf.size(); ++i)
        images[i] = static_cast<double>(buf[i]);
	
	winBeign = images.cbegin();
	winEnd = images.cbegin(); 

    labels.resize(N);
	std::vector<uint8_t> lblTemp(N);
    if (std::fread(lblTemp.data(), 1, N, flbl) != size_t(N)) throw std::runtime_error("Loader: Label reading error");
	labels.assign(lblTemp.begin(), lblTemp.end());

	std::cout << "images.size(): " << images.size() << " labels.size(): " << labels.size() <<std::endl;
}

inline Loader::~Loader()
{
	std::fclose(fimg);
    std::fclose(flbl);
}

inline bool Loader::load()
{
	if(++pos >= labels.size()) return false;

	winBeign = images.cbegin()+pos*784;
	winEnd = images.cbegin()+(pos+1)*784;
	return true;
}

#endif // ! Loader_H
