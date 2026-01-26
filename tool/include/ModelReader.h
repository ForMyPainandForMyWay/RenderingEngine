//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_MODELREADER_H
#define UNTITLED_MODELREADER_H
#include <unordered_map>
#include <vector>

struct Film;
class Mesh;
class Material;
struct TextureMap;
struct ObjFace;

class ModelReader {
public:
    static std::vector<std::string> readObjFile(
        bool Gamma,
        const std::string &filename,
        std::unordered_map<std::string, std::shared_ptr<Mesh>> &meshes,
        std::unordered_map<std::string, std::shared_ptr<Material>> &materialMap,
        std::unordered_map<std::string, std::shared_ptr<TextureMap>> &textureMap,
        std::unordered_map<std::string, std::shared_ptr<TextureMap>> &normalMap);
    static void readMTLFile(
        bool Gamma,
        const std::string &mtlFilename,
        std::unordered_map<std::string, std::shared_ptr<Material>> &materialMap,
        std::unordered_map<std::string, std::shared_ptr<TextureMap>> &textureMap,
        std::unordered_map<std::string, std::shared_ptr<TextureMap>> &normalMap);
    static void splitPoly2Tri(const ObjFace& face, const std::shared_ptr<Mesh>& mesh);
    static void GammaCorrect(const std::unique_ptr<Film> &img);
};


#endif //UNTITLED_MODELREADER_H