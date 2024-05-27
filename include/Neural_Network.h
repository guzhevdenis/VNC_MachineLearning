#include <vector>
template <typename Type>
class Network
{
    private:
        std::vector<Layer<Type> *> layers;
        
    public:
    Network()
    {

    }

    void addLayer (Layer *layer)
    {
        this->layers.push_back(layer)
    }
}