//
// Created by azureuser on 1/5/26.
//

#ifndef FLASHMOE_VT_CUH
#define FLASHMOE_VT_CUH

#define MAX_ALIGNMENT 16
namespace flashmoe {
    template<typename T, int Alignment = MAX_ALIGNMENT>
    struct VectorTypeDescriptor {
        using VectorWidth = cute::C<Alignment / sizeof(T)>;
        using VectorType = cutlass::AlignedArray<T, VectorWidth::value, Alignment>;
    };
    template<typename Element, int dim>
    constexpr int ElementWidth = cute::min(dim, MAX_ALIGNMENT / sizeof(Element));
    template<typename Element, int dim>
    constexpr int ElementAlignment = (cutlass::is_pow2<ElementWidth<Element, dim>>::value ?
        ElementWidth<Element, dim> : 1) * sizeof(Element);
}
#endif //FLASHMOE_VT_CUH