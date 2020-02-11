#pragma once

#include "polynom.h"
#include "grid.h"

#include <util/generic/vector.h>


namespace NMonoForest {

    void trainLasso(
            TPolynom* poly,
            const TVector<TVector<double>>& data,
            const TVector<double>& label,
            const IGrid& grid,
            double lambda,
            double eps,
            size_t maxSteps
    );

    TVector<TVector<size_t>> applyPolynom(
            const TVector<TVector<double>>& data,
            const TPolynom& poly,
            const IGrid& grid
    );

    TVector<double> apply(const TPolynom& poly, const IGrid& grid, const TVector<TVector<double>>& data);
}
