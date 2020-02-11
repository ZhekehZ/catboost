#include "lasso.h"

#include <util/generic/ymath.h>

namespace NMonoForest {

    TVector<TVector<size_t>> applyPolynom(
            const TVector<TVector<double>>& data,
            const TPolynom& poly,
            const IGrid& grid
    ) {
        TVector<TVector<size_t>> result(data.size());
        size_t rowNum = 0;
        for (const auto& row : data) {

            size_t monomNum = 0;
            for (const auto& [monom, stat] : poly.MonomsEnsemble) {
                bool positive = true;
                for (const auto &split : monom.Splits) {

                    auto border = grid.Border(split.FeatureId, split.BinIdx);
                    auto value = row[grid.ExternalFlatFeatureIndex(split.FeatureId)];

                    if (split.SplitType == EBinSplitType::TakeGreater) {
                        positive = value > border;
                    } else if (split.SplitType == EBinSplitType::TakeBin) {
                        positive = Abs(value - border) < std::numeric_limits<double>::epsilon();
                    }

                    if (!positive) {
                        break;
                    }

                }
                if (positive) {
                    result[rowNum].push_back(monomNum);
                }
                ++monomNum;
            }

            ++rowNum;
        }
        return result;
    }


    void trainLasso(
            TPolynom* poly,
            const TVector<TVector<double>>& data,
            const TVector<double>& label,
            const IGrid& grid,
            double lambda,
            double eps = std::numeric_limits<double>::epsilon(),
            size_t maxSteps = 100
    ) {

        auto sparseTable = applyPolynom(data, *poly, grid);

        TVector<double *> weightsP;
        weightsP.reserve(poly->MonomsEnsemble.size());
        for (auto&[_, stat] : poly->MonomsEnsemble) {
            weightsP.push_back(&stat.Value[0]);
        }

        size_t numMonoms = poly->MonomsEnsemble.size();
        TVector<TVector<size_t>> columns(numMonoms);
        for (size_t i = 0; i < sparseTable.size(); ++i) {
            for (size_t idx : sparseTable[i]) {
                columns[idx].push_back(i);
            }
        }


        for (size_t iter = 0; iter < maxSteps; ++iter) {

            double maxDiff = 0;
            for (size_t j = 0; j < numMonoms; ++j) {
                double prevW = *weightsP[j];

                double rho = 0;
                for (size_t x: columns[j]) {
                    rho += label[x] * 2 - 1 + *weightsP[j];
                    for (size_t idx: sparseTable[x]) {
                        rho -= *weightsP[idx];
                    }
                }

                if (rho < -lambda) {
                    *weightsP[j] = (rho + lambda) / columns[j].size();
                } else if (rho > lambda) {
                    *weightsP[j] = (rho - lambda) / columns[j].size();
                } else {
                    *weightsP[j] = 0;
                }

                maxDiff = Max(Abs(*weightsP[j] - prevW), maxDiff);
            }

            if (maxDiff < eps) {
                break;
            }

        }

        for (auto it = poly->MonomsEnsemble.begin(); it != poly->MonomsEnsemble.end();) {
            if (Abs(it->second.Value[0]) < std::numeric_limits<double>::epsilon()) {
                auto toDelete = it;
                ++it;
                poly->MonomsEnsemble.erase(toDelete);
            } else {
                ++it;
            }
        }

    }

    TVector<double> apply(const TPolynom& poly, const IGrid& grid, const TVector<TVector<double>>& data) {
        const auto& sparseTable = applyPolynom(data, poly, grid);

        TVector<double> res;
        res.reserve(sparseTable.size());

        TVector<const double *> weightsP;
        weightsP.reserve(poly.MonomsEnsemble.size());
        for (const auto& [_, stat] : poly.MonomsEnsemble) {
            weightsP.push_back(&stat.Value[0]);
        }

        for (const auto& row : sparseTable) {
            double y = 0;
            for (const auto& idx : row) {
                y += *weightsP[idx];
            }
            res.push_back(y);
        }

        return res;
    }


}