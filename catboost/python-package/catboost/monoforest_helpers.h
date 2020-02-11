#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/monoforest/enums.h>
#include <catboost/libs/monoforest/interpretation.h>

#include <util/generic/fwd.h>

namespace NMonoForest {
    struct THumanReadableSplit {
        int FeatureIdx;
        EBinSplitType SplitType;
        float Border;
    };

    struct THumanReadableMonom {
        TVector<THumanReadableSplit> Splits;
        TVector<double> Value;
        double Weight;
    };

    // to manage with weak support of namespaces in Cython
    using EMonoForestFeatureType = EFeatureType;

    TVector<THumanReadableMonom> ConvertFullModelToPolynom(const TFullModel& fullModel);
    TString ConvertFullModelToPolynomString(const TFullModel& fullModel);
    TVector<TFeatureExplanation> ExplainFeatures(const TFullModel& fullModel);

    TVector<TVector<double>> TestPolynomLasso(
            const TFullModel& fullModel,
            const TVector<TVector<double>>& train_data,
            const TVector<double>& train_labels,
            double lambda,
            double eps,
            size_t maxSteps,
            const TVector<TVector<double>>& test_data
    );
}
