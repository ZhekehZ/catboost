#pragma once

enum class EDocumentStrengthType {
    PerObject,
    PerPool,
    Raw
};

enum class EUpdateType {
    SinglePoint,
    AllPoints,
    TopKLeaves
};
