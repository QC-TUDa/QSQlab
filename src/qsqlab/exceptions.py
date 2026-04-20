# =========================== Base ============================
class QSQLabError(Exception):
    """ Base class for qc_benchmarking_suite errors """
    pass

# ====================== PostProcessing =======================
class MixedDataException(QSQLabError):
    pass
class UnevenDepth(QSQLabError):
    pass