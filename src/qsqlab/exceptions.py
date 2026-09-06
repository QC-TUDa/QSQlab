# =========================== Base ============================


# ====================== PostProcessing =======================
class PostProcessingError(Exception):
    """Base class for PostProcessing errors.
    """

class MixedDataError(PostProcessingError):
    """Data contains sequences of mixed gate or measurement basis.
    """
    pass

class UnevenDepthError(PostProcessingError):
    """Sequence contains an uneven depth of pi/2 rotations.
    """
    pass

class GateNotSupportedError(PostProcessingError):
    """Gate not supported by analysis functions.
    """
    pass